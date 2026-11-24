"""Trace frozen Mission 2 score and alarm decisions for the four misses."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
MISSION = ROOT.parent / "esa-mission2"
sys.path.insert(0, str(MISSION))
import replay_transfer as replay
import annotations

annotations.CHANNELS = replay.CHANNELS


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    seal = replay.json.loads((MISSION / "residual/replay-seal.json").read_text())
    freeze = replay.json.loads((MISSION / "freeze.json").read_text())
    assert replay.ingest.checksum(MISSION / "freeze.json") == seal["freeze_sha256"]
    grid = pd.date_range(seal["grid_start"], seal["end_inclusive"], freq="h")
    checks = replay.json.loads((MISSION / "label-inputs.json").read_text())
    for name, digest in checks.items():
        assert replay.ingest.checksum(MISSION / "inputs" / name) == digest
    labels = annotations.load_annotations(MISSION / "inputs/labels.csv", MISSION / "inputs/anomaly_types.csv")
    scored = replay.json.loads((MISSION / "residual/ledger.json").read_text())["residual"]
    calculate = replay.forecast.calculate_scores
    captured = {}

    def capture(*args):
        raw, smoothed = calculate(*args)
        captured["error"] = raw["lstm"]
        return raw, smoothed

    replay.forecast.calculate_scores = capture
    rows = []
    for c, ledger in scored.items():
        if not ledger["events"]:
            continue
        record = freeze["model_records"][c]
        source = MISSION / "inputs" / f"{c}.zip"
        assert replay.ingest.checksum(source) == record["source_sha256"]
        series = replay.ingest.channel_series(pd.read_pickle(source, compression="zip"), c)
        threshold = freeze["thresholds"][c]
        eligible, smoothed, above, emitted = replay.make_replay(c, record, threshold, series, grid)
        path = MISSION / "residual" / f"{c}_replay.npz"
        assert replay.ingest.checksum(path) == seal["replay_sha256"][c]
        with np.load(path, allow_pickle=False) as prior:
            np.testing.assert_array_equal(eligible, prior["eligible"])
            np.testing.assert_array_equal(above, prior["residual_raw"])
            np.testing.assert_array_equal(emitted, prior["residual_emitted"])
            np.testing.assert_allclose(smoothed[eligible], prior["residual_scores"], rtol=1e-5, atol=1e-7)
        raw = np.full(len(grid), np.nan)
        raw[eligible] = captured["error"]
        calibration = eligible & (grid <= replay.SPLIT)
        cal_raw, cal_smooth = raw[calibration], smoothed[calibration]
        _, _, _, truths = annotations.evaluation_inputs(grid, eligible, emitted, labels[c])
        for identity in ledger["events"]:
            mask = truths[identity] & (grid > replay.SPLIT)
            timestamps = grid[mask]
            row = dict(channel=c, incident=identity, first=timestamps[0].isoformat(),
                       last=timestamps[-1].isoformat(), eligible_hours=int(mask.sum()),
                       raw_peak=float(raw[mask].max()), raw_calibration_p99=float(np.quantile(cal_raw,.99)),
                       raw_peak_calibration_percentile=float((cal_raw <= raw[mask].max()).mean()),
                       smoothed_peak=float(smoothed[mask].max()), smoothed_calibration_p99=float(np.quantile(cal_smooth,.99)),
                       threshold=threshold, peak_threshold_ratio=float(smoothed[mask].max()/threshold),
                       above=int(above[mask].sum()), emitted=int(emitted[mask].sum()),
                       suppressed_high=int((above & ~emitted & mask).sum()))
            rows.append(row)
            print(row, flush=True)
            np.savez_compressed(ROOT / f"{c}_trace.npz", timestamps=grid.as_unit("ns").asi8,
                                raw=raw, smoothed=smoothed, eligible=eligible, event=mask,
                                exceeds=above, emitted=emitted)
    assert len(rows) == 4
    replay.save(ROOT / "trace.json", rows)


if __name__ == "__main__":
    main()
