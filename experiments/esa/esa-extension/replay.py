"""Extend the sealed ESA replay with frozen checkpoints and thresholds."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
BUDGET = ROOT.parent / "esa-budget"
sys.path.insert(0, str(BUDGET))
from run import METHODS, PILOT, UPLOAD, alarm_flags, ingest, read, save, score_channel

WARMUP = pd.Timestamp("2008-01-01T01:00Z")
BOUNDARY = pd.Timestamp("2009-01-01T00:00Z")
END = pd.Timestamp("2013-12-31T23:00Z")
GRID = pd.date_range(WARMUP, END, freq="h")


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    freeze = read(BUDGET / "freeze.json")
    prior = read(BUDGET / "test-seal.json")
    freeze_hash = ingest.checksum(BUDGET / "freeze.json")
    assert freeze_hash == prior["freeze_sha256"]
    direct = read(PILOT / "models/manifest.json")["channels"]
    for name, digest in freeze["source_sha256"].items():
        assert ingest.checksum(UPLOAD / name) == digest
    paths = {}
    for channel in ingest.CHANNELS:
        residual = read(BUDGET / "models" / f"{channel}.json")
        assert residual["checkpoint_sha256"] == freeze["residual_checkpoints"][channel]
        assert direct[channel]["checkpoint_sha256"] == freeze["direct_checkpoints"][channel]
        frame = pd.read_pickle(UPLOAD / f"{channel}.zip", compression="zip")
        series = ingest.channel_series(frame, channel)
        assert series.index.max() >= END
        start = WARMUP - pd.Timedelta(hours=ingest.HISTORY)
        grid, values, eligible = ingest.prepare_hourly(series.loc[start-ingest.STALE:END], start, END)
        del frame, series
        inputs = ROOT / f"{channel}_input.npz"
        np.savez_compressed(inputs, timestamp_unix_ns=grid.as_unit("ns").asi8, values=values,
                            eligible_target=eligible, test=grid >= WARMUP)
        data = score_channel(inputs, channel, "test", direct[channel], residual, GRID)
        outputs = dict(eligible=data[0], positions=data[1])
        old_path = BUDGET / f"{channel}_replay.npz"
        assert ingest.checksum(old_path) == prior["replay_sha256"][channel]
        length = int((GRID <= BOUNDARY).sum())
        with np.load(old_path, allow_pickle=False) as old:
            np.testing.assert_array_equal(outputs["eligible"][:length], old["eligible"])
            for method in METHODS:
                threshold = freeze["methods"][method]["selected"]["thresholds"][channel]
                raw = np.zeros(len(GRID), dtype=bool)
                scores = data[2][method]
                # The longer replay may change the last floating-point batch.
                # Preserve the exact sealed prefix; verify the new prefix's decisions.
                candidate, _ = alarm_flags(method, data, threshold)
                np.testing.assert_array_equal(candidate[:length], old[f"{method}_raw"])
                raw[:] = candidate
                from evaluate import emit
                emitted = emit(raw, data[0])
                np.testing.assert_array_equal(emitted[:length], old[f"{method}_emitted"])
                outputs.update({f"{method}_raw": raw, f"{method}_emitted": emitted,
                                f"{method}_scores": scores})
        path = ROOT / f"{channel}_replay.npz"
        np.savez_compressed(path, **outputs)
        paths[channel] = ingest.checksum(path)
        print(f"{channel}: {data[0][GRID > BOUNDARY].sum()} new eligible hours", flush=True)
    save(ROOT / "replay-seal.json", {"freeze_sha256": freeze_hash, "prior_seal_sha256": ingest.checksum(BUDGET / "test-seal.json"),
                                    "replay_sha256": paths, "new_start_exclusive": BOUNDARY.isoformat(),
                                    "end_inclusive": END.isoformat()})


if __name__ == "__main__":
    main()
