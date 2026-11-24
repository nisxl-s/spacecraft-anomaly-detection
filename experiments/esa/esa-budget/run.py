"""Calibrate a common alarm budget and evaluate the frozen ESA detectors."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import forecast
import ingest
from annotations import load_annotations
from train import PILOT, ROOT, ResidualForecast

sys.path.insert(0, str(PILOT))
from evaluate import as_of, emit, episodes, evaluate, pool, save

METHODS = ("residual", "direct", "persistence", "level")
UPLOAD = ROOT.parent / "upload"


def read(path):
    return json.loads(path.read_text())


def score_channel(path, channel, phase, direct, residual, grid):
    signal, windows, eligible, masks, timestamps = forecast.prepare_arrays(path, direct["center"], direct["scale"])
    selected = forecast.target_indices(eligible, masks[phase])
    scores = {}
    for method, model, weights, digest in [
        ("direct", forecast.Forecast(), PILOT / "models" / f"{channel}.pt", direct["checkpoint_sha256"]),
        ("residual", ResidualForecast(), ROOT / "models" / f"{channel}.pt", residual["checkpoint_sha256"]),
    ]:
        if ingest.checksum(weights) != digest:
            raise ValueError(f"Checkpoint mismatch for {channel} {method}")
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        _, calculated = forecast.calculate_scores(model, signal, windows, selected)
        scores[method] = calculated["lstm"]
        scores.update({key: calculated[key] for key in ("persistence", "level")})
    times = pd.to_datetime(timestamps[selected], unit="ns", utc=True)
    positions = grid.get_indexer(times)
    if (positions < 0).any():
        raise ValueError(f"Targets outside the frozen phase: {channel}")
    eligible = np.zeros(len(grid), dtype=bool)
    eligible[positions] = True
    return eligible, positions, scores


def alarm_flags(method, data, threshold):
    eligible, positions, scores = data
    raw = np.zeros(len(eligible), dtype=bool)
    if threshold is not None:
        raw[positions] = scores[method] > threshold
    return raw, emit(raw, eligible)


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    protocol_path = ROOT / "protocol.json"
    protocol = read(protocol_path)
    start = pd.Timestamp(protocol["test_start_exclusive"])
    stop = pd.Timestamp(protocol["test_end_inclusive"])
    fit = read(PILOT / "fit/manifest.json")
    for name, digest in fit["source_member_sha256"].items():
        if ingest.checksum(UPLOAD / name) != digest:
            raise ValueError(f"Input checksum changed: {name}")
    direct = read(PILOT / "models/manifest.json")["channels"]
    residual = {c: read(ROOT / "models" / f"{c}.json") for c in ingest.CHANNELS}
    base = {c: dict(direct[c]["base_thresholds"], direct=direct[c]["base_thresholds"]["lstm"],
                    residual=residual[c]["base_threshold"]) for c in ingest.CHANNELS}
    _, _, input_rows = forecast.read_input(PILOT / "calibration-input", "calibration")
    calibration_grid = pd.date_range(ingest.CALIBRATION_START, ingest.OFFICIAL_SPLIT, freq="h")
    calibration = {}
    for c in ingest.CHANNELS:
        calibration[c] = score_channel(PILOT / "calibration-input" / f"{c}.npz", c, "calibration",
                                       direct[c], residual[c], calibration_grid)
        np.savez_compressed(ROOT / f"{c}_calibration.npz", eligible=calibration[c][0], **calibration[c][2])
    exposure = sum(int(item[0].sum()) for item in calibration.values())
    freeze = {}
    for method in METHODS:
        candidates = []
        for multiplier in [10**(k/40) for k in range(-40, 41)] + [None]:
            thresholds = {c: base[c][method]*multiplier if multiplier is not None else None for c in ingest.CHANNELS}
            flags = [alarm_flags(method, data, thresholds[c])[1] for c, data in calibration.items()]
            count = sum(len(episodes(flag)) for flag in flags)
            flagged = sum(int(flag.sum()) for flag in flags)
            candidates.append(dict(multiplier=multiplier, thresholds=thresholds, episodes=count, flagged=flagged,
                                   feasible=1000*count <= 2*exposure and 20*flagged <= exposure))
        candidates_ok = [r for r in candidates if r["feasible"] and r["episodes"]]
        selected = max(candidates_ok, key=lambda r: (r["episodes"], -r["flagged"], r["multiplier"])) if candidates_ok else candidates[-1]
        freeze[method] = {"selected": selected, "candidates": candidates}
        print(f"{method}: {selected['episodes']} calibration episodes across {exposure} channel-hours", flush=True)
    freeze_path = ROOT / "freeze.json"
    save(freeze_path, {"calibration_exposure": exposure, "methods": freeze,
                       "protocol_sha256": ingest.checksum(protocol_path),
                       "code_sha256": {name: ingest.checksum(ROOT / name) for name in ("train.py", "run.py")},
                       "source_sha256": fit["source_member_sha256"],
                       "calibration_input_sha256": {c: row["output_sha256"] for c,row in input_rows.items()},
                       "calibration_scores_sha256": {c: ingest.checksum(ROOT / f"{c}_calibration.npz") for c in ingest.CHANNELS},
                       "direct_checkpoints": {c: direct[c]["checkpoint_sha256"] for c in ingest.CHANNELS},
                       "residual_checkpoints": {c: residual[c]["checkpoint_sha256"] for c in ingest.CHANNELS}})
    grid = pd.date_range(start + pd.Timedelta(hours=1), stop, freq="h")
    initial = grid[0] - pd.Timedelta(hours=ingest.HISTORY)
    hashes = {}
    for c in ingest.CHANNELS:
        frame = pd.read_pickle(UPLOAD / f"{c}.zip", compression="zip")
        series = ingest.channel_series(frame, c).loc[initial-ingest.STALE:stop]
        del frame
        hourly, values, eligible = ingest.prepare_hourly(series, initial, stop)
        path = ROOT / f"{c}_input.npz"
        np.savez_compressed(path, timestamp_unix_ns=hourly.as_unit("ns").asi8, values=values,
                            eligible_target=eligible, test=hourly > start)
        data = score_channel(path, c, "test", direct[c], residual[c], grid)
        outputs = dict(eligible=data[0], positions=data[1])
        for m in METHODS:
            raw, emitted = alarm_flags(m, data, freeze[m]["selected"]["thresholds"][c])
            outputs.update({f"{m}_raw": raw, f"{m}_emitted": emitted, f"{m}_scores": data[2][m]})
        path = ROOT / f"{c}_replay.npz"
        np.savez_compressed(path, **outputs)
        hashes[c] = ingest.checksum(path)
        print(f"{c}: sealed {data[0].sum()} test targets", flush=True)
    save(ROOT / "test-seal.json", {"freeze_sha256": ingest.checksum(freeze_path), "replay_sha256": hashes})
    annotations = as_of(load_annotations(UPLOAD / "labels.csv", UPLOAD / "anomaly_types.csv"), grid[0], stop)
    ledgers = {m: {} for m in METHODS}
    for c, digest in hashes.items():
        path = ROOT / f"{c}_replay.npz"
        assert ingest.checksum(path) == digest
        with np.load(path, allow_pickle=False) as data:
            for m in METHODS:
                ledgers[m][c] = evaluate(grid, data["eligible"], data[f"{m}_emitted"], data[f"{m}_raw"], annotations[c])
    save(ROOT / "ledger.json", ledgers)
    save(ROOT / "summary.json", {m: pool(rows.values()) for m, rows in ledgers.items()})
    print("Evaluation complete", flush=True)


if __name__ == "__main__":
    main()
