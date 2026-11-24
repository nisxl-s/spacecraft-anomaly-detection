"""Freeze Mission 2 thresholds and seal continuous replays before label intake."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from train_transfer import CHANNELS, ROOT, Residual, forecast, ingest

sys.path.insert(0, str(ROOT.parent / "esa-run"))
from evaluate import emit, episodes, save

SPLIT = pd.Timestamp("2001-10-01T00:00Z")
CAL_START = SPLIT - pd.DateOffset(months=3)
MODELS = ROOT / "models"


def calibrate(records):
    calibration = {}
    for channel, record in records.items():
        path = MODELS / f"{channel}_calibration.npz"
        if ingest.checksum(path) != record["calibration_sha256"]:
            raise ValueError(f"Calibration scores changed for {channel}")
        with np.load(path, allow_pickle=False) as arrays:
            eligible = arrays["eligible"]
            scores = np.full(len(eligible), np.nan)
            scores[eligible] = arrays["residual"]
        calibration[channel] = eligible, scores
    exposure = sum(int(e.sum()) for e, _ in calibration.values())
    candidates = []
    for k in range(-40, 41):
        multiplier = 10 ** (k / 40)
        count = flagged = 0
        for channel, (eligible, scores) in calibration.items():
            alarms = emit(scores > records[channel]["base"] * multiplier, eligible)
            count += len(episodes(alarms))
            flagged += int(alarms.sum())
        candidates.append(dict(multiplier=multiplier, episodes=count, flagged_hours=flagged,
                               feasible=0 < 1000 * count <= 2 * exposure and 20 * flagged <= exposure))
    feasible = [c for c in candidates if c["feasible"]]
    chosen = max(feasible, key=lambda c: (c["episodes"], -c["flagged_hours"], c["multiplier"]), default=None)
    thresholds = {c: record["base"] * chosen["multiplier"] if chosen else None for c, record in records.items()}
    return dict(training_endpoint=SPLIT.isoformat(), calibration_start=CAL_START.isoformat(),
                exposure=exposure, candidates=candidates, selected=chosen, thresholds=thresholds,
                model_records=records)


def make_replay(channel, record, threshold, series, grid):
    checkpoint = MODELS / f"{channel}.pt"
    if ingest.checksum(checkpoint) != record["checkpoint_sha256"]:
        raise ValueError(f"Weights changed for {channel}")
    model = Residual()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    start = CAL_START - pd.Timedelta(hours=ingest.HISTORY)
    times, values, eligible = ingest.prepare_hourly(series.loc[start - ingest.STALE:grid[-1]], start, grid[-1])
    signal = (values - record["center"]) / record["scale"]
    windows = np.lib.stride_tricks.sliding_window_view(signal, ingest.HISTORY)[:-1]
    keep = times >= CAL_START
    selected = np.flatnonzero(eligible & keep)
    _, predicted = forecast.calculate_scores(model, signal, windows, selected)
    observable = eligible[keep]
    scores = np.full(len(grid), np.nan)
    scores[observable] = predicted["lstm"]
    with np.load(MODELS / f"{channel}_calibration.npz", allow_pickle=False) as fitted:
        calibration = grid <= SPLIT
        np.testing.assert_array_equal(observable[calibration], fitted["eligible"])
        np.testing.assert_allclose(scores[calibration][fitted["eligible"]], fitted["residual"],
                                   rtol=1e-5, atol=1e-7)
    raw = scores > threshold if threshold is not None else np.zeros_like(observable)
    return observable, scores, raw, emit(raw, observable)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    records = {c: json.loads((MODELS / f"{c}.json").read_text()) for c in CHANNELS}
    if any(pd.Timestamp(record["split"]) != SPLIT for record in records.values()):
        raise ValueError("Training split differs from the protocol")
    freeze = calibrate(records)
    freeze_path = ROOT / "freeze.json"
    save(freeze_path, freeze)
    freeze_hash = ingest.checksum(freeze_path)
    print("Calibrated and froze the Mission 2 residual thresholds.", flush=True)
    sources = {}
    for channel, record in records.items():
        path = ROOT / "inputs" / f"{channel}.zip"
        if ingest.checksum(path) != record["source_sha256"]:
            raise ValueError(f"Source changed for {channel}")
        sources[channel] = ingest.channel_series(pd.read_pickle(path, compression="zip"), channel)
    end = min(series.index.max().floor("h") for series in sources.values())
    if end <= SPLIT:
        raise ValueError("No common post-training coverage")
    grid = pd.date_range(CAL_START, end, freq="h")
    destination = ROOT / "residual"
    destination.mkdir(exist_ok=True)
    hashes = {}
    for channel, record in records.items():
        eligible, scores, raw, emitted = make_replay(channel, record, freeze["thresholds"][channel], sources.pop(channel), grid)
        path = destination / f"{channel}_replay.npz"
        np.savez_compressed(path, eligible=eligible, positions=np.flatnonzero(eligible),
                            residual_emitted=emitted, residual_raw=raw, residual_scores=scores[eligible])
        hashes[channel] = ingest.checksum(path)
        print(f"{channel}: {int(eligible[grid > SPLIT].sum())} eligible test hours sealed.", flush=True)
    save(destination / "replay-seal.json", dict(freeze_sha256=freeze_hash, replay_sha256=hashes,
         grid_start=grid[0].isoformat(), new_start_exclusive=SPLIT.isoformat(), end_inclusive=end.isoformat()))
    ticket_file = ROOT.parent / "esa-ticket" / "run.py"
    spec = importlib.util.spec_from_file_location("mission_ticket", ticket_file)
    ticket = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ticket)
    ticket.EXT, ticket.GRID = destination, grid
    pages = {}
    for policy, quorum, window in [("opening", 0, 1), ("escalation", 3, 6)]:
        folder = ROOT / policy
        ledger = ticket.replay(folder, 24, quorum, window, None)
        pages[policy] = ingest.checksum(folder / "replay-seal.json")
        count = sum(a["page"] and grid[a["index"]] > SPLIT for t in ledger for a in t["actions"])
        print(f"{policy}: {count} test pages sealed.", flush=True)
    save(ROOT / "transfer-seal.json", dict(freeze_sha256=freeze_hash,
         residual_seal_sha256=ingest.checksum(destination / "replay-seal.json"), policy_seals=pages,
         runner_sha256=ingest.checksum(Path(__file__)), ticket_runner_sha256=ingest.checksum(ticket_file)))


if __name__ == "__main__":
    main()
