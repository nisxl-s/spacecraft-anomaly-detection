"""Calibrate and evaluate the frozen ESA pilot detectors."""
import dataclasses
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

import forecast
import ingest
from annotations import evaluation_inputs, load_annotations
from match_edges import match_edges

ROOT = Path(__file__).resolve().parent
UPLOAD = ROOT.parent / "upload"


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def replay(phase, fit_manifest):
    inputs = ROOT / f"{phase}-input"
    output = ROOT / f"{phase}-scores"
    inputs.mkdir(exist_ok=True)
    output.mkdir(exist_ok=True)
    channels = []
    for channel in ingest.CHANNELS:
        frame = pd.read_pickle(UPLOAD / f"{channel}.zip", compression="zip")
        channels.append(ingest.save_channel(inputs, channel, ingest.channel_series(frame, channel), phase))
    manifest = dict(fit_manifest, phase=phase, channels=channels)
    save(inputs / "manifest.json", manifest)
    forecast.read_input(inputs, phase)
    reports = forecast.replay(SimpleNamespace(models=ROOT / "models", input=inputs,
                                               output=output, stage=phase), manifest)
    save(output / "manifest.json", reports)
    print(f"{phase}: forecasts saved for {len(reports)} channels", flush=True)
    return reports


def load_scores(phase, reports, retained):
    data = {}
    for channel in retained:
        path = ROOT / f"{phase}-scores" / f"{channel}_scores.npz"
        assert ingest.checksum(path) == reports[channel]["score_sha256"]
        with np.load(path, allow_pickle=False) as npz:
            times = pd.to_datetime(npz["timestamp_unix_ns"], unit="ns", utc=True)
            grid = pd.date_range(times[0], times[-1], freq="h")
            positions = grid.get_indexer(times)
            eligible = np.zeros(len(grid), dtype=bool)
            eligible[positions] = True
            data[channel] = (grid, positions, eligible, {m: npz[f"{m}_smoothed"] for m in forecast.METHODS})
    return data


def episodes(flags):
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def emit(raw, eligible):
    result = np.zeros_like(raw)
    active = clear = 0
    locked = False
    for i in range(len(raw)):
        if not eligible[i]:
            active = clear = 0
            locked = False
        elif locked:
            clear = 0 if raw[i] else clear + 1
            if clear == 5:
                locked = False
                active = clear = 0
        elif raw[i]:
            result[i] = True
            active += 1
            locked = active == 30
        else:
            active = 0
    return result


def evaluate(grid, eligible, emitted, raw, annotations):
    observable, flags, rare, truths = evaluation_inputs(grid, eligible, emitted, annotations)
    alarms = episodes(flags)
    supported = sorted((np.flatnonzero(mask)[0].item(), identity, mask)
                       for identity, mask in truths.items() if mask.any())
    edges = []
    for ti, (onset, identity, mask) in enumerate(supported):
        for ai, (start, stop) in enumerate(alarms):
            overlap = np.flatnonzero(mask[start:stop])
            if overlap.size:
                edges.append((onset, start, ti, ai, ti, ai, start + int(overlap[0]) - onset))
    matches = match_edges(sorted(edges), len(supported), len(alarms))
    return dict(tp=len(matches), fp=len(alarms)-len(matches), fn=len(supported)-len(matches),
                exposure=int(observable.sum()), flagged=int(flags.sum()),
                raw_high=int(raw.sum()), emitted=int(emitted.sum()),
                streaming_exposure=int(eligible.sum()), streaming_episode_durations=[b-a for a,b in episodes(emitted)],
                rare_overlap=sum(bool(rare[a:b].any()) for a,b in alarms),
                excluded_events=sorted(identity for identity, mask in truths.items() if not mask.any()),
                events=[identity for _, identity, _ in supported], alarms=alarms,
                grid_start=grid[0].isoformat(), matches=matches)


def pool(rows):
    out = {k: sum(row[k] for row in rows) for k in ("tp", "fp", "fn", "exposure", "flagged", "emitted", "raw_high", "streaming_exposure", "rare_overlap")}
    positives, alerts = out["tp"] + out["fn"], out["tp"] + out["fp"]
    delays = [match[2] for row in rows for match in row["matches"]]
    out.update(recall=out["tp"]/positives if positives else None,
               precision=out["tp"]/alerts if alerts else None,
               false_per_1000=1000*out["fp"]/out["exposure"] if out["exposure"] else None,
               flagged_fraction=out["flagged"]/out["exposure"] if out["exposure"] else None,
               median_delay_hours=statistics.median(delays) if delays else None)
    return out


def as_of(annotations, start, stop):
    return {channel: [dataclasses.replace(row, start=max(start, row.start), end=min(stop, row.end))
                      for row in rows if row.start <= stop and row.end >= start]
            for channel, rows in annotations.items()}


def run_candidate(method, multiplier, data, reports, annotations, fixed=None):
    details = {}
    for channel, (grid, positions, eligible, scores) in data.items():
        threshold = None if multiplier is None else reports[channel]["base_thresholds"][method] * multiplier
        raw = np.zeros(len(grid), dtype=bool)
        if threshold is not None:
            raw[positions] = scores[method] > threshold
        emitted = emit(raw, eligible)
        if fixed is not None:
            fixed[channel][f"{method}_raw"] = raw
            fixed[channel][f"{method}_emitted"] = emitted
        if annotations is not None:
            details[channel] = evaluate(grid, eligible, emitted, raw, annotations[channel])
    return details


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    fit_manifest = json.loads((ROOT / "fit/manifest.json").read_text())
    for name, digest in fit_manifest["source_member_sha256"].items():
        assert ingest.checksum(UPLOAD / name) == digest
    calibration = replay("calibration", fit_manifest)
    retained = [c for c, report in calibration.items() if "excluded" not in report]
    if not retained:
        raise ValueError("No channels have sufficient calibration exposure")
    data = load_scores("calibration", calibration, retained)
    # The source CSV spans both periods. Only calibration intervals enter selection.
    annotations = load_annotations(UPLOAD / "labels.csv", UPLOAD / "anomaly_types.csv")
    cal_labels = as_of(annotations, ingest.CALIBRATION_START, ingest.OFFICIAL_SPLIT)
    del annotations
    freeze = {}
    for method in forecast.METHODS:
        candidates = []
        for multiplier in [10**(k/40) for k in range(-40, 41)] + [None]:
            metrics = pool(run_candidate(method, multiplier, data, calibration, cal_labels).values())
            feasible = (metrics["exposure"] > 0 and 1000*metrics["fp"] <= 2*metrics["exposure"]
                        and 20*metrics["flagged"] <= metrics["exposure"])
            candidates.append(dict(multiplier=multiplier, feasible=feasible, **metrics))
        finite = [r for r in candidates if r["multiplier"] is not None and r["feasible"]
                  and r["emitted"] and r["recall"] is not None]
        winner = max(finite, key=lambda r: (r["recall"], -r["flagged_fraction"], -r["false_per_1000"], r["multiplier"])) if finite else candidates[-1]
        thresholds = {c: None if winner["multiplier"] is None else calibration[c]["base_thresholds"][method]*winner["multiplier"] for c in retained}
        freeze[method] = dict(selected=winner, thresholds=thresholds, candidates=candidates)
    freeze_path = ROOT / "threshold-freeze.json"
    save(freeze_path, freeze)
    freeze_hash = ingest.checksum(freeze_path)
    print("Thresholds frozen; replaying held-out period", flush=True)
    test = replay("test", fit_manifest)
    if any("excluded" in test[c] for c in retained):
        raise ValueError("A retained channel has no test exposure")
    data = load_scores("test", test, retained)
    flags = {c: {} for c in retained}
    for method in forecast.METHODS:
        run_candidate(method, freeze[method]["selected"]["multiplier"], data, test, None, flags)
    digests = {}
    for channel, arrays in flags.items():
        path = ROOT / "test-scores" / f"{channel}_flags.npz"
        np.savez_compressed(path, **arrays)
        digests[channel] = ingest.checksum(path)
    save(ROOT / "test-emissions.json", {"freeze_sha256": freeze_hash, "flags_sha256": digests,
                                      "scores_sha256": {c: test[c]["score_sha256"] for c in retained}})
    annotations = load_annotations(UPLOAD / "labels.csv", UPLOAD / "anomaly_types.csv")
    test_labels = as_of(annotations, ingest.OFFICIAL_SPLIT + pd.Timedelta(hours=1), ingest.TEST_END)
    ledgers, summary = {}, {}
    for method in forecast.METHODS:
        ledgers[method] = {}
        for c, (grid, _, eligible, _) in data.items():
            path = ROOT / "test-scores" / f"{c}_flags.npz"
            assert ingest.checksum(path) == digests[c]
            with np.load(path) as saved:
                ledgers[method][c] = evaluate(grid, eligible, saved[f"{method}_emitted"],
                                              saved[f"{method}_raw"], test_labels[c])
        summary[method] = pool(ledgers[method].values())
    save(ROOT / "test-ledger.json", ledgers)
    save(ROOT / "test-summary.json", {"channels": retained, "metrics": summary})
    print("Held-out scoring complete", flush=True)


if __name__ == "__main__":
    main()
