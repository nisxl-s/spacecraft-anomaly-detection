"""Replay frozen SMAP/MSL forecasts, then score labeled events.

Exploratory evaluation: prior project work used this official test split.
The completed replay verified 78 channels, 234 detector ledgers, and
472,832 scored readings. On 100 labeled events, direct LSTM detected 76
(2,196 unmatched alarms), residual LSTM detected 77 (2,116), and persistence
detected 69 (1,076). Median delay was 26, 26, and 38 readings respectively.

Supply the pinned intake manifest, official telemetry archive and label CSV,
original direct-model directory, and the completed residual-screen ZIP.
The output directory receives score arrays, the replay manifest, the event
ledger, and pooled and mission-level summary JSON.
"""
import argparse
import io
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from event_scoring import score_events
from intake import parse_metadata
from residual_screen import ARMS, MANIFEST_SHA, ResidualForecast, confirmed_episodes, init, recipe, save_json, sha, smooth

def replay(job):
    name, channel, record, row, args = job
    stem = name.replace("/", "_")
    destination = (args.output / "scores") / f"{stem}.npz"
    with ZipFile(args.archive) as archive, ZipFile(args.checkpoint) as checkpoint:
        raw = archive.read(record["member"])
        if sha(raw) != record["sha256"]:
            raise ValueError(f"Test hash mismatch for {name}")
        test = np.load(io.BytesIO(raw), allow_pickle=False)
        residual_weights = checkpoint.read(f"results/{stem}.pt")
        calibration = checkpoint.read(f"results/{stem}.npz")
    start, stop = channel["test_target_range_half_open"]
    if (start, stop) != (recipe.WINDOW, len(test)):
        raise ValueError(f"Invalid test bounds for {name}")
    thresholds = {arm: row["detectors"][arm]["threshold"] for arm in ARMS}
    with np.load(io.BytesIO(calibration), allow_pickle=False) as errors:
        for arm, threshold in thresholds.items():
            scores = np.sort(smooth(errors[f"calibration_{arm}"]))
            if scores[math.ceil(0.99 * len(scores)) - 1] != threshold:
                raise ValueError(f"Calibration mismatch for {name} {arm}")
    direct_weights = (args.models / name).with_suffix(".pt").read_bytes()
    if sha(direct_weights) != row["direct_checkpoint_sha256"]:
        raise ValueError(f"Direct checkpoint mismatch for {name}")
    features = torch.from_numpy(np.ascontiguousarray(test, dtype=np.float32))
    target = torch.from_numpy(test[:, 0].astype(np.float64))
    windows = features.unfold(0, start, 1)[:-1].transpose(1, 2)
    indices = torch.arange(start, stop)
    direct = recipe.Forecast(features.shape[1])
    direct.load_state_dict(torch.load(io.BytesIO(direct_weights), weights_only=True)["model_state_dict"])
    residual = ResidualForecast(features.shape[1])
    residual.load_state_dict(torch.load(io.BytesIO(residual_weights), weights_only=True))
    errors = {
        "direct": recipe.residuals(direct, windows, target, indices),
        "residual": recipe.residuals(residual, windows, target, indices),
        "persistence": (target[indices] - target[indices - 1]).abs().numpy(),
    }
    arrays = {}
    for arm, error in errors.items():
        scores = smooth(error)
        if len(scores) != stop - start or not np.isfinite(scores).all():
            raise ValueError(f"Invalid scores for {name} {arm}")
        arrays[f"{arm}_scores"] = scores
        arrays[f"{arm}_flags"] = confirmed_episodes(scores > thresholds[arm])
    pending = destination.with_suffix(".tmp.npz")
    np.savez_compressed(pending, index=np.arange(start, stop), **arrays)
    pending.replace(destination)
    return name, sha(destination.read_bytes()), thresholds


def score(channel, stage, label, args):
    name, digest, thresholds = stage
    start, stop = channel["test_target_range_half_open"]
    if label["issues"] or label["declared_test_length"] != stop:
        raise ValueError(f"Invalid labels for {name}")
    truths = [(event["start"], event["end"]) for event in label["events"]]
    raw = ((args.output / "scores") / f"{name.replace('/', '_')}.npz").read_bytes()
    if sha(raw) != digest:
        raise ValueError(f"Replay changed for {name}")
    detectors = {}
    with np.load(io.BytesIO(raw), allow_pickle=False) as records:
        for arm in ARMS:
            flags = records[f"{arm}_flags"]
            edges = np.flatnonzero(np.diff(np.r_[False, flags, False])) + start
            alarms = [(int(a), int(b - 1)) for a, b in zip(edges[::2], edges[1::2], strict=True)]
            result = score_events(truths, alarms, start, stop - 1)
            detectors[arm] = dict(result, alarms=alarms, threshold=thresholds[arm])
    return dict(channel=name, readings=stop-start, truths=truths, detectors=detectors)


def summarize(rows):
    readings = sum(row["readings"] for row in rows)
    result = {}
    for arm in ARMS:
        detectors = [row["detectors"][arm] for row in rows]
        tp, fp, fn = (sum(record[key] for record in detectors) for key in ("tp", "fp", "fn"))
        delays = [match[2] for record in detectors for match in record["matches"]]
        result[arm] = dict(tp=tp, fp=fp, fn=fn, recall=tp/(tp+fn) if tp+fn else None,
                           false_episodes_per_1000=1000*fp/readings,
                           median_delay=statistics.median(delays) if delays else None,
                           excluded_truths=sum(len(record["excluded_truths"]) for record in detectors),
                           clipped_truths=sum(len(record["partly_clipped_truths"]) for record in detectors))
    return dict(channels=len(rows), readings=readings, detectors=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("manifest", "archive", "labels", "models", "checkpoint", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    raw = args.manifest.read_bytes()
    if sha(raw) != MANIFEST_SHA:
        raise ValueError("Intake manifest changed")
    manifest = json.loads(raw)
    channels = {f"{c['mission']}/{c['channel']}": c for c in manifest["channels"] if c["eligible"]}
    records = {r["member"]: r for r in manifest["arrays"]}
    with ZipFile(args.checkpoint) as checkpoint:
        rows = [json.loads(checkpoint.read(path)) for path in checkpoint.namelist()
                if path.startswith("results/") and path.endswith(".json") and not path.endswith("/summary.json")]
    if len(rows) != 78 or {row["channel"] for row in rows} != channels.keys():
        raise ValueError("Checkpoint and test cohorts differ")
    jobs = [(name, channels[name], records[channels[name]["array_members"]["test"]], row, args)
            for row in rows for name in [row["channel"]]]
    (args.output / "scores").mkdir(parents=True, exist_ok=False)
    print(f"Replaying {len(jobs)} channels with frozen thresholds", flush=True)
    staged = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init, initargs=(1,)) as pool:
        from concurrent.futures import as_completed
        for future in as_completed([pool.submit(replay, job) for job in jobs]):
            staged.append(future.result())
            print(f"{len(staged)}/78 {staged[-1][0]}", flush=True)
    save_json(args.output / "replay-manifest.json", sorted(staged))
    raw_labels = args.labels.read_bytes()
    if sha(raw_labels) != manifest["labels"]["sha256"]:
        raise ValueError("Test labels changed")
    labels = {f"{r['mission']}/{r['channel']}": r for r in parse_metadata(raw_labels)}
    scored = [score(channels[stage[0]], stage, labels[stage[0]], args) for stage in sorted(staged)]
    save_json(args.output / "ledger.json", scored)
    summary = {mission: summarize([row for row in scored if mission == "overall" or row["channel"].startswith(mission + "/")])
               for mission in ("overall", "SMAP", "MSL")}
    save_json(args.output / "summary.json", summary)
    print("Scored all channels against pinned labels", flush=True)


if __name__ == "__main__":
    main()
