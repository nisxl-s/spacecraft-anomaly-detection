"""Calibrate baselines, persist held-out scores, and score eligible benchmark channels."""

import argparse
import json
from math import ceil
from pathlib import Path
import zipfile

import numpy as np
import torch

from intake import file_hash, parse_metadata
from pilot_baselines import smooth
from pilot_lstm import Forecast, residuals
from replay_p1 import FREEZE_SHA256, MANIFEST_SHA256, detector_ledger, digest, load_array, read_verified, require


def baseline_scores(train, test, channel):
    fit_end = channel["train_row_ranges_half_open"]["fit"][1]
    first, last = channel["train_row_ranges_half_open"]["calibration"]
    start, stop = channel["test_target_range_half_open"]
    fit = train[:fit_end]
    mean, scale = fit.mean(), max(fit.std(), 1e-8)
    calibration = {
        "level": np.abs(train[first:last] - mean) / scale,
        "persistence": np.abs(np.diff(train[first - 1:last])),
    }
    index = ceil(0.99 * (last - first)) - 1
    thresholds = {name: float(np.sort(smooth(score))[index]) for name, score in calibration.items()}
    return thresholds, {
        "level": smooth(np.abs(test[start:stop] - mean) / scale),
        "persistence": smooth(np.abs(np.diff(test[start - 1:stop]))),
    }


def lstm_scores(test, start, path, record, channel, expected_config):
    report = json.loads(path.with_suffix(".json").read_bytes())
    require(report["configuration"] == expected_config, f"training configuration mismatch: {path}")
    require(report["training_member_sha256"] == record["sha256"], f"training data mismatch: {path}")
    require(report["target_ranges_half_open"] == channel["train_target_ranges_half_open"],
            f"split mismatch: {path}")
    read_verified(path, report["checkpoint"]["sha256"])
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    require(checkpoint["input_features"] == test.shape[1]
            and checkpoint["training_member_sha256"] == record["sha256"]
            and checkpoint["best_epoch"] == report["best_epoch"]
            and checkpoint["threshold"] == report["threshold"], f"checkpoint metadata mismatch: {path}")
    model = Forecast(test.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    features = torch.from_numpy(np.ascontiguousarray(test, dtype=np.float32))
    targets = torch.from_numpy(test[:, 0].astype(float))
    windows = features.unfold(0, start, 1)[:-1].transpose(1, 2)
    return report["threshold"], smooth(residuals(model, windows, targets, torch.arange(start, len(test))))


def run(args):
    raw_manifest = read_verified(args.manifest, MANIFEST_SHA256)
    manifest = json.loads(raw_manifest)
    require(file_hash(args.archive) == manifest["archive"]["sha256"], "archive hash mismatch")
    channels = [row for row in manifest["channels"] if row["eligible"]]
    records = {row["member"]: row for row in manifest["arrays"]}
    config = None
    if args.models:
        freeze = json.loads(args.pilot.read_bytes())["pilot_freeze"]
        require(digest(json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode())
                == FREEZE_SHA256, "pilot freeze hash mismatch")
        config = freeze["lstm"]
    if args.models:
        for row in channels:
            stem = args.models / row["mission"] / row["channel"]
            require(stem.with_suffix(".pt").is_file() and stem.with_suffix(".json").is_file(),
                    f"missing trained model: {stem}")
    args.output.mkdir()
    score_dir = args.output / "scores"
    score_dir.mkdir()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    staged = []
    with zipfile.ZipFile(args.archive) as archive:
        for channel in channels:
            mission, name = channel["mission"], channel["channel"]
            start, stop = channel["test_target_range_half_open"]
            train_rec = records[channel["array_members"]["train"]]
            train = load_array(archive, train_rec)
            test = load_array(archive, records[channel["array_members"]["test"]])
            require(len(test) == stop and start == 250, f"invalid test range: {name}")
            thresholds, scores = baseline_scores(train[:, 0].astype(float), test[:, 0].astype(float), channel)
            if args.models:
                thresholds["lstm"], scores["lstm"] = lstm_scores(
                    test, start, args.models / mission / (name + ".pt"), train_rec, channel, config)
            require(all(len(x) == stop - start and np.isfinite(x).all() for x in scores.values())
                    and all(np.isfinite(x) for x in thresholds.values()), f"invalid scores: {name}")
            flags = {key: score > thresholds[key] for key, score in scores.items()}
            path = score_dir / f"{mission}-{name}.npz"
            np.savez_compressed(path, index=np.arange(start, stop), telemetry=test[start:stop, 0],
                                **{f"{k}_score": x for k, x in scores.items()},
                                **{f"{k}_threshold": t for k, t in thresholds.items()},
                                **{f"{k}_flag": x for k, x in flags.items()})
            staged.append((channel, path, file_hash(path), thresholds))

    labels = parse_metadata(read_verified(args.labels, manifest["labels"]["sha256"]))
    by_key = {(row["mission"], row["channel"]): row for row in labels}
    results = []
    for channel, path, sha, thresholds in staged:
        mission, name = channel["mission"], channel["channel"]
        label = by_key[mission, name]
        start, stop = channel["test_target_range_half_open"]
        events = sorted(label["events"], key=lambda e: e["source_event_order"])
        expected = sorted(channel["events"], key=lambda e: e["source_event_order"])
        require(not label["issues"] and label["declared_test_length"] == stop
                and events == [{k: e[k] for k in ("start", "end", "class", "source_event_order")}
                               for e in expected], f"labels changed: {mission}/{name}")
        read_verified(path, sha)
        with np.load(path, allow_pickle=False) as stored:
            detectors = {key: {"threshold": value, **detector_ledger(stored[f"{key}_flag"], start, stop, events)}
                         for key, value in thresholds.items()}
        results.append({"channel": f"{mission}/{name}", "mission": mission,
                        "scores": {"filename": path.name, "sha256": sha}, "detectors": detectors})
    (args.output / "ledger.json").write_text(json.dumps({"channels": results}, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "manifest", "labels", "output"):
        parser.add_argument(f"--{field}", required=True, type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--pilot", type=Path)
    args = parser.parse_args()
    if bool(args.models) != bool(args.pilot):
        parser.error("--models and --pilot must be supplied together")
    run(args)
