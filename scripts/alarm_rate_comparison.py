"""Retrospectively compare frozen detectors at training-side alarm-rate anchors."""

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from intake import file_hash
from pilot_baselines import smooth
from pilot_lstm import Forecast, WINDOW, residuals
from replay_p1 import (
    FREEZE_SHA256, MANIFEST_SHA256, detector_ledger, digest, load_array, read_verified, require,
)
from summarize_benchmark import summarize

BASELINES = ("level", "persistence")
DETECTORS = ("level", "persistence", "lstm")
FROZEN_PERCENTILE_BPS = 9900
GRID_BPS = np.arange(9000, 10001)


def quantile_thresholds(scores, ranks_bps):
    ranks = (len(scores) * ranks_bps + 9999) // 10000
    return np.sort(scores)[ranks - 1]


def episode_counts(scores, thresholds):
    flags = scores[:, None] > thresholds
    return flags[0].astype(int) + (flags[1:] & ~flags[:-1]).sum(axis=0)


def lstm_calibration(train, first, last, checkpoint_path, channel, record, config):
    report = json.loads(checkpoint_path.with_suffix(".json").read_bytes())
    require(report["configuration"] == config, f"training configuration mismatch: {checkpoint_path}")
    require(report["training_member_sha256"] == record["sha256"], f"training data mismatch: {checkpoint_path}")
    require(report["target_ranges_half_open"] == channel["train_target_ranges_half_open"],
            f"split mismatch: {checkpoint_path}")
    read_verified(checkpoint_path, report["checkpoint"]["sha256"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["input_features"] == train.shape[1]
            and checkpoint["training_member_sha256"] == record["sha256"]
            and checkpoint["best_epoch"] == report["best_epoch"]
            and checkpoint["threshold"] == report["threshold"], f"checkpoint metadata mismatch: {checkpoint_path}")
    model = Forecast(train.shape[1])
    model.load_state_dict(checkpoint["model_state_dict"])
    features = torch.from_numpy(np.ascontiguousarray(train, dtype=np.float32))
    targets = torch.from_numpy(train[:, 0].astype(float))
    windows = features.unfold(0, WINDOW, 1)[:-1].transpose(1, 2)
    scores = smooth(residuals(model, windows, targets, torch.arange(first, last)))
    np.testing.assert_allclose(quantile_thresholds(scores, FROZEN_PERCENTILE_BPS), report["threshold"],
                               rtol=1e-6, atol=1e-7)
    return scores, report["threshold"], report["checkpoint"]["sha256"]


def calibrate(args):
    manifest = json.loads(read_verified(args.manifest, MANIFEST_SHA256))
    require(file_hash(args.archive) == manifest["archive"]["sha256"], "archive hash mismatch")
    channels = [row for row in manifest["channels"] if row["eligible"]]
    records = {row["member"]: row for row in manifest["arrays"]}
    freeze = json.loads(args.pilot.read_bytes())["pilot_freeze"]
    require(digest(json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode())
            == FREEZE_SHA256, "pilot freeze hash mismatch")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    lstm_grid_counts = np.zeros(len(GRID_BPS), dtype=int)
    anchors = dict.fromkeys(BASELINES, 0)
    rows = {}
    with zipfile.ZipFile(args.archive) as archive:
        for channel in channels:
            mission, name = channel["mission"], channel["channel"]
            key = f"{mission}/{name}"
            record = records[channel["array_members"]["train"]]
            train = load_array(archive, record)
            signal = train[:, 0].astype(float)
            fit_end = channel["train_row_ranges_half_open"]["fit"][1]
            first, last = channel["train_target_ranges_half_open"]["calibration"]
            fit = signal[:fit_end]
            scores = {
                "level": smooth(np.abs(signal[first:last] - fit.mean()) / max(fit.std(), 1e-8)),
                "persistence": smooth(np.abs(np.diff(signal[first - 1:last]))),
            }
            checkpoint_path = args.models / mission / f"{name}.pt"
            scores["lstm"], frozen_lstm_threshold, model_hash = lstm_calibration(
                train, first, last, checkpoint_path, channel, record, freeze["lstm"])
            frozen = {key: float(quantile_thresholds(values, FROZEN_PERCENTILE_BPS))
                      for key, values in scores.items()}
            frozen["lstm"] = frozen_lstm_threshold
            counts = {key: int(episode_counts(values, [frozen[key]])[0]) for key, values in scores.items()}
            for base in BASELINES:
                anchors[base] += counts[base]
            thresholds = quantile_thresholds(scores["lstm"], GRID_BPS)
            lstm_grid_counts += episode_counts(scores["lstm"], thresholds)
            rows[key] = {
                "calibration_readings": last - first,
                "checkpoint_sha256": model_hash,
                "frozen_thresholds": frozen,
                "frozen_episode_counts": counts,
                "lstm_calibration_scores": scores["lstm"],
            }
    chosen = {}
    for base, anchor in anchors.items():
        differences = np.abs(lstm_grid_counts - anchor)
        index = int(np.flatnonzero(differences == differences.min())[-1])
        chosen[base] = {
            "lstm_percentile_bps": int(GRID_BPS[index]),
            "baseline_episodes": anchor,
            "lstm_episodes": int(lstm_grid_counts[index]),
        }
    for row in rows.values():
        scores = row.pop("lstm_calibration_scores")
        row["matched_lstm"] = {}
        for base, operating_point in chosen.items():
            threshold = float(quantile_thresholds(scores, operating_point["lstm_percentile_bps"]))
            row["matched_lstm"][base] = {
                "threshold": threshold,
                "episodes": int(episode_counts(scores, [threshold])[0]),
            }
    exposure = sum(row["calibration_readings"] for row in rows.values())
    for base, op in chosen.items():
        require(op["lstm_episodes"] == sum(row["matched_lstm"][base]["episodes"] for row in rows.values()),
                f"calibration total mismatch: {base}")
        op["baseline_per_1000"] = 1000 * op["baseline_episodes"] / exposure
        op["lstm_per_1000"] = 1000 * op["lstm_episodes"] / exposure
    return {
        "method": "Minimize absolute pooled calibration alarm-episode count difference to each frozen baseline. "
                  "Use a shared LSTM percentile, with channel-specific thresholds and highest-percentile ties.",
        "grid_bps": [int(GRID_BPS[0]), int(GRID_BPS[-1]), 1],
        "manifest_sha256": MANIFEST_SHA256,
        "pilot_freeze_sha256": FREEZE_SHA256,
        "calibration_readings": exposure,
        "operating_points": chosen,
        "channels": rows,
    }


def evaluate(args, frozen):
    original = json.loads((args.replay / "ledger.json").read_bytes())["channels"]
    require(set(frozen["channels"]) == {row["channel"] for row in original}
            and len(frozen["channels"]) == len(original), "calibration and replay channel mismatch")
    results = []
    for row in original:
        key, mission = row["channel"], row["mission"]
        channel = frozen["channels"][key]
        score_path = args.replay / "scores" / row["scores"]["filename"]
        read_verified(score_path, row["scores"]["sha256"])
        events = [
            {field: event[field] for field in ("start", "end", "class", "source_event_order")}
            for event in row["detectors"]["level"]["truths"]
        ]
        with np.load(score_path, allow_pickle=False) as stored:
            indices = stored["index"]
            start, stop = int(indices[0]), int(indices[-1]) + 1
            np.testing.assert_array_equal(indices, np.arange(start, stop))
            for detector in DETECTORS:
                threshold = channel["frozen_thresholds"][detector]
                require(threshold == stored[f"{detector}_threshold"].item()
                        == row["detectors"][detector]["threshold"], f"frozen threshold changed: {key} {detector}")
                np.testing.assert_array_equal(stored[f"{detector}_score"] > threshold, stored[f"{detector}_flag"])
            matched = {}
            for base in BASELINES:
                threshold = channel["matched_lstm"][base]["threshold"]
                flags = stored["lstm_score"] > threshold
                matched[base] = {"threshold": threshold, **detector_ledger(flags, start, stop, events)}
        results.append({"channel": key, "mission": mission, "baseline": {key: row["detectors"][key] for key in BASELINES},
                        "matched_lstm": matched, "scores": row["scores"]})
    groups = {"overall": results, **{
        mission: [row for row in results if row["mission"] == mission]
        for mission in sorted({row["mission"] for row in results})
    }, "without_pilot_P-1": [row for row in results if row["channel"] != "SMAP/P-1"]}
    summary = {}
    for group, rows in groups.items():
        summary[group] = {}
        for base in BASELINES:
            baseline = summarize([row["baseline"][base] for row in rows])
            lstm = summarize([row["matched_lstm"][base] for row in rows])
            summary[group][base] = {"baseline": baseline, "lstm": lstm}
    return results, summary


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibrate", "evaluate"))
    for field in ("archive", "manifest", "models", "pilot", "replay", "output"):
        parser.add_argument(f"--{field}", type=Path)
    args = parser.parse_args()
    if args.output is None:
        parser.error("--output is required")
    frozen_path = args.output / "calibration.json"
    if args.phase == "calibrate":
        if any(getattr(args, name) is None for name in ("archive", "manifest", "models", "pilot")):
            parser.error("calibration requires --archive, --manifest, --models and --pilot")
        args.output.mkdir()
        write_json(frozen_path, calibrate(args))
    else:
        if args.replay is None:
            parser.error("evaluation requires --replay")
        frozen_bytes = frozen_path.read_bytes()
        results, summary = evaluate(args, json.loads(frozen_bytes))
        write_json(args.output / "ledger.json", {"channels": results})
        write_json(args.output / "summary.json", {
            "calibration_sha256": digest(frozen_bytes),
            "source_replay_ledger_sha256": file_hash(args.replay / "ledger.json"),
            "channel_count": len(results),
            **summary,
        })


if __name__ == "__main__":
    main()
