"""Replay frozen P-1 detectors and write scores before opening the label CSV."""

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch

from event_scoring import score_events
from intake import file_hash, parse_metadata
from pilot_baselines import smooth
from pilot_lstm import Forecast, residuals

DETECTORS = ("level", "persistence", "lstm")
MANIFEST_SHA256 = "4aa489f270b397e48dfca2410816e86120ff1047819b592f2aacc2732c68ea78"
FREEZE_SHA256 = "e50fd7bf5a5a0305ca7042a77ef444d36293718eeab60e6e41176adffacf5f0b"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_verified(path, expected):
    raw = path.read_bytes()
    require(digest(raw) == expected, f"SHA-256 mismatch for {path}")
    return raw


def load_array(archive, record):
    raw = archive.read(record["member"])
    require(digest(raw) == record["sha256"], f"member SHA-256 mismatch for {record['member']}")
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    require(list(array.shape) == record["shape"] and str(array.dtype) == record["dtype"],
            f"array metadata mismatch for {record['member']}")
    require(np.isfinite(array).all(), f"nonfinite value in {record['member']}")
    return array


def alarm_runs(flags, start):
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    transitions = np.diff(padded)
    return list(zip(
        (np.flatnonzero(transitions == 1) + start).tolist(),
        (np.flatnonzero(transitions == -1) + start - 1).tolist(),
    ))


def detector_ledger(flags, start, stop, events):
    alarms = alarm_runs(flags, start)
    truths = [(event["start"], event["end"]) for event in events]
    matches = score_events(truths, alarms, start, stop - 1)
    alarm_match = {alarm: truth for truth, alarm, _ in matches["matches"]}
    truth_match = {truth: (alarm, delay) for truth, alarm, delay in matches["matches"]}
    excluded = set(matches["excluded_truths"])
    read_count = stop - start
    delay_values = [delay for _, _, delay in matches["matches"]]
    return {
        "exposure_readings": read_count,
        "eligible_truths": len(truths) - len(excluded),
        "excluded_truths": len(excluded),
        "clipped_truths": len(matches["partly_clipped_truths"]),
        "alarm_episodes": len(alarms),
        "matches": matches["tp"],
        "false_alarms": matches["fp"],
        "misses": matches["fn"],
        "precision": matches["precision"],
        "recall": matches["recall"],
        "false_alarms_per_1000_readings": 1000 * matches["fp"] / read_count,
        "flagged_readings": int(flags.sum()),
        "fraction_flagged": float(flags.mean()),
        "median_delay_readings": float(np.median(delay_values)) if delay_values else None,
        "p90_delay_readings": float(np.percentile(delay_values, 90, method="linear")) if delay_values else None,
        "alarms": [
            {
                "alarm_id": i,
                "bounds_inclusive": [first, last],
                "status": "matched" if i in alarm_match else "false_alarm",
                "matched_truth_id": alarm_match.get(i),
            }
            for i, (first, last) in enumerate(alarms)
        ],
        "truths": [
            {
                **event,
                "truth_id": i,
                "observable_bounds_inclusive": (
                    None if i in excluded else [max(start, event["start"]), min(stop - 1, event["end"])]
                ),
                "status": "excluded" if i in excluded else ("matched" if i in truth_match else "miss"),
                "matched_alarm_id": truth_match[i][0] if i in truth_match else None,
                "delay_readings": truth_match[i][1] if i in truth_match else None,
            }
            for i, event in enumerate(events)
        ],
    }


def replay(args):
    raw_manifest = read_verified(args.manifest, MANIFEST_SHA256)
    manifest = json.loads(raw_manifest)
    pilot = json.loads(args.pilot.read_bytes())
    freeze = pilot["pilot_freeze"]
    require(digest(json.dumps(freeze, sort_keys=True, separators=(",", ":")).encode()) ==
            pilot["pilot_freeze_sha256"] == FREEZE_SHA256, "pilot configuration SHA-256 mismatch")
    require(pilot["channel"] == "SMAP/P-1", "freeze channel is not SMAP/P-1")
    require(pilot["configuration"] == freeze["lstm"], "configuration differs from freeze")
    require(pilot["threshold"] == freeze["thresholds"]["lstm"], "LSTM threshold differs from freeze")
    for code, expected in freeze["code_sha256"].items():
        read_verified(Path(__file__).parent / code, expected)
    require(file_hash(args.archive) == manifest["archive"]["sha256"], "archive SHA-256 mismatch")
    checkpoint_raw = read_verified(args.checkpoint, freeze["checkpoint_sha256"])

    channel, = (r for r in manifest["channels"] if (r["mission"], r["channel"]) == ("SMAP", "P-1"))
    require(channel["eligible"], "P-1 was excluded at intake")
    require(channel["train_target_ranges_half_open"] == freeze["target_ranges_half_open"],
            "manifest split differs from freeze")
    records = {
        row["split"]: row for row in manifest["arrays"]
        if row["member"] in channel["array_members"].values()
    }
    require(records["train"]["sha256"] == pilot["training_member_sha256"], "training hash differs from pilot")
    with zipfile.ZipFile(args.archive) as archive:
        train = load_array(archive, records["train"])
        test = load_array(archive, records["test"])
    start, stop = channel["test_target_range_half_open"]
    require(start == freeze["lstm"]["history_rows"] and stop == len(test) and start < stop,
            "test array differs from common scored range")
    thresholds = freeze["thresholds"]
    require(set(thresholds) == set(DETECTORS) and all(math.isfinite(v) for v in thresholds.values()),
            "missing or invalid frozen threshold")

    # Calibrate neither baseline on the held-out file. Verify the frozen values from training.
    fit_stop = freeze["target_ranges_half_open"]["fit"][1]
    calibration_start, calibration_end = freeze["target_ranges_half_open"]["calibration"]
    training_signal = train[:, 0].astype(float)
    fit_signal = training_signal[:fit_stop]
    mean, scale = float(fit_signal.mean()), max(float(fit_signal.std()), 1e-8)
    calibration_signal = training_signal[calibration_start:calibration_end]
    calibration_prev = training_signal[calibration_start - 1:calibration_end - 1]
    calibration_scores = {
        "level": np.abs(calibration_signal - mean) / scale,
        "persistence": np.abs(calibration_signal - calibration_prev),
    }
    for name, score in calibration_scores.items():
        position = math.ceil(freeze["lstm"]["calibration_percentile"] * len(score)) - 1
        require(np.sort(smooth(score))[position] == thresholds[name], f"training threshold changed for {name}")

    torch.set_num_threads(pilot["environment"]["torch_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    checkpoint = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
    require(checkpoint["threshold"] == thresholds["lstm"] and checkpoint["best_epoch"] == pilot["best_epoch"]
            and checkpoint["training_member_sha256"] == records["train"]["sha256"]
            and checkpoint["input_features"] == train.shape[1] == test.shape[1],
            "checkpoint metadata differs from frozen training record")
    model = Forecast(checkpoint["input_features"])
    model.load_state_dict(checkpoint["model_state_dict"])

    def raw_scores(data):
        features = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))
        telemetry = data[:, 0].astype(float)
        windows = features.unfold(0, start, 1)[:-1].transpose(1, 2)
        np.testing.assert_array_equal(windows[0].numpy(), features[:start].numpy())
        return {
            "level": np.abs(telemetry[start:] - mean) / scale,
            "persistence": np.abs(telemetry[start:] - telemetry[start - 1:-1]),
            "lstm": residuals(model, windows, torch.from_numpy(telemetry), torch.arange(start, len(data))),
        }

    raw = raw_scores(test)
    scores = {name: smooth(raw[name]) for name in DETECTORS}
    exposure = stop - start
    for name in DETECTORS:
        require(len(scores[name]) == exposure and np.isfinite(scores[name]).all(), f"invalid common exposure for {name}")
    flags = {name: scores[name] > thresholds[name] for name in DETECTORS}
    prefix_length = exposure // 2
    prefix_raw = raw_scores(test[:start + prefix_length])
    prefix_check = {}
    for name in DETECTORS:
        prefix_scores = smooth(prefix_raw[name])
        np.testing.assert_allclose(prefix_scores, scores[name][:prefix_length], rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(prefix_scores > thresholds[name], flags[name][:prefix_length])
        prefix_check[name] = float(np.max(np.abs(prefix_scores - scores[name][:prefix_length])))

    header = ["index", "telemetry"] + [
        f"{name}_{field}" for name in DETECTORS for field in ("raw_score", "score", "threshold", "flag")
    ]
    with args.scores.open("x", encoding="utf-8", newline="") as stream:
        import csv

        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header)
        for i, index in enumerate(range(start, stop)):
            row = [index, float(test[index, 0])]
            for name in DETECTORS:
                row.extend((float(raw[name][i]), float(scores[name][i]), thresholds[name], int(flags[name][i])))
            writer.writerow(row)

    # The predictions are closed and hashed before reading the separate label CSV.
    score_hash = file_hash(args.scores)
    label_raw = read_verified(args.labels, manifest["labels"]["sha256"])
    label, = (r for r in parse_metadata(label_raw) if (r["mission"], r["channel"]) == ("SMAP", "P-1"))
    require(not label["issues"] and label["declared_test_length"] == stop, "P-1 label validation failed")
    events = sorted(label["events"], key=lambda event: event["source_event_order"])
    require(
        events == sorted(
            (
                {key: event[key] for key in ("start", "end", "class", "source_event_order")}
                for event in channel["events"]
            ),
            key=lambda event: event["source_event_order"],
        ),
        "P-1 labels differ from frozen intake",
    )
    return {
        "channel": "SMAP/P-1",
        "test_target_range_half_open": [start, stop],
        "exposure_readings": exposure,
        "input_provenance": {
            "intake_manifest_sha256": MANIFEST_SHA256,
            "pilot_freeze_sha256": FREEZE_SHA256,
            "archive_sha256": manifest["archive"]["sha256"],
            "training_member_sha256": records["train"]["sha256"],
            "test_member_sha256": records["test"]["sha256"],
            "label_csv_sha256": manifest["labels"]["sha256"],
            "checkpoint_sha256": freeze["checkpoint_sha256"],
            "replay_code_sha256": file_hash(Path(__file__)),
        },
        "scores": {"filename": args.scores.name, "sha256": score_hash, "rows": exposure},
        "baseline_level_fit_mean": mean,
        "baseline_level_fit_std_floored": scale,
        "verification": {
            "scores_closed_and_hashed_before_label_csv_read": True,
            "prefix_readings": prefix_length,
            "prefix_max_absolute_score_difference": prefix_check,
            "prefix_flags_identical": True,
            "delay_percentile_method": "linear",
        },
        "detectors": {
            name: {"threshold": thresholds[name], **detector_ledger(flags[name], start, stop, events)}
            for name in DETECTORS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("archive", "manifest", "pilot", "checkpoint", "labels", "scores", "report"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    args = parser.parse_args()
    if args.scores.resolve() == args.report.resolve() or args.scores.exists() or args.report.exists():
        parser.error("provide different unused paths for --scores and --report")
    result = replay(args)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
