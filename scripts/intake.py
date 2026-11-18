"""Validate a Telemanom archive without extracting members to disk."""

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

MISSIONS = ("SMAP", "MSL")
WINDOW = 250
MIN_CALIBRATION = 100
EXPECTED_CHANNELS = {"SMAP": 55, "MSL": 27}
EXPECTED_INTERVALS = 105


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def inspect_arrays(archive):
    arrays = defaultdict(list)
    for member in archive.infolist():
        path = PurePosixPath(member.filename)
        if member.is_dir() or len(path.parts) < 2 or path.suffix != ".npy":
            continue
        if path.parent.name not in {"train", "test"}:
            continue
        raw = archive.read(member)
        info = {
            "member": member.filename,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "issues": [],
        }
        try:
            array = np.load(io.BytesIO(raw), allow_pickle=False)
            if not isinstance(array, np.ndarray):
                array.close()
                raise ValueError("member is not a NumPy array")
            info["shape"] = list(array.shape)
            info["dtype"] = str(array.dtype)
            if array.ndim != 2 or array.shape[1] < 1:
                info["issues"].append("expected_2d_array_with_target_column")
            if array.dtype.kind not in "iuf":
                info["issues"].append("expected_real_numeric_dtype")
            elif not np.isfinite(array).all():
                info["issues"].append("nonfinite_values")
        except (ValueError, EOFError, OSError, TypeError):
            info["issues"].append("invalid_numpy_array")
        arrays[(path.stem, path.parent.name)].append(info)
    return arrays


def parse_metadata(raw):
    stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
    reader = csv.DictReader(stream)
    required = {"spacecraft", "chan_id", "anomaly_sequences", "class", "num_values"}
    if reader.fieldnames is None or not required.issubset(reader.fieldnames):
        raise ValueError("label CSV is missing required column names")
    rows = []
    for order, row in enumerate(reader):
        info = {
            "source_row_order": order,
            "mission": (row["spacecraft"] or "").strip(),
            "channel": (row["chan_id"] or "").strip(),
            "issues": [],
            "events": [],
        }
        if info["mission"] not in MISSIONS:
            info["issues"].append("unknown_mission")
        if not info["channel"]:
            info["issues"].append("blank_channel_id")
        try:
            length = int(row["num_values"])
            if length < 1:
                raise ValueError
            info["declared_test_length"] = length
        except (ValueError, TypeError):
            info["issues"].append("invalid_num_values")
            length = 0
        try:
            intervals = ast.literal_eval(row["anomaly_sequences"])
            class_field = (row["class"] or "").strip()
            if not class_field.startswith("[") or not class_field.endswith("]"):
                raise ValueError
            class_items = class_field[1:-1].strip()
            classes = [item.strip() for item in class_items.split(",")] if class_items else []
            if not isinstance(intervals, (list, tuple)):
                raise ValueError
            info["declared_interval_count"] = len(intervals)
            if len(intervals) != len(classes):
                raise ValueError
            for event_order, (bounds, event_class) in enumerate(zip(intervals, classes)):
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError
                start, end = bounds
                if type(start) is not int or type(end) is not int or not 0 <= start <= end < length:
                    raise ValueError
                if event_class not in ("point", "contextual"):
                    raise ValueError
                info["events"].append({
                    "start": start,
                    "end": end,
                    "class": event_class,
                    "source_event_order": event_order,
                })
            info["events"].sort(key=lambda event: (event["start"], event["end"], event["source_event_order"]))
        except (ValueError, SyntaxError, TypeError):
            info["events"] = []
            info["issues"].append("invalid_anomaly_intervals_or_classes")
        rows.append(info)
    return rows


def reconcile(rows, arrays):
    ids = Counter(row["channel"] for row in rows)
    keys = Counter((row["mission"], row["channel"]) for row in rows)
    for row in rows:
        channel = row["channel"]
        issues = row["issues"]
        if keys[(row["mission"], channel)] > 1:
            issues.append("duplicate_metadata_record")
        if ids[channel] > keys[(row["mission"], channel)]:
            issues.append("ambiguous_channel_across_missions")
        paired = {}
        for split in ("train", "test"):
            candidates = arrays.get((channel, split), [])
            if not candidates:
                issues.append(f"missing_{split}_array")
            elif len(candidates) > 1:
                issues.append(f"duplicate_{split}_arrays")
            else:
                paired[split] = candidates[0]
                issues.extend(f"{split}_{issue}" for issue in candidates[0]["issues"])
        row["array_members"] = {split: entry["member"] for split, entry in paired.items()}
        if "test" in paired and len(paired["test"].get("shape", [])) == 2:
            test_length = paired["test"]["shape"][0]
            if test_length != row.get("declared_test_length"):
                issues.append("declared_test_length_mismatch")
            if test_length <= WINDOW:
                issues.append("no_test_target")
        if "train" in paired and len(paired["train"].get("shape", [])) == 2:
            train_length = paired["train"]["shape"][0]
            fit_stop = 70 * train_length // 100
            validation_stop = 85 * train_length // 100
            row["train_target_ranges_half_open"] = {
                "fit": [WINDOW, max(WINDOW, fit_stop)],
                "validation": [max(WINDOW, fit_stop), max(WINDOW, validation_stop)],
                "calibration": [max(WINDOW, validation_stop), max(WINDOW, train_length)],
            }
            row["train_row_ranges_half_open"] = {
                "fit": [0, fit_stop],
                "validation": [fit_stop, validation_stop],
                "calibration": [validation_stop, train_length],
            }
            if fit_stop <= WINDOW:
                issues.append("no_fit_target")
            if validation_stop <= max(WINDOW, fit_stop):
                issues.append("no_validation_target")
            if train_length - max(WINDOW, validation_stop) < MIN_CALIBRATION:
                issues.append("fewer_than_100_calibration_scores")
            if "test" in paired and len(paired["test"].get("shape", [])) == 2:
                if paired["train"]["shape"][1] != paired["test"]["shape"][1]:
                    issues.append("feature_width_mismatch")
        row["eligible"] = not issues
        if row["eligible"]:
            row["test_target_range_half_open"] = [WINDOW, test_length]
            for event in row["events"]:
                if event["end"] < WINDOW:
                    event["scoring_status"] = "outside_scored_range"
                elif event["start"] < WINDOW:
                    event["scoring_status"] = "clipped_to_scored_range"
                else:
                    event["scoring_status"] = "eligible"
                if event["end"] >= WINDOW:
                    event["scored_bounds_inclusive"] = [max(WINDOW, event["start"]), event["end"]]
    return rows


def build_manifest(archive_path, labels_path, source, commit, retrieved):
    with zipfile.ZipFile(archive_path) as archive:
        arrays = inspect_arrays(archive)
    raw_labels = labels_path.read_bytes()
    rows = reconcile(parse_metadata(raw_labels), arrays)
    counts = {}
    for mission in MISSIONS:
        mission_rows = [row for row in rows if row["mission"] == mission]
        eligible = [row for row in mission_rows if row["eligible"]]
        counts[mission] = {
            "expected_channels": EXPECTED_CHANNELS[mission],
            "metadata_rows": len(mission_rows),
            "unique_channel_ids": len({row["channel"] for row in mission_rows}),
            "eligible_channels": len(eligible),
            "excluded_metadata_rows": len(mission_rows) - len(eligible),
            "exclusion_reasons": dict(Counter(issue for row in mission_rows for issue in row["issues"])),
            "declared_intervals": sum(row.get("declared_interval_count", 0) for row in mission_rows),
            "validated_intervals": sum(len(row["events"]) for row in mission_rows),
            "eligible_intervals": sum(
                event["scoring_status"] != "outside_scored_range"
                for row in eligible for event in row["events"]
            ),
            "scored_readings": sum(row["test_target_range_half_open"][1] - WINDOW for row in eligible),
        }
    label_ids = {row["channel"] for row in rows}
    return {
        "schema_version": 1,
        "source": source,
        "repository_commit": commit,
        "retrieved_utc_date": retrieved,
        "archive": {"filename": archive_path.name, "sha256": file_hash(archive_path)},
        "labels": {
            "filename": labels_path.name,
            "sha256": hashlib.sha256(raw_labels).hexdigest(),
            "repository_url": (
                f"https://github.com/khundman/telemanom/blob/{commit}/labeled_anomalies.csv"
            ),
        },
        "protocol": {"history_rows": WINDOW, "minimum_calibration_scores": MIN_CALIBRATION},
        "expected_intervals": EXPECTED_INTERVALS,
        "declared_intervals": sum(row.get("declared_interval_count", 0) for row in rows),
        "mission_counts": counts,
        "unmapped_metadata_rows": sum(row["mission"] not in MISSIONS for row in rows),
        "arrays": [
            {"channel": channel, "split": split, "has_metadata": channel in label_ids, **entry}
            for (channel, split), entries in sorted(arrays.items())
            for entry in entries
        ],
        "channels": rows,
    }


def main():
    from datetime import date

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--source", required=True, help="Release archive URL")
    parser.add_argument("--commit", required=True, help="Repository commit for the label file")
    parser.add_argument("--retrieved", required=True, help="UTC retrieval date, YYYY-MM-DD")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.resolve() in {args.archive.resolve(), args.labels.resolve()}:
        parser.error("--output must differ from input paths")
    try:
        retrieved = date.fromisoformat(args.retrieved).isoformat()
        manifest = build_manifest(args.archive, args.labels, args.source, args.commit, retrieved)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, ValueError, csv.Error, zipfile.BadZipFile) as error:
        parser.exit(1, f"intake failed: {error}\n")


if __name__ == "__main__":
    main()
