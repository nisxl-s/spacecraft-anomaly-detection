"""Compare one- and two-high confirmation at unchanged NASA thresholds."""

import argparse
import json
from pathlib import Path

import numpy as np

from duration_reset import CLEAR_READINGS_TO_REARM, MAX_FLAG_READINGS, durations, limit_episodes
from intake import file_hash
from replay_p1 import detector_ledger
from summarize_benchmark import summarize

DETECTORS = ("level", "persistence", "lstm")
CONFIRM_READINGS = 2


def confirmed_episodes(raw_flags, required=CONFIRM_READINGS):
    """Keep output causal. The threshold-crossing reading is never restored."""
    if required < 1:
        raise ValueError("Confirmation requires at least one high reading")
    emitted = np.zeros_like(raw_flags)
    consecutive_high = consecutive_clear = active_readings = 0
    lockout = False
    for i, high in enumerate(raw_flags):
        if lockout:
            consecutive_clear = 0 if high else consecutive_clear + 1
            if consecutive_clear == CLEAR_READINGS_TO_REARM:
                lockout = False
            continue
        if not high:
            consecutive_high = active_readings = 0
            continue
        consecutive_high += 1
        if consecutive_high < required:
            continue
        emitted[i] = True
        active_readings += 1
        if active_readings == MAX_FLAG_READINGS:
            lockout = True
            consecutive_high = consecutive_clear = active_readings = 0
    return emitted


def events_for(channel):
    return [
        {key: truth[key] for key in ("start", "end", "class", "source_event_order")}
        for truth in channel["detectors"]["level"]["truths"]
    ]


def paired_status(one_high, two_high):
    one, two = one_high["truths"], two_high["truths"]
    states = []
    for a, b in zip(one, two, strict=True):
        identity = ("truth_id", "source_event_order", "start", "end", "class", "observable_bounds_inclusive")
        if any(a[key] != b[key] for key in identity):
            raise ValueError("Paired truth identity changed")
        if a["status"] == b["status"] == "excluded":
            continue
        if {a["status"], b["status"]} - {"matched", "miss"}:
            raise ValueError("Paired truth eligibility changed")
        states.append({
            "truth_id": a["truth_id"], "source_event_order": a["source_event_order"],
            "class": a["class"],
            "one_high_detected": a["status"] == "matched", "two_high_detected": b["status"] == "matched",
            "one_high_delay": a["delay_readings"], "two_high_delay": b["delay_readings"],
        })
    return states


def evaluate_channel(channel, previous, replay):
    mission, identity = channel["mission"], channel["channel"]
    path = replay / "scores" / channel["scores"]["filename"]
    if file_hash(path) != channel["scores"]["sha256"]:
        raise ValueError(f"Score file hash mismatch for {identity}")
    result = {}
    with np.load(path, allow_pickle=False) as data:
        indices = data["index"]
        start, stop = int(indices[0]), int(indices[-1]) + 1
        np.testing.assert_array_equal(indices, np.arange(start, stop))
        for detector in DETECTORS:
            threshold = data[f"{detector}_threshold"].item()
            if threshold != channel["detectors"][detector]["threshold"]:
                raise ValueError(f"Original threshold changed for {identity}, {detector}")
            raw = data[f"{detector}_score"] > threshold
            np.testing.assert_array_equal(raw, data[f"{detector}_flag"])
            one, _, _ = limit_episodes(raw)
            np.testing.assert_array_equal(confirmed_episodes(raw, required=1), one)
            two = confirmed_episodes(raw)
            events = events_for(channel)
            one_ledger = detector_ledger(one, start, stop, events)
            if one_ledger != previous["policies"][detector]["limited"]:
                raise ValueError(f"One-high ledger did not replay for {identity}, {detector}")
            two_ledger = detector_ledger(two, start, stop, events)
            result[detector] = {
                "threshold": threshold, "one_high": one_ledger, "two_high": two_ledger,
                "paired_truths": paired_status(one_ledger, two_ledger),
            }
    return {"channel": identity, "mission": mission, "detectors": result}


def summarize_paired(rows, one_count, two_count):
    paired = {"both_detected": 0, "two_high_gains": 0, "two_high_losses": 0, "neither_detected": 0}
    differences = []
    for channel, truth in rows:
        one, two = truth["one_high_detected"], truth["two_high_detected"]
        if one and two:
            paired["both_detected"] += 1
            differences.append(truth["two_high_delay"] - truth["one_high_delay"])
        elif one:
            paired["two_high_losses"] += 1
        elif two:
            paired["two_high_gains"] += 1
        else:
            paired["neither_detected"] += 1
    if paired["both_detected"] + paired["two_high_losses"] != one_count:
        raise ValueError("One-high paired detection count mismatch")
    if paired["both_detected"] + paired["two_high_gains"] != two_count:
        raise ValueError("Two-high paired detection count mismatch")
    return paired | {
        "common_detected_delay_change_median_readings": float(np.median(differences)) if differences else None,
        "common_detected_delay_change_p90_readings": float(np.percentile(differences, 90)) if differences else None,
        "event_changes": [{"channel": channel, **truth} for channel, truth in rows
                          if truth["one_high_detected"] != truth["two_high_detected"]],
    }


def summarize_group(channels):
    group = {}
    for detector in DETECTORS:
        variants = {}
        for name in ("one_high", "two_high"):
            ledgers = [channel["detectors"][detector][name] for channel in channels]
            duration = durations(ledgers)
            if duration["over_30_readings"]:
                raise ValueError("Episode duration cap exceeded")
            variants[name] = summarize(ledgers) | {"episode_duration": duration}
        pairs = [(channel["channel"], truth)
                 for channel in channels for truth in channel["detectors"][detector]["paired_truths"]]
        group[detector] = variants | {
            "paired": summarize_paired(pairs, variants["one_high"]["matches"], variants["two_high"]["matches"]),
        }
    return group


def write_json(path, data, compact=False):
    path.write_text(json.dumps(data, indent=None if compact else 2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--previous-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    previous_path = args.previous_policy / "ledger.json"
    replay_path = args.replay / "ledger.json"
    original = json.loads(replay_path.read_bytes())["channels"]
    previous = {row["channel"]: row for row in json.loads(previous_path.read_bytes())["channels"]}
    if len(original) != 78 or {row["channel"] for row in original} != set(previous):
        raise ValueError("Expected identical coverage of 78 unique channels")
    args.output.mkdir()
    config_path = args.output / "config.json"
    write_json(config_path, {
        "scope": "exploratory replay of the same NASA test scores on all 78 channels",
        "thresholds": "original frozen per-channel thresholds for all three detectors",
        "triggers": {"one_high": 1, "two_high": CONFIRM_READINGS},
        "policy": "Strict exceedance. The confirming reading is the first emitted flag. A clear reading "
                  "after a short alarm resets confirmation. The emitted cap starts lockout until five "
                  "consecutive at-or-below-threshold readings rearm the detector on a future reading.",
        "max_emitted_readings": MAX_FLAG_READINGS,
        "clear_readings_required_after_cap": CLEAR_READINGS_TO_REARM,
        "source_replay_ledger_sha256": file_hash(replay_path),
        "prior_one_high_ledger_sha256": file_hash(previous_path),
        "code_sha256": file_hash(Path(__file__)),
    })
    rows = [evaluate_channel(channel, previous[channel["channel"]], args.replay) for channel in original]
    groups = {"overall": rows, **{mission: [row for row in rows if row["mission"] == mission]
                                  for mission in ("SMAP", "MSL")}}
    write_json(args.output / "ledger.json", {"channels": rows}, compact=True)
    write_json(args.output / "summary.json", {
        "config_sha256": file_hash(config_path), "channel_count": len(rows),
        **{group: summarize_group(channels) for group, channels in groups.items()},
    })


if __name__ == "__main__":
    main()
