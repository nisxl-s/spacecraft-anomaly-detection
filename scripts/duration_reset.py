"""Replay a causal episode cap and clear-run reset on saved benchmark scores."""

import argparse
import json
from pathlib import Path
from statistics import median

import numpy as np

from intake import file_hash
from replay_p1 import alarm_runs, detector_ledger
from summarize_benchmark import summarize

MAX_FLAG_READINGS = 30
CLEAR_READINGS_TO_REARM = 5
FROZEN = {detector: (detector, 1.0) for detector in ("level", "persistence", "lstm")}


def limit_episodes(raw_flags, max_flags=MAX_FLAG_READINGS, clear_required=CLEAR_READINGS_TO_REARM):
    """Return emitted flags and counts of cap entries and completed clear resets."""
    emitted = np.zeros_like(raw_flags)
    capped = resets = run = consecutive_clear = 0
    suppressed = False
    for index, exceeds in enumerate(raw_flags):
        if suppressed:
            consecutive_clear = 0 if exceeds else consecutive_clear + 1
            if consecutive_clear == clear_required:
                suppressed = False
                resets += 1
            continue
        if not exceeds:
            run = 0
            continue
        emitted[index] = True
        run += 1
        if run == max_flags:
            capped += 1
            suppressed = True
            run = consecutive_clear = 0
    return emitted, capped, resets


def evaluate_channel(channel, replay, policies):
    identity = channel["channel"]
    path = replay / "scores" / channel["scores"]["filename"]
    if file_hash(path) != channel["scores"]["sha256"]:
        raise ValueError(f"Score file hash mismatch for {identity}")
    events = [{field: truth[field] for field in ("start", "end", "class", "source_event_order")}
              for truth in channel["detectors"]["level"]["truths"]]
    results = {}
    with np.load(path, allow_pickle=False) as arrays:
        indices = arrays["index"]
        start, stop = int(indices[0]), int(indices[-1]) + 1
        np.testing.assert_array_equal(indices, np.arange(start, stop))
        for name, (detector, multiplier) in policies.items():
            scores = arrays[f"{detector}_score"]
            threshold = arrays[f"{detector}_threshold"].item() * multiplier
            if not np.isfinite(scores).all() or not np.isfinite(threshold):
                raise ValueError(f"Nonfinite score or threshold for {identity} {name}")
            raw = scores > threshold
            original = detector_ledger(raw, start, stop, events)
            if multiplier == 1:
                expected = channel["detectors"][detector]
                np.testing.assert_array_equal(raw, arrays[f"{detector}_flag"])
                if {"threshold": threshold, **original} != expected:
                    raise ValueError(f"Original ledger changed for {identity} {name}")
            emitted, capped, resets = limit_episodes(raw)
            limited = detector_ledger(emitted, start, stop, events)
            runs = alarm_runs(raw, 0)
            blocked_attempts = sum(not emitted[begin] for begin, _ in runs)
            suppressed = int(np.count_nonzero(raw & ~emitted))
            if len(runs) != original["alarm_episodes"] or len(runs) - blocked_attempts != limited["alarm_episodes"]:
                raise ValueError(f"Episode onset count mismatch for {identity} {name}")
            results[name] = {
                "threshold": threshold,
                "raw": original,
                "limited": limited,
                "policy_counts": {
                    "attempted_activations": len(runs),
                    "blocked_activation_attempts": int(blocked_attempts),
                    "episodes_reaching_cap": capped,
                    "completed_clear_resets": resets,
                    "suppressed_exceedance_readings": suppressed,
                },
            }
    return {"channel": identity, "mission": channel["mission"], "policies": results}


def durations(ledgers):
    values = [alarm["bounds_inclusive"][1] - alarm["bounds_inclusive"][0] + 1
              for ledger in ledgers for alarm in ledger["alarms"]]
    return {
        "episodes": len(values),
        "mean_readings": float(np.mean(values)) if values else None,
        "median_readings": median(values) if values else None,
        "p90_readings": float(np.percentile(values, 90)) if values else None,
        "max_readings": max(values, default=None),
        "over_30_readings": sum(duration > MAX_FLAG_READINGS for duration in values),
    }


def summarize_policies(channels, policies):
    result = {}
    for name in policies:
        rows = [channel["policies"][name] for channel in channels]
        counts = {key: sum(row["policy_counts"][key] for row in rows) for key in rows[0]["policy_counts"]}
        metrics = {}
        for kind in ("raw", "limited"):
            ledgers = [row[kind] for row in rows]
            metrics[kind] = summarize(ledgers) | {"episode_duration": durations(ledgers)}
        if metrics["limited"]["episode_duration"]["over_30_readings"]:
            raise ValueError(f"Duration cap exceeded for {name}")
        if metrics["raw"]["flagged_readings"] - metrics["limited"]["flagged_readings"] != counts["suppressed_exceedance_readings"]:
            raise ValueError(f"Suppressed reading count mismatch for {name}")
        result[name] = metrics | {"policy_counts": counts}
    return result


def write_json(path, data, compact=False):
    indentation = None if compact else 2
    path.write_text(json.dumps(data, indent=indentation, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--sweep-comparisons", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    replay_ledger = args.replay / "ledger.json"
    original = json.loads(replay_ledger.read_bytes())["channels"]
    comparisons = json.loads(args.sweep_comparisons.read_bytes())
    policies = FROZEN.copy()
    for comparison in comparisons:
        reference = comparison["reference"]
        if reference["group"] == "overall":
            multiplier = comparison["selected_lstm"]["threshold_multiplier"]
            if not isinstance(multiplier, (int, float)) or not np.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("Expected a positive finite sweep multiplier")
            policies[f"lstm_sweep_{reference['detector']}"] = ("lstm", multiplier)
    if set(policies) != set(FROZEN) | {"lstm_sweep_level", "lstm_sweep_persistence"}:
        raise ValueError("Expected one overall LSTM sweep multiplier for each baseline")
    args.output.mkdir()
    config = {
        "policy": "Flag strict exceedances. On reaching the episode cap, suppress output until a complete run of "
                  "clear readings at or below threshold. The next reading can start a new episode. Below cap, one "
                  "clear reading ends the episode and the next reading can rearm. Initialize per channel.",
        "maximum_flagged_readings_per_episode": MAX_FLAG_READINGS,
        "clear_readings_required_after_cap": CLEAR_READINGS_TO_REARM,
        "source_replay_ledger_sha256": file_hash(replay_ledger),
        "source_sweep_comparisons_sha256": file_hash(args.sweep_comparisons),
        "variants": {key: {"detector": detector, "threshold_multiplier": multiplier,
                           "original_frozen_threshold": key in FROZEN}
                     for key, (detector, multiplier) in policies.items()},
    }
    write_json(args.output / "config.json", config)
    results = [evaluate_channel(channel, args.replay, policies) for channel in original]
    groups = {"overall": results, **{
        mission: [row for row in results if row["mission"] == mission]
        for mission in sorted({row["mission"] for row in results})
    }, "without_pilot_P-1": [row for row in results if row["channel"] != "SMAP/P-1"]}
    write_json(args.output / "ledger.json", {"channels": results}, compact=True)
    write_json(args.output / "summary.json", {
        "config_sha256": file_hash(args.output / "config.json"),
        "channel_count": len(results),
        **{group: summarize_policies(rows, policies) for group, rows in groups.items()},
    })


if __name__ == "__main__":
    main()
