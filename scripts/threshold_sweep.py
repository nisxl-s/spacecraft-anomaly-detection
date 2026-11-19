"""Draw an exploratory test-set threshold sweep for the frozen scored arrays."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from event_scoring import score_events
from intake import file_hash
from replay_p1 import alarm_runs

DETECTORS = ("level", "persistence", "lstm")
GROUPS = ("overall", "SMAP", "MSL")
MULTIPLIERS = (0.0, *(10.0 ** (step / 40) for step in range(-40, 41)), float("inf"))
LABELS = {"level": "Level baseline", "persistence": "Persistence baseline", "lstm": "LSTM"}
COLORS = {"level": "#1f6a56", "persistence": "#ad5816", "lstm": "#2354a3"}


def score_channel(source, channel, totals):
    path = source / "scores" / channel["scores"]["filename"]
    if file_hash(path) != channel["scores"]["sha256"]:
        raise ValueError(f"Score array hash mismatch: {path}")
    events = [(truth["start"], truth["end"]) for truth in channel["detectors"]["level"]["truths"]]
    with np.load(path, allow_pickle=False) as stored:
        indices = stored["index"]
        start, end = int(indices[0]), int(indices[-1])
        np.testing.assert_array_equal(indices, np.arange(start, end + 1))
        for detector in DETECTORS:
            series = stored[f"{detector}_score"]
            frozen = stored[f"{detector}_threshold"].item()
            if not np.isfinite(series).all() or not np.isfinite(frozen) or frozen < 0:
                raise ValueError(f"Invalid scores or threshold: {path} {detector}")
            for row, multiplier in enumerate(MULTIPLIERS):
                threshold = float("inf") if np.isinf(multiplier) else multiplier * frozen
                flags = series > threshold
                if multiplier == 1:
                    np.testing.assert_array_equal(flags, stored[f"{detector}_flag"])
                scores = score_events(events, alarm_runs(flags, start), start, end)
                counts = (series.size, scores["tp"] + scores["fn"], scores["tp"], scores["fp"], int(flags.sum()))
                for group in ("overall", channel["mission"]):
                    totals[group, detector][row] += counts


def sweep(source):
    totals = {(group, detector): np.zeros((len(MULTIPLIERS), 5), dtype=np.int64)
              for group in GROUPS for detector in DETECTORS}
    channels = json.loads((source / "ledger.json").read_bytes())["channels"]
    for channel in channels:
        score_channel(source, channel, totals)
    rows = []
    for (group, detector), counts in totals.items():
        for multiplier, values in zip(MULTIPLIERS, counts):
            readings, events, matched, false_alarms, flagged = map(int, values)
            rows.append({
                "group": group,
                "detector": detector,
                "threshold_multiplier": multiplier,
                "scored_readings": readings,
                "eligible_events": events,
                "detected_events": matched,
                "event_recall": matched / events,
                "false_alarms": false_alarms,
                "false_alarms_per_1000_readings": 1000 * false_alarms / readings,
                "flagged_readings": flagged,
                "fraction_flagged": flagged / readings,
            })
    return rows


def reference_comparisons(rows):
    comparisons = []
    for group in GROUPS:
        for baseline in DETECTORS[:2]:
            reference = next(r for r in rows if r["group"] == group and r["detector"] == baseline
                             and r["threshold_multiplier"] == 1)
            feasible = [
                r for r in rows if r["group"] == group and r["detector"] == "lstm"
                and r["false_alarms"] <= reference["false_alarms"]
            ]
            best = min(feasible, key=lambda r: (
                -r["detected_events"], r["flagged_readings"], r["false_alarms"], -r["threshold_multiplier"]
            ))
            comparisons.append({"reference": reference, "selected_lstm": best})
    return comparisons


def write_plot(rows, path):
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(11.8, 4.15), sharey=True)
    for group, axis in zip(GROUPS, axes):
        for detector in DETECTORS:
            selected = [row for row in rows if row["group"] == group and row["detector"] == detector]
            far = [row["false_alarms_per_1000_readings"] for row in selected]
            recall = [100 * row["event_recall"] for row in selected]
            axis.scatter(far, recall, s=11, alpha=0.55, ec="none", color=COLORS[detector], label=LABELS[detector])
            frozen = next(row for row in selected if row["threshold_multiplier"] == 1)
            axis.scatter(frozen["false_alarms_per_1000_readings"], 100 * frozen["event_recall"],
                         marker="D", s=37, fc=COLORS[detector], ec="white", linewidth=0.8, zorder=3)
        axis.set_title(f"{group.capitalize() if group == 'overall' else group} · {selected[0]['eligible_events']} events",
                       loc="left", fontweight="bold")
        axis.set_xlabel("False alarms per 1,000 readings")
        axis.set_ylim(-3, 103)
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="#e0e4e7", linewidth=0.7)
    axes[0].set_ylabel("Event recall (%)")
    axes[0].set_yticks(np.arange(0, 101, 20))
    figure.suptitle("Event recall and false alarms across threshold settings", x=0.078, ha="left",
                    fontsize=15, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.069, 0.90), ncols=3, frameon=False)
    figure.text(0.078, 0.065, "Dots show sampled test-set thresholds. Diamonds show the original frozen settings. "
                "See the CSV for the share of readings flagged.", fontsize=8.5, color="#414850")
    figure.subplots_adjust(left=0.078, right=0.983, bottom=0.21, top=0.76, wspace=0.28)
    figure.savefig(path)
    plt.close(figure)


def json_safe(value):
    return {key: ("inf" if isinstance(item, float) and np.isinf(item) else item) for key, item in value.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True)
    rows = sweep(args.replay)
    with (args.output / "threshold_sweep.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    comparisons = [{key: json_safe(value) for key, value in pair.items()} for pair in reference_comparisons(rows)]
    (args.output / "reference_comparisons.json").write_text(json.dumps(comparisons, indent=2, allow_nan=False) + "\n")
    write_plot(rows, args.output / "recall_false_alarm_sweep.pdf")


if __name__ == "__main__":
    main()
