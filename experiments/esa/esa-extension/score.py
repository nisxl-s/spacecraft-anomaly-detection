"""Score the sealed temporal extension and attribute results by UTC year."""
import statistics
from collections import Counter, defaultdict

import numpy as np
from annotations import evaluation_inputs, load_annotations
from replay import BUDGET, BOUNDARY, END, GRID, METHODS, ROOT, UPLOAD, ingest, read, save
from evaluate import as_of, episodes, evaluate

YEARS = tuple(range(2009, 2014))
NEW = np.asarray(GRID > BOUNDARY)


def year(index):
    return GRID[index].year


def main():
    seal = read(ROOT / "replay-seal.json")
    freeze = read(BUDGET / "freeze.json")
    assert ingest.checksum(BUDGET / "freeze.json") == seal["freeze_sha256"]
    for name in ("labels.csv", "anomaly_types.csv"):
        assert ingest.checksum(UPLOAD / name) == freeze["source_sha256"][name]
    for channel, digest in seal["replay_sha256"].items():
        assert ingest.checksum(ROOT / f"{channel}_replay.npz") == digest
    annotations = as_of(load_annotations(UPLOAD / "labels.csv", UPLOAD / "anomaly_types.csv"), GRID[0], END)
    results = {}
    ledgers = {}
    for method in METHODS:
        counts = {y: Counter() for y in YEARS}
        delays = defaultdict(list)
        incident_year, hits, event_keys = {}, set(), set()
        ledgers[method] = {}
        for channel in ingest.CHANNELS:
            with np.load(ROOT / f"{channel}_replay.npz", allow_pickle=False) as saved:
                eligible = saved["eligible"]
                emitted = saved[f"{method}_emitted"]
                raw = saved[f"{method}_raw"]
            observable, flags, _, truths = evaluation_inputs(GRID, eligible, emitted, annotations[channel])
            first = {key: int(np.flatnonzero(mask)[0]) for key, mask in truths.items() if mask.any()}
            carry = {key for key, pos in first.items() if not NEW[pos]}
            for key, pos in first.items():
                incident_year[key] = min(incident_year.get(key, pos), pos)
            new_labels = [row for row in annotations[channel] if row.identity not in carry]
            primary = emitted.copy()
            carry_alarms = []
            for start, stop in episodes(flags):
                if not NEW[start] and NEW[stop-1]:
                    primary[start:stop] = False
                    carry_alarms.append([start, stop])
            primary &= NEW
            row = evaluate(GRID, eligible & NEW, primary, raw & NEW, new_labels)
            row["carry_in_events_overlapping_test"] = sorted(key for key in carry if (truths[key] & NEW).any())
            row["carry_in_observable_alarms"] = carry_alarms
            row["carry_in_alarm_overlaps_new_events"] = sorted(
                key for key, pos in first.items() if NEW[pos]
                and any((truths[key][a:b] & NEW[a:b]).any() for a,b in carry_alarms))
            row["events_active_at_endpoint"] = sorted(key for key in row["events"] if truths[key][-1])
            row["alarm_active_at_endpoint"] = bool(emitted[-1])
            ledgers[method][channel] = row
            matches = {ti: (ai, delay) for ti, ai, delay in row["matches"]}
            matched_alarms = {ai for ai, _ in matches.values()}
            for index, key in enumerate(row["events"]):
                y = year(first[key])
                counts[y]["events"] += 1
                if index in matches:
                    counts[y]["detected"] += 1
                    delays[y].append(matches[index][1])
                    hits.add(key)
                    event_keys.add((channel, key))
            for index, (start, _) in enumerate(row["alarms"]):
                counts[year(start)]["matched_alarms" if index in matched_alarms else "false_alarms"] += 1
            for start, _ in episodes(emitted):
                if NEW[start]:
                    counts[year(start)]["emitted_episodes"] += 1
            for y in YEARS:
                mask = NEW & (GRID.year == y)
                counts[y]["eligible_hours"] += int((eligible & mask).sum())
                counts[y]["observable_hours"] += int((observable & mask).sum())
                counts[y]["flagged_hours"] += int((emitted & mask).sum())
        fresh = {key for key, pos in incident_year.items() if NEW[pos]}
        for key in fresh:
            y = year(incident_year[key])
            counts[y]["incidents"] += 1
            counts[y]["incidents_detected"] += key in hits
        def metrics(c, d):
            keys = ("events", "detected", "incidents", "incidents_detected", "matched_alarms", "false_alarms",
                    "emitted_episodes", "eligible_hours", "observable_hours", "flagged_hours")
            return dict({key: c[key] for key in keys},
                        recall=c["detected"]/c["events"] if c["events"] else None,
                        precision=c["matched_alarms"]/(c["matched_alarms"]+c["false_alarms"]) if c["matched_alarms"]+c["false_alarms"] else None,
                        false_per_1000=1000*c["false_alarms"]/c["observable_hours"] if c["observable_hours"] else None,
                        workload_per_1000=1000*c["emitted_episodes"]/c["eligible_hours"] if c["eligible_hours"] else None,
                        flagged_fraction=c["flagged_hours"]/c["eligible_hours"] if c["eligible_hours"] else None,
                        median_delay_hours=statistics.median(d) if d else None)
        results[method] = dict(pooled=metrics(sum(counts.values(), Counter()), sum(delays.values(), [])),
                               annual={y: metrics(counts[y], delays[y]) for y in YEARS},
                               detected_incident_ids=sorted(hits & fresh), detected_channel_events=sorted(event_keys))
    save(ROOT / "ledger.json", ledgers)
    save(ROOT / "summary.json", results)
    print("Scored four detectors across all six channels and five reporting years.", flush=True)


if __name__ == "__main__":
    main()
