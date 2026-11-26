import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from frozen_alarm import confirmed_episodes
from hourly import channel_series, checksum, prepare_hourly

ROOT = Path(__file__).resolve().parent
INPUT = ROOT.parent.parent / "esa-mission2" / "inputs"
CHANNELS = ["channel_81", "channel_97", "channel_96"]
PRIOR = [f"channel_{i}" for i in range(9, 21)]
SPLIT = pd.Timestamp("2001-10-01", tz="UTC")


def write(filename, value):
    (ROOT / filename).write_text(json.dumps(value, indent=2) + "\n")


def pages(flags):
    return np.flatnonzero(flags & ~np.r_[False, flags[:-1]])


def freeze():
    seal = {}
    for channel in CHANNELS:
        source = INPUT / f"{channel}.zip"
        series = channel_series(pd.read_pickle(source, compression="zip"), channel)
        dates, values, _ = prepare_hourly(series, series.index.min().ceil("h"), series.index.max().floor("h"))
        scores = np.abs(np.diff(values, prepend=np.nan))
        finite = np.isfinite(scores)
        threshold = float(np.quantile(scores[finite & (dates <= SPLIT)], .99))
        valid = finite[dates > SPLIT]
        high = (scores[dates > SPLIT] > threshold) & valid
        dates = dates[dates > SPLIT]
        edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
        legacy = np.zeros_like(high)
        for start, stop in zip(edges[::2], edges[1::2], strict=True):
            legacy[start:stop] = confirmed_episodes(high[start:stop], required=2)
        sustained = high & np.r_[False, high[:-1]]
        original_pages = pages(legacy)
        assert np.all(~legacy | sustained)
        path = ROOT / f"{channel}.npz"
        np.savez_compressed(path, dates=dates.as_unit("ns").asi8, valid=valid, legacy=legacy,
                            sustained=sustained, original_pages=original_pages)
        seal[channel] = dict(source=checksum(source), replay=checksum(path), threshold=threshold)
        print(channel, len(original_pages), int(sustained.sum()), flush=True)
    write("seal.json", {"channels": seal, "protocol": checksum(ROOT / "protocol.md"),
                        "runner": checksum(Path(__file__)), "paging_rule": checksum(ROOT / "frozen_alarm.py")})


def score():
    seal = json.loads((ROOT / "seal.json").read_text())
    assert seal["protocol"] == checksum(ROOT / "protocol.md")
    assert seal["runner"] == checksum(Path(__file__))
    assert seal["paging_rule"] == checksum(ROOT / "frozen_alarm.py")
    seal = seal["channels"]
    labels = pd.read_csv(INPUT / "labels.csv")
    categories = pd.read_csv(INPUT / "anomaly_types.csv").set_index("ID").Category
    previously_seen = set(labels.loc[labels.Channel.isin(PRIOR), "ID"])
    rows, metrics = [], []
    panel = labels.loc[labels.Channel.isin(CHANNELS)].copy()
    for field in ("StartTime", "EndTime"):
        panel[field] = pd.to_datetime(panel[field], utc=True, format="mixed")
    for channel in CHANNELS:
        path = ROOT / f"{channel}.npz"
        assert checksum(path) == seal[channel]["replay"]
        with np.load(path, allow_pickle=False) as saved:
            dates, valid, legacy, sustained, original = (saved[key] for key in
                ("dates", "valid", "legacy", "sustained", "original_pages"))
        delivered = pages(legacy)
        np.testing.assert_array_equal(delivered, original)
        timestamps = pd.to_datetime(dates, utc=True)
        transitions = np.flatnonzero(np.diff(np.r_[False, sustained, False]))
        metrics.append(dict(channel=channel, eligible=int(valid.sum()), paging=int(legacy.sum()),
                            record=int(sustained.sum()), pages=len(delivered),
                            max_record=int(np.max(transitions[1::2]-transitions[::2], initial=0))))
        for event in panel.loc[panel.Channel.eq(channel) & panel.StartTime.gt(SPLIT)].itertuples():
            if categories[event.ID] != "Anomaly":
                continue
            start = int(timestamps.searchsorted(event.StartTime))
            end = int(timestamps.searchsorted(event.EndTime, side="right"))
            row = dict(channel=channel, id=event.ID, start=event.StartTime.isoformat(), end=event.EndTime.isoformat())
            if event.ID in previously_seen:
                row["status"] = "prior incident"
            elif start == end or event.EndTime > timestamps[-1] or not valid[start:end].all():
                row["status"] = "no complete hourly coverage"
            else:
                row.update(status="eligible", legacy=bool(legacy[start:end].any()),
                           sustained=bool(sustained[start:end].any()),
                           page=bool(np.any((delivered >= start) & (delivered < end))))
            rows.append(row)
    eligible = [row for row in rows if row["status"] == "eligible"]
    totals = {key: sum(row[key] for row in metrics) for key in ("eligible", "paging", "record", "pages")}
    summary = dict(totals=totals, eligible_events=len(eligible), incidents=len({x["id"] for x in eligible}),
                   legacy_hits=sum(row["legacy"] for row in eligible),
                   sustained_hits=sum(row["sustained"] for row in eligible),
                   page_hits=sum(row["page"] for row in eligible),
                   unpaged_gain=sum(row["sustained"] and not row["legacy"] and not row["page"] for row in eligible),
                   exclusions=pd.Series([row["status"] for row in rows]).value_counts().to_dict(),
                   metrics=metrics)
    summary["annotation_hashes"] = {name: checksum(INPUT / name)
                                    for name in ("labels.csv", "anomaly_types.csv")}
    summary["gates"] = {
        "exact_paging_parity": True,
        "additional_event_overlap": summary["sustained_hits"] > summary["legacy_hits"],
        "occupancy_at_most_10_percent": totals["record"] <= .10 * totals["eligible"],
    }
    write("event-ledger.json", rows)
    write("summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("freeze", "score"))
    {"freeze": freeze, "score": score}[parser.parse_args().step]()
