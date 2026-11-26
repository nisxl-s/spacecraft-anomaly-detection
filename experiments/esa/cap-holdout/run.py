"""Evaluate a fixed seven-day duration on frozen hourly ESA replays."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = 7 * 24


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def bounded_records(active):
    bounded = np.zeros_like(active)
    edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        bounded[start:min(stop, start + HORIZON)] = True
    return bounded


def evaluate(panel):
    result = {"eligible_hours": 0, "bounded_hours": 0, "pages": 0, "events": [],
              "longest_bounded_hours": 0}
    events = json.loads((panel / "event-ledger.json").read_text())
    for channel, checksums in json.loads((panel / "seal.json").read_text())["channels"].items():
        path = panel / f"{channel}.npz"
        assert digest(path) == checksums["replay"]
        with np.load(path, allow_pickle=False) as replay:
            legacy = replay["legacy"]
            sustained = replay["sustained"]
            bounded = bounded_records(sustained)
            dates = replay["dates"]
            original = dates[replay["original_pages"]]
            delivered = dates[legacy & ~np.r_[False, legacy[:-1]]]
            np.testing.assert_array_equal(delivered, original)
            result["eligible_hours"] += int(replay["valid"].sum())
        result["pages"] += len(delivered)
        result["bounded_hours"] += int(bounded.sum())
        edges = np.flatnonzero(np.diff(np.r_[False, bounded, False]))
        result["longest_bounded_hours"] = max(
            result["longest_bounded_hours"], int(np.max(edges[1::2] - edges[::2], initial=0))
        )
        for event in events:
            if event["channel"] == channel and event["status"] == "eligible":
                timestamps = pd.to_datetime(dates, utc=True)
                start = timestamps.searchsorted(pd.Timestamp(event["start"]))
                stop = timestamps.searchsorted(pd.Timestamp(event["end"]), side="right")
                result["events"].append({**event, "bounded": bool(bounded[start:stop].any())})
    result["bounded_occupancy"] = result["bounded_hours"] / result["eligible_hours"]
    result["preserved_all_sustained_overlaps"] = all(
        row["bounded"] for row in result["events"] if row["sustained"]
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("panel", type=Path)
    print(json.dumps(evaluate(parser.parse_args().panel), indent=2))
