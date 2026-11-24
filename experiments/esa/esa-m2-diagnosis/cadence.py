"""Check whether the hourly last-value representation hid native telemetry variation."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MISSION = ROOT.parent / "esa-mission2"
sys.path.insert(0, str(MISSION))
import replay_transfer as replay

CHANNEL_METRICS = ("span", "max_step", "last_step")


def main():
    trace = json.loads((ROOT / "trace.json").read_text())
    frozen = json.loads((MISSION / "freeze.json").read_text())
    results = []
    for row in trace:
        channel = row["channel"]
        path = MISSION / "inputs" / f"{channel}.zip"
        assert replay.ingest.checksum(path) == frozen["model_records"][channel]["source_sha256"]
        series = replay.ingest.channel_series(pd.read_pickle(path, compression="zip"), channel)
        end = pd.Timestamp(row["last"])
        native = series.loc[replay.CAL_START - pd.Timedelta(hours=2):end]
        grouped = native.resample("h", closed="right", label="right")
        features = pd.DataFrame({"span": grouped.max() - grouped.min(),
                                 "max_step": native.diff().abs().resample("h", closed="right", label="right").max(),
                                 "count": grouped.count(), "last_step": grouped.last().diff().abs()})
        cal = features.loc[replay.CAL_START + pd.Timedelta(hours=1):replay.SPLIT]
        event = features.loc[end]
        overview = {}
        for metric in CHANNEL_METRICS:
            reference = cal.loc[cal["count"] >= 2, metric].dropna()
            value = float(event[metric])
            overview[metric] = dict(value=value, calibration_p99=float(reference.quantile(.99)),
                                    calibration_count=len(reference),
                                    percentile=float((reference <= value).mean()),
                                    above_or_equal_count=int((reference >= value).sum()))
        results.append(dict(channel=channel, hour_end=end.isoformat(), count=int(event["count"]), metrics=overview))
        print(results[-1], flush=True)
    replay.save(ROOT / "cadence.json", results)


if __name__ == "__main__":
    main()
