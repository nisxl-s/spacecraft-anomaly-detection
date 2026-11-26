"""Existing ESA hourly preprocessing and source hashing."""
import hashlib

import numpy as np
import pandas as pd

STALE = pd.Timedelta(hours=2)
HISTORY = 250

def checksum(path, algorithm="sha256"):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def utc_index(raw):
    index = pd.DatetimeIndex(pd.to_datetime(raw, utc=True, format="mixed", errors="raise"))
    if index.hasnans:
        raise ValueError("Missing timestamp")
    return index


def channel_series(frame, channel):
    timestamps = utc_index(frame.index)
    values = pd.to_numeric(frame[channel], errors="raise").to_numpy(dtype=float)
    return pd.Series(values, index=timestamps, name=channel)


def prepare_hourly(series, start, end):
    if not series.index.is_monotonic_increasing:
        raise ValueError(f"Unordered observations for {series.name}")
    if series.index.has_duplicates:
        counts = series[series.index.duplicated(keep=False)].groupby(level=0).nunique(dropna=False)
        if counts.gt(1).any():
            raise ValueError(f"Conflicting duplicate observations for {series.name}")
        series = series.loc[~series.index.duplicated(keep="first")]
    grid = pd.date_range(start, end, freq="1h")
    source = series.index.get_indexer(grid, method="pad", tolerance=STALE)
    available = source >= 0
    values = np.full(len(grid), np.nan, dtype=float)
    values[available] = series.to_numpy()[source[available]]
    if not (series.index[source[available]] <= grid[available]).all():
        raise AssertionError("Future observation selected")
    available = np.isfinite(values)
    values[~available] = np.nan
    counter = np.concatenate(([0], np.cumsum(~available)))
    eligible = np.zeros(len(grid), dtype=bool)
    eligible[HISTORY:] = (counter[HISTORY + 1:] - counter[:-HISTORY - 1]) == 0
    return grid, values, eligible
