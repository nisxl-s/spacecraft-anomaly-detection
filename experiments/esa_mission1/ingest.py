"""Validate pinned ESA Mission1 inputs and prepare causal hourly pilot windows."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

import numpy as np
import pandas as pd

MISSION_ARCHIVE_MD5 = "9770ad12ed730238f37c42d5c27ab436"
CHANNELS = tuple(f"channel_{channel}" for channel in range(41, 47))
FIT_START = pd.Timestamp("2005-10-01", tz="UTC")
VALIDATION_START = pd.Timestamp("2006-07-01", tz="UTC")
CALIBRATION_START = pd.Timestamp("2006-10-01", tz="UTC")
OFFICIAL_SPLIT = pd.Timestamp("2007-01-01", tz="UTC")
TEST_END = pd.Timestamp("2008-01-01", tz="UTC")
HISTORY = 250
STALE = pd.Timedelta(hours=2)


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


def load_normalized_channel(path, channel):
    frame = pd.read_csv(path)
    if set(frame.columns) != {"timestamp", channel}:
        raise ValueError(f"Expected timestamp and {channel} columns")
    return channel_series(frame.set_index("timestamp"), channel)


def load_official_channel(archive, channel):
    member = f"ESA-Mission1/channels/{channel}.zip"
    with archive.open(member) as stream:
        # The containing official Mission1 archive is verified before unpickling.
        frame = pd.read_pickle(stream, compression="zip")
    if not isinstance(frame, pd.DataFrame) or channel not in frame or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"Unexpected ESA channel table in {member}")
    return channel_series(frame, channel)


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


def choose_phase_grid(phase):
    if phase == "fit":
        return FIT_START, CALIBRATION_START - pd.Timedelta(hours=1)
    if phase == "calibration":
        return FIT_START, OFFICIAL_SPLIT
    return OFFICIAL_SPLIT + pd.Timedelta(hours=1 - HISTORY), TEST_END


def phase_mask(grid, phase):
    if phase == "fit":
        return {"fit": grid < VALIDATION_START, "validation": grid >= VALIDATION_START}
    if phase == "calibration":
        return {"calibration": grid >= CALIBRATION_START}
    return {"test": grid > OFFICIAL_SPLIT}


def archive_channel_names(archive):
    return {
        name.removesuffix(".zip").rsplit("/", 1)[-1]
        for name in archive.namelist()
        if re.fullmatch(r"ESA-Mission1/channels/channel_\d+\.zip", name)
    }


def source_member_hashes(archive):
    members = [f"ESA-Mission1/channels/{channel}.zip" for channel in CHANNELS]
    members.extend(f"ESA-Mission1/{name}.csv" for name in ("channels", "labels", "anomaly_types"))
    digests = {}
    for name in members:
        if archive.namelist().count(name) != 1:
            raise ValueError(f"Missing or ambiguous source member: {name}")
        with archive.open(name) as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return digests


def check_archive(path):
    actual = checksum(path, "md5")
    if actual != MISSION_ARCHIVE_MD5:
        raise ValueError(f"ESA Mission1 release MD5 mismatch: {actual}")


def save_channel(out, channel, series, phase):
    start, end = choose_phase_grid(phase)
    if phase == "fit":
        series = series.loc[(series.index >= FIT_START) & (series.index < CALIBRATION_START)]
    elif phase == "calibration":
        series = series.loc[(series.index >= FIT_START) & (series.index <= OFFICIAL_SPLIT)]
    else:
        series = series.loc[(series.index >= start - STALE) & (series.index <= TEST_END)]
    grid, values, eligible = prepare_hourly(series, start, end)
    ranges = phase_mask(grid, phase)
    path = out / f"{channel}.npz"
    np.savez_compressed(path, timestamp_unix_ns=grid.as_unit("ns").asi8, values=values,
                        eligible_target=eligible, **ranges)
    return {
        "channel": channel,
        "grid_start_utc": grid[0].isoformat(),
        "grid_end_utc": grid[-1].isoformat(),
        "grid_rows": len(grid),
        "missing_grid_rows": int(np.isnan(values).sum()),
        "eligible_targets": {name: int((eligible & mask).sum()) for name, mask in ranges.items()},
        "output_sha256": checksum(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--official-archive", type=Path)
    source.add_argument("--synthetic-fixture", type=Path, help="Directory of validation-only fixture CSVs")
    parser.add_argument("--phase", choices=("fit", "calibration", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = args.official_archive
    if archive:
        check_archive(archive)
        with zipfile.ZipFile(archive) as release:
            if not set(CHANNELS) <= archive_channel_names(release):
                raise ValueError("Official release is missing a frozen channel")
            member_hashes = source_member_hashes(release)
            args.output.mkdir()
            rows = [save_channel(args.output, channel, load_official_channel(release, channel), args.phase)
                    for channel in CHANNELS]
        data_source = {"kind": "official_esa_mission1_v2", "archive_md5": MISSION_ARCHIVE_MD5,
                       "archive_sha256": checksum(archive)}
    else:
        args.output.mkdir()
        rows = []
        for channel in CHANNELS:
            path = args.synthetic_fixture / f"{channel}.csv"
            row = save_channel(args.output, channel, load_normalized_channel(path, channel), args.phase)
            rows.append(row | {"fixture_sha256": checksum(path)})
        data_source = {"kind": "synthetic_fixture_only"}
    result = {"data_source": data_source, "phase": args.phase, "channels": rows}
    if archive:
        result["source_member_sha256"] = member_hashes
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
