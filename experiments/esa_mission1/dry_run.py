"""Exercise the independent-pilot ingestion contract on generated fixtures."""

import argparse
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from annotations import evaluation_inputs, load_annotations
from ingest import CHANNELS, HISTORY, checksum, prepare_hourly


def verify_fixture(workspace):
    inputs = workspace / "fixture"
    inputs.mkdir()
    base = pd.date_range("2005-10-01", "2008-01-01 01:00", freq="1h", tz="UTC")
    observations = np.arange(len(base), dtype=float)
    for offset, channel in enumerate(CHANNELS):
        keep = np.ones(len(base), dtype=bool)
        keep[600:606] = False
        pd.DataFrame({"timestamp": base[keep] + pd.Timedelta(minutes=10),
                      channel: observations[keep] + offset}).to_csv(inputs / f"{channel}.csv", index=False)
    wrong_archive = workspace / "ESA-Mission1.zip"
    wrong_archive.write_bytes(b"synthetic input must fail before any archive member is opened")
    rejected_output = workspace / "wrong-archive"
    rejected = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("ingest.py")),
         "--official-archive", str(wrong_archive), "--phase", "fit", "--output", str(rejected_output)],
        text=True, capture_output=True, check=False,
    )
    assert rejected.returncode != 0 and "release MD5 mismatch" in rejected.stderr
    assert not rejected_output.exists()
    phases = {}
    for phase in ("fit", "calibration", "test"):
        output = workspace / phase
        subprocess.run([sys.executable, str(Path(__file__).with_name("ingest.py")),
                        "--synthetic-fixture", str(inputs), "--phase", phase, "--output", str(output)], check=True)
        manifest = json.loads((output / "manifest.json").read_bytes())
        assert manifest["data_source"]["kind"] == "synthetic_fixture_only"
        assert tuple(row["channel"] for row in manifest["channels"]) == CHANNELS
        for row in manifest["channels"]:
            channel = row["channel"]
            assert checksum(output / f"{channel}.npz") == row["output_sha256"]
            with np.load(output / f"{channel}.npz", allow_pickle=False) as data:
                values = data["values"]
                eligible = data["eligible_target"]
                assert not eligible[:HISTORY].any()
                assert eligible[HISTORY if phase == "test" else HISTORY + 1]
                assert all(count > 0 for count in row["eligible_targets"].values())
                if phase == "fit":
                    assert np.isnan(values[0])
                    assert values[1] == CHANNELS.index(channel)
                    assert np.isfinite(values[601]) and np.isnan(values[602:607]).all()
                    assert not eligible[602:857].any() and eligible[857]
                if phase == "test":
                    assert pd.to_datetime(data["timestamp_unix_ns"][data["test"]][0], utc=True).isoformat() == "2007-01-01T01:00:00+00:00"
                    assert pd.to_datetime(data["timestamp_unix_ns"][-1], utc=True).isoformat() == "2008-01-01T00:00:00+00:00"
                if phase == "calibration":
                    selected = pd.to_datetime(data["timestamp_unix_ns"][data["calibration"]], utc=True)
                    assert selected[0].isoformat() == "2006-10-01T00:00:00+00:00"
                    assert selected[-1].isoformat() == "2007-01-01T00:00:00+00:00"
        phases[phase] = {row["channel"]: row["eligible_targets"] for row in manifest["channels"]}
    with_duplicates = pd.Series(
        [4.0, 4.0, np.inf],
        index=pd.DatetimeIndex(["2006-01-01 00:00", "2006-01-01 00:00", "2006-01-01 00:10"], tz="UTC"),
        name="channel_41",
    )
    grid_start = pd.Timestamp("2006-01-01 00:00", tz="UTC")
    _, values, _ = prepare_hourly(with_duplicates, grid_start, grid_start + pd.Timedelta(hours=1))
    assert values[0] == 4.0 and np.isnan(values[1])
    with_duplicates.iloc[1] = 5.0
    try:
        prepare_hourly(with_duplicates, grid_start, grid_start + pd.Timedelta(hours=1))
    except ValueError as error:
        assert "Conflicting duplicate" in str(error)
    else:
        raise AssertionError("Conflicting duplicate accepted")
    labels = """ID,Channel,StartTime,EndTime
a,channel_41,2007-01-01T04:00Z,2007-01-01T06:00Z
b,channel_41,2007-01-01T05:00Z,2007-01-01T05:00Z
c,channel_41,2007-01-01T07:00Z,2007-01-01T08:00Z
"""
    types = """ID,Category
a,Anomaly
b,Communication Gap
c,Rare Event
"""
    annotations = load_annotations(StringIO(labels), StringIO(types))
    grid = pd.date_range("2007-01-01 01:00", periods=9, freq="1h", tz="UTC")
    eligible = np.ones(len(grid), dtype=bool)
    emitted = np.ones(len(grid), dtype=bool)
    observable, censored, rare, events = evaluation_inputs(grid, eligible, emitted, annotations["channel_41"])
    assert np.flatnonzero(~observable).tolist() == [4]
    assert np.flatnonzero(~censored).tolist() == [4]
    assert np.flatnonzero(rare).tolist() == [6, 7]
    assert list(events) == ["a"] and np.flatnonzero(events["a"]).tolist() == [3, 5]
    assert eligible.all() and emitted.all()
    return phases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with TemporaryDirectory() as temporary:
        phases = verify_fixture(Path(temporary))
    result = {
        "data": "synthetic_fixture_only",
        "real_esa_scores_generated": False,
        "checks": [
            "publisher digest mismatch rejects an input before creating any official intake output",
            "all six channels and phase-specific target windows validated",
            "off-grid observations carried forward without future lookup",
            "two-hour staleness and 250-observation history recovery verified",
            "exact duplicates collapse; conflicting duplicates fail; newer nonfinite values block carry",
            "anomaly and rare labels separated; quality censor splits observable truth without altering flags",
        ],
        "eligible_target_counts": phases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
