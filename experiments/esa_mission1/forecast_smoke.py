"""Exercise fit and score stages on a small generated six-channel input."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from forecast import METHODS, smooth_at_targets
from ingest import CHANNELS, HISTORY, checksum


def generate_input(directory, stage):
    directory.mkdir()
    size = 390
    timeline = np.arange(size)
    eligible = timeline >= HISTORY
    if stage == "fit":
        masks = {"fit": timeline < 270, "validation": timeline >= 270}
    else:
        masks = {stage: timeline >= 270}
    rows = []
    for ordinal, channel in enumerate(CHANNELS):
        target = np.sin(timeline / 11 + ordinal) + 0.1 * np.cos(timeline / 5)
        path = directory / f"{channel}.npz"
        np.savez_compressed(path, timestamp_unix_ns=timeline * 3_600_000_000_000,
                            values=target, eligible_target=eligible, **masks)
        rows.append({"channel": channel, "output_sha256": checksum(path)})
    (directory / "manifest.json").write_text(json.dumps({
        "phase": stage, "data_source": {"kind": "synthetic_fixture_only"}, "channels": rows,
    }))


def verify(directory, stage, expected_scored):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    assert manifest["stage"] == stage and manifest["synthetic_smoke"]
    assert manifest["fit_max_epochs"] == 1
    assert tuple(manifest["channels"]) == CHANNELS
    for channel, row in manifest["channels"].items():
        path = directory / f"{channel}_scores.npz"
        if stage == "fit":
            assert row["fit_targets"] == 20 and row["validation_targets"] == expected_scored
            assert row["epochs_run"] == row["best_epoch"] == 1
            assert checksum(directory / f"{channel}.pt") == row["checkpoint_sha256"]
            assert checksum(path) == row["validation_scores_sha256"]
        else:
            assert row["scored_targets"] == expected_scored and checksum(path) == row["score_sha256"]
        assert all(np.isfinite(value) and value >= 0 for value in row["base_thresholds"].values())
        with np.load(path, allow_pickle=False) as data:
            assert len(data["target_grid_index"]) == expected_scored
            for method in METHODS:
                raw, filtered = data[f"{method}_raw"], data[f"{method}_smoothed"]
                assert np.isfinite(raw).all() and np.isfinite(filtered).all()
                assert raw[0] == filtered[0]
    return {"channels": len(CHANNELS), "score_targets_per_channel": expected_scored}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    np.testing.assert_allclose(smooth_at_targets(np.array([3., 6., 0., 9.]), np.array([5, 6, 13, 14])),
                               [3., 3. + 6 / 31, 0., 18 / 31])
    result = {"synthetic_fixture_only": True, "maximum_epochs": 1, "esa_dataset_scored": False}
    with TemporaryDirectory() as temporary:
        workspace = Path(temporary)
        for name in ("fit", "calibration", "test"):
            input_dir, output_dir = workspace / f"{name}_in", workspace / f"{name}_out"
            generate_input(input_dir, name)
            cmd = [sys.executable, str(Path(__file__).with_name("forecast.py")), name,
                   "--input", str(input_dir), "--output", str(output_dir), "--synthetic-smoke"]
            if name != "fit":
                cmd += ["--models", str(workspace / "fit_out")]
            subprocess.run(cmd, check=True)
            result[name] = verify(output_dir, name, 120)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
