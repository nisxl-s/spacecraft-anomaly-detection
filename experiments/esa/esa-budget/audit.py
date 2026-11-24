"""Verify frozen selection, replay hashes and reported totals."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text())


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


freeze, seal = load("freeze.json"), load("test-seal.json")
ledger, summary = load("ledger.json"), load("summary.json")
assert freeze["protocol_sha256"] == digest("protocol.json")
assert all(digest(name) == expected for name, expected in freeze["code_sha256"].items())
assert digest("freeze.json") == seal["freeze_sha256"]
exposure = freeze["calibration_exposure"]
for method, record in freeze["methods"].items():
    assert len(record["candidates"]) == 82
    finite = [r for r in record["candidates"] if r["episodes"] and
              1000*r["episodes"] <= 2*exposure and 20*r["flagged"] <= exposure]
    expected = max(finite, key=lambda r: (r["episodes"], -r["flagged"], r["multiplier"])) if finite else record["candidates"][-1]
    assert record["selected"] == expected
    rows = ledger[method]
    for key in ("tp", "fp", "fn", "exposure", "flagged", "emitted", "streaming_exposure"):
        assert summary[method][key] == sum(row[key] for row in rows.values())
for channel, expected in seal["replay_sha256"].items():
    name = f"{channel}_replay.npz"
    assert digest(name) == expected
    assert digest(f"{channel}_calibration.npz") == freeze["calibration_scores_sha256"][channel]
    with np.load(ROOT / name, allow_pickle=False) as data:
        assert len(data["eligible"]) == 8784
        for method, record in freeze["methods"].items():
            threshold = record["selected"]["thresholds"][channel]
            raw = data[f"{method}_raw"]
            emitted = data[f"{method}_emitted"]
            if threshold is None:
                assert not raw.any()
            else:
                assert np.array_equal(raw[data["positions"]], data[f"{method}_scores"] > threshold)
            assert not (emitted & ~raw).any()
            row = ledger[method][channel]
            assert row["emitted"] == int(emitted.sum())
            assert row["tp"] + row["fn"] == len(row["events"])
            assert row["tp"] + row["fp"] == len(row["alarms"])
print("Verified calibration selection, protocol and replay hashes, and all 24 detector ledgers.")
