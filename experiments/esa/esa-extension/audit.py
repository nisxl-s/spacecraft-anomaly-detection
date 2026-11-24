"""Verify extension inputs and aggregated metrics."""
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent


def load(name):
    return json.loads((R / name).read_text())


seal = load("replay-seal.json")
summary, ledger = load("summary.json"), load("ledger.json")
assert len(seal["replay_sha256"]) == 6
for c, digest in seal["replay_sha256"].items():
    assert hashlib.sha256((R / f"{c}_replay.npz").read_bytes()).hexdigest() == digest
for m, result in summary.items():
    p = result["pooled"]
    assert len(ledger[m]) == 6 and len(result["annual"]) == 5
    for key in ("eligible_hours", "observable_hours", "events", "detected", "incidents", "incidents_detected",
                "matched_alarms", "false_alarms", "emitted_episodes", "flagged_hours"):
        assert p[key] == sum(v[key] for v in result["annual"].values())
    assert p["events"] == sum(len(v["events"]) for v in ledger[m].values())
    assert p["detected"] == p["matched_alarms"] == sum(len(v["matches"]) for v in ledger[m].values())
    assert p["false_alarms"] + p["matched_alarms"] == sum(len(v["alarms"]) for v in ledger[m].values())
    assert p["incidents_detected"] == len(result["detected_incident_ids"])
print("Verified six replay hashes, 24 channel-detector ledgers and five-year rollups.")
