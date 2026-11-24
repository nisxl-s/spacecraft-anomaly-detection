"""Score label-blind, sealed Mission 2 telemetry and page ledgers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "esa-extension"))
import score
from replay import ingest, read, save
from train_transfer import CHANNELS


def main():
    seal = read(ROOT / "transfer-seal.json")
    score.ingest.CHANNELS = CHANNELS
    import annotations
    annotations.CHANNELS = CHANNELS
    score.UPLOAD = ROOT / "inputs"
    source = read(ROOT / "residual/replay-seal.json")
    score.GRID = pd.date_range(source["grid_start"], source["end_inclusive"], freq="h")
    boundary = pd.Timestamp(source["new_start_exclusive"])
    score.END = score.GRID[-1]
    score.NEW = np.asarray(score.GRID > boundary)
    score.YEARS = tuple(range(boundary.year, score.END.year + 1))
    # The original scorer checks a source manifest; retain the already sealed
    # detector hash and provide separately verified annotation hashes for scoring.
    label_hashes = {name: ingest.checksum(score.UPLOAD / name) for name in ("labels.csv", "anomaly_types.csv")}
    save(ROOT / "label-inputs.json", label_hashes)
    datasets = {"residual": "residual", "opening": "page", "escalation": "page"}
    outputs = {}
    for name, method in datasets.items():
        directory = ROOT / name
        source_seal = directory / "replay-seal.json"
        expected = seal["residual_seal_sha256"] if name == "residual" else seal["policy_seals"][name]
        assert ingest.checksum(source_seal) == expected
        local = read(source_seal)
        assert local["freeze_sha256"] == seal["freeze_sha256"]
        assert all(ingest.checksum(directory / f"{c}_replay.npz") == digest
                   for c, digest in local["replay_sha256"].items())
        if name != "residual":
            assert ingest.checksum(directory / "tickets.json") == local["tickets_sha256"]
        # Reuse the common event scorer with this mission's previously verified hashes.
        score.ROOT = directory
        score.METHODS = (method,)
        score.BUDGET = ROOT
        original_read, original_checksum = score.read, ingest.checksum

        def read_manifest(path):
            if path == ROOT / "freeze.json":
                return {"source_sha256": label_hashes}
            return original_read(path)

        assert original_checksum(ROOT / "freeze.json") == local["freeze_sha256"]
        score.read = read_manifest
        try:
            score.main()
        finally:
            score.read = original_read
        result = read(directory / "summary.json")[method]
        if name != "residual":
            tickets = read(directory / "tickets.json")
            times = [score.GRID[a["index"]] for ticket in tickets for a in ticket["actions"]
                     if a["page"] and score.NEW[a["index"]]]
            assert len(set(times)) == len(times)
            for year, row in result["annual"].items():
                row["mission_pages"] = sum(t.year == int(year) for t in times)
                row["mission_pages_per_1000"] = 1000*row["mission_pages"]/row["eligible_hours"] if row["eligible_hours"] else None
            result["pooled"]["mission_pages"] = len(times)
            result["pooled"]["mission_pages_per_1000"] = 1000*len(times)/result["pooled"]["eligible_hours"]
        outputs[name] = result
    base = set(outputs["residual"]["detected_incident_ids"])
    for method in ("opening", "escalation"):
        found = set(outputs[method]["detected_incident_ids"])
        outputs[method]["detector_incidents_retained"] = sorted(found & base)
        outputs[method]["detector_incidents_missed"] = sorted(base - found)
    save(ROOT / "summary.json", outputs)
    for method, result in outputs.items():
        print(method, result["pooled"], flush=True)


if __name__ == "__main__":
    main()
