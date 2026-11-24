"""Replay and score per-channel 24-hour repeat-notification suppression."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXT = ROOT.parent / "esa-extension"
sys.path.insert(0, str(EXT))
import score
from replay import GRID, ingest, read, save
from evaluate import episodes

QUIET_HOURS = 24


def main():
    source = read(EXT / "replay-seal.json")
    decisions, hashes = {}, {}
    for c, digest in source["replay_sha256"].items():
        path = EXT / f"{c}_replay.npz"
        assert ingest.checksum(path) == digest
        with np.load(path, allow_pickle=False) as saved:
            eligible = saved["eligible"]
            emitted = saved["residual_emitted"]
        flags = np.zeros_like(emitted)
        gaps = np.r_[0, (~eligible).cumsum()]
        previous_end = None
        decisions[c] = []
        for start, stop in episodes(emitted):
            send = (previous_end is None or start - previous_end >= QUIET_HOURS
                    or gaps[start] != gaps[previous_end])
            if send:
                flags[start:stop] = True
            decisions[c].append(dict(start=start, stop_exclusive=stop, utc=GRID[start].isoformat(), sent=bool(send)))
            previous_end = stop
        output = ROOT / f"{c}_replay.npz"
        np.savez_compressed(output, eligible=eligible, notification_emitted=flags, notification_raw=flags)
        hashes[c] = ingest.checksum(output)
    save(ROOT / "decisions.json", decisions)
    seal = dict(freeze_sha256=source["freeze_sha256"], replay_sha256=hashes,
                decisions_sha256=ingest.checksum(ROOT / "decisions.json"),
                source_seal_sha256=ingest.checksum(EXT / "replay-seal.json"), quiet_hours=QUIET_HOURS)
    save(ROOT / "replay-seal.json", seal)
    print("Sealed notification decisions before reading labels.", flush=True)
    score.ROOT = ROOT
    score.METHODS = ("notification",)
    score.main()
    frozen = read(EXT / "summary.json")["residual"]
    notification = read(ROOT / "summary.json")["notification"]
    f, n = frozen["pooled"], notification["pooled"]
    assert n["eligible_hours"] == f["eligible_hours"]
    for y, metrics in notification["annual"].items():
        assert metrics["emitted_episodes"] == sum(
            row["sent"] and row["utc"].startswith(y) and bool(score.NEW[row["start"]])
            for rows in decisions.values() for row in rows)
    lost = sorted(set(map(tuple, frozen["detected_channel_events"]))
                  - set(map(tuple, notification["detected_channel_events"])))
    gained = sorted(set(map(tuple, notification["detected_channel_events"]))
                    - set(map(tuple, frozen["detected_channel_events"])))
    ledger = read(EXT / "ledger.json")["residual"]
    diagnostics = []
    for channel, event in lost:
        row = ledger[channel]
        matched = next(ai for ti, ai, _ in row["matches"] if row["events"][ti] == event)
        start, stop = row["alarms"][matched]
        prior = next((item for item in decisions[channel] if item["start"] <= start < item["stop_exclusive"]), None)
        diagnostics.append(dict(channel=channel, event=event, original_alarm=[start, stop], notification=prior))
    comparison = dict(frozen=frozen, notification=notification, lost_channel_events=lost,
                      gained_channel_events=gained, lost_event_diagnostics=diagnostics,
                      lost_incidents=sorted(set(frozen["detected_incident_ids"])-set(notification["detected_incident_ids"])),
                      passed=(all(1000*r["emitted_episodes"] <= 2*r["eligible_hours"] for r in notification["annual"].values())
                              and n["incidents_detected"] == 3 and n["detected"] >= 14))
    save(ROOT / "comparison.json", comparison)
    print(n)
    print([(y, r["workload_per_1000"],r["detected"]) for y,r in notification["annual"].items()])
    print(comparison["lost_incidents"], lost, gained, comparison["passed"])


if __name__ == "__main__":
    main()
