"""Replay mission-level paging, with optional cross-channel escalation."""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / "esa-extension"
sys.path.insert(0, str(EXT))
import score
from replay import GRID, ingest, read, save
from evaluate import episodes


def replay(destination, quiet_hours, escalation, window, cooldown):
    seal = read(EXT / "replay-seal.json")
    channels = sorted(seal["replay_sha256"])
    eligible, alarms = {}, {}
    starts = {}
    for channel in channels:
        path = EXT / f"{channel}_replay.npz"
        assert ingest.checksum(path) == seal["replay_sha256"][channel]
        with np.load(path, allow_pickle=False) as data:
            eligible[channel] = data["eligible"]
            alarms[channel] = data["residual_emitted"]
        for start, stop in episodes(alarms[channel]):
            starts.setdefault(start, []).append((channel, stop))
    all_eligible = np.logical_and.reduce(list(eligible.values()))
    any_alarm = np.logical_or.reduce(list(alarms.values()))
    flags = {c: np.zeros(len(GRID), dtype=bool) for c in channels}
    tickets = []
    ticket = None
    quiet = 0
    for i in range(len(GRID)):
        items = starts.get(i, [])
        if not all_eligible[i]:
            if ticket is not None:
                ticket["closed"] = i
                ticket["close_reason"] = "coverage_gap"
            ticket, quiet = None, 0
        if items:
            opening = ticket is None
            if opening:
                ticket = dict(opened=i, actions=[], closed=None)
                tickets.append(ticket)
                last_seen = {}
                last_page = i
                escalated = False
            evidence = set(c for c, _ in items)
            armed = not escalated if cooldown is None else i - last_page >= cooldown
            if cooldown is None or armed:
                last_seen.update(dict.fromkeys(evidence, i))
            quorum = sum(i - hour < window for hour in last_seen.values())
            upgrade = not opening and armed and escalation and quorum >= escalation
            paged = opening or bool(upgrade)
            escalated |= bool(upgrade)
            if paged:
                last_page = i
                if cooldown is not None:
                    last_seen.clear()
            ticket["actions"].append(dict(index=i, utc=GRID[i].isoformat(), page=paged,
                                          kind="opening" if opening else "escalation" if upgrade else "update",
                                          episodes=items))
            if paged:
                for c, stop in items:
                    flags[c][i:stop] = True
        if ticket is not None:
            quiet = quiet + 1 if all_eligible[i] and not any_alarm[i] else 0
            if quiet >= quiet_hours:
                ticket["closed"] = i + 1
                ticket["close_reason"] = "quiet"
                ticket = None
    if ticket is not None:
        ticket["close_reason"] = "endpoint_censored"
    destination.mkdir(exist_ok=True)
    hashes = {}
    for channel in channels:
        path = destination / f"{channel}_replay.npz"
        np.savez_compressed(path, eligible=eligible[channel], page_emitted=flags[channel], page_raw=flags[channel])
        hashes[channel] = ingest.checksum(path)
    save(destination / "tickets.json", tickets)
    save(destination / "replay-seal.json", dict(freeze_sha256=seal["freeze_sha256"], replay_sha256=hashes,
         tickets_sha256=ingest.checksum(destination / "tickets.json"), source_seal_sha256=ingest.checksum(EXT/"replay-seal.json"),
         quiet_hours=quiet_hours, escalation=escalation, window=window, cooldown=cooldown))
    assert sum(len(a["episodes"]) for t in tickets for a in t["actions"]) == sum(len(episodes(f)) for f in alarms.values())
    return tickets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--quiet", type=int, default=24)
    parser.add_argument("--escalation", type=int, default=0)
    parser.add_argument("--window", type=int, default=1)
    parser.add_argument("--cooldown", type=int)
    args = parser.parse_args()
    destination = HERE / args.name
    tickets = replay(destination, args.quiet, args.escalation, args.window, args.cooldown)
    print("Sealed pages and updates before event scoring.", flush=True)
    score.ROOT, score.METHODS = destination, ("page",)
    score.main()
    reference = read(EXT / "summary.json")["residual"]
    page = read(destination / "summary.json")["page"]
    annual = {}
    for y, data in page["annual"].items():
        actions = [a for t in tickets for a in t["actions"] if score.NEW[a["index"]] and a["utc"].startswith(y)]
        pages = sum(a["page"] for a in actions)
        days = int(sum(score.NEW & (GRID.year == int(y))))/24
        annual[y] = dict(pages=pages, updates=len(actions)-pages,
                         rate=1000*pages/data["eligible_hours"], pages_per_mission_day=pages/days)
    valid = [t for t in tickets if any(score.NEW[a["index"]] for a in t["actions"])]
    durations = [(t["closed"] or len(GRID))-t["opened"] for t in valid]
    output = dict(annual=annual, pooled_pages=sum(r["pages"] for r in annual.values()),
                  page_detection=page, ticket_evidence=reference,
                  median_ticket_hours=float(np.median(durations)), max_ticket_hours=max(durations),
                  tickets=len(valid), updates_per_ticket=sum(r["updates"] for r in annual.values())/len(valid),
                  lost_incidents=sorted(set(reference["detected_incident_ids"])-set(page["detected_incident_ids"])),
                  passed=all(r["rate"] <= 2 for r in annual.values())
                    and page["pooled"]["incidents_detected"] == 3 and reference["pooled"]["detected"] >= 14)
    save(destination / "comparison.json", output)
    print(args.name, output["pooled_pages"], output["lost_incidents"],
          [round(v["rate"],3) for v in annual.values()], output["median_ticket_hours"], output["max_ticket_hours"], output["passed"])


if __name__ == "__main__":
    main()
