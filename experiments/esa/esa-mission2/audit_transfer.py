"""Verify frozen Mission 2 pages and detector coverage."""
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


seal = read("transfer-seal.json")
source = read("residual/replay-seal.json")
assert digest("freeze.json") == seal["freeze_sha256"]
assert digest("residual/replay-seal.json") == seal["residual_seal_sha256"]
assert digest("replay_transfer.py") == seal["runner_sha256"]
assert digest("../esa-ticket/run.py") == seal["ticket_runner_sha256"]
grid = pd.date_range(source["grid_start"], source["end_inclusive"], freq="h")
new = grid > pd.Timestamp(source["new_start_exclusive"])
summary = read("summary.json")
starts = defaultdict(set)
coverage, busy = [], []
for c in source["replay_sha256"]:
    with np.load(ROOT / "residual" / f"{c}_replay.npz", allow_pickle=False) as f:
        eligible, emitted = f["eligible"], f["residual_emitted"]
        coverage.append(eligible)
        busy.append(emitted)
        for i in np.flatnonzero(emitted & ~np.r_[False, emitted[:-1]]):
            starts[int(i)].add(c)
all_eligible = np.logical_and.reduce(coverage)
quiet = all_eligible & ~np.logical_or.reduce(busy)
closures = np.flatnonzero(np.convolve(quiet.astype(int), np.ones(24, dtype=int), "valid") == 24) + 24


def independent_pages(escalate):
    pages, history = [], deque()
    previous = None
    upgraded = False
    for i, batch in sorted(starts.items()):
        if previous is None:
            opening = True
        else:
            index = np.searchsorted(closures, previous, side="right")
            opening = not all_eligible[previous+1:i+1].all() or (
                index < len(closures) and closures[index] <= i)
        if opening:
            pages.append(i)
            history.clear()
            upgraded = False
        history.append((i, batch))
        while i - history[0][0] >= 6:
            history.popleft()
        if escalate and not opening and not upgraded and len(set().union(*(c for _, c in history))) >= 3:
            pages.append(i)
            upgraded = True
        previous = i
    return [i for i in pages if new[i]]


for policy, expected in seal["policy_seals"].items():
    assert digest(f"{policy}/replay-seal.json") == expected
    record = read(f"{policy}/replay-seal.json")
    assert record["tickets_sha256"] == digest(f"{policy}/tickets.json")
    tickets = read(f"{policy}/tickets.json")
    pages = []
    evidence = {c: np.zeros(len(grid), dtype=bool) for c in source["replay_sha256"]}
    credited = {c: np.zeros(len(grid), dtype=bool) for c in evidence}
    for ticket in tickets:
        assert sum(a["page"] for a in ticket["actions"]) in (1, 2)
        last_start = {}
        for action in ticket["actions"]:
            i = action["index"]
            for channel, stop in action["episodes"]:
                assert not evidence[channel][i:stop].any()
                evidence[channel][i:stop] = True
                last_start[channel] = i
                if action["page"]:
                    credited[channel][i:stop] = True
            if action["kind"] == "escalation":
                assert policy == "escalation" and sum(i-t < 6 for t in last_start.values()) >= 3
            if new[i] and action["page"]:
                pages.append(i)
    assert pages == independent_pages(policy == "escalation")
    assert len(set(pages)) == len(pages) == summary[policy]["pooled"]["mission_pages"]
    for y, row in summary[policy]["annual"].items():
        assert row["mission_pages"] == sum(grid[i].year == int(y) for i in pages)
    for c in evidence:
        assert digest(f"{policy}/{c}_replay.npz") == record["replay_sha256"][c]
        assert digest(f"residual/{c}_replay.npz") == source["replay_sha256"][c]
        with np.load(ROOT / "residual" / f"{c}_replay.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(f["residual_emitted"], evidence[c])
        with np.load(ROOT / policy / f"{c}_replay.npz", allow_pickle=False) as f:
            np.testing.assert_array_equal(f["page_emitted"], credited[c])
print("Verified both policy seals, all alarm evidence and page-credit masks, and annual mission page counts.")
