"""Audit the selected mission-ticket policy against the sealed alarm replay."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent
source = root.parent / "esa-extension"
chosen = root / "k3w6"
grid = pd.date_range("2008-01-01T01:00Z", "2013-12-31T23:00Z", freq="h")
boundary = pd.Timestamp("2009-01-01T00:00Z")


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


tickets = read(chosen / "tickets.json")
result = read(chosen / "comparison.json")
seal = read(chosen / "replay-seal.json")
assert digest(chosen / "tickets.json") == seal["tickets_sha256"]
assert digest(source / "replay-seal.json") == seal["source_seal_sha256"]
channels = seal["replay_sha256"]
flags = {c: np.zeros(len(grid), dtype=bool) for c in channels}
evidence = {c: np.zeros(len(grid), dtype=bool) for c in channels}
page_times = []
for ticket in tickets:
    assert ticket["actions"][0]["kind"] == "opening"
    assert 1 <= sum(a["page"] for a in ticket["actions"]) <= 2
    seen = {}
    for a in ticket["actions"]:
        i = a["index"]
        for c, stop in a["episodes"]:
            assert not evidence[c][i:stop].any()
            evidence[c][i:stop] = True
            seen[c] = i
            if a["page"]:
                flags[c][i:stop] = True
        if a["kind"] == "escalation":
            assert sum(i - seen[c] < 6 for c in seen) >= 3
        if a["page"] and grid[i] > boundary:
            page_times.append(i)
assert len(page_times) == len(set(page_times)) == result["pooled_pages"]
for y, metrics in result["annual"].items():
    assert metrics["pages"] == sum(grid[i].year == int(y) for i in page_times)
for c, expected in channels.items():
    path = chosen / f"{c}_replay.npz"
    assert digest(path) == expected
    with np.load(path, allow_pickle=False) as page, np.load(source / path.name, allow_pickle=False) as original:
        np.testing.assert_array_equal(page["page_emitted"], flags[c])
        np.testing.assert_array_equal(evidence[c], original["residual_emitted"])
print("Verified mission pages, escalation quorum, all source-alarm evidence, and yearly page counts.")

# Independently reconstruct ticket boundaries from the alarm-free intervals.
from collections import defaultdict, deque
starts = defaultdict(set)
coverage = []
active = []
for c in channels:
    with np.load(source / f"{c}_replay.npz", allow_pickle=False) as original:
        emitted, eligible = original["residual_emitted"], original["eligible"]
        coverage.append(eligible)
        active.append(emitted)
        for i in np.flatnonzero(emitted & ~np.r_[False, emitted[:-1]]):
            starts[int(i)].add(c)
all_eligible = np.logical_and.reduce(coverage)
busy = np.logical_or.reduce(active)
quiet = all_eligible & ~busy
closed_at = np.flatnonzero(np.convolve(quiet.astype(int), np.ones(24, dtype=int), "valid") == 24) + 24
expected = []
last = None
history = deque()
escalated = False
for i, batch in sorted(starts.items()):
    boundary_hit = last is None or not all_eligible[last + 1:i + 1].all()
    if last is not None:
        j = np.searchsorted(closed_at, last, side="right")
        boundary_hit |= j < len(closed_at) and closed_at[j] <= i
    if boundary_hit:
        expected.append(i)
        history.clear()
        escalated = False
    history.append((i, batch))
    while i - history[0][0] >= 6:
        history.popleft()
    if not boundary_hit and not escalated and len(set().union(*(c for _, c in history))) >= 3:
        expected.append(i)
        escalated = True
    last = i
assert [i for i in expected if grid[i] > boundary] == page_times
page_lookup = {i: a for t in tickets for a in t["actions"] if a["page"] for i in [a["index"]]}
for c, row in read(chosen / "ledger.json")["page"].items():
    for _, ai, _ in row["matches"]:
        start, _ = row["alarms"][ai]
        assert any(channel == c for channel, _ in page_lookup[start]["episodes"])
print("Independently reproduced exact page timestamps and verified every matched alarm began on a page.")
