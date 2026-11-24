"""Replay causal monthly calibration on the already examined ESA period."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXT = ROOT.parent / "esa-extension"
sys.path.insert(0, str(EXT))
from replay import BOUNDARY, GRID, ingest, read, save
from evaluate import emit, episodes


def main():
    freeze = read(EXT / "freeze.json")
    original = read(EXT / "replay-seal.json")
    assert ingest.checksum(EXT / "freeze.json") == original["freeze_sha256"]
    method = freeze["methods"]["residual"]
    base = method["candidates"][40]["thresholds"]
    assert method["candidates"][40]["multiplier"] == 1
    data = {}
    for channel, digest in original["replay_sha256"].items():
        path = EXT / f"{channel}_replay.npz"
        assert ingest.checksum(path) == digest
        with np.load(path, allow_pickle=False) as saved:
            eligible = saved["eligible"]
            scores = np.full(len(GRID), np.nan)
            scores[saved["positions"]] = saved["residual_scores"]
            data[channel] = (eligible, scores, saved["residual_raw"].copy())
    shadow = {}
    for k in range(-40, 41):
        multiplier = 10**(k/40)
        starts = np.zeros(len(GRID), dtype=np.int64)
        flagged = starts.copy()
        for c,(eligible,scores,_) in data.items():
            flags = emit(scores > base[c]*multiplier, eligible)
            starts += flags & ~np.r_[False, flags[:-1]]
            flagged += flags
        shadow[multiplier] = (np.r_[0, starts.cumsum()], np.r_[0, flagged.cumsum()])
    schedule = []
    for month in pd.date_range("2009-01-01", "2013-12-01", freq="MS", tz="UTC"):
        start = max(month, BOUNDARY + pd.Timedelta(hours=1))
        end = month + pd.offsets.MonthBegin()
        a,b = GRID.searchsorted([month-pd.Timedelta(days=90), month])
        lo,hi = GRID.searchsorted([start,end])
        exposure = sum(int(e[a:b].sum()) for e, _, _ in data.values())
        options = []
        for multiplier,(starts, flags) in shadow.items():
            count = int(starts[b] - starts[a])
            flagged = int(flags[b] - flags[a])
            if 0 < 1000*count <= 2*exposure and 20*flagged <= exposure:
                options.append((count,-flagged,multiplier))
        count,neg_flagged,multiplier = max(options, default=(0,0,None))
        for c,(_,scores,raw) in data.items():
            raw[lo:hi] = False if multiplier is None else scores[lo:hi] > base[c]*multiplier
        schedule.append(dict(month=month.isoformat(), history_start=GRID[a].isoformat(),
                             history_end_exclusive=month.isoformat(), exposure=exposure,
                             episodes=count, flagged=-neg_flagged, multiplier=multiplier,
                             thresholds={c:base[c]*multiplier if multiplier is not None else None for c in data}))
    save(ROOT / "monthly-thresholds.json", schedule)
    hashes = {}
    for c,(eligible,_,raw) in data.items():
        path = ROOT / f"{c}_replay.npz"
        np.savez_compressed(path, eligible=eligible, adaptive_raw=raw, adaptive_emitted=emit(raw,eligible))
        hashes[c] = ingest.checksum(path)
    save(ROOT / "replay-seal.json", dict(freeze_sha256=original["freeze_sha256"], replay_sha256=hashes,
                                       schedule_sha256=ingest.checksum(ROOT/"monthly-thresholds.json"),
                                       source_seal_sha256=ingest.checksum(EXT/"replay-seal.json")))
    print("Sealed adaptive replay and 60 monthly selections.", flush=True)


if __name__ == "__main__":
    main()
