"""Development trial of native hourly peak-to-peak scores."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MISSION = HERE.parent / "esa-mission2"
OUT = HERE / "range-trial"
sys.path.insert(0, str(MISSION))
import replay_transfer as transfer
sys.path.insert(0, str(HERE.parent / "esa-extension"))
import score
import annotations
from evaluate import emit, episodes

r = transfer


def main():
    OUT.mkdir(exist_ok=True)
    original = json.loads((MISSION / "residual/replay-seal.json").read_text())
    freeze = json.loads((MISSION / "freeze.json").read_text())
    assert r.ingest.checksum(MISSION / "freeze.json") == original["freeze_sha256"]
    grid = pd.date_range(original["grid_start"], original["end_inclusive"], freq="h")
    calibration = grid <= r.SPLIT
    arrays, bases = {}, {}
    for c in r.CHANNELS:
        source = MISSION / "inputs" / f"{c}.zip"
        assert r.ingest.checksum(source) == freeze["model_records"][c]["source_sha256"]
        seal_path = MISSION / "residual" / f"{c}_replay.npz"
        assert r.ingest.checksum(seal_path) == original["replay_sha256"][c]
        with np.load(seal_path, allow_pickle=False) as data:
            eligible = data["eligible"]
        series = r.ingest.channel_series(pd.read_pickle(source,compression="zip"), c)
        start = r.SPLIT - pd.DateOffset(months=6)
        buckets = series.loc[(series.index > start-pd.Timedelta(hours=1)) &
                             (series.index <= grid[-1])].resample("h",closed="right",label="right")
        count = buckets.count()
        span = (buckets.max() - buckets.min()).where(count>=2)
        selection = span.loc[(span.index>=start) & (span.index<r.CAL_START)].dropna().to_numpy()
        assert len(selection)>=100
        bases[c] = float(np.sort(selection)[(99*len(selection)+99)//100-1])
        values = span.reindex(grid).to_numpy()
        arrays[c] = (eligible,values)
    exposure = sum(int((e & calibration).sum()) for e,_ in arrays.values())
    candidates = []
    for k in range(-40,41):
        m = 10**(k/40)
        outputs = [emit(v[calibration]>bases[c]*m,e[calibration]) for c,(e,v) in arrays.items()]
        count = sum(len(episodes(f)) for f in outputs)
        hours = sum(int(f.sum()) for f in outputs)
        candidates.append(dict(multiplier=m,episodes=count,flagged=hours,
                               feasible=0<1000*count<=2*exposure and 20*hours<=exposure))
    selected = max((x for x in candidates if x["feasible"]),
                   key=lambda x:(x["episodes"],-x["flagged"],x["multiplier"]),default=None)
    labels = json.loads((MISSION / "label-inputs.json").read_text())
    for name,digest in labels.items():
        assert r.ingest.checksum(MISSION/"inputs"/name)==digest
    freeze_path=OUT/"freeze.json"
    r.save(freeze_path,dict(source_sha256=labels,source_seal_sha256=r.ingest.checksum(MISSION/"residual/replay-seal.json"),
                            exposure=exposure, bases=bases,candidates=candidates,selected=selected))
    hashes={}
    for c,(e,v) in arrays.items():
        raw=(v>bases[c]*selected["multiplier"]) & e if selected else np.zeros(len(grid),dtype=bool)
        path=OUT/f"{c}_replay.npz"
        np.savez_compressed(path,eligible=e,range_emitted=emit(raw,e),range_raw=raw,score=v)
        hashes[c]=r.ingest.checksum(path)
    r.save(OUT/"replay-seal.json",dict(replay_sha256=hashes,freeze_sha256=r.ingest.checksum(freeze_path)))
    annotations.CHANNELS = score.ingest.CHANNELS = r.CHANNELS
    score.GRID,score.END,score.NEW=grid,grid[-1],np.asarray(~calibration)
    score.YEARS=tuple(range(r.SPLIT.year,grid[-1].year+1))
    score.BUDGET=score.ROOT=OUT
    score.UPLOAD=MISSION/"inputs"
    score.METHODS=("range",)
    score.main()
    result=json.loads((OUT/"summary.json").read_text())["range"]
    assert result["pooled"]["eligible_hours"]==91866
    print(selected,result["pooled"],flush=True)


if __name__=="__main__":
    main()
