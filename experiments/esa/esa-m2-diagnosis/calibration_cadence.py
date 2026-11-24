"""Measure within-hour variation and eligibility using calibration telemetry only."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MISSION = ROOT.parent / "esa-mission2"
sys.path.insert(0, str(MISSION))
import replay_transfer as r


def main():
    result = []
    start, end = r.CAL_START, r.SPLIT
    for channel in ('channel_9', 'channel_12', 'channel_13', 'channel_14'):
        path = MISSION / "inputs" / f"{channel}.zip"
        series = r.ingest.channel_series(pd.read_pickle(path, compression="zip"), channel)
        first = start - pd.Timedelta(hours=r.ingest.HISTORY)
        series = series.loc[first - r.ingest.STALE:end]
        assert series.index.is_monotonic_increasing and series.index.is_unique
        native = series.loc[start:end]
        spacing = np.diff(native.index.as_unit("ns").asi8)/1e9
        _, hourly, hour_eligible = r.ingest.prepare_hourly(series, first, end)
        values = pd.Series(hourly, index=pd.date_range(first,end,freq="h"))
        binned = native.loc[native.index > start].resample("h", closed="right", label="right")
        stats = pd.DataFrame({"low":binned.min(),"high":binned.max(),"count":binned.count()})
        before = values.shift().reindex(stats.index)
        change = (values.reindex(stats.index)-before).abs()
        cutoff = float(change.quantile(.95))
        excursion = np.maximum((stats.low-before).abs(),(stats.high-before).abs())
        multi = (stats["count"]>=2) & before.notna() & change.notna()
        returned = multi & (excursion>cutoff) & (change<=cutoff)
        grid = pd.date_range(first,end,freq="15min")
        positions = series.index.get_indexer(grid, method="pad", tolerance=r.ingest.STALE)
        observed = positions >= 0
        observed[observed] &= np.isfinite(series.to_numpy()[positions[observed]])
        counter = np.r_[0,(~observed).cumsum()]
        history = r.ingest.HISTORY*4
        eligible = np.zeros(len(grid),dtype=bool)
        eligible[history:] = counter[history+1:]-counter[:-history-1]==0
        usable = eligible & (grid>=start)
        fresh = np.r_[False,positions[1:] > positions[:-1]]
        result.append(dict(channel=channel, spacing_median_seconds=float(np.median(spacing)),
                           spacing_p90_seconds=float(np.quantile(spacing,.9)), bins=len(stats),
                           multiple_observations=int(multi.sum()), excursion_cutoff=cutoff,
                           excursion_and_return=int(returned.sum()),
                           eligible_hourly=int(hour_eligible[-(int((end-start).total_seconds()/3600)+1):].sum()),
                           eligible_15min=int(usable.sum()), fresh_15min=int((usable & fresh).sum())))
    r.save(ROOT/"calibration-cadence.json",result)
    for row in result:
        print(row)


if __name__=="__main__":
    main()
