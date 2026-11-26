"""Compare sustained NASA event records and causal operator page timestamps."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np

from event_scoring import score_events
from frozen_alarm import confirmed_episodes
from frozen_alarm import MAX_FLAG_READINGS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervals(flags, start):
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False])) + start
    return list(zip(edges[::2].tolist(), (edges[1::2]-1).tolist(), strict=True))


def evaluate(source, output):
    output.mkdir(exist_ok=True)
    expected = {c: (h, t) for c,h,t in json.loads((source/"replay-manifest.json").read_text())}
    ledger = json.loads((source/"ledger.json").read_text())
    results, actions = [], []
    for item in ledger:
        c = item["channel"]
        path = source/"scores"/(c.replace("/","_")+".npz")
        file_hash, thresholds = expected[c]
        assert digest(path) == file_hash
        previous = item["detectors"]["residual"]
        assert thresholds["residual"] == previous["threshold"]
        with np.load(path, allow_pickle=False) as saved:
            lo,hi=map(int,saved["index"][[0,-1]])
            assert np.array_equal(saved["index"],np.arange(lo,hi+1))
            high=saved["residual_scores"]>previous["threshold"]
            legacy_flags=confirmed_episodes(high, required=2)
            np.testing.assert_array_equal(legacy_flags,saved["residual_flags"])
        legacy=intervals(legacy_flags, lo)
        assert legacy==[tuple(a) for a in previous["alarms"]]
        sustained=intervals(high & np.r_[False,high[:-1]],lo)
        starts=[(a,a) for a,_ in legacy]
        # One prospective reminder at the original cap boundary for each
        # sustained run, sent only if the signal is still high at that reading.
        reminders=[(a+MAX_FLAG_READINGS,a+MAX_FLAG_READINGS) for a,b in sustained if a+MAX_FLAG_READINGS<=b]
        repeat=sorted(set(starts+reminders))
        variants={"legacy_overlap":legacy,"sustained_record":sustained,"legacy_pages":starts,"reminder_pages":repeat}
        matched={k:score_events(item["truths"],v,lo,hi) for k,v in variants.items()}
        assert json.loads(json.dumps(matched["legacy_overlap"]["matches"]))==previous["matches"]
        hits={k:{a for a,_,_ in m["matches"]} for k,m in matched.items()}
        assert set(starts)<=set(repeat)
        results.append(dict(channel=c,readings=hi-lo+1,legacy_active_readings=int(legacy_flags.sum()),
                            sustained_active_readings=sum(b-a+1 for a,b in sustained),
                            longest_record_readings=max((b-a+1 for a,b in sustained),default=0),
                            recovered_record_events=sorted(hits["sustained_record"]-hits["legacy_overlap"]),
                            recovered_record_events_unpaged=sorted((hits["sustained_record"]-hits["legacy_overlap"])-hits["legacy_pages"]),
                            matched=matched))
        actions.append(dict(channel=c,truths=item["truths"],legacy=legacy,sustained=sustained,
                            original_page_indices=[a for a,_ in starts],
                            reminder_page_indices=[a for a,_ in sorted(set(repeat)-set(starts))]))
    assert len(results)==len(expected)==78
    readings=sum(r["readings"] for r in results)
    summary={"readings":readings,"channels":len(results),"recovered_record_events_already_unpaged":sum(
        len(r["recovered_record_events_unpaged"]) for r in results),"variants":{}}
    for name in results[0]["matched"]:
        records=[r["matched"][name] for r in results]
        row={k:sum(r[k] for r in records) for k in ("tp","fp","fn")}
        total=row["tp"]+row["fp"]
        delays=[d for m in records for _,_,d in m["matches"]]
        row.update(episodes_or_pages=total,rate_per_1000=1000*total/readings,
                   median_delay_readings=statistics.median(delays) if delays else None)
        summary["variants"][name]=row
    for stream in ("legacy","sustained"):
        summary[f"{stream}_active_fraction"]=sum(r[f"{stream}_active_readings"] for r in results)/readings
    summary["max_record_readings"]=max(r["longest_record_readings"] for r in results)
    for name,payload in (("summary",summary),("ledger",results),("actions",actions)):
        (output/f"{name}.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source",type=Path)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    evaluate(args.source,args.output)
