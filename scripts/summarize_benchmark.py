"""Aggregate replay_p1 detector ledgers and paired mission-stratified bootstrap.

Input JSON: {"channels": [{"channel": "SMAP/P-1", "mission": "SMAP",
"detectors": {"level": ledger, "persistence": ledger, "lstm": ledger}}]}.
Each ledger is the output of replay_p1.detector_ledger; truths retain class and
status, with delays for matched truths. Each channel occurs once. Undefined
ratios and percentile intervals are represented as JSON null.
"""
import argparse
import json
from collections import defaultdict
import numpy as np

DETECTORS = ('level', 'persistence', 'lstm')


def summarize(ledgers):
    total = {k: sum(int(x[k]) for x in ledgers) for k in
             ('exposure_readings', 'eligible_truths', 'alarm_episodes', 'matches', 'false_alarms', 'misses', 'flagged_readings')}
    if total['eligible_truths'] != total['matches'] + total['misses'] or total['alarm_episodes'] != total['matches'] + total['false_alarms']:
        raise ValueError('inconsistent event counts')
    exposure = total['exposure_readings']
    total['precision'] = total['matches'] / total['alarm_episodes'] if total['alarm_episodes'] else None
    total['recall'] = total['matches'] / total['eligible_truths'] if total['eligible_truths'] else None
    total['false_alarms_per_1000_readings'] = 1000 * total['false_alarms'] / exposure if exposure else None
    total['fraction_flagged'] = total['flagged_readings'] / exposure if exposure else None
    classes = defaultdict(lambda: {'eligible': 0, 'matched': 0})
    delays = []
    for ledger in ledgers:
        for event in ledger['truths']:
            if event['status'] == 'excluded':
                continue
            if event['status'] not in ('matched', 'miss'):
                raise ValueError('unknown truth status')
            cls = event['class'].strip().lower()
            classes[cls]['eligible'] += 1
            if event['status'] == 'matched':
                classes[cls]['matched'] += 1
                delays.append(event['delay_readings'])
    if sum(v['eligible'] for v in classes.values()) != total['eligible_truths'] or len(delays) != total['matches']:
        raise ValueError('truth ledger and aggregate counts disagree')
    total['class_recall'] = {k: {**v, 'recall': v['matched']/v['eligible']} for k, v in sorted(classes.items())}
    total['median_delay_readings'] = float(np.percentile(delays, 50, method='linear')) if delays else None
    total['p90_delay_readings'] = float(np.percentile(delays, 90, method='linear')) if delays else None
    return total


def run(data, draws=2000, seed=42, baseline_only=False):
    detectors = DETECTORS[:2] if baseline_only else DETECTORS
    channels = data['channels']
    ids = [x['channel'] for x in channels]
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate channel IDs')
    groups = defaultdict(list)
    for c in channels:
        if not set(detectors).issubset(c['detectors']):
            raise ValueError('each channel must have all requested detector ledgers')
        groups[c['mission']].append(c)
    groups = {k: sorted(v, key=lambda x: x['channel']) for k, v in sorted(groups.items())}
    def table(selected):
        return {d: summarize([c['detectors'][d] for c in selected]) for d in detectors}
    result = {'channel_count': len(channels), 'overall': table(channels),
              'by_mission': {m: table(cs) for m, cs in groups.items()},
              'without_pilot_P-1': table([c for c in channels if c['channel'] not in ('P-1', 'SMAP/P-1')]),
              'bootstrap': {'draws': draws, 'seed': seed, 'stratification': 'mission', 'differences': {}}}
    if baseline_only:
        result['bootstrap'] = None
        return result
    rng = np.random.default_rng(seed)
    values = {base: {'recall': [], 'false_alarms_per_1000_readings': []} for base in ('level', 'persistence')}
    for _ in range(draws):
        sample = [group[i] for group in groups.values() for i in rng.integers(0, len(group), size=len(group))]
        pooled = table(sample)
        for base, metrics in values.items():
            for metric, vals in metrics.items():
                a, b = pooled['lstm'][metric], pooled[base][metric]
                if a is not None and b is not None:
                    vals.append(a - b)
    for base, metrics in values.items():
        result['bootstrap']['differences'][f'lstm_minus_{base}'] = {
            metric: {'defined_draws': len(vals), 'percentile_95_interval': [float(x) for x in np.percentile(vals, [2.5, 97.5], method='linear')] if vals else None}
            for metric, vals in metrics.items()}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input_json'); p.add_argument('output_json')
    p.add_argument('--baseline-only', action='store_true', help='aggregate level and persistence without LSTM or bootstrap comparisons')
    args = p.parse_args()
    with open(args.input_json) as f: data = json.load(f)
    result = run(data, baseline_only=args.baseline_only)
    with open(args.output_json, 'w') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')

if __name__ == '__main__':
    main()
