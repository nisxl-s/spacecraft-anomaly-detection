"""Independently check saved scores and alarm summaries in a screening run."""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def smoothed(values):
    previous = float(values[0])
    output = []
    for value in values:
        previous = previous + (float(value) - previous) * (2.0 / 31.0)
        output.append(previous)
    return np.asarray(output)


def alarms(high):
    active = False
    locked = False
    consecutive = clear = duration = 0
    positions = []
    flagged = 0
    for index, value in enumerate(high):
        if locked:
            clear = 0 if value else clear + 1
            if clear == 5:
                locked = False
            continue
        if not value:
            consecutive = duration = 0
            active = False
            continue
        consecutive += 1
        if consecutive < 2:
            continue
        if not active:
            positions.append(index)
            active = True
        flagged += 1
        duration += 1
        if duration == 30:
            locked = True
            consecutive = clear = duration = 0
            active = False
    return positions, flagged


def audit(path):
    rows = []
    for file in sorted(path.glob('*.json')):
        if file.name == 'summary.json':
            continue
        row = json.loads(file.read_text())
        ranges = row['target_ranges']
        n = ranges['evaluation'][1]
        assert ranges == {
            'fit': [250, n * 70 // 100],
            'early_stopping': [n * 70 // 100, n * 85 // 100],
            'calibration': [n * 85 // 100, n * 925 // 1000],
            'evaluation': [n * 925 // 1000, n],
        }
        assert row['calibration_rows'] == ranges['calibration'][1] - ranges['calibration'][0]
        assert row['evaluation_rows'] == ranges['evaluation'][1] - ranges['evaluation'][0]
        with np.load(file.with_suffix('.npz'), allow_pickle=False) as data:
            for arm in ('direct', 'residual', 'persistence'):
                cal = data['calibration_' + arm]
                val = data['evaluation_' + arm]
                assert len(cal) == row['calibration_rows'] and len(val) == row['evaluation_rows']
                assert np.isfinite(cal).all() and np.isfinite(val).all()
                assert (cal >= 0).all() and (val >= 0).all()
                threshold = np.sort(smoothed(cal))[math.ceil(len(cal) * 0.99) - 1]
                result = row['detectors'][arm]
                assert math.isclose(threshold, result['threshold'], rel_tol=1e-12, abs_tol=1e-14)
                assert math.isclose(float(np.mean(val)), result['mae'], rel_tol=1e-12, abs_tol=1e-14)
                starts, flags = alarms(smoothed(val) > threshold)
                assert starts == result['alarm_episodes'], (file.name, arm)
                assert flags == result['flagged_readings'], (file.name, arm)
        rows.append(row)
    if not rows:
        raise ValueError('No completed channel outputs found')
    summary_file = path / 'summary.json'
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        if summary['overall']['channels'] != len(rows):
            print('Summary represents a different completion point; skipping aggregate checks')
            print(f'Independently checked {len(rows)} channels and {3 * len(rows)} detector outputs')
            return
        for name in ('overall', 'SMAP', 'MSL'):
            subset = [row for row in rows if name == 'overall' or row['channel'].startswith(name + '/')]
            if not subset:
                assert name not in summary
                continue
            calculated = summary[name]
            exposure = sum(row['evaluation_rows'] for row in subset)
            assert calculated['channels'] == len(subset)
            assert calculated['evaluation_readings'] == exposure
            for arm in ('direct', 'residual', 'persistence'):
                records = [row['detectors'][arm] for row in subset]
                episodes = sum(len(record['alarm_episodes']) for record in records)
                flags = sum(record['flagged_readings'] for record in records)
                observed = calculated['models'][arm]
                assert observed['episodes'] == episodes and observed['flagged_readings'] == flags
                assert math.isclose(observed['macro_mae'], float(np.mean([r['mae'] for r in records])), rel_tol=1e-12, abs_tol=1e-14)
                assert math.isclose(observed['episodes_per_1000'], 1000 * episodes / exposure, rel_tol=1e-12, abs_tol=1e-14)
                assert math.isclose(observed['flagged_share'], flags / exposure, rel_tol=1e-12, abs_tol=1e-14)
                if arm != 'residual':
                    wins = sum(row['detectors']['residual']['mae'] < row['detectors'][arm]['mae'] for row in subset)
                    assert calculated['residual_mae_wins'][arm] == wins
        models = summary['overall']['models']
        residual, direct, persistence = (models[arm] for arm in ('residual', 'direct', 'persistence'))
        gate = (len(rows) == 78 and residual['macro_mae'] < direct['macro_mae']
                and residual['macro_mae'] <= persistence['macro_mae']
                and residual['episodes'] <= direct['episodes']
                and residual['flagged_readings'] <= direct['flagged_readings'])
        assert summary['advance'] == gate
        print('Independently checked summary aggregation and advancement gate')
    print(f'Independently checked {len(rows)} channels and {3 * len(rows)} detector outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    audit(parser.parse_args().output)
