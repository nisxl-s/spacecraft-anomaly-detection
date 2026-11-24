import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
def read(name):
    return json.loads((root / name).read_text())

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

freeze = read('threshold-freeze.json')
sealed = read('test-emissions.json')
summary = read('test-summary.json')
ledger = read('test-ledger.json')
assert digest(root / 'threshold-freeze.json') == sealed['freeze_sha256']
assert len(summary['channels']) == 6
assert all(len(record['candidates']) == 82 for record in freeze.values())
for c in summary['channels']:
    for kind in ('scores', 'flags'):
        assert digest(root / 'test-scores' / f'{c}_{kind}.npz') == sealed[f'{kind}_sha256'][c]
    for method, record in freeze.items():
        assert record['selected']['recall'] is None and record['thresholds'][c] is None
        with np.load(root / 'test-scores' / f'{c}_flags.npz', allow_pickle=False) as flags:
            assert not flags[f'{method}_raw'].any() and not flags[f'{method}_emitted'].any()
        row = ledger[method][c]
        assert row['tp'] == row['fp'] == row['flagged'] == 0
        assert row['fn'] == len(row['events'])
for method, result in summary['metrics'].items():
    for key in ('tp', 'fp', 'fn', 'exposure', 'flagged'):
        assert result[key] == sum(row[key] for row in ledger[method].values())
print('Verified the frozen thresholds, six score and flag file pairs, and 18 channel-detector ledgers.')
