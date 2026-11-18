"""Training-side level and persistence calibration for the SMAP P-1 pilot."""
import argparse
import io
import json
import math
import zipfile

import numpy as np


def smooth(scores):
    result = np.empty(len(scores), dtype=float)
    current = scores[0]
    for i, score in enumerate(scores):
        current += (2 / 31) * (score - current)
        result[i] = current
    return result


def run(archive):
    with zipfile.ZipFile(archive) as z:
        member = "data/data/train/P-1.npy"
        data = np.load(io.BytesIO(z.read(member)), allow_pickle=False)
    signal = data[:, 0].astype(float)
    fit_end, cal_start = len(signal) * 70 // 100, len(signal) * 85 // 100
    fit = signal[:fit_end]
    center = fit.mean()
    scale = max(fit.std(), 1e-8)
    result = {"channel": "SMAP/P-1", "training_member": member,
              "fit_end_exclusive": fit_end, "calibration_start": cal_start,
              "training_rows": len(signal), "first_forecast_index": 250, "detectors": {}}
    for name, scores in {
        "level": np.abs(signal[cal_start:] - center) / scale,
        "persistence": np.abs(signal[cal_start:] - signal[cal_start-1:-1])
    }.items():
        filtered = smooth(scores)
        cut = len(scores) // 2
        np.testing.assert_array_equal(smooth(scores[:cut]), filtered[:cut])
        rank = math.ceil(0.99 * len(scores))
        threshold = np.sort(filtered)[rank-1]
        result["detectors"][name] = {"threshold": float(threshold),
            "calibration_rows": len(scores), "strict_exceedances": int((filtered > threshold).sum()),
            "prefix_check_rows": cut}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    print(json.dumps(run(parser.parse_args().archive), indent=2))
