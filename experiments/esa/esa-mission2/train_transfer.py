"""Fit the unchanged residual recipe to the chronological Mission 2 training side."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import forecast
import ingest

ROOT = Path(__file__).resolve().parent
CHANNELS = tuple(f"channel_{i}" for i in range(9, 15))


class Residual(forecast.Forecast):
    def forward(self, history):
        return history[:, -1, 0] + super().forward(history)


def init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    forecast.Forecast = Residual


def fit(channel, split):
    from evaluate import save
    source = ROOT / "inputs" / f"{channel}.zip"
    series = ingest.channel_series(pd.read_pickle(source, compression="zip"), channel)
    fit_start = split - pd.DateOffset(months=15)
    validation = split - pd.DateOffset(months=6)
    calibration = split - pd.DateOffset(months=3)
    if series.index.min() > fit_start or series.index.max() <= split:
        raise ValueError(f"Insufficient chronological coverage for {channel}")
    times, values, eligible = ingest.prepare_hourly(series.loc[fit_start:split], fit_start, split)
    fit_mask = times < validation
    valid_mask = (times >= validation) & (times < calibration)
    cal_mask = times >= calibration
    fit_values = values[fit_mask & np.isfinite(values)]
    if not len(fit_values) or not float(fit_values.std()) > 0:
        raise ValueError(f"Missing or constant fit data for {channel}")
    center, scale = float(fit_values.mean()), max(float(fit_values.std()), 1e-8)
    signal = (values - center) / scale
    windows = np.lib.stride_tricks.sliding_window_view(signal, ingest.HISTORY)[:-1]
    fit_indices, valid_indices, cal_indices = [np.flatnonzero(mask & eligible) for mask in (fit_mask, valid_mask, cal_mask)]
    if len(fit_indices) == 0 or min(len(valid_indices), len(cal_indices)) < 100:
        raise ValueError(f"Insufficient eligible targets for {channel}")
    model, record = forecast.fit_model(signal, windows, fit_indices, valid_indices, forecast.EPOCHS)
    _, valid_scores = forecast.calculate_scores(model, signal, windows, valid_indices)
    _, cal_scores = forecast.calculate_scores(model, signal, windows, cal_indices)
    directory = ROOT / "models"
    checkpoint = directory / f"{channel}.pt"
    pending = checkpoint.with_suffix(".tmp")
    torch.save(model.state_dict(), pending)
    pending.replace(checkpoint)
    cal_path = directory / f"{channel}_calibration.npz"
    np.savez_compressed(cal_path, eligible=eligible[cal_mask], residual=cal_scores["lstm"])
    errors = valid_scores["lstm"]
    record.update(source_sha256=ingest.checksum(source), checkpoint_sha256=ingest.checksum(checkpoint),
                  calibration_sha256=ingest.checksum(cal_path), center=center, scale=scale,
                  base=float(np.sort(errors)[(99*len(errors)+99)//100-1]),
                  fit_targets=len(fit_indices), validation_targets=len(valid_indices),
                  calibration_targets=len(cal_indices), split=split.isoformat())
    save(directory / f"{channel}.json", record)
    return channel, record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    args = parser.parse_args()
    split = pd.Timestamp(args.split)
    (ROOT / "models").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6, initializer=init) as executor:
        futures = [executor.submit(fit, c, split) for c in CHANNELS]
        for future in as_completed(futures):
            channel, record = future.result()
            print(channel, record, flush=True)


if __name__ == "__main__":
    main()
