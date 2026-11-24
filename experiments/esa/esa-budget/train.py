"""Fit the residual LSTM with the ESA pilot's original training recipe."""
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import forecast
from ingest import CHANNELS, checksum

ROOT = Path(__file__).resolve().parent
PILOT = ROOT.parent / "esa-run"


class ResidualForecast(forecast.Forecast):
    def forward(self, history):
        return history[:, -1, 0] + super().forward(history)


def init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    forecast.Forecast = ResidualForecast


def fit(channel):
    destination = ROOT / "models" / f"{channel}.json"
    weights = destination.with_suffix(".pt")
    if destination.exists():
        record = json.loads(destination.read_text())
        if checksum(weights) != record["checkpoint_sha256"]:
            raise ValueError(f"Bad residual checkpoint for {channel}")
        return channel, record
    prior = json.loads((PILOT / "models/manifest.json").read_text())["channels"][channel]
    path = PILOT / "fit" / f"{channel}.npz"
    signal, windows, eligible, masks, _ = forecast.prepare_arrays(path, prior["center"], prior["scale"])
    fit_indices, valid_indices = [forecast.target_indices(eligible, masks[key]) for key in ("fit", "validation")]
    model, record = forecast.fit_model(signal, windows, fit_indices, valid_indices, forecast.EPOCHS)
    _, scores = forecast.calculate_scores(model, signal, windows, valid_indices)
    pending = weights.with_suffix(".tmp")
    torch.save(model.state_dict(), pending)
    pending.replace(weights)
    residual = scores["lstm"]
    record.update(checkpoint_sha256=checksum(weights),
                  base_threshold=float(np.sort(residual)[(99*len(residual)+99)//100-1]))
    pending = destination.with_suffix(".tmp")
    pending.write_text(json.dumps(record, indent=2) + "\n")
    pending.replace(destination)
    return channel, record


if __name__ == "__main__":
    forecast.read_input(PILOT / "fit", "fit")
    (ROOT / "models").mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=6, initializer=init) as executor:
        for number, future in enumerate(as_completed([executor.submit(fit, c) for c in CHANNELS]), 1):
            channel, record = future.result()
            print(f"{number}/6 {channel}: {record}", flush=True)
