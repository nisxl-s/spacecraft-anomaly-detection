"""Train and calibrate the SMAP P-1 LSTM using only the official training member."""

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time
import zipfile

import numpy as np
import torch
from torch import nn

from pilot_baselines import smooth

TRAIN_MEMBER = "data/data/train/P-1.npy"
WINDOW = 250
BATCH_SIZE = 64
SEED = 42
MAX_EPOCHS = 50
PATIENCE = 5
LEARNING_RATE = 0.001


class Forecast(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.sequence = nn.LSTM(
            feature_count, hidden_size=80, num_layers=2, dropout=0.30, batch_first=True
        )
        self.readout = nn.Linear(80, 1)

    def forward(self, windows):
        _, (hidden, _) = self.sequence(windows)
        return self.readout(hidden[-1]).squeeze(-1)


def batches(windows, telemetry, indices):
    for start in range(0, len(indices), BATCH_SIZE):
        selection = indices[start:start + BATCH_SIZE]
        yield windows[selection - WINDOW], telemetry[selection]


def validation_loss(model, windows, telemetry, indices):
    squared_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for features, targets in batches(windows, telemetry, indices):
            squared_sum += (model(features) - targets).square().sum().item()
    return squared_sum / len(indices)


def residuals(model, windows, telemetry, indices):
    chunks = []
    model.eval()
    with torch.inference_mode():
        for features, targets in batches(windows, telemetry, indices):
            chunks.append((model(features).double() - targets).abs().numpy())
    return np.concatenate(chunks)


def run(archive_path, checkpoint_path, threads):
    started = time.monotonic()
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    shuffle = torch.Generator().manual_seed(SEED)
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(TRAIN_MEMBER)
    series = np.load(io.BytesIO(raw), allow_pickle=False)
    if series.ndim != 2 or series.shape[1] < 1 or not np.isfinite(series).all():
        raise ValueError("P-1 training array must contain finite tabular telemetry")
    n_rows, n_features = series.shape
    fit_end, calibration_start = n_rows * 70 // 100, n_rows * 85 // 100
    if not WINDOW < fit_end < calibration_start <= n_rows - 100:
        raise ValueError("P-1 training array is too short for the prescribed splits")

    all_features = torch.from_numpy(np.ascontiguousarray(series, dtype=np.float32))
    fit_telemetry = all_features[:, 0]
    telemetry = torch.from_numpy(np.asarray(series[:, 0], dtype=np.float64).copy())
    windows = all_features.unfold(0, WINDOW, 1)[:-1].transpose(1, 2)
    np.testing.assert_array_equal(windows[0].numpy(), series[:WINDOW].astype(np.float32))
    np.testing.assert_array_equal(windows[-1].numpy(), series[-WINDOW - 1:-1].astype(np.float32))
    assert telemetry[WINDOW].item() == float(series[WINDOW, 0])

    model = Forecast(n_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    fit_indices = torch.arange(WINDOW, fit_end)
    validation_indices = torch.arange(fit_end, calibration_start)
    calibration_indices = torch.arange(calibration_start, n_rows)
    history = []
    best_loss, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        fit_sum = 0.0
        shuffled_indices = fit_indices[torch.randperm(len(fit_indices), generator=shuffle)]
        for features, targets in batches(windows, fit_telemetry, shuffled_indices):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(features), targets)
            loss.backward()
            optimizer.step()
            fit_sum += loss.item() * len(targets)
        held_out_loss = validation_loss(model, windows, fit_telemetry, validation_indices)
        if not math.isfinite(held_out_loss):
            raise ValueError("nonfinite validation loss")
        history.append({
            "epoch": epoch,
            "fit_mse": fit_sum / len(fit_indices),
            "validation_mse": held_out_loss,
        })
        print(f"epoch {epoch:2d}: validation MSE {held_out_loss:.7g}", flush=True)
        if held_out_loss < best_loss:
            best_loss, best_epoch, best_state = held_out_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    if validation_loss(model, windows, fit_telemetry, validation_indices) != best_loss:
        raise RuntimeError("selected checkpoint did not reproduce validation loss")

    calibration = smooth(residuals(model, windows, telemetry, calibration_indices))
    rank = math.ceil(0.99 * len(calibration))
    threshold = float(np.sort(calibration)[rank - 1])
    prefix_size = len(calibration_indices) // 2
    prefix_end = calibration_start + prefix_size
    # Build the shorter replay from truncated physical rows, including its final partial batch.
    prefix_windows = all_features[:prefix_end].unfold(0, WINDOW, 1)[:-1].transpose(1, 2)
    prefix = smooth(residuals(model, prefix_windows, telemetry[:prefix_end], calibration_indices[:prefix_size]))
    np.testing.assert_allclose(prefix, calibration[:prefix_size], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(prefix > threshold, calibration[:prefix_size] > threshold)

    checkpoint = {
        "model_state_dict": best_state,
        "input_features": n_features,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "training_member_sha256": hashlib.sha256(raw).hexdigest(),
    }
    with checkpoint_path.open("xb") as stream:
        torch.save(checkpoint, stream)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(restored["model_state_dict"])
    np.testing.assert_array_equal(
        smooth(residuals(model, windows, telemetry, calibration_indices)), calibration
    )
    return {
        "channel": "SMAP/P-1",
        "training_member": TRAIN_MEMBER,
        "training_member_sha256": checkpoint["training_member_sha256"],
        "training_shape": list(series.shape),
        "target_ranges_half_open": {
            "fit": [WINDOW, fit_end],
            "validation": [fit_end, calibration_start],
            "calibration": [calibration_start, n_rows],
        },
        "configuration": {
            "seed": SEED, "history_rows": WINDOW, "units": [80, 80], "dropout": 0.30,
            "batch_size": BATCH_SIZE, "optimizer": "Adam", "learning_rate": LEARNING_RATE,
            "loss": "MSE", "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "score": "absolute residual", "ewma_span": 30, "calibration_percentile": 0.99,
            "alarm_rule": "strict exceedance",
        },
        "epochs": history,
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "threshold": threshold,
        "calibration_rows": len(calibration),
        "calibration_rank_one_based": rank,
        "calibration_strict_exceedances": int((calibration > threshold).sum()),
        "verification": {
            "first_forecast": {"input_half_open": [0, WINDOW], "target_index": WINDOW},
            "prefix_rows": prefix_size,
            "prefix_max_absolute_score_difference": float(np.max(np.abs(prefix - calibration[:prefix_size]))),
            "prefix_flags_identical": True,
            "reloaded_checkpoint_scores_identical": True,
        },
        "checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
            "platform": platform.platform(), "device": "cpu", "torch_threads": threads,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "runtime_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.output.resolve() == args.checkpoint.resolve() or args.output.exists() or args.checkpoint.exists():
        parser.error("provide different unused paths for --output and --checkpoint")
    result = run(args.archive, args.checkpoint, args.threads)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
