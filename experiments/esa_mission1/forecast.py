"""Fit sealed ESA pilot detectors and generate scores without reading labels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ingest import CHANNELS, HISTORY, checksum

METHODS = ("level", "persistence", "lstm")
BATCH = 64
EPOCHS = 50
PATIENCE = 5
SEED = 42


class Forecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.LSTM(1, 80, num_layers=2, dropout=0.30, batch_first=True)
        self.readout = nn.Linear(80, 1)

    def forward(self, history):
        _, (hidden, _) = self.encoder(history)
        return self.readout(hidden[-1]).squeeze(-1)


def smooth_at_targets(raw, indices):
    filtered = np.empty(len(indices), dtype=float)
    for i, value in enumerate(raw):
        if i == 0 or indices[i] != indices[i - 1] + 1:
            current = value
        else:
            current += (2 / 31) * (value - current)
        filtered[i] = current
    return filtered


def prepare_arrays(path, center, scale):
    with np.load(path, allow_pickle=False) as data:
        signal = (data["values"] - center) / scale
        eligible = data["eligible_target"]
        masks = {key: data[key] for key in data.files if key in ("fit", "validation", "calibration", "test")}
        timestamps = data["timestamp_unix_ns"]
    windows = np.lib.stride_tricks.sliding_window_view(signal, HISTORY)[:-1]
    return signal, windows, eligible, masks, timestamps


def target_indices(eligible, mask):
    return np.flatnonzero(eligible & mask)


def forward(model, signal, windows, indices):
    for start in range(0, len(indices), BATCH):
        i = indices[start:start + BATCH]
        history = torch.tensor(windows[i - HISTORY, :, None], dtype=torch.float32)
        target = torch.tensor(signal[i], dtype=torch.float32)
        yield model(history), target


def fit_model(signal, windows, fit, valid, epochs):
    require(len(fit) > 0 and len(valid) >= 100, "Insufficient fit or validation targets")
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = Forecast()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss = nn.MSELoss()
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for prediction, target in forward(model, signal, windows, rng.permutation(fit)):
            optimizer.zero_grad()
            loss(prediction, target).backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_mse = sum((prediction - target).square().sum().item()
                                 for prediction, target in forward(model, signal, windows, valid)) / len(valid)
        require(np.isfinite(validation_mse), "Nonfinite validation loss")
        if validation_mse < best_loss:
            best_loss = validation_mse
            best_epoch = epoch
            best_state = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        if epoch - best_epoch >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "epochs_run": epoch, "validation_mse": best_loss}


def calculate_scores(model, signal, windows, indices):
    model.eval()
    with torch.inference_mode():
        residual = np.concatenate([(prediction.double() - target.double()).abs().numpy()
                                   for prediction, target in forward(model, signal, windows, indices)])
    raw = {
        "level": np.abs(signal[indices]),
        "persistence": np.abs(signal[indices] - signal[indices - 1]),
        "lstm": residual,
    }
    filtered = {method: smooth_at_targets(raw[method], indices) for method in METHODS}
    require(all(np.isfinite(score).all() for score in filtered.values()), "Nonfinite detector score")
    return raw, filtered


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_input(directory, phase):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    require(manifest["phase"] == phase, f"Expected {phase} input manifest")
    channels = {row["channel"]: row for row in manifest["channels"]}
    require(tuple(channels) == CHANNELS and len(manifest["channels"]) == len(CHANNELS), "Expected six frozen channels")
    for channel, row in channels.items():
        require(checksum(directory / f"{channel}.npz") == row["output_sha256"], f"Input mismatch for {channel}")
    return manifest_path, manifest, channels


def save_scores(out, channel, timestamps, selected, raw, smoothed):
    path = out / f"{channel}_scores.npz"
    np.savez_compressed(path, target_grid_index=selected, timestamp_unix_ns=timestamps[selected],
                        **{f"{method}_raw": raw[method] for method in METHODS},
                        **{f"{method}_smoothed": smoothed[method] for method in METHODS})
    return checksum(path)


def fit(args, manifest, epochs):
    report = {}
    for channel in CHANNELS:
        path = args.input / f"{channel}.npz"
        with np.load(path, allow_pickle=False) as data:
            fit_values = data["values"][data["fit"]]
        fit_values = fit_values[np.isfinite(fit_values)]
        if not len(fit_values):
            report[channel] = {"excluded": "no finite fit observations"}
            continue
        center, scale = float(fit_values.mean()), max(float(fit_values.std()), 1e-8)
        signal, windows, eligible, masks, timestamps = prepare_arrays(path, center, scale)
        fit_indices, valid_indices = [target_indices(eligible, masks[block]) for block in ("fit", "validation")]
        if not len(fit_indices) or len(valid_indices) < 100:
            report[channel] = {"excluded": "insufficient eligible fit or checkpoint-selection targets",
                               "fit_targets": len(fit_indices), "validation_targets": len(valid_indices)}
            continue
        model, training = fit_model(signal, windows, fit_indices, valid_indices, epochs)
        raw, filtered = calculate_scores(model, signal, windows, valid_indices)
        thresholds = {method: float(np.sort(scores)[(99 * len(scores) + 99) // 100 - 1])
                      for method, scores in filtered.items()}
        checkpoint = args.output / f"{channel}.pt"
        torch.save(model.state_dict(), checkpoint)
        report[channel] = {
            "fit_targets": len(fit_indices), "validation_targets": len(valid_indices),
            "center": center, "scale": scale, "base_thresholds": thresholds, **training,
            "checkpoint_sha256": checksum(checkpoint),
            "validation_scores_sha256": save_scores(args.output, channel, timestamps, valid_indices, raw, filtered),
        }
    return report


def replay(args, manifest):
    report = {}
    frozen_path = args.models / "manifest.json"
    frozen = json.loads(frozen_path.read_bytes())
    require(frozen["stage"] == "fit" and frozen["data_source"] == manifest["data_source"],
            "Fitted model source mismatch")
    for channel in CHANNELS:
        model_metadata = frozen["channels"][channel]
        if "excluded" in model_metadata:
            report[channel] = {"excluded": model_metadata["excluded"]}
            continue
        checkpoint = args.models / f"{channel}.pt"
        require(checksum(checkpoint) == model_metadata["checkpoint_sha256"], f"Checkpoint mismatch for {channel}")
        model = Forecast()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        path = args.input / f"{channel}.npz"
        signal, windows, eligible, masks, timestamps = prepare_arrays(
            path, model_metadata["center"], model_metadata["scale"]
        )
        selected = target_indices(eligible, masks[args.stage])
        if not len(selected) or (args.stage == "calibration" and len(selected) < 100):
            report[channel] = {"excluded": "insufficient eligible scoring targets", "scored_targets": len(selected)}
            continue
        raw, filtered = calculate_scores(model, signal, windows, selected)
        report[channel] = {
            "scored_targets": len(selected), "base_thresholds": model_metadata["base_thresholds"],
            "score_sha256": save_scores(args.output, channel, timestamps, selected, raw, filtered),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fit", "calibration", "test"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--synthetic-smoke", action="store_true")
    args = parser.parse_args()
    source_path, manifest, _ = read_input(args.input, args.stage)
    synthetic = manifest["data_source"]["kind"] == "synthetic_fixture_only"
    require(args.synthetic_smoke == synthetic, "Synthetic fixtures require explicit --synthetic-smoke and vice versa")
    require(args.stage == "fit" or args.models is not None, "Scoring requires --models")
    if args.stage == "test" and not synthetic:
        parser.error("The official test replay remains gated on implementing and freezing labelled calibration")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir()
    epochs = 1 if synthetic else EPOCHS
    report = fit(args, manifest, epochs) if args.stage == "fit" else replay(args, manifest)
    result = {
        "stage": args.stage, "data_source": manifest["data_source"],
        "input_manifest_sha256": checksum(source_path), "code_sha256": checksum(Path(__file__)),
        "numpy_version": np.__version__, "torch_version": torch.__version__,
        "synthetic_smoke": synthetic, "fit_max_epochs": epochs,
        "model_manifest_sha256": checksum(args.models / "manifest.json") if args.models else None,
        "channels": report,
    }
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
