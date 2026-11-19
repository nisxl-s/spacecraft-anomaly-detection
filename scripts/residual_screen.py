"""Screen persistence-residual forecasts on official NASA training arrays."""
import argparse
import copy
import hashlib
import io
import json
import math
import multiprocessing as mp
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pilot_lstm as recipe
from confirm_alarm import confirmed_episodes
from pilot_baselines import smooth

ARMS = ("direct", "residual", "persistence")
OUTPUT = Path(__file__).parent
MANIFEST_SHA = "4aa489f270b397e48dfca2410816e86120ff1047819b592f2aacc2732c68ea78"


class ResidualForecast(recipe.Forecast):
    def forward(self, windows):
        return windows[:, -1, 0] + super().forward(windows)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save_json(path, value):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


def fit(features, windows, fit_indices, stopping_indices):
    torch.manual_seed(recipe.SEED)
    shuffle = torch.Generator().manual_seed(recipe.SEED)
    model = ResidualForecast(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.LEARNING_RATE)
    best_loss, best_epoch, best_state = math.inf, 0, None
    target = features[:, 0]
    history = []
    for epoch in range(1, recipe.MAX_EPOCHS + 1):
        model.train()
        indices = fit_indices[torch.randperm(len(fit_indices), generator=shuffle)]
        for context, actual in recipe.batches(windows, target, indices):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(model(context), actual)
            loss.backward()
            optimizer.step()
        loss = recipe.validation_loss(model, windows, target, stopping_indices)
        if not math.isfinite(loss):
            raise ValueError("nonfinite stopping MSE")
        history.append(loss)
        if loss < best_loss:
            best_loss, best_epoch, best_state = loss, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= recipe.PATIENCE:
            break
    model.load_state_dict(best_state)
    if recipe.validation_loss(model, windows, target, stopping_indices) != best_loss:
        raise ValueError("Residual checkpoint did not reproduce its stopping loss")
    return model, history, best_epoch


def evaluate(calibration, evaluation):
    cutoff = np.sort(smooth(calibration))[math.ceil(0.99 * len(calibration)) - 1]
    evaluation_scores = smooth(evaluation)
    emitted = confirmed_episodes(evaluation_scores > cutoff)
    starts = emitted & ~np.r_[False, emitted[:-1]]
    return {
        "threshold": float(cutoff), "mae": float(evaluation.mean()),
        "alarm_episodes": np.flatnonzero(starts).tolist(),
        "flagged_readings": int(emitted.sum()),
    }


def run_channel(job):
    name, member, expected, output, archive_path, models = job
    started = time.monotonic()
    output = Path(output)
    path = output / (name.replace("/", "_") + ".json")
    if path.exists():
        return json.loads(path.read_text())
    print(f"START {name}", flush=True)
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(member)
    if sha(raw) != expected or not member.startswith("data/data/train/"):
        raise ValueError(f"Unexpected training member for {name}")
    series = np.load(io.BytesIO(raw), allow_pickle=False)
    n = len(series)
    fit_end, cal_start, eval_start = n * 70 // 100, n * 85 // 100, n * 925 // 1000
    fit_indices = torch.arange(recipe.WINDOW, fit_end)
    stopping_indices = torch.arange(fit_end, cal_start)
    calibration_indices = torch.arange(cal_start, eval_start)
    evaluation_indices = torch.arange(eval_start, n)
    if min(map(len, (fit_indices, stopping_indices, calibration_indices, evaluation_indices))) < 1:
        raise ValueError(f"Empty split for {name}")
    target = torch.from_numpy(series[:, 0].astype(np.float64))
    features = torch.from_numpy(np.ascontiguousarray(series, dtype=np.float32))
    windows = features.unfold(0, recipe.WINDOW, 1)[:-1].transpose(1, 2)
    np.testing.assert_array_equal(windows[eval_start - recipe.WINDOW].numpy(), features[eval_start-recipe.WINDOW:eval_start].numpy())
    original = models / name
    checkpoint_bytes = original.with_suffix(".pt").read_bytes()
    report = json.loads(original.with_suffix(".json").read_bytes())
    if (report["channel"] != name or report["training_member"] != member
        or report["training_member_sha256"] != expected
        or report["checkpoint"]["sha256"] != sha(checkpoint_bytes)
        or report["target_ranges_half_open"] != {
            "fit": [recipe.WINDOW, fit_end], "validation": [fit_end, cal_start],
            "calibration": [cal_start, n],
        }):
        raise ValueError(f"Original direct checkpoint provenance differs for {name}")
    expected_config = {
        "seed": recipe.SEED, "history_rows": recipe.WINDOW, "units": [80, 80],
        "dropout": 0.30, "batch_size": recipe.BATCH_SIZE, "optimizer": "Adam",
        "learning_rate": recipe.LEARNING_RATE, "loss": "MSE", "max_epochs": recipe.MAX_EPOCHS,
        "patience": recipe.PATIENCE, "score": "absolute residual", "ewma_span": 30,
        "calibration_percentile": 0.99, "alarm_rule": "strict exceedance",
    }
    if report["configuration"] != expected_config:
        raise ValueError(f"Original direct configuration differs for {name}")
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    if (checkpoint["training_member_sha256"] != expected
        or checkpoint["best_epoch"] != report["best_epoch"]
        or checkpoint["input_features"] != series.shape[1]):
        raise ValueError(f"Original direct checkpoint metadata differs for {name}")
    direct = recipe.Forecast(series.shape[1])
    direct.load_state_dict(checkpoint["model_state_dict"])
    residual, history, best_epoch = fit(features, windows, fit_indices, stopping_indices)
    calibration, evaluation = {}, {}
    for detector, model in (("direct", direct), ("residual", residual)):
        calibration[detector] = recipe.residuals(model, windows, target, calibration_indices)
        evaluation[detector] = recipe.residuals(model, windows, target, evaluation_indices)
    for destination, indices in ((calibration, calibration_indices), (evaluation, evaluation_indices)):
        destination["persistence"] = (target[indices] - target[indices - 1]).abs().numpy()
    torch.save(residual.state_dict(), path.with_suffix(".pt"))
    np.savez_compressed(path.with_suffix(".npz"),
                        **{f"calibration_{key}": values for key, values in calibration.items()},
                        **{f"evaluation_{key}": values for key, values in evaluation.items()})
    row = {
        "channel": name, "training_array_sha256": expected,
        "direct_checkpoint_sha256": sha(checkpoint_bytes), "direct_best_epoch": checkpoint["best_epoch"],
        "residual_best_epoch": best_epoch, "residual_stopping_mse": history,
        "calibration_rows": len(calibration_indices), "evaluation_rows": len(evaluation_indices),
        "target_ranges": {"fit": [recipe.WINDOW, fit_end], "early_stopping": [fit_end, cal_start],
                          "calibration": [cal_start, eval_start], "evaluation": [eval_start, n]},
        "detectors": {key: evaluate(calibration[key], evaluation[key]) for key in ARMS},
        "elapsed_seconds": time.monotonic() - started,
    }
    save_json(path, row)
    return row


def init(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def group(rows):
    exposure = sum(row["evaluation_rows"] for row in rows)
    scores = {}
    for arm in ARMS:
        data = [row["detectors"][arm] for row in rows]
        episode_count = sum(len(item["alarm_episodes"]) for item in data)
        flags = sum(item["flagged_readings"] for item in data)
        scores[arm] = {
            "macro_mae": float(np.mean([item["mae"] for item in data])),
            "episodes": episode_count, "episodes_per_1000": 1000 * episode_count / exposure,
            "flagged_readings": flags, "flagged_share": flags / exposure,
        }
    return {
        "channels": len(rows), "evaluation_readings": exposure,
        "models": scores,
        "residual_mae_wins": {arm: sum(row["detectors"]["residual"]["mae"] < row["detectors"][arm]["mae"]
                                       for row in rows) for arm in ("direct", "persistence")},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--channel", action="append")
    parser.add_argument("--external-pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=OUTPUT / "results")
    args = parser.parse_args()
    manifest_raw = args.manifest.read_bytes()
    if sha(manifest_raw) != MANIFEST_SHA:
        raise ValueError("Frozen intake manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != manifest["archive"]["sha256"]:
            raise ValueError("Frozen archive hash mismatch")
    members = {entry["member"]: entry for entry in manifest["arrays"] if entry["split"] == "train"}
    rows = [row for row in manifest["channels"] if row["eligible"]]
    if len(rows) != 78:
        raise ValueError("Original cohort does not have 78 eligible channels")
    jobs = []
    for row in rows:
        key = f'{row["mission"]}/{row["channel"]}'
        if args.channel and key not in args.channel:
            continue
        member = row["array_members"]["train"]
        jobs.append((key, member, members[member]["sha256"], args.output, args.archive, args.models))
    if args.channel and {job[0] for job in jobs} != set(args.channel):
        raise ValueError("A selected channel is not in the frozen cohort")
    args.output.mkdir(parents=True, exist_ok=True)
    jobs.sort(key=lambda job: members[job[1]]["shape"][0])
    if args.external_pilot:
        if args.channel:
            parser.error("--external-pilot requires the full cohort")
        jobs = [job for job in jobs if job[0] != "SMAP/P-1"]
    completed = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("fork"),
                             initializer=init, initargs=(args.threads,)) as pool:
        futures = [pool.submit(run_channel, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            completed.append(row)
            print(f'{row["channel"]} {len(completed)}/{len(jobs)} {row["elapsed_seconds"]:.1f}s', flush=True)
    if args.external_pilot:
        completed.append(json.loads((args.output / "SMAP_P-1.json").read_text()))
    completed.sort(key=lambda row: row["channel"])
    summary = {"overall": group(completed)}
    for mission in ("SMAP", "MSL"):
        mission_rows = [row for row in completed if row["channel"].startswith(mission + "/")]
        if mission_rows:
            summary[mission] = group(mission_rows)
    models = summary["overall"]["models"]
    r, d, p = (models[key] for key in ("residual", "direct", "persistence"))
    summary["advance"] = len(completed) == 78 and (
        r["macro_mae"] < d["macro_mae"] and r["macro_mae"] <= p["macro_mae"]
        and r["episodes"] <= d["episodes"] and r["flagged_readings"] <= d["flagged_readings"]
    )
    save_json(args.output / "summary.json", summary)


if __name__ == "__main__":
    main()
