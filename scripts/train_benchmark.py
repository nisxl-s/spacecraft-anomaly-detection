"""Run the frozen LSTM training recipe for each eligible intake channel."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

from intake import file_hash

INTAKE_SHA256 = "4aa489f270b397e48dfca2410816e86120ff1047819b592f2aacc2732c68ea78"
PILOT_CODE = {
    "pilot_lstm.py": "89ea2ed77c46ef9007d2e1ba649c366d880034c70e700aa7bf4c499a20f2cc1a",
    "pilot_baselines.py": "fbec241c38899e1d1989f8222e006c051824609bb986db88c8ebf8af0efe7bb5",
}
PILOT_CHECKPOINT_SHA256 = "501d9d767faa19b26afcb5bbcb814f964554b3c2283317d68c523299f0f36f94"
PILOT_REPORT_SHA256 = "782f0c212ac0380e1365026fbd16ac4490d2d571ef9f42254990dd850373c1b5"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    pending = path.with_suffix(path.suffix + ".tmp")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    pending.replace(path)


def check_result(result, job, checkpoint):
    require(result["channel"] == job["channel"], f"channel mismatch for {checkpoint}")
    require(result["training_member"] == job["training_member"], f"member mismatch for {checkpoint}")
    require(result["training_member_sha256"] == job["training_member_sha256"], f"train hash mismatch for {checkpoint}")
    require(result["target_ranges_half_open"] == job["target_ranges_half_open"], f"split mismatch for {checkpoint}")
    require(result["configuration"] == job["configuration"], f"configuration mismatch for {checkpoint}")
    require(result["checkpoint"]["filename"] == checkpoint.name, f"checkpoint filename mismatch for {checkpoint}")
    require(checkpoint.is_file() and file_hash(checkpoint) == result["checkpoint"]["sha256"],
            f"checkpoint hash mismatch for {checkpoint}")


def check_checkpoint_metadata(checkpoint, result):
    import torch

    model = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(model["training_member_sha256"] == result["training_member_sha256"], "checkpoint train hash mismatch")
    require(model["threshold"] == result["threshold"], "checkpoint threshold mismatch")
    require(model["best_epoch"] == result["best_epoch"], "checkpoint best epoch mismatch")
    require(model["input_features"] == result["training_shape"][1], "checkpoint feature width mismatch")


def publish_model(models, result, checkpoint):
    mission, channel = result["channel"].split("/")
    model_path = models / mission / f"{channel}.pt"
    report_path = model_path.with_suffix(".json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        pending = model_path.with_suffix(".pt.tmp")
        with pending.open("wb") as stream:
            with checkpoint.open("rb") as source:
                shutil.copyfileobj(source, stream)
        pending.replace(model_path)
    require(file_hash(model_path) == result["checkpoint"]["sha256"],
            f"published checkpoint hash mismatch for {result['channel']}")
    if result["channel"] == "SMAP/P-1":
        require("pilot_freeze" in result, "publication requires the P-1 verified report")
        published = result
    else:
        published = result | {"checkpoint": result["checkpoint"] | {"filename": model_path.name}}
    if report_path.exists():
        require(read_json(report_path) == published, f"published report differs for {result['channel']}")
    else:
        report_path.with_suffix(".json.tmp").unlink(missing_ok=True)
        write_json(report_path, published)


@contextmanager
def channel_lock(root):
    with (root / "worker.lock").open("a", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def worker(path):
    with channel_lock(path.parent):
        train_channel(path)


def train_channel(path):
    import pilot_lstm

    job = read_json(path)
    archive_path = Path(job["archive"])
    checkpoint = path.parent / "pilot_lstm_checkpoint.pt"
    result_path = path.parent / "training.json"
    with zipfile.ZipFile(archive_path) as archive:
        raw = archive.read(job["training_member"])
    require(hashlib.sha256(raw).hexdigest() == job["training_member_sha256"],
            f"archive train member hash mismatch for {job['channel']}")

    if result_path.exists():
        result = read_json(result_path)
        check_result(result, job, checkpoint)
        check_checkpoint_metadata(checkpoint, result)
    else:
        # A checkpoint without its report may come from an interrupted run. Keep it for inspection.
        if checkpoint.exists():
            suffix = 0
            while (prior := checkpoint.with_suffix(f".interrupted-{suffix}.pt")).exists():
                suffix += 1
            checkpoint.rename(prior)
        pending = result_path.with_suffix(result_path.suffix + ".tmp")
        if pending.exists():
            suffix = 0
            while (prior := pending.with_suffix(f".interrupted-{suffix}.tmp")).exists():
                suffix += 1
            pending.rename(prior)

        # The frozen module selects one train member; each process overrides it in memory.
        pilot_lstm.TRAIN_MEMBER = job["training_member"]
        result = pilot_lstm.run(archive_path, checkpoint, job["threads"])
        result["channel"] = job["channel"]
        check_result(result, job, checkpoint)
        check_checkpoint_metadata(checkpoint, result)
        write_json(result_path, result)

    if job["channel"] != "SMAP/P-1":
        models = Path(job["models"]) if "models" in job else path.parents[3] / "models"
        publish_model(models, result, checkpoint)


def build_jobs(args):
    require(file_hash(args.manifest) == INTAKE_SHA256, "intake manifest SHA-256 mismatch")
    manifest = read_json(args.manifest)
    require(file_hash(args.archive) == manifest["archive"]["sha256"], "archive SHA-256 mismatch")
    for filename, expected in PILOT_CODE.items():
        require(file_hash(Path(__file__).parent / filename) == expected, f"frozen code hash mismatch for {filename}")
    p1 = read_json(args.pilot_result)
    require(file_hash(args.pilot_result) == PILOT_REPORT_SHA256, "P-1 verified report hash mismatch")
    require(file_hash(args.pilot_checkpoint) == PILOT_CHECKPOINT_SHA256 ==
            p1["checkpoint"]["sha256"], "frozen P-1 checkpoint hash mismatch")
    members = {entry["member"]: entry for entry in manifest["arrays"]}
    jobs = {}
    for row in manifest["channels"]:
        if not row["eligible"]:
            continue
        mission, channel = row["mission"], row["channel"]
        require(re.fullmatch(r"[A-Za-z0-9-]+", channel), f"unexpected path characters for {channel}")
        key = f"{mission}/{channel}"
        member = row["array_members"]["train"]
        job = {
            "channel": key,
            "archive": str(args.archive.resolve()),
            "training_member": member,
            "training_member_sha256": members[member]["sha256"],
            "target_ranges_half_open": row["train_target_ranges_half_open"],
            "configuration": p1["configuration"],
            "threads": args.threads,
        }
        jobs[key] = job
        if args.models:
            job["models"] = str(args.models.resolve())

    pilot_dir = args.output / "SMAP" / "P-1"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    pilot_checkpoint = pilot_dir / "pilot_lstm_checkpoint.pt"
    pilot_result = pilot_dir / "training.json"
    if not pilot_checkpoint.exists():
        with pilot_checkpoint.open("xb") as stream:
            stream.write(args.pilot_checkpoint.read_bytes())
    check_result(p1, jobs["SMAP/P-1"], pilot_checkpoint)
    if not pilot_result.exists():
        write_json(pilot_result, p1)
    else:
        check_result(read_json(pilot_result), jobs["SMAP/P-1"], pilot_checkpoint)
    publish_model(args.models or args.output.parent / "models", p1, pilot_checkpoint)
    return jobs


def run_batch(args):
    jobs = build_jobs(args)
    selected = set(args.channel or jobs)
    require(selected.issubset(jobs), "selected channel is absent or ineligible")

    def dispatch(key):
        job = jobs[key]
        root = args.output / key
        root.mkdir(parents=True, exist_ok=True)
        job_path = root / "job.json"
        if job_path.exists():
            previous = read_json(job_path)
            require(previous == job, f"job configuration changed for {key}")
        else:
            write_json(job_path, job)
        with (root / "training.log").open("a", encoding="utf-8") as log:
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker-job", str(job_path)],
                stdout=log, stderr=subprocess.STDOUT, check=True,
            )
        return key

    failures = {}
    completed = []
    ordered = sorted(selected, key=lambda key: (key != "SMAP/P-1", jobs[key]["target_ranges_half_open"]["fit"][1]))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(dispatch, key): key for key in ordered}
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
                completed.append(key)
                print(f"verified {key} ({len(completed)}/{len(selected)})", flush=True)
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                failures[key] = str(error)
                print(f"failed {key}: {error}", file=sys.stderr, flush=True)
    summary = args.output / "batch-summary.json"
    write_json(summary, {"selected": len(selected), "completed": sorted(completed), "failed": failures})
    return bool(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-job", type=Path, help=argparse.SUPPRESS)
    for name in ("archive", "manifest", "pilot-result", "pilot-checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--models", type=Path, help="Published model directory, defaults to models beside --output")
    parser.add_argument("--channel", action="append", help="Select one or more eligible MISSION/CHANNEL IDs")
    args = parser.parse_args()
    if args.worker_job:
        worker(args.worker_job)
        return 0
    if any(getattr(args, name) is None for name in ("archive", "manifest", "pilot_result", "pilot_checkpoint", "output")):
        parser.error("provide --archive, --manifest, --pilot-result, --pilot-checkpoint, and --output")
    if args.workers < 1 or args.threads < 1:
        parser.error("worker and thread counts must be positive")
    return run_batch(args)


if __name__ == "__main__":
    raise SystemExit(main())
