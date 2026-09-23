"""Build paired one-step clipping-advantage labels from stored checkpoints.

The builder uses only a deterministically declared public subset of CIFAR-10.
For each checkpoint it draws two public Poisson batches, selects two anchors
from each batch, and compares a C=1 baseline with two one-anchor clipping
counterfactuals.  Checkpoint and history tensors are referenced rather than
duplicated in the output SQLite dataset.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F
from torchvision import datasets

from model import WideResNet


CIFAR10_MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
CIFAR10_STD = torch.tensor((0.2470, 0.2435, 0.2616)).view(1, 3, 1, 1)
PARAMETER_COUNT = 176_602


@dataclass(frozen=True)
class Config:
    source_remote: str
    output_remote: str
    data_dir: str
    output_dir: str
    stage_dir: str
    runs: int
    steps: int
    public_seed: int
    experiment_seed: int
    public_size: int
    update_size: int
    q: float
    anchors: int
    candidates: tuple[float, ...]
    sigma: float
    cap: float
    learning_rate: float
    physical_grad_batch: int
    probe_batch: int
    notify_seconds: int
    upload_seconds: int
    ntfy_topic: str
    max_checkpoints: int | None
    stage_whole_run: bool
    upload: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-remote",
        default=(
            "rclone_s3:dpsgd-as-a-optimization-problem/current-work/"
            "adaptive-clipping-wrn/collection_10/restart_20260922"
        ),
    )
    parser.add_argument(
        "--output-remote",
        default=(
            "rclone_s3:dpsgd-as-a-optimization-problem/current-work/"
            "adaptive-clipping-wrn/counterfactual_advantage_40800/v1_20260923"
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/counterfactual_advantage_40800_v1"),
    )
    parser.add_argument(
        "--stage-dir", type=Path,
        default=Path("runs/counterfactual_advantage_40800_v1/checkpoints"),
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=510)
    parser.add_argument("--public-seed", type=int, default=20260923)
    parser.add_argument("--experiment-seed", type=int, default=314159)
    parser.add_argument("--public-size", type=int, default=5_000)
    parser.add_argument("--update-size", type=int, default=4_000)
    parser.add_argument("--q", type=float, default=1 / 3)
    parser.add_argument("--anchors", type=int, default=2)
    parser.add_argument("--candidates", type=float, nargs="+", default=(0.25, 0.75))
    parser.add_argument("--sigma", type=float, default=4.888246618)
    parser.add_argument("--cap", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=4.0)
    parser.add_argument("--physical-grad-batch", type=int, default=512)
    parser.add_argument("--probe-batch", type=int, default=512)
    parser.add_argument("--notify-seconds", type=int, default=600)
    parser.add_argument("--upload-seconds", type=int, default=600)
    parser.add_argument("--ntfy-topic", default="")
    parser.add_argument("--max-checkpoints", type=int)
    parser.add_argument("--single-checkpoint-download", action="store_true")
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    if args.runs < 1 or args.runs > 10:
        raise ValueError("runs must be in [1, 10]")
    if args.steps < 1 or args.steps > 510:
        raise ValueError("steps must be in [1, 510]")
    if args.public_size != 5_000:
        raise ValueError("this experiment declares exactly 5,000 public examples")
    if not 2 <= args.update_size < args.public_size:
        raise ValueError("update-size must leave a non-empty probe set")
    if args.anchors != 2:
        raise ValueError("the registered design uses exactly two anchors")
    if len(args.candidates) != 2:
        raise ValueError("the registered design uses exactly two candidate clips")
    if not all(0 < value <= args.cap for value in args.candidates):
        raise ValueError("candidate clips must be in (0, cap]")
    if not args.source_remote.startswith("rclone_s3:"):
        raise ValueError("source remote must use the approved rclone_s3 remote")
    if not args.no_upload and not args.output_remote.startswith("rclone_s3:"):
        raise ValueError("output remote must use the approved rclone_s3 remote")
    return Config(
        source_remote=args.source_remote.rstrip("/"),
        output_remote=args.output_remote.rstrip("/"),
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        stage_dir=str(args.stage_dir),
        runs=args.runs,
        steps=args.steps,
        public_seed=args.public_seed,
        experiment_seed=args.experiment_seed,
        public_size=args.public_size,
        update_size=args.update_size,
        q=args.q,
        anchors=args.anchors,
        candidates=tuple(float(value) for value in args.candidates),
        sigma=args.sigma,
        cap=args.cap,
        learning_rate=args.learning_rate,
        physical_grad_batch=args.physical_grad_batch,
        probe_batch=args.probe_batch,
        notify_seconds=args.notify_seconds,
        upload_seconds=args.upload_seconds,
        ntfy_topic=args.ntfy_topic,
        max_checkpoints=args.max_checkpoints,
        stage_whole_run=not args.single_checkpoint_download,
        upload=not args.no_upload,
    )


def notify(topic: str, title: str, message: str, tags: str = "bar_chart") -> None:
    print(f"NOTIFY {title}: {message}", flush=True)
    if not topic:
        return
    request = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=message.encode("utf-8"),
        method="POST",
        headers={"Title": title, "Tags": tags},
    )
    try:
        with urllib.request.urlopen(request, timeout=15):
            pass
    except Exception as exc:
        print(f"ntfy notification failed: {exc}", flush=True)


def stable_seed(master: int, run: int, step: int, replicate: int, stream: int) -> int:
    sequence = np.random.SeedSequence([master, run, step, replicate, stream])
    return int(sequence.generate_state(1, dtype=np.uint64)[0]) & ((1 << 63) - 1)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()


def make_public_split(
    labels: np.ndarray, public_size: int, update_size: int, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    if public_size % 10 or update_size % 10:
        raise ValueError("public-size and update-size must be divisible by 10")
    public_per_class = public_size // 10
    update_per_class = update_size // 10
    rng = np.random.default_rng(seed)
    update: list[int] = []
    probe: list[int] = []
    for label in range(10):
        indices = np.flatnonzero(labels == label).copy()
        rng.shuffle(indices)
        chosen = indices[:public_per_class]
        update.extend(int(value) for value in chosen[:update_per_class])
        probe.extend(int(value) for value in chosen[update_per_class:])
    update_array = np.array(update, dtype=np.int64)
    probe_array = np.array(probe, dtype=np.int64)
    rng.shuffle(update_array)
    rng.shuffle(probe_array)
    serialized = {
        "definition": "class-stratified declared-public CIFAR-10 train subset",
        "seed": seed,
        "public_size": public_size,
        "update_indices": update_array.tolist(),
        "probe_indices": probe_array.tolist(),
    }
    payload = json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
    serialized["sha256"] = sha256_bytes(payload)
    return update_array, probe_array, serialized


def initialize_database(path: Path, manifest_hash: str) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS advantages (
            run_id INTEGER NOT NULL,
            checkpoint_step INTEGER NOT NULL,
            batch_replicate INTEGER NOT NULL,
            anchor_slot INTEGER NOT NULL,
            example_id INTEGER NOT NULL,
            action_name TEXT NOT NULL,
            candidate_clip REAL NOT NULL,
            baseline_probe_ce REAL NOT NULL,
            candidate_probe_ce REAL NOT NULL,
            advantage_raw REAL NOT NULL,
            advantage_scaled REAL,
            gradient_norm REAL NOT NULL,
            baseline_factor REAL NOT NULL,
            candidate_factor REAL NOT NULL,
            parameter_delta_norm REAL NOT NULL,
            logical_batch_size INTEGER NOT NULL,
            batch_seed INTEGER NOT NULL,
            augmentation_seed INTEGER NOT NULL,
            noise_seed INTEGER NOT NULL,
            checkpoint_uri TEXT NOT NULL,
            history_remote TEXT NOT NULL,
            history_end_step INTEGER NOT NULL,
            PRIMARY KEY (
                run_id, checkpoint_step, batch_replicate,
                anchor_slot, action_name
            )
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS completed (
            run_id INTEGER NOT NULL,
            checkpoint_step INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            rows_added INTEGER NOT NULL,
            completed_utc TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_id, checkpoint_step)
        )
        """
    )
    existing = connection.execute(
        "SELECT value FROM metadata WHERE key='manifest_sha256'"
    ).fetchone()
    if existing is not None and existing[0] != manifest_hash:
        raise RuntimeError("existing database was created with a different manifest")
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key,value) VALUES('manifest_sha256',?)",
        (manifest_hash,),
    )
    connection.commit()
    return connection


class CounterfactualEngine:
    def __init__(
        self,
        config: Config,
        images: np.ndarray,
        labels: np.ndarray,
        update_indices: np.ndarray,
        probe_indices: np.ndarray,
        device: torch.device,
    ) -> None:
        self.config = config
        self.images = images
        self.labels = labels
        self.update_indices = update_indices
        self.device = device
        self.model = WideResNet(
            num_classes=10, depth=16, width=1, groups=16, dropout_rate=0.0
        ).to(device).eval()
        if sum(parameter.numel() for parameter in self.model.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("unexpected model parameter count")
        self.buffers = dict(self.model.named_buffers())

        def single_loss(
            params: dict[str, torch.Tensor],
            buffers: dict[str, torch.Tensor],
            image: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            logits = functional_call(
                self.model, (params, buffers), (image.unsqueeze(0),)
            )
            return F.cross_entropy(logits, target.unsqueeze(0))

        self.batched_grad = vmap(
            grad(single_loss), in_dims=(None, None, 0, 0), randomness="different"
        )
        probe_images = torch.from_numpy(images[probe_indices].copy()).permute(0, 3, 1, 2)
        self.probe_images = self._normalize(probe_images).to(device)
        self.probe_targets = torch.from_numpy(labels[probe_indices].copy()).long().to(device)

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        images = images.float().div_(255.0)
        return images.sub_(CIFAR10_MEAN).div_(CIFAR10_STD)

    def augmented_batch(self, indices: np.ndarray, seed: int) -> torch.Tensor:
        images = torch.from_numpy(self.images[indices].copy()).permute(0, 3, 1, 2)
        images = images.float().div_(255.0)
        padded = F.pad(images, (4, 4, 4, 4), mode="reflect")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        count = images.shape[0]
        tops = torch.randint(0, 9, (count,), generator=generator)
        lefts = torch.randint(0, 9, (count,), generator=generator)
        windows = padded.unfold(2, 32, 1).unfold(3, 32, 1)
        rows = torch.arange(count)
        images = windows[rows, :, tops, lefts].contiguous()
        flips = torch.rand(count, generator=generator) < 0.5
        images[flips] = images[flips].flip(-1)
        return images.sub_(CIFAR10_MEAN).div_(CIFAR10_STD)

    def load_checkpoint(self, path: Path) -> dict:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()
        return payload

    def aggregate_and_anchors(
        self,
        batch_indices: np.ndarray,
        anchor_positions: np.ndarray,
        augmentation_seed: int,
    ) -> tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]], list[float]]:
        images = self.augmented_batch(batch_indices, augmentation_seed)
        targets = torch.from_numpy(self.labels[batch_indices].copy()).long()
        params = dict(self.model.named_parameters())
        sums = {name: torch.zeros_like(parameter) for name, parameter in params.items()}
        anchor_lookup = {int(position): slot for slot, position in enumerate(anchor_positions)}
        anchor_gradients: list[dict[str, torch.Tensor] | None] = [None] * len(anchor_positions)
        anchor_norms: list[float | None] = [None] * len(anchor_positions)

        chunk_size = self.config.physical_grad_batch
        for start in range(0, len(batch_indices), chunk_size):
            stop = min(start + chunk_size, len(batch_indices))
            chunk_images = images[start:stop].to(self.device, non_blocking=True)
            chunk_targets = targets[start:stop].to(self.device, non_blocking=True)
            with torch.enable_grad():
                per_sample = self.batched_grad(
                    params, self.buffers, chunk_images, chunk_targets
                )
            squared_norm = torch.zeros(stop - start, device=self.device)
            for value in per_sample.values():
                squared_norm.add_(value.reshape(stop - start, -1).square().sum(1))
            norms = squared_norm.sqrt()
            factors = torch.clamp(self.config.cap / norms.clamp_min(1e-30), max=1.0)
            for name, value in per_sample.items():
                shape = (stop - start,) + (1,) * (value.ndim - 1)
                sums[name].add_((value * factors.view(shape)).sum(0).detach())
            for global_position in range(start, stop):
                if global_position not in anchor_lookup:
                    continue
                slot = anchor_lookup[global_position]
                local = global_position - start
                anchor_gradients[slot] = {
                    name: value[local].detach().clone()
                    for name, value in per_sample.items()
                }
                anchor_norms[slot] = float(norms[local].item())
            del per_sample, squared_norm, norms, factors, chunk_images, chunk_targets

        if any(value is None for value in anchor_gradients + anchor_norms):
            raise RuntimeError("failed to capture every selected anchor gradient")
        return (
            sums,
            [value for value in anchor_gradients if value is not None],
            [float(value) for value in anchor_norms if value is not None],
        )

    def probe_loss(self, params: dict[str, torch.Tensor]) -> float:
        total = 0.0
        count = self.probe_targets.numel()
        with torch.inference_mode():
            for start in range(0, count, self.config.probe_batch):
                stop = min(start + self.config.probe_batch, count)
                logits = functional_call(
                    self.model,
                    (params, self.buffers),
                    (self.probe_images[start:stop],),
                )
                losses = F.cross_entropy(
                    logits, self.probe_targets[start:stop], reduction="none"
                )
                total += losses.double().sum().item()
        return total / count

    def process_batch(
        self,
        run: int,
        step: int,
        replicate: int,
    ) -> tuple[list[dict], dict]:
        cfg = self.config
        batch_seed = stable_seed(cfg.experiment_seed, run, step, replicate, 0)
        augmentation_seed = stable_seed(cfg.experiment_seed, run, step, replicate, 1)
        noise_seed = stable_seed(cfg.experiment_seed, run, step, replicate, 2)
        rng = np.random.default_rng(batch_seed)
        selected = rng.random(len(self.update_indices)) < cfg.q
        batch_indices = self.update_indices[selected]
        if len(batch_indices) < cfg.anchors:
            raise RuntimeError("Poisson batch too small for requested anchors")
        anchor_positions = np.sort(
            rng.choice(len(batch_indices), size=cfg.anchors, replace=False)
        )

        clipped_sum, anchor_gradients, anchor_norms = self.aggregate_and_anchors(
            batch_indices, anchor_positions, augmentation_seed
        )
        params = dict(self.model.named_parameters())
        noise_generator = torch.Generator(device=self.device).manual_seed(noise_seed)
        # The public batch estimates the background population gradient.  Noise
        # and the one-record intervention retain the original N=50,000 scale.
        background_denominator = cfg.q * cfg.update_size
        deployment_denominator = cfg.q * 50_000
        baseline_params: dict[str, torch.Tensor] = {}
        for name, parameter in params.items():
            noise = torch.randn(
                parameter.shape,
                generator=noise_generator,
                device=self.device,
                dtype=parameter.dtype,
            ).mul_(cfg.sigma * cfg.cap / deployment_denominator)
            update = clipped_sum[name].div(background_denominator).add(noise)
            baseline_params[name] = parameter.detach() - cfg.learning_rate * update
        baseline_loss = self.probe_loss(baseline_params)

        rows: list[dict] = []
        for slot, (position, gradient, norm) in enumerate(
            zip(anchor_positions, anchor_gradients, anchor_norms, strict=True), start=1
        ):
            baseline_factor = min(1.0, cfg.cap / max(norm, 1e-30))
            for action_index, candidate in enumerate(cfg.candidates):
                candidate_factor = min(1.0, candidate / max(norm, 1e-30))
                factor_delta = (candidate_factor - baseline_factor) / deployment_denominator
                counterfactual_params = {
                    name: baseline_params[name]
                    - cfg.learning_rate * factor_delta * gradient[name]
                    for name in params
                }
                candidate_loss = self.probe_loss(counterfactual_params)
                parameter_delta_norm = (
                    cfg.learning_rate * abs(factor_delta) * norm
                )
                rows.append({
                    "run_id": run,
                    "checkpoint_step": step,
                    "batch_replicate": replicate,
                    "anchor_slot": slot,
                    "example_id": int(batch_indices[int(position)]),
                    "action_name": "low" if action_index == 0 else "high",
                    "candidate_clip": candidate,
                    "baseline_probe_ce": baseline_loss,
                    "candidate_probe_ce": candidate_loss,
                    "advantage_raw": baseline_loss - candidate_loss,
                    "gradient_norm": norm,
                    "baseline_factor": baseline_factor,
                    "candidate_factor": candidate_factor,
                    "parameter_delta_norm": parameter_delta_norm,
                    "logical_batch_size": len(batch_indices),
                    "batch_seed": batch_seed,
                    "augmentation_seed": augmentation_seed,
                    "noise_seed": noise_seed,
                })

        # Exact no-op invariant on the first batch of the first checkpoint.
        if run == 0 and step == 1 and replicate == 1:
            diagnostic_loss = self.probe_loss(baseline_params)
            if diagnostic_loss != baseline_loss:
                raise RuntimeError(
                    f"C=1 diagnostic failed: {baseline_loss} != {diagnostic_loss}"
                )
        return rows, {
            "batch_size": len(batch_indices),
            "baseline_loss": baseline_loss,
            "anchor_ids": [int(batch_indices[int(value)]) for value in anchor_positions],
        }


def stage_run(config: Config, run: int) -> Path:
    destination = Path(config.stage_dir) / f"run_{run:02d}"
    destination.mkdir(parents=True, exist_ok=True)
    if config.stage_whole_run:
        command = [
            "rclone", "copy",
            f"{config.source_remote}/run_{run:02d}", str(destination),
            "--include", "step_*/checkpoint.pt",
            "--transfers", "16", "--checkers", "32",
        ]
        subprocess.run(command, check=True)
    return destination


def ensure_checkpoint(config: Config, run: int, step: int, run_stage: Path) -> Path:
    path = run_stage / f"step_{step:06d}" / "checkpoint.pt"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "rclone", "copyto",
            f"{config.source_remote}/run_{run:02d}/step_{step:06d}/checkpoint.pt",
            str(path),
        ], check=True)
    return path


def insert_checkpoint(
    connection: sqlite3.Connection,
    rows: list[dict],
    run: int,
    step: int,
    elapsed: float,
    config: Config,
) -> None:
    history_remote = f"{config.source_remote}/run_{run:02d}"
    checkpoint_uri = (
        f"{history_remote}/step_{step:06d}/checkpoint.pt"
    )
    columns = (
        "run_id", "checkpoint_step", "batch_replicate", "anchor_slot",
        "example_id", "action_name", "candidate_clip", "baseline_probe_ce",
        "candidate_probe_ce", "advantage_raw", "gradient_norm",
        "baseline_factor", "candidate_factor", "parameter_delta_norm",
        "logical_batch_size", "batch_seed", "augmentation_seed", "noise_seed",
        "checkpoint_uri", "history_remote", "history_end_step",
    )
    placeholders = ",".join("?" for _ in columns)
    with connection:
        for row in rows:
            enriched = dict(row)
            enriched.update({
                "checkpoint_uri": checkpoint_uri,
                "history_remote": history_remote,
                "history_end_step": step,
            })
            connection.execute(
                f"INSERT INTO advantages({','.join(columns)}) VALUES({placeholders})",
                tuple(enriched[name] for name in columns),
            )
        connection.execute(
            "INSERT INTO completed(run_id,checkpoint_step,elapsed_seconds,rows_added) "
            "VALUES(?,?,?,?)",
            (run, step, elapsed, len(rows)),
        )


def database_stats(connection: sqlite3.Connection) -> dict:
    row = connection.execute(
        """
        SELECT COUNT(*), AVG(advantage_raw), AVG(advantage_raw*advantage_raw),
               MIN(advantage_raw), MAX(advantage_raw),
               SUM(CASE WHEN advantage_raw > 1e-12 THEN 1 ELSE 0 END),
               SUM(CASE WHEN advantage_raw < -1e-12 THEN 1 ELSE 0 END),
               SUM(CASE WHEN ABS(advantage_raw) <= 1e-12 THEN 1 ELSE 0 END),
               AVG(gradient_norm), AVG(baseline_probe_ce)
        FROM advantages
        """
    ).fetchone()
    count = int(row[0] or 0)
    mean = float(row[1] or 0.0)
    variance = max(0.0, float(row[2] or 0.0) - mean * mean)
    action_rows = connection.execute(
        "SELECT action_name,COUNT(*),AVG(advantage_raw),"
        "AVG(CASE WHEN advantage_raw>0 THEN 1.0 ELSE 0.0 END) "
        "FROM advantages GROUP BY action_name ORDER BY action_name"
    ).fetchall()
    completed = connection.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
    return {
        "completed_checkpoints": int(completed),
        "rows": count,
        "advantage_mean": mean,
        "advantage_std": math.sqrt(variance),
        "advantage_min": float(row[3] or 0.0),
        "advantage_max": float(row[4] or 0.0),
        "positive": int(row[5] or 0),
        "negative": int(row[6] or 0),
        "neutral": int(row[7] or 0),
        "mean_gradient_norm": float(row[8] or 0.0),
        "mean_baseline_probe_ce": float(row[9] or 0.0),
        "by_action": {
            name: {"rows": int(n), "mean": float(avg), "positive_fraction": float(pos)}
            for name, n, avg, pos in action_rows
        },
    }


def format_progress(
    stats: dict, total_checkpoints: int, elapsed: float, run: int, step: int
) -> str:
    completed = stats["completed_checkpoints"]
    rate = completed / max(elapsed, 1e-9)
    remaining = max(0, total_checkpoints - completed)
    eta_seconds = remaining / rate if rate else float("inf")
    positive_fraction = stats["positive"] / max(stats["rows"], 1)
    low = stats["by_action"].get("low", {})
    high = stats["by_action"].get("high", {})
    return (
        f"Checkpoint {completed:,}/{total_checkpoints:,} ({completed/total_checkpoints:.1%}); "
        f"current run={run:02d} step={step:03d}; rows={stats['rows']:,}; "
        f"elapsed={elapsed/3600:.2f}h; ETA={eta_seconds/3600:.2f}h; "
        f"rate={rate*60:.2f} checkpoints/min. "
        f"Advantage mean={stats['advantage_mean']:.3e}, sd={stats['advantage_std']:.3e}, "
        f"range=[{stats['advantage_min']:.3e},{stats['advantage_max']:.3e}], "
        f"positive={positive_fraction:.1%}, neutral={stats['neutral']:,}. "
        f"Low mean={low.get('mean', 0.0):.3e} (positive {low.get('positive_fraction', 0.0):.1%}); "
        f"high mean={high.get('mean', 0.0):.3e} "
        f"(positive {high.get('positive_fraction', 0.0):.1%}). "
        f"Mean gradient norm={stats['mean_gradient_norm']:.3f}; "
        f"mean baseline probe CE={stats['mean_baseline_probe_ce']:.4f}."
    )


def upload_snapshot(
    connection: sqlite3.Connection,
    config: Config,
    output_dir: Path,
    progress: dict,
) -> None:
    atomic_json(output_dir / "progress.json", progress)
    if not config.upload:
        return
    snapshot = output_dir / "advantages.snapshot.sqlite3"
    with contextlib.closing(sqlite3.connect(snapshot)) as destination:
        connection.backup(destination)
    for local, remote_name in (
        (snapshot, "advantages.sqlite3"),
        (output_dir / "manifest.json", "manifest.json"),
        (output_dir / "public_split.json", "public_split.json"),
        (output_dir / "progress.json", "progress.json"),
    ):
        subprocess.run(
            ["rclone", "copyto", str(local), f"{config.output_remote}/{remote_name}"],
            check=True,
        )


def export_jsonl(connection: sqlite3.Connection, path: Path) -> None:
    cursor = connection.execute("SELECT * FROM advantages ORDER BY run_id,checkpoint_step,batch_replicate,anchor_slot,action_name")
    names = [description[0] for description in cursor.description]
    with gzip.open(path, "wt", encoding="utf-8") as destination:
        for values in cursor:
            destination.write(json.dumps(dict(zip(names, values, strict=True))) + "\n")


def main() -> None:
    args = parse_args()
    config = make_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(config.stage_dir).mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10(config.data_dir, train=True, download=False)
    images = np.asarray(dataset.data)
    labels = np.asarray(dataset.targets, dtype=np.int64)
    update_indices, probe_indices, public_split = make_public_split(
        labels, config.public_size, config.update_size, config.public_seed
    )
    atomic_json(output_dir / "public_split.json", public_split)

    manifest = {
        "schema_version": 1,
        "description": "paired one-anchor clipping advantage dataset",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "config": asdict(config),
        "public_split_sha256": public_split["sha256"],
        "normalization": {
            "background_gradient": "1 / (q * n_update)",
            "one_record_intervention": "1 / (q * 50000)",
            "gaussian_noise": "sigma * C_cap / (q * 50000)",
        },
        "row_count_expected": config.runs * config.steps * 2 * config.anchors * len(config.candidates),
        "history_definition": "existing 4275-D sketch plus six block norms for public anchor through checkpoint step",
    }
    manifest_identity = dict(manifest)
    manifest_identity.pop("created_utc")
    manifest_payload = json.dumps(
        manifest_identity, sort_keys=True, separators=(",", ":")
    ).encode()
    manifest_hash = sha256_bytes(manifest_payload)
    manifest["manifest_sha256"] = manifest_hash
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("manifest_sha256") != manifest_hash:
            raise RuntimeError("existing output manifest differs from requested experiment")
    else:
        atomic_json(manifest_path, manifest)

    connection = initialize_database(output_dir / "advantages.sqlite3", manifest_hash)
    engine = CounterfactualEngine(
        config, images, labels, update_indices, probe_indices, device
    )
    total_checkpoints = config.runs * config.steps
    if config.max_checkpoints is not None:
        total_checkpoints = min(total_checkpoints, config.max_checkpoints)
    already_complete = {
        (int(run), int(step))
        for run, step in connection.execute("SELECT run_id,checkpoint_step FROM completed")
    }
    target_order = [
        (run, step)
        for run in range(config.runs)
        for step in range(1, config.steps + 1)
    ][:total_checkpoints]

    notify(
        config.ntfy_topic,
        "Counterfactual dataset started",
        f"Starting {len(target_order):,} checkpoints / "
        f"{len(target_order)*2*config.anchors*len(config.candidates):,} rows; "
        f"q={config.q:.6f}, public update={config.update_size:,}, "
        f"probe={config.public_size-config.update_size:,}, b={config.anchors}, "
        f"actions={config.candidates}, GPU={torch.cuda.get_device_name(device)}. "
        f"Remote: {config.output_remote}",
        "rocket,bar_chart",
    )
    started = time.monotonic()
    last_notify = started
    last_upload = started
    processed_this_execution = 0
    current_run = -1
    run_stage: Path | None = None

    try:
        for run, step in target_order:
            if (run, step) in already_complete:
                continue
            if run != current_run:
                current_run = run
                notify(
                    config.ntfy_topic,
                    "Counterfactual dataset staging",
                    f"Staging checkpoints for run {run:02d}; completed "
                    f"{database_stats(connection)['completed_checkpoints']:,}/{len(target_order):,}.",
                    "floppy_disk",
                )
                run_stage = stage_run(config, run)
            assert run_stage is not None
            checkpoint_path = ensure_checkpoint(config, run, step, run_stage)
            checkpoint_started = time.monotonic()
            payload = engine.load_checkpoint(checkpoint_path)
            if int(payload.get("step", step)) != step:
                raise RuntimeError("checkpoint step does not match its path")
            rows: list[dict] = []
            batch_summaries = []
            for replicate in (1, 2):
                batch_rows, batch_summary = engine.process_batch(run, step, replicate)
                rows.extend(batch_rows)
                batch_summaries.append(batch_summary)
            checkpoint_elapsed = time.monotonic() - checkpoint_started
            insert_checkpoint(
                connection, rows, run, step, checkpoint_elapsed, config
            )
            processed_this_execution += 1
            print(
                f"run={run:02d} step={step:03d} rows={len(rows)} "
                f"batch_sizes={[item['batch_size'] for item in batch_summaries]} "
                f"seconds={checkpoint_elapsed:.2f} "
                f"adv_mean={np.mean([row['advantage_raw'] for row in rows]):.3e}",
                flush=True,
            )

            now = time.monotonic()
            stats = database_stats(connection)
            elapsed = now - started
            progress = {
                "status": "running",
                "run": run,
                "step": step,
                "elapsed_seconds_this_execution": elapsed,
                "processed_this_execution": processed_this_execution,
                "statistics": stats,
                "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            if now - last_upload >= config.upload_seconds:
                upload_snapshot(connection, config, output_dir, progress)
                last_upload = time.monotonic()
            if now - last_notify >= config.notify_seconds:
                notify(
                    config.ntfy_topic,
                    "Counterfactual dataset progress",
                    format_progress(stats, len(target_order), elapsed, run, step),
                    "hourglass_flowing_sand,bar_chart",
                )
                last_notify = time.monotonic()

        stats = database_stats(connection)
        # Compute the median absolute advantage exactly in Python for 40,800 rows.
        absolute = np.fromiter(
            (abs(value[0]) for value in connection.execute("SELECT advantage_raw FROM advantages")),
            dtype=np.float64,
        )
        advantage_scale = float(np.median(absolute)) if len(absolute) else 0.0
        connection.execute(
            "INSERT OR REPLACE INTO metadata(key,value) VALUES('advantage_median_abs',?)",
            (repr(advantage_scale),),
        )
        connection.execute(
            "UPDATE advantages SET advantage_scaled="
            "advantage_raw/(? + 1e-12)",
            (advantage_scale,),
        )
        connection.commit()
        elapsed = time.monotonic() - started
        summary = {
            "status": "complete",
            "elapsed_seconds_this_execution": elapsed,
            "statistics": stats,
            "advantage_median_abs": advantage_scale,
            "advantage_multiplier": 1.0 / (advantage_scale + 1e-12),
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        atomic_json(output_dir / "summary.json", summary)
        export_jsonl(connection, output_dir / "advantages.jsonl.gz")
        upload_snapshot(connection, config, output_dir, summary)
        if config.upload:
            for filename in ("summary.json", "advantages.jsonl.gz"):
                subprocess.run([
                    "rclone", "copyto", str(output_dir / filename),
                    f"{config.output_remote}/{filename}",
                ], check=True)
        notify(
            config.ntfy_topic,
            "Counterfactual dataset complete",
            format_progress(stats, len(target_order), elapsed, config.runs - 1, config.steps)
            + f" Median |advantage|={advantage_scale:.3e}; output={config.output_remote}",
            "white_check_mark,bar_chart",
        )
    except Exception:
        failure = traceback.format_exc()
        print(failure, flush=True)
        stats = database_stats(connection)
        progress = {
            "status": "failed",
            "error": failure,
            "statistics": stats,
            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with contextlib.suppress(Exception):
            upload_snapshot(connection, config, output_dir, progress)
        notify(
            config.ntfy_topic,
            "Counterfactual dataset failed",
            f"Stopped after {stats['completed_checkpoints']:,}/{len(target_order):,} checkpoints. "
            f"Last error: {failure.splitlines()[-1]}",
            "x,warning",
        )
        raise
    finally:
        connection.close()


if __name__ == "__main__":
    main()
