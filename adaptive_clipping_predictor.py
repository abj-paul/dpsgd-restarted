"""Materialize and train the constrained adaptive clipping predictor.

The pipeline is resumable by phase:
  1. project every source checkpoint with one fixed Gaussian map;
  2. range-read the K-step per-example gradient history from object storage;
  3. fit train-only feature normalization;
  4. optimize the weighted policy loss with an epoch-level dual update;
  5. evaluate and upload the compact, reproducible artifact.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import os
import signal
import sqlite3
import subprocess
import time
import traceback
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from clipping_predictor import AdaptiveClippingPredictor, parameter_count
from model import WideResNet


GRADIENT_DIM = 4_275
NORM_DIM = 6
PARAMETER_COUNT = 176_602
RUNS = 10
SOURCE_STEPS = 510
SIGN_TOLERANCE = 1e-12
MEDIAN_ABSOLUTE_ADVANTAGE = 4.868945805358571e-05


@dataclass(frozen=True)
class Config:
    dataset_dir: str
    checkpoint_dir: str
    source_remote: str
    output_dir: str
    output_remote: str
    ntfy_topic: str
    context_size: int
    deployment_steps: int
    weight_dim: int
    weight_seed: int
    policy_seed: int
    http_port: int
    http_workers: int
    checkpoint_batch: int
    batch_size: int
    workers: int
    epochs: int
    learning_rate: float
    weight_decay: float
    dual_learning_rate: float
    dual_max: float
    constraint_tolerance: float
    dropout: float
    patience: int
    notify_seconds: int
    upload: bool
    prepare_only: bool
    train_only: bool
    smoke_rows: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("runs/counterfactual_advantage_binary_20400_v1"),
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("runs/counterfactual_advantage_40800_v1/checkpoints"),
    )
    parser.add_argument(
        "--source-remote",
        default=(
            "rclone_s3:dpsgd-as-a-optimization-problem/current-work/"
            "adaptive-clipping-wrn/collection_10/restart_20260922"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("runs/adaptive_clipping_predictor_v1"),
    )
    parser.add_argument(
        "--output-remote",
        default=(
            "rclone_s3:dpsgd-as-a-optimization-problem/current-work/"
            "adaptive-clipping-wrn/adaptive_clipping_predictor/v1_20260924"
        ),
    )
    parser.add_argument(
        "--ntfy-topic", default="wrn16-1-sketch-progress-20260921-6b2b63b8"
    )
    parser.add_argument("--context-size", type=int, default=8)
    parser.add_argument("--deployment-steps", type=int, default=1_020)
    parser.add_argument("--weight-dim", type=int, default=512)
    parser.add_argument("--weight-seed", type=int, default=20_260_924)
    parser.add_argument("--policy-seed", type=int, default=1_729)
    parser.add_argument("--http-port", type=int, default=18_779)
    parser.add_argument("--http-workers", type=int, default=64)
    parser.add_argument("--checkpoint-batch", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dual-learning-rate", type=float, default=0.1)
    parser.add_argument("--dual-max", type=float, default=100.0)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--notify-seconds", type=int, default=300)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--smoke-rows", type=int)
    return parser.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    if args.context_size != 8:
        raise ValueError("the registered experiment uses CTX_SIZE=8")
    if args.deployment_steps != 2 * SOURCE_STEPS:
        raise ValueError("the registered experiment uses T'=2T=1020")
    if args.weight_dim != 512:
        raise ValueError("the registered experiment uses a 512-D weight projection")
    if args.prepare_only and args.train_only:
        raise ValueError("prepare-only and train-only are mutually exclusive")
    if not 1 <= args.http_workers <= 128:
        raise ValueError("http-workers must be in [1, 128]")
    if args.smoke_rows is not None and args.smoke_rows < 30:
        raise ValueError("smoke-rows must be at least 30")
    if not args.source_remote.startswith("rclone_s3:"):
        raise ValueError("source remote must use rclone_s3")
    if not args.no_upload and not args.output_remote.startswith("rclone_s3:"):
        raise ValueError("output remote must use rclone_s3")
    return Config(
        dataset_dir=str(args.dataset_dir),
        checkpoint_dir=str(args.checkpoint_dir),
        source_remote=args.source_remote.rstrip("/"),
        output_dir=str(args.output_dir),
        output_remote=args.output_remote.rstrip("/"),
        ntfy_topic=args.ntfy_topic,
        context_size=args.context_size,
        deployment_steps=args.deployment_steps,
        weight_dim=args.weight_dim,
        weight_seed=args.weight_seed,
        policy_seed=args.policy_seed,
        http_port=args.http_port,
        http_workers=args.http_workers,
        checkpoint_batch=args.checkpoint_batch,
        batch_size=args.batch_size,
        workers=args.workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dual_learning_rate=args.dual_learning_rate,
        dual_max=args.dual_max,
        constraint_tolerance=args.constraint_tolerance,
        dropout=args.dropout,
        patience=args.patience,
        notify_seconds=args.notify_seconds,
        upload=not args.no_upload,
        prepare_only=args.prepare_only,
        train_only=args.train_only,
        smoke_rows=args.smoke_rows,
    )


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def atomic_torch_save(payload: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
    ).strip()


class Notifier:
    def __init__(self, topic: str, interval: int) -> None:
        self.topic = topic
        self.interval = interval
        self.last = 0.0

    def send(
        self, title: str, message: str, tags: str = "bar_chart", force: bool = False
    ) -> None:
        now = time.monotonic()
        if not force and now - self.last < self.interval:
            return
        print(f"NOTIFY {title}: {message}", flush=True)
        self.last = now
        if not self.topic:
            return
        request = urllib.request.Request(
            f"https://ntfy.sh/{self.topic}",
            data=message.encode("utf-8"),
            method="POST",
            headers={"Title": title, "Tags": tags},
        )
        try:
            with urllib.request.urlopen(request, timeout=15):
                pass
        except Exception as exc:
            print(f"ntfy notification failed: {exc}", flush=True)


def load_registered_rows(config: Config, output_dir: Path) -> dict[str, np.ndarray]:
    database = Path(config.dataset_dir) / "binary_advantages.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(database)
    with sqlite3.connect(database) as connection:
        records = connection.execute(
            """
            SELECT run_id,checkpoint_step,batch_replicate,anchor_slot,
                   example_id,dataset_split,advantage_low_vs_high,
                   advantage_scaled,gradient_norm
            FROM binary_advantages
            WHERE checkpoint_step>=?
            ORDER BY run_id,checkpoint_step,batch_replicate,anchor_slot
            """,
            (config.context_size,),
        ).fetchall()
    if len(records) != 20_120:
        raise RuntimeError(f"expected 20,120 post-warm-up rows, found {len(records):,}")
    if config.smoke_rows is not None:
        # Keep every split represented while preserving deterministic order.
        by_split = {
            name: [record for record in records if record[5] == name]
            for name in ("train", "validation", "test")
        }
        train_n = max(10, config.smoke_rows - 20)
        records = by_split["train"][:train_n] + by_split["validation"][:10] + by_split["test"][:10]
        records.sort(key=lambda row: (row[0], row[1], row[2], row[3]))

    split_code = {"train": 0, "validation": 1, "test": 2}
    arrays = {
        "run_id": np.asarray([row[0] for row in records], dtype=np.int16),
        "checkpoint_step": np.asarray([row[1] for row in records], dtype=np.int16),
        "batch_replicate": np.asarray([row[2] for row in records], dtype=np.int8),
        "anchor_slot": np.asarray([row[3] for row in records], dtype=np.int8),
        "example_id": np.asarray([row[4] for row in records], dtype=np.int32),
        "split": np.asarray([split_code[row[5]] for row in records], dtype=np.int8),
        "advantage": np.asarray([row[6] for row in records], dtype=np.float64),
        "advantage_scaled": np.asarray([row[7] for row in records], dtype=np.float32),
        "gradient_norm": np.asarray([row[8] for row in records], dtype=np.float32),
    }
    np.savez_compressed(output_dir / "rows.npz", **arrays)
    return arrays


def ensure_raw_memmap(path: Path, dtype: str, shape: tuple[int, ...]) -> np.memmap:
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if path.exists():
        if path.stat().st_size != expected:
            raise RuntimeError(
                f"unexpected size for {path}: {path.stat().st_size} != {expected}"
            )
        return np.memmap(path, dtype=dtype, mode="r+", shape=shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    mapping = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    mapping[:] = 0
    mapping.flush()
    return mapping


def parameter_names() -> list[str]:
    model = WideResNet(
        num_classes=10, depth=16, width=1, groups=16, dropout_rate=0.0
    )
    names = [name for name, _ in model.named_parameters()]
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise RuntimeError("WRN-16-1 parameter count changed")
    return names


def generate_weight_projection(config: Config, feature_dir: Path) -> Path:
    path = feature_dir / "weight_projection.npy"
    expected_shape = (config.weight_dim, PARAMETER_COUNT)
    if path.exists():
        array = np.load(path, mmap_mode="r")
        if array.shape != expected_shape or array.dtype != np.float32:
            raise RuntimeError("existing weight projection has the wrong shape or dtype")
        return path

    temporary = path.with_suffix(path.suffix + ".tmp")
    projection = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=expected_shape
    )
    generator = np.random.Generator(np.random.PCG64(config.weight_seed))
    scale = np.float32(1.0 / math.sqrt(config.weight_dim))
    for start in range(0, config.weight_dim, 8):
        stop = min(config.weight_dim, start + 8)
        projection[start:stop] = generator.standard_normal(
            (stop - start, PARAMETER_COUNT), dtype=np.float32
        ) * scale
    projection.flush()
    del projection
    os.replace(temporary, path)
    return path


def materialize_weight_features(
    config: Config,
    rows: dict[str, np.ndarray],
    feature_dir: Path,
    notifier: Notifier,
    device: torch.device,
) -> Path:
    projection_path = generate_weight_projection(config, feature_dir)
    features_path = feature_dir / "checkpoint_weight_features.npy"
    complete_path = feature_dir / "checkpoint_weight_complete.u8"
    shape = (RUNS, SOURCE_STEPS + 1, config.weight_dim)
    if features_path.exists():
        features = np.lib.format.open_memmap(features_path, mode="r+")
        if features.shape != shape or features.dtype != np.float32:
            raise RuntimeError("existing checkpoint features have wrong schema")
    else:
        features = np.lib.format.open_memmap(
            features_path, mode="w+", dtype=np.float32, shape=shape
        )
        features[:] = np.nan
        features.flush()
    complete = ensure_raw_memmap(
        complete_path, "u1", (RUNS, SOURCE_STEPS + 1)
    )
    required = sorted({
        (int(run), int(step))
        for run, step in zip(rows["run_id"], rows["checkpoint_step"], strict=True)
    })
    missing = [(run, step) for run, step in required if not complete[run, step]]
    if not missing:
        return features_path

    notifier.send(
        "Clipping predictor: weight projection",
        f"Projecting {len(missing):,} checkpoints to {config.weight_dim} dimensions.",
        "gear,bar_chart",
        force=True,
    )
    projection_np = np.load(projection_path, mmap_mode="r")
    projection = torch.from_numpy(projection_np).to(device=device, dtype=torch.float32)
    names = parameter_names()
    started = time.monotonic()
    done = 0
    for batch_start in range(0, len(missing), config.checkpoint_batch):
        batch_keys = missing[batch_start:batch_start + config.checkpoint_batch]
        vectors = []
        for run, step in batch_keys:
            checkpoint = (
                Path(config.checkpoint_dir)
                / f"run_{run:02d}"
                / f"step_{step:06d}"
                / "checkpoint.pt"
            )
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload["model"]
            vector = torch.cat([state[name].reshape(-1) for name in names])
            if vector.numel() != PARAMETER_COUNT:
                raise RuntimeError(f"parameter mismatch in {checkpoint}")
            vectors.append(vector)
        matrix = torch.stack(vectors).to(device, non_blocking=True)
        with torch.inference_mode():
            projected = matrix @ projection.T
        values = projected.cpu().numpy()
        for key, value in zip(batch_keys, values, strict=True):
            run, step = key
            features[run, step] = value
            complete[run, step] = 1
        features.flush()
        complete.flush()
        done += len(batch_keys)
        elapsed = time.monotonic() - started
        rate = done / max(elapsed, 1e-9)
        notifier.send(
            "Clipping predictor: weight projection",
            f"{done:,}/{len(missing):,} checkpoints; {rate:.1f}/s; "
            f"ETA {(len(missing)-done)/max(rate,1e-9)/60:.1f} min.",
        )
        del matrix, projected, values, vectors
    del projection
    return features_path


def http_get(url: str, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "adaptive-clipping-predictor/1"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    error: Exception | None = None
    for attempt in range(6):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
            if byte_range is not None:
                expected = byte_range[1] - byte_range[0] + 1
                if len(payload) != expected:
                    raise IOError(f"range returned {len(payload)} bytes, expected {expected}")
            return payload
        except Exception as exc:
            error = exc
            time.sleep(min(10.0, 0.25 * 2**attempt))
    raise RuntimeError(f"failed to read {url}: {error}")


@contextlib.contextmanager
def rclone_http_gateway(config: Config, feature_dir: Path) -> Iterator[str]:
    address = f"127.0.0.1:{config.http_port}"
    log_path = feature_dir / "rclone_http.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            [
                "rclone", "serve", "http", config.source_remote,
                "--addr", address, "--read-only", "--no-modtime",
                "--log-level", "NOTICE",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        base = f"http://{address}"
        health = f"{base}/run_00/step_000001/manifest.json"
        try:
            for _ in range(60):
                if process.poll() is not None:
                    raise RuntimeError(f"rclone HTTP gateway exited with {process.returncode}")
                try:
                    http_get(health)
                    break
                except Exception:
                    time.sleep(0.5)
            else:
                raise TimeoutError("rclone HTTP gateway did not become ready")
            yield base
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def bounded_futures(
    executor: concurrent.futures.ThreadPoolExecutor,
    function,
    items: Iterable,
    bound: int,
) -> Iterator:
    iterator = iter(items)
    pending: set[concurrent.futures.Future] = set()
    for _ in range(bound):
        try:
            pending.add(executor.submit(function, next(iterator)))
        except StopIteration:
            break
    while pending:
        done, pending = concurrent.futures.wait(
            pending, return_when=concurrent.futures.FIRST_COMPLETED
        )
        for future in done:
            yield future.result()
            try:
                pending.add(executor.submit(function, next(iterator)))
            except StopIteration:
                pass


def materialize_history_features(
    config: Config,
    rows: dict[str, np.ndarray],
    feature_dir: Path,
    notifier: Notifier,
) -> tuple[Path, Path]:
    count = len(rows["run_id"])
    k = config.context_size
    sketches_path = feature_dir / "history_sketches.bf16"
    norms_path = feature_dir / "history_norms.f32"
    complete_path = feature_dir / "history_complete.u8"
    sketches = ensure_raw_memmap(
        sketches_path, "<u2", (count, k, GRADIENT_DIM)
    )
    norms = ensure_raw_memmap(norms_path, "<f4", (count, k, NORM_DIM))
    complete = ensure_raw_memmap(complete_path, "u1", (count, k))

    groups: dict[tuple[int, int], dict[int, list[tuple[int, int]]]] = {}
    for row_index, (run, step, example) in enumerate(zip(
        rows["run_id"], rows["checkpoint_step"], rows["example_id"], strict=True
    )):
        start = int(step) - k + 1
        for offset, history_step in enumerate(range(start, int(step) + 1)):
            if complete[row_index, offset]:
                continue
            example_map = groups.setdefault((int(run), history_step), {})
            example_map.setdefault(int(example), []).append((row_index, offset))
    if not groups:
        if not bool(np.all(complete)):
            raise RuntimeError("history group list empty but completion mask is incomplete")
        return sketches_path, norms_path

    total_cells = count * k
    initial_cells = int(complete.sum())
    notifier.send(
        "Clipping predictor: history extraction",
        f"Starting {len(groups):,} remote shards; {initial_cells:,}/{total_cells:,} "
        f"history cells already complete; {config.http_workers} workers.",
        "floppy_disk,bar_chart",
        force=True,
    )

    tasks = [(run, step, example_map) for (run, step), example_map in groups.items()]
    started = time.monotonic()

    def fetch_shard(task):
        run, history_step, example_map = task
        prefix = f"{gateway}/run_{run:02d}/step_{history_step:06d}"
        norm_payload = http_get(f"{prefix}/norms.f32")
        if len(norm_payload) != 50_000 * NORM_DIM * 4:
            raise IOError(f"bad norms size for run={run}, step={history_step}")
        norm_array = np.frombuffer(norm_payload, dtype="<f4").reshape(50_000, NORM_DIM)
        outputs = []
        for example, destinations in example_map.items():
            start_byte = example * GRADIENT_DIM * 2
            payload = http_get(
                f"{prefix}/sketch.bf16",
                (start_byte, start_byte + GRADIENT_DIM * 2 - 1),
            )
            outputs.append((destinations, payload, norm_array[example].copy()))
        return run, history_step, outputs

    completed_shards = 0
    with rclone_http_gateway(config, feature_dir) as gateway:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.http_workers
        ) as executor:
            for run, history_step, outputs in bounded_futures(
                executor, fetch_shard, tasks, config.http_workers * 2
            ):
                for destinations, payload, norm_value in outputs:
                    sketch_value = np.frombuffer(payload, dtype="<u2")
                    for row_index, offset in destinations:
                        sketches[row_index, offset] = sketch_value
                        norms[row_index, offset] = norm_value
                        complete[row_index, offset] = 1
                completed_shards += 1
                if completed_shards % 64 == 0:
                    sketches.flush()
                    norms.flush()
                    complete.flush()
                elapsed = time.monotonic() - started
                rate = completed_shards / max(elapsed, 1e-9)
                cells = initial_cells + int(complete.sum()) - initial_cells
                notifier.send(
                    "Clipping predictor: history extraction",
                    f"{completed_shards:,}/{len(tasks):,} shards; "
                    f"{cells:,}/{total_cells:,} cells; {rate:.2f} shards/s; "
                    f"ETA {(len(tasks)-completed_shards)/max(rate,1e-9)/60:.1f} min; "
                    f"last run={run:02d} step={history_step:03d}.",
                )
    sketches.flush()
    norms.flush()
    complete.flush()
    missing = int((complete == 0).sum())
    if missing:
        raise RuntimeError(f"history extraction left {missing:,} cells incomplete")
    return sketches_path, norms_path


def bf16_bits_to_float32(values: np.ndarray) -> np.ndarray:
    expanded = values.astype(np.uint32)
    expanded <<= np.uint32(16)
    return expanded.view(np.float32)


def fit_normalization(
    config: Config,
    rows: dict[str, np.ndarray],
    feature_dir: Path,
    notifier: Notifier,
) -> Path:
    output = feature_dir / "normalization.npz"
    if output.exists():
        values = np.load(output)
        if values["sketch_mean"].shape != (GRADIENT_DIM,):
            raise RuntimeError("normalization file has wrong schema")
        return output
    notifier.send(
        "Clipping predictor: normalization",
        "Fitting coordinate-wise statistics on runs 0-7 only.",
        "abacus,bar_chart",
        force=True,
    )
    count = len(rows["run_id"])
    k = config.context_size
    sketches = np.memmap(
        feature_dir / "history_sketches.bf16", dtype="<u2", mode="r",
        shape=(count, k, GRADIENT_DIM),
    )
    norms = np.memmap(
        feature_dir / "history_norms.f32", dtype="<f4", mode="r",
        shape=(count, k, NORM_DIM),
    )
    train_indices = np.flatnonzero(rows["split"] == 0)
    sketch_sum = np.zeros(GRADIENT_DIM, dtype=np.float64)
    sketch_sq = np.zeros(GRADIENT_DIM, dtype=np.float64)
    norm_sum = np.zeros(NORM_DIM, dtype=np.float64)
    norm_sq = np.zeros(NORM_DIM, dtype=np.float64)
    observations = 0
    started = time.monotonic()
    for start in range(0, len(train_indices), 64):
        selected = train_indices[start:start + 64]
        sketch = bf16_bits_to_float32(np.asarray(sketches[selected])).reshape(-1, GRADIENT_DIM)
        norm = np.log1p(np.asarray(norms[selected], dtype=np.float32)).reshape(-1, NORM_DIM)
        sketch64 = sketch.astype(np.float64)
        norm64 = norm.astype(np.float64)
        sketch_sum += sketch64.sum(axis=0)
        sketch_sq += np.square(sketch64).sum(axis=0)
        norm_sum += norm64.sum(axis=0)
        norm_sq += np.square(norm64).sum(axis=0)
        observations += sketch.shape[0]
        notifier.send(
            "Clipping predictor: normalization",
            f"Processed {min(start+64,len(train_indices)):,}/{len(train_indices):,} "
            f"training rows in {(time.monotonic()-started)/60:.1f} min.",
        )
    sketch_mean = sketch_sum / observations
    norm_mean = norm_sum / observations
    sketch_var = np.maximum(sketch_sq / observations - sketch_mean**2, 1e-12)
    norm_var = np.maximum(norm_sq / observations - norm_mean**2, 1e-12)

    weights = np.load(feature_dir / "checkpoint_weight_features.npy", mmap_mode="r")
    # Fit this transform on the registered training rows only. Besides avoiding
    # validation/test leakage, this also makes reduced smoke runs well-defined.
    train_weights = np.asarray([
        weights[int(rows["run_id"][row]), int(rows["checkpoint_step"][row])]
        for row in train_indices
    ], dtype=np.float32)
    weight_mean = train_weights.mean(axis=0, dtype=np.float64)
    weight_std = train_weights.std(axis=0, dtype=np.float64)
    np.savez(
        output,
        sketch_mean=sketch_mean.astype(np.float32),
        sketch_std=np.sqrt(sketch_var).astype(np.float32),
        norm_mean=norm_mean.astype(np.float32),
        norm_std=np.sqrt(norm_var).astype(np.float32),
        weight_mean=weight_mean.astype(np.float32),
        weight_std=np.maximum(weight_std, 1e-6).astype(np.float32),
        observations=np.asarray(observations, dtype=np.int64),
    )
    return output


class PredictorDataset(Dataset):
    def __init__(
        self,
        rows: dict[str, np.ndarray],
        indices: np.ndarray,
        feature_dir: Path,
        context_size: int,
        deployment_steps: int,
    ) -> None:
        self.rows = rows
        self.indices = np.asarray(indices, dtype=np.int64)
        self.context_size = context_size
        self.deployment_steps = deployment_steps
        count = len(rows["run_id"])
        self.sketches = np.memmap(
            feature_dir / "history_sketches.bf16", dtype="<u2", mode="r",
            shape=(count, context_size, GRADIENT_DIM),
        )
        self.norms = np.memmap(
            feature_dir / "history_norms.f32", dtype="<f4", mode="r",
            shape=(count, context_size, NORM_DIM),
        )
        self.weights = np.load(
            feature_dir / "checkpoint_weight_features.npy", mmap_mode="r"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        row = int(self.indices[item])
        run = int(self.rows["run_id"][row])
        step = int(self.rows["checkpoint_step"][row])
        raw = torch.from_numpy(np.array(self.sketches[row], copy=True))
        sketch = raw.view(torch.bfloat16)
        norms = torch.from_numpy(np.array(self.norms[row], copy=True))
        weight = torch.from_numpy(np.array(self.weights[run, step], copy=True))
        advantage = float(self.rows["advantage"][row])
        label = float(advantage > SIGN_TOLERANCE)
        importance = abs(advantage) / (MEDIAN_ABSOLUTE_ADVANTAGE + 1e-12)
        normalized_step = (step + 1) / self.deployment_steps
        return (
            sketch,
            norms,
            weight,
            torch.tensor(normalized_step, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(importance, dtype=torch.float32),
            torch.tensor(advantage, dtype=torch.float64),
            torch.tensor(row, dtype=torch.int64),
        )


def move_batch(batch, device: torch.device):
    return tuple(value.to(device, non_blocking=True) for value in batch)


def forward_batch(
    model: AdaptiveClippingPredictor,
    batch,
    normalization: dict[str, Tensor],
) -> Tensor:
    sketches, norms, weights, steps, *_ = batch
    return model(
        sketches, norms, weights, steps,
        normalization["sketch_mean"], normalization["sketch_std"],
        normalization["norm_mean"], normalization["norm_std"],
        normalization["weight_mean"], normalization["weight_std"],
    )


@torch.inference_mode()
def evaluate(
    model: AdaptiveClippingPredictor,
    loader: DataLoader,
    normalization: dict[str, Tensor],
    device: torch.device,
    beta: float,
) -> dict:
    model.eval()
    logits_all = []
    labels_all = []
    importance_all = []
    advantages_all = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        logits_all.append(forward_batch(model, batch, normalization).cpu())
        labels_all.append(batch[4].cpu())
        importance_all.append(batch[5].cpu())
        advantages_all.append(batch[6].cpu())
    logits = torch.cat(logits_all).double()
    labels = torch.cat(labels_all).double()
    importance = torch.cat(importance_all).double()
    advantages = torch.cat(advantages_all).double()
    probabilities = logits.sigmoid()
    decisions = probabilities >= 0.5
    weighted_losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    ) * importance
    weight_sum = importance.sum().clamp_min(1e-12)
    expected_c2 = (0.0625 * probabilities + 1.0 - probabilities).mean()
    oracle = advantages.clamp_min(0).mean()
    expected_value = (probabilities * advantages).mean()
    deterministic_value = (decisions.double() * advantages).mean()
    return {
        "rows": int(len(logits)),
        "weighted_bce": float(weighted_losses.sum() / weight_sum),
        "expected_low_fraction": float(probabilities.mean()),
        "deterministic_low_fraction": float(decisions.double().mean()),
        "expected_c2": float(expected_c2),
        "constraint": float(expected_c2 - beta),
        "expected_policy_value": float(expected_value),
        "deterministic_policy_value": float(deterministic_value),
        "oracle_policy_value": float(oracle),
        "expected_regret": float(oracle - expected_value),
        "deterministic_regret": float(oracle - deterministic_value),
        "weighted_sign_accuracy": float(
            (importance * (decisions == labels.bool()).double()).sum() / weight_sum
        ),
        "ordinary_sign_accuracy": float((decisions == labels.bool()).double().mean()),
        "probability_mean": float(probabilities.mean()),
        "probability_std": float(probabilities.std(unbiased=True)),
    }


def stochastic_evaluation(
    probabilities: np.ndarray,
    advantages: np.ndarray,
    beta: float,
    seed: int,
    repetitions: int = 100,
) -> dict:
    generator = np.random.default_rng(seed)
    values = []
    low_rates = []
    constraints = []
    for _ in range(repetitions):
        low = generator.random(len(probabilities)) < probabilities
        values.append(float(np.mean(low * advantages)))
        low_rates.append(float(np.mean(low)))
        constraints.append(float(np.mean(np.where(low, 0.0625, 1.0)) - beta))
    return {
        "repetitions": repetitions,
        "policy_value_mean": float(np.mean(values)),
        "policy_value_sample_std": float(np.std(values, ddof=1)),
        "low_fraction_mean": float(np.mean(low_rates)),
        "low_fraction_sample_std": float(np.std(low_rates, ddof=1)),
        "constraint_mean": float(np.mean(constraints)),
        "constraint_sample_std": float(np.std(constraints, ddof=1)),
    }


@torch.inference_mode()
def collect_probabilities(
    model: AdaptiveClippingPredictor,
    loader: DataLoader,
    normalization: dict[str, Tensor],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities = []
    advantages = []
    for cpu_batch in loader:
        batch = move_batch(cpu_batch, device)
        probabilities.append(forward_batch(model, batch, normalization).sigmoid().cpu())
        advantages.append(batch[6].cpu())
    return torch.cat(probabilities).numpy(), torch.cat(advantages).numpy()


def baseline_statistics(advantages: np.ndarray) -> dict:
    return {
        "always_high_value": 0.0,
        "always_low_value": float(advantages.mean()),
        "oracle_value": float(np.maximum(advantages, 0).mean()),
        "always_low_fraction": 1.0,
        "always_high_fraction": 0.0,
    }


def train_predictor(
    config: Config,
    rows: dict[str, np.ndarray],
    output_dir: Path,
    feature_dir: Path,
    notifier: Notifier,
    device: torch.device,
) -> dict:
    torch.manual_seed(config.policy_seed)
    np.random.seed(config.policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.policy_seed)
    split_indices = {
        "train": np.flatnonzero(rows["split"] == 0),
        "validation": np.flatnonzero(rows["split"] == 1),
        "test": np.flatnonzero(rows["split"] == 2),
    }
    datasets = {
        name: PredictorDataset(
            rows, indices, feature_dir, config.context_size, config.deployment_steps
        )
        for name, indices in split_indices.items()
    }

    def loader_for(name: str, shuffle: bool, epoch: int = 0) -> DataLoader:
        generator = torch.Generator().manual_seed(config.policy_seed + epoch)
        return DataLoader(
            datasets[name], batch_size=config.batch_size, shuffle=shuffle,
            num_workers=config.workers, pin_memory=True,
            persistent_workers=False, generator=generator,
        )

    evaluation_loaders = {
        name: loader_for(name, False) for name in datasets
    }
    stats_file = np.load(feature_dir / "normalization.npz")
    normalization = {
        key: torch.from_numpy(stats_file[key]).to(device)
        for key in (
            "sketch_mean", "sketch_std", "norm_mean", "norm_std",
            "weight_mean", "weight_std",
        )
    }
    model = AdaptiveClippingPredictor(dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    beta = (SOURCE_STEPS - config.context_size) / (
        config.deployment_steps - config.context_size
    )
    dual = 0.0
    start_epoch = 1
    best_epoch = 0
    best_feasible = False
    best_value = -math.inf
    best_violation = math.inf
    stale_epochs = 0
    metrics_path = output_dir / "epoch_metrics.json"
    last_path = output_dir / "last_predictor.pt"
    best_path = output_dir / "best_predictor.pt"
    epoch_metrics: list[dict] = []
    if last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        dual = float(payload["dual"])
        start_epoch = int(payload["epoch"]) + 1
        best_epoch = int(payload["best_epoch"])
        best_feasible = bool(payload.get("best_feasible", False))
        best_value = float(payload["best_value"])
        best_violation = float(payload["best_violation"])
        stale_epochs = int(payload["stale_epochs"])
        if metrics_path.exists():
            epoch_metrics = json.loads(metrics_path.read_text())
    notifier.send(
        "Clipping predictor: training",
        f"Starting epoch {start_epoch}/{config.epochs}; model={parameter_count(model):,} "
        f"parameters; train={len(datasets['train']):,}; beta={beta:.6f}; "
        f"GPU={torch.cuda.get_device_name(device)}.",
        "rocket,bar_chart",
        force=True,
    )
    started = time.monotonic()
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        train_loader = loader_for("train", True, epoch)
        running_loss = 0.0
        batches = 0
        for cpu_batch in train_loader:
            batch = move_batch(cpu_batch, device)
            logits = forward_batch(model, batch, normalization)
            weighted = F.binary_cross_entropy_with_logits(
                logits, batch[4], reduction="none"
            ) * batch[5]
            predictor_loss = weighted.sum() / batch[5].sum().clamp_min(1e-12)
            probabilities = logits.sigmoid()
            constraint = (0.0625 * probabilities + 1.0 - probabilities).mean() - beta
            loss = predictor_loss + dual * constraint
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(predictor_loss.detach())
            batches += 1

        train_metrics = evaluate(
            model, evaluation_loaders["train"], normalization, device, beta
        )
        dual = min(
            config.dual_max,
            max(0.0, dual + config.dual_learning_rate * train_metrics["constraint"]),
        )
        validation_metrics = evaluate(
            model, evaluation_loaders["validation"], normalization, device, beta
        )
        feasible = train_metrics["constraint"] <= config.constraint_tolerance
        value = validation_metrics["expected_policy_value"]
        violation = max(0.0, train_metrics["constraint"])
        improved = False
        if feasible and (not best_feasible or value > best_value):
            improved = True
        elif not best_feasible and violation < best_violation:
            improved = True
        if improved:
            best_epoch = epoch
            best_feasible = feasible
            best_value = value
            best_violation = violation
            stale_epochs = 0
            atomic_torch_save(
                {
                    "epoch": epoch, "model": model.state_dict(), "dual": dual,
                    "feasible": feasible, "config": asdict(config), "beta": beta,
                    "normalization": {
                        key: value.cpu() for key, value in normalization.items()
                    },
                },
                best_path,
            )
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "mean_minibatch_predictor_loss": running_loss / max(batches, 1),
            "dual": dual,
            "feasible": feasible,
            "train": train_metrics,
            "validation": validation_metrics,
            "elapsed_seconds": time.monotonic() - started,
        }
        epoch_metrics.append(record)
        atomic_json(metrics_path, epoch_metrics)
        atomic_torch_save(
            {
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "dual": dual,
                "best_epoch": best_epoch, "best_value": best_value,
                "best_feasible": best_feasible, "best_violation": best_violation, "stale_epochs": stale_epochs,
                "config": asdict(config),
            },
            last_path,
        )
        print(json.dumps(record, sort_keys=True), flush=True)
        notifier.send(
            "Clipping predictor: training",
            f"Epoch {epoch}/{config.epochs}; dual={dual:.4g}; "
            f"train g={train_metrics['constraint']:.3e}; "
            f"val expected value={value:.3e}; val deterministic "
            f"value={validation_metrics['deterministic_policy_value']:.3e}; "
            f"best epoch={best_epoch}; stale={stale_epochs}/{config.patience}.",
        )
        if stale_epochs >= config.patience and best_feasible:
            break

    if not best_path.exists():
        raise RuntimeError("training did not produce a best predictor checkpoint")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final_metrics = {
        name: evaluate(model, loader, normalization, device, beta)
        for name, loader in evaluation_loaders.items()
    }
    probabilities, test_advantages = collect_probabilities(
        model, evaluation_loaders["test"], normalization, device
    )
    summary = {
        "status": "complete",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "best_epoch": int(best["epoch"]),
        "selected_checkpoint_feasible": bool(best["feasible"]),
        "dual": float(best["dual"]),
        "beta": beta,
        "required_expected_low_fraction": (1.0 - beta) / 0.9375,
        "model_parameters": parameter_count(model),
        "splits": final_metrics,
        "test_stochastic_100": stochastic_evaluation(
            probabilities, test_advantages, beta, config.policy_seed + 100_000
        ),
        "test_baselines": baseline_statistics(test_advantages),
        "epochs_completed": len(epoch_metrics),
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_json(output_dir / "summary.json", summary)
    np.savez_compressed(
        output_dir / "test_predictions.npz",
        probabilities=probabilities.astype(np.float32),
        advantages=test_advantages.astype(np.float64),
        row_indices=split_indices["test"],
    )
    notifier.send(
        "Clipping predictor: training complete",
        f"Best epoch={summary['best_epoch']}; test expected value="
        f"{final_metrics['test']['expected_policy_value']:.3e}; deterministic="
        f"{final_metrics['test']['deterministic_policy_value']:.3e}; "
        f"test g={final_metrics['test']['constraint']:.3e}; "
        f"always-low={summary['test_baselines']['always_low_value']:.3e}; "
        f"oracle={summary['test_baselines']['oracle_value']:.3e}.",
        "white_check_mark,bar_chart",
        force=True,
    )
    return summary


def write_manifest(
    config: Config,
    output_dir: Path,
    rows: dict[str, np.ndarray],
) -> None:
    dataset_database = Path(config.dataset_dir) / "binary_advantages.sqlite3"
    manifest = {
        "schema_version": 1,
        "description": "constrained adaptive clipping predictor, K=8, T'=1020",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_head": git_head(),
        "config": asdict(config),
        "source_database_sha256": sha256_file(dataset_database),
        "rows": int(len(rows["run_id"])),
        "split_rows": {
            "train": int((rows["split"] == 0).sum()),
            "validation": int((rows["split"] == 1).sum()),
            "test": int((rows["split"] == 2).sum()),
        },
        "action": {"low": 0.25, "high": 1.0},
        "reference_steps": SOURCE_STEPS,
        "deployment_steps": config.deployment_steps,
        "beta": (SOURCE_STEPS - config.context_size)
        / (config.deployment_steps - config.context_size),
    }
    atomic_json(output_dir / "manifest.json", manifest)


def upload_result(config: Config, output_dir: Path, notifier: Notifier) -> None:
    if not config.upload:
        return
    notifier.send(
        "Clipping predictor: upload",
        f"Uploading artifact to {config.output_remote}.",
        "outbox_tray,floppy_disk",
        force=True,
    )
    process = subprocess.Popen([
        "rclone", "copy", str(output_dir), config.output_remote,
        "--transfers", "8", "--checkers", "16", "--stats", "30s",
    ])
    while process.poll() is None:
        notifier.send(
            "Clipping predictor: upload",
            f"Upload is still active: {output_dir} -> {config.output_remote}.",
        )
        time.sleep(min(30, config.notify_seconds))
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, process.args)
    subprocess.run(
        ["rclone", "check", str(output_dir), config.output_remote, "--size-only"],
        check=True,
    )


def main() -> None:
    config = make_config(parse_args())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    output_dir = Path(config.output_dir)
    feature_dir = output_dir / "features"
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    notifier = Notifier(config.ntfy_topic, config.notify_seconds)
    notifier.send(
        "Adaptive clipping predictor started",
        f"K={config.context_size}, T'={config.deployment_steps}, d_w={config.weight_dim}, "
        f"epochs={config.epochs}, topic={config.ntfy_topic}, output={output_dir}.",
        "rocket,bar_chart",
        force=True,
    )
    try:
        rows = load_registered_rows(config, output_dir)
        write_manifest(config, output_dir, rows)
        if not config.train_only:
            materialize_weight_features(
                config, rows, feature_dir, notifier, device
            )
            materialize_history_features(config, rows, feature_dir, notifier)
            fit_normalization(config, rows, feature_dir, notifier)
        if config.prepare_only:
            notifier.send(
                "Adaptive clipping predictor prepared",
                "Feature materialization and train-only normalization completed.",
                "white_check_mark,floppy_disk",
                force=True,
            )
            return
        summary = train_predictor(
            config, rows, output_dir, feature_dir, notifier, device
        )
        upload_result(config, output_dir, notifier)
        notifier.send(
            "Adaptive clipping predictor complete",
            f"Best epoch={summary['best_epoch']}; artifact={config.output_remote}.",
            "white_check_mark,tada",
            force=True,
        )
    except Exception:
        notifier.send(
            "Adaptive clipping predictor failed",
            traceback.format_exc()[-3_500:],
            "x,warning",
            force=True,
        )
        raise


if __name__ == "__main__":
    main()
