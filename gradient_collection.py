"""Collect fixed, block-wise random projections of all CIFAR-10 gradients.

Each step is a transactional local staging directory. A background worker copies
it to the object store and deletes the local copy only after rclone verifies it.
The gradients are *unclipped* and are NOT differentially private data.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import WideResNet


BLOCKS = ("stem", "block1", "block2", "block3", "final_norm", "classifier")
DIMENSIONS = (131, 601, 1110, 2209, 69, 155)
assert sum(DIMENSIONS) == 4275


def block_for_name(name: str) -> str:
    block = name.split(".", 1)[0]
    if block not in BLOCKS:
        raise ValueError(f"unrecognized parameter block: {name}")
    return block


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class VerifiedUploader:
    """Bounded asynchronous uploader; never drops an unverified local step."""

    def __init__(self, remote: str, max_pending: int = 2) -> None:
        self.remote = remote.rstrip("/")
        self.pending: queue.Queue[Path | None] = queue.Queue(max_pending)
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._run, name="rclone-uploader", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            folder = self.pending.get()
            try:
                if folder is None:
                    return
                destination = f"{self.remote}/{folder.name}"
                for attempt in range(1, 4):
                    try:
                        subprocess.run(
                            ["rclone", "copy", str(folder), destination, "--transfers", "4"],
                            check=True, timeout=1800, capture_output=True, text=True,
                        )
                        subprocess.run(
                            ["rclone", "check", str(folder), destination, "--size-only"],
                            check=True, timeout=1800, capture_output=True, text=True,
                        )
                        break
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        if attempt == 3:
                            raise
                        time.sleep(10 * attempt)
                shutil.rmtree(folder)
                print(f"uploaded and verified {destination}", flush=True)
            except Exception as exc:
                self.error = exc
                print(f"UPLOAD FAILED; local data retained in {folder}: {exc}", flush=True)
                return
            finally:
                self.pending.task_done()

    def submit(self, folder: Path) -> None:
        while True:
            if self.error:
                raise RuntimeError("upload failed; stopped before local disk fills") from self.error
            try:
                self.pending.put(folder, timeout=5)
                return
            except queue.Full:
                pass

    def finish(self) -> None:
        while self.pending.unfinished_tasks:
            if self.error:
                raise RuntimeError("upload failed; local data retained") from self.error
            time.sleep(1)
        self.pending.put(None)
        self.thread.join()


class GradientCollector:
    def __init__(
        self,
        *,
        data_dir: Path,
        output_dir: Path,
        remote: str,
        device: torch.device,
        projection_seed: int,
        batch_size: int,
        workers: int,
        depth: int,
        width: int,
        groups: int,
        limit: int | None = None,
    ) -> None:
        self.device = device
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        archive = data_dir / "cifar-10-python.tar.gz"
        self.dataset_sha256 = file_sha256(archive) if archive.exists() else None
        self.batch_size = batch_size
        # Creating the shadow model must not advance the training RNG stream.
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            self.shadow = WideResNet(depth=depth, width=width, groups=groups, dropout_rate=0.0).to(device).eval()
        self.names: dict[str, list[str]] = {block: [] for block in BLOCKS}
        for name, parameter in self.shadow.named_parameters():
            self.names[block_for_name(name)].append(name)
        self.sizes = {
            block: sum(dict(self.shadow.named_parameters())[name].numel() for name in names)
            for block, names in self.names.items()
        }
        if (depth, width, groups) == (16, 1, 16):
            assert tuple(self.sizes.values()) == (464, 9792, 33344, 132224, 128, 650)

        projection_path = output_dir / "projection.pt"
        if projection_path.exists():
            payload = torch.load(projection_path, map_location="cpu", weights_only=True)
            if payload["seed"] != projection_seed or payload["dimensions"] != DIMENSIONS:
                raise ValueError("existing projection has different configuration")
            self.matrices = payload["matrices"]
        else:
            generator = torch.Generator(device="cpu").manual_seed(projection_seed)
            self.matrices = {
                block: torch.randn(self.sizes[block], DIMENSIONS[index], generator=generator)
                .mul_(1 / math.sqrt(DIMENSIONS[index]))
                for index, block in enumerate(BLOCKS)
            }
            temporary = projection_path.with_suffix(".tmp")
            torch.save({"seed": projection_seed, "dimensions": DIMENSIONS, "matrices": self.matrices}, temporary)
            os.replace(temporary, projection_path)
        self.projection_sha256 = file_sha256(projection_path)
        self.uploader = VerifiedUploader(remote)
        self.uploader.submit(self._projection_folder(projection_path))

        canonical = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=canonical)
        if limit is not None:
            if not 1 <= limit <= len(dataset):
                raise ValueError("invalid collection limit")
            dataset = torch.utils.data.Subset(dataset, range(limit))
        self.count = len(dataset)
        self.loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
            pin_memory=device.type == "cuda", persistent_workers=workers > 0,
            generator=torch.Generator().manual_seed(projection_seed + 1),
        )

    def _projection_folder(self, projection_path: Path) -> Path:
        folder = self.output_dir / "projection_spec"
        folder.mkdir(exist_ok=True)
        # A hard link avoids a second 3 GB local copy.
        linked = folder / "projection.pt"
        if not linked.exists():
            os.link(projection_path, linked)
        (folder / "manifest.json").write_text(json.dumps({
            "sha256": self.projection_sha256, "blocks": BLOCKS,
            "dimensions": DIMENSIONS, "parameter_sizes": self.sizes,
            "normalization": "R_ij ~ N(0, 1/k_l)",
        }, indent=2) + "\n")
        return folder

    def upload_initial(self, payload: dict) -> None:
        folder = self.output_dir / "initial"
        folder.mkdir(exist_ok=False)
        torch.save(payload, folder / "initial.pt")
        (folder / "manifest.json").write_text(json.dumps({
            "description": "model and RNG state before the first private update",
            "sha256": file_sha256(folder / "initial.pt"),
            "dataset_archive_sha256": self.dataset_sha256,
            "python_version": platform.python_version(),
            "package_versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "torchvision", "opacus", "numpy")
            },
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, text=True
            ).strip(),
        }, indent=2) + "\n")
        self.uploader.submit(folder)

    def collect(self, *, step: int, trained_model: nn.Module, checkpoint: dict) -> dict:
        started = time.monotonic()
        folder = self.output_dir / f"step_{step:06d}"
        folder.mkdir(exist_ok=False)
        torch.save(checkpoint, folder / "checkpoint.pt")
        self.shadow.load_state_dict(trained_model.state_dict())
        self.shadow.eval()
        parameters = dict(self.shadow.named_parameters())
        buffers = dict(self.shadow.named_buffers())

        def single_loss(params: dict, state: dict, image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            logits = functional_call(self.shadow, (params, state), (image.unsqueeze(0),))
            return nn.functional.cross_entropy(logits, target.unsqueeze(0))

        batched_grad = vmap(grad(single_loss), in_dims=(None, None, 0, 0), randomness="different")
        gpu_matrices = {block: matrix.to(self.device, non_blocking=False) for block, matrix in self.matrices.items()}
        sketches = np.memmap(folder / "sketch.bf16", dtype=np.uint16, mode="w+", shape=(self.count, 4275))
        norms = np.memmap(folder / "norms.f32", dtype=np.float32, mode="w+", shape=(self.count, len(BLOCKS)))
        offset = 0
        for images, targets in self.loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with torch.enable_grad():
                per_sample = batched_grad(parameters, buffers, images, targets)
            features = []
            layer_norms = []
            for block in BLOCKS:
                flat = torch.cat([per_sample[name].reshape(images.shape[0], -1) for name in self.names[block]], dim=1).detach()
                layer_norms.append(torch.linalg.vector_norm(flat, dim=1))
                features.append(flat @ gpu_matrices[block])
            combined = torch.cat(features, dim=1).to(torch.bfloat16).contiguous()
            next_offset = offset + images.shape[0]
            sketches[offset:next_offset] = combined.view(torch.uint16).cpu().numpy()
            norms[offset:next_offset] = torch.stack(layer_norms, dim=1).detach().float().cpu().numpy()
            offset = next_offset
            del per_sample, flat, features, combined, images, targets
        sketches.flush()
        norms.flush()
        del sketches, norms, gpu_matrices
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        manifest = {
            "step": step, "examples": offset, "feature_dim": 4275,
            "block_order": BLOCKS, "block_dims": DIMENSIONS,
            "gradient_definition": "unclipped per-example cross-entropy at post-update theta; canonical unaugmented CIFAR-10",
            "sample_order": "CIFAR-10 train original index order",
            "sketch_dtype": "bfloat16 little-endian raw uint16, row-major",
            "norm_dtype": "float32 little-endian row-major",
            "projection_sha256": self.projection_sha256,
            "files": {name: {"bytes": (folder / name).stat().st_size, "sha256": file_sha256(folder / name)}
                      for name in ("checkpoint.pt", "sketch.bf16", "norms.f32")},
            "elapsed_seconds": time.monotonic() - started,
        }
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        self.uploader.submit(folder)
        return manifest

    def finish(self) -> None:
        self.uploader.finish()
