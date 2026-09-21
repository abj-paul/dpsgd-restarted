"""Fixed-step, large-logical-batch DP-SGD baseline for CIFAR-10."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import WideResNet

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--output-dir", type=Path, default=Path("runs/dp_baseline_wrn16_1_q33"))
    p.add_argument("--steps", type=int, default=500)
    # Opacus derives Poisson q as 1 / len(loader). Three logical batches per
    # epoch gives q=1/3 and an expected batch of 16,667 examples.
    p.add_argument("--logical-batch-size", type=int, default=16_667)
    p.add_argument("--physical-batch-size", type=int, default=2_048)
    p.add_argument("--test-batch-size", type=int, default=1_000)
    p.add_argument("--lr", type=float, default=4.0)
    p.add_argument("--noise-multiplier", type=float, default=5.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--width", type=int, default=1)
    p.add_argument("--groups", type=int, default=16)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-augmentation", action="store_true")
    return p.parse_args()


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_ops = []
    if not args.no_augmentation:
        train_ops += [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
        ]
    train_ops += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ]
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    train_set = datasets.CIFAR10(
        args.data_dir, train=True, download=True, transform=transforms.Compose(train_ops)
    )
    test_set = datasets.CIFAR10(
        args.data_dir, train=False, download=True, transform=test_transform
    )
    common = dict(
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    return (
        DataLoader(
            train_set,
            batch_size=args.logical_batch_size,
            shuffle=True,
            drop_last=False,
            **common,
        ),
        DataLoader(
            test_set,
            batch_size=args.test_batch_size,
            shuffle=False,
            **common,
        ),
    )


@torch.no_grad()
def update_ema(
    ema_model: nn.Module, private_model: nn.Module, step: int, decay: float
) -> None:
    warm_decay = min(decay, (1.0 + step) / (10.0 + step))
    for averaged, current in zip(
        ema_model.parameters(), private_model._module.parameters(), strict=True
    ):
        averaged.mul_(warm_decay).add_(current, alpha=1.0 - warm_decay)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    model.eval()
    loss_sum = correct = total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += nn.functional.cross_entropy(
            logits, targets, reduction="sum"
        ).item()
        correct += logits.argmax(1).eq(targets).sum().item()
        total += targets.numel()
    return loss_sum / total, correct / total


def serializable_args(args: argparse.Namespace) -> dict:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if not 0 < args.logical_batch_size <= 50_000:
        raise ValueError("logical batch size must be in [1, 50000]")
    if args.physical_batch_size > args.logical_batch_size:
        raise ValueError("physical batch size cannot exceed logical batch size")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    train_loader, test_loader = make_loaders(args)
    model = WideResNet(
        num_classes=10,
        depth=args.depth,
        width=args.width,
        groups=args.groups,
        dropout_rate=0.0,
    ).to(device)
    ema_model = copy.deepcopy(model).eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    criterion = nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine(accountant="prv", secure_mode=False)
    model, optimizer, criterion, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        poisson_sampling=True,
        clipping="flat",
        grad_sample_mode="ghost",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(serializable_args(args), indent=2) + "\n", encoding="utf-8"
    )
    sample_rate = 1.0 / len(train_loader)
    params = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"device={device} WRN-{args.depth}-{args.width} params={params:,} "
        f"logical_batch={args.logical_batch_size} "
        f"physical_batch={args.physical_batch_size} q={sample_rate:.5f} "
        f"sigma={args.noise_multiplier} C={args.max_grad_norm}",
        flush=True,
    )

    started = time.monotonic()
    update = 0
    history: list[dict] = []
    loss_sum = correct = examples = 0
    model.train()
    with BatchMemoryManager(
        data_loader=train_loader,
        max_physical_batch_size=args.physical_batch_size,
        optimizer=optimizer,
    ) as memory_safe_loader:
        while update < args.steps:
            for images, targets in memory_safe_loader:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item() * targets.numel()
                correct += logits.detach().argmax(1).eq(targets).sum().item()
                examples += targets.numel()
                if optimizer._is_last_step_skipped:
                    continue

                update += 1
                update_ema(ema_model, model, update, args.ema_decay)
                epsilon = privacy_engine.get_epsilon(args.delta)
                metrics = {
                    "step": update,
                    "train_loss": loss_sum / examples,
                    "train_accuracy": correct / examples,
                    "logical_examples": examples,
                    "epsilon_prv": epsilon,
                    "elapsed_seconds": time.monotonic() - started,
                }
                loss_sum = correct = examples = 0
                if update % args.eval_every == 0 or update == args.steps:
                    test_loss, test_accuracy = evaluate(ema_model, test_loader, device)
                    metrics["ema_test_loss"] = test_loss
                    metrics["ema_test_accuracy"] = test_accuracy
                    model.train()
                history.append(metrics)
                if update % args.log_every == 0 or update == 1:
                    seen = sum(item["logical_examples"] for item in history)
                    rate = seen / (time.monotonic() - started)
                    test_text = (
                        f" ema_test_acc={metrics['ema_test_accuracy']:.2%}"
                        if "ema_test_accuracy" in metrics
                        else ""
                    )
                    print(
                        f"step={update:04d}/{args.steps} "
                        f"loss={metrics['train_loss']:.4f} "
                        f"acc={metrics['train_accuracy']:.2%} "
                        f"eps={epsilon:.4f} rate={rate:.0f} example/s{test_text}",
                        flush=True,
                    )
                if update >= args.steps:
                    break

    elapsed = time.monotonic() - started
    epsilon = privacy_engine.get_epsilon(args.delta)
    summary = {
        "status": "complete",
        "steps": update,
        "sample_rate": sample_rate,
        "noise_multiplier": args.noise_multiplier,
        "max_grad_norm": args.max_grad_norm,
        "delta": args.delta,
        "epsilon_prv": epsilon,
        "elapsed_seconds": elapsed,
        "hyperparameters": serializable_args(args),
        "final": history[-1],
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "model": model._module.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "summary": summary,
        },
        args.output_dir / "checkpoint.pt",
    )
    print(
        f"complete steps={update} epsilon={epsilon:.4f} delta={args.delta:g} "
        f"elapsed={elapsed / 60:.1f}min output={args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
