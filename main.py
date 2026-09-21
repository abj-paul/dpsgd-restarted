"""Train the DP-friendly WideResNet on CIFAR-10 with an SGD baseline."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import WideResNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/wrn16-4"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA AMP.")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Use a deterministic subset for quick smoke tests.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    train_set = datasets.CIFAR10(
        args.data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        args.data_dir, train=False, download=True, transform=test_transform
    )
    if args.max_train_samples is not None:
        if not 1 <= args.max_train_samples <= len(train_set):
            raise ValueError("--max-train-samples must be between 1 and 50000")
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(train_set), generator=generator)[
            : args.max_train_samples
        ].tolist()
        train_set = Subset(train_set, indices)

    common = {
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=args.test_batch_size, shuffle=False, **common
    )
    return train_loader, test_loader


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_enabled: bool = False,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    correct = 0
    total = 0

    context = torch.enable_grad if training else torch.no_grad
    with context():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = targets.size(0)
            loss_sum += loss.detach().item() * batch_size
            correct += logits.detach().argmax(dim=1).eq(targets).sum().item()
            total += batch_size
    return loss_sum / total, correct / total


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available to PyTorch")
    seed_everything(args.seed)
    device = torch.device(args.device)
    amp_enabled = device.type == "cuda" and not args.no_amp

    train_loader, test_loader = make_loaders(args)
    model = WideResNet(
        num_classes=10,
        depth=args.depth,
        width=args.width,
        dropout_rate=args.dropout,
        groups=args.groups,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device} model=WRN-{args.depth}-{args.width} params={parameter_count:,}")
    best_accuracy = 0.0
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled
        )
        test_loss, test_accuracy = run_epoch(
            model, test_loader, criterion, device, amp_enabled=amp_enabled
        )
        scheduler.step()
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.monotonic() - started,
        }
        history.append(metrics)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} "
            f"test_loss={test_loss:.4f} test_acc={test_accuracy:.2%} "
            f"time={metrics['seconds']:.1f}s"
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "test_accuracy": test_accuracy,
                },
                args.output_dir / "best.pt",
            )
        (args.output_dir / "metrics.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
