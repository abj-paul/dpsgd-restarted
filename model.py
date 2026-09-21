"""DP-friendly WideResNet for CIFAR-sized images.

This is a PyTorch port of the WideResNet used by the historical JAX Privacy
CIFAR experiments.  In particular, it uses scaled weight-standardized
convolutions and GroupNorm, so it does not couple examples through batch
statistics.  The default configuration is WRN-16-4.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class WSConv2d(nn.Conv2d):
    """Conv2d with scaled weight standardization and per-channel gain."""

    def __init__(self, *args, eps: float = 1e-4, **kwargs) -> None:
        kwargs.setdefault("bias", True)
        super().__init__(*args, **kwargs)
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(self.out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Haiku's VarianceScaling(1.0, "fan_in", "normal") equivalent.
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        nn.init.normal_(self.weight, mean=0.0, std=math.sqrt(1.0 / fan_in))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if hasattr(self, "gain"):
            nn.init.ones_(self.gain)

    def forward(self, inputs: Tensor) -> Tensor:
        reduce_dims = (1, 2, 3)
        mean = self.weight.mean(dim=reduce_dims, keepdim=True)
        variance = self.weight.var(dim=reduce_dims, unbiased=False, keepdim=True)
        fan_in = self.weight[0].numel()
        scale = torch.rsqrt(torch.clamp(variance * fan_in, min=self.eps))
        scale = scale * self.gain.view(-1, 1, 1, 1)
        weight = (self.weight - mean) * scale
        return F.conv2d(
            inputs,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def _normalization(channels: int, groups: int) -> nn.GroupNorm:
    if channels % groups:
        raise ValueError(
            f"GroupNorm groups ({groups}) must divide channels ({channels})."
        )
    return nn.GroupNorm(groups, channels)


class WideResidualUnit(nn.Module):
    """One activation-norm-conv residual unit from the reference model."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        groups: int,
        use_skip_init: bool,
        use_skip_path: bool,
        project_skip: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip_path = use_skip_path
        self.norm1 = _normalization(in_channels, groups)
        self.conv1 = WSConv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.norm2 = _normalization(out_channels, groups)
        self.conv2 = WSConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        if use_skip_path and project_skip:
            # The JAX model applies activation + norm before its projection.
            self.skip_norm: nn.Module | None = _normalization(in_channels, groups)
            self.skip_conv: nn.Module | None = WSConv2d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )
        else:
            self.skip_norm = None
            self.skip_conv = None

        self.residual_scale = (
            nn.Parameter(torch.zeros(())) if use_skip_init else None
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.use_skip_path:
            if self.skip_conv is None:
                skip = inputs
            else:
                skip = self.skip_conv(self.skip_norm(F.relu(inputs)))

        outputs = self.conv1(self.norm1(F.relu(inputs)))
        outputs = self.conv2(self.norm2(F.relu(outputs)))
        if self.residual_scale is not None:
            outputs = outputs * self.residual_scale
        return outputs + skip if self.use_skip_path else outputs


class WideResNet(nn.Module):
    """WideResNet for 32x32 images, defaulting to DP-friendly WRN-16-4."""

    def __init__(
        self,
        num_classes: int = 10,
        depth: int = 16,
        width: int = 4,
        dropout_rate: float = 0.0,
        groups: int = 16,
        use_skip_init: bool = False,
        use_skip_paths: bool = True,
    ) -> None:
        super().__init__()
        if depth < 10 or (depth - 4) % 6 != 0:
            raise ValueError("depth must have the form 6n + 4 and be at least 10")
        if width < 1:
            raise ValueError("width must be positive")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")

        blocks_per_group = (depth - 4) // 6
        channels = (16, 16 * width, 32 * width, 64 * width)
        self.stem = WSConv2d(3, channels[0], kernel_size=3, padding=1)
        self.block1 = self._make_group(
            channels[0], channels[1], blocks_per_group, 1, groups,
            use_skip_init, use_skip_paths
        )
        self.block2 = self._make_group(
            channels[1], channels[2], blocks_per_group, 2, groups,
            use_skip_init, use_skip_paths
        )
        self.block3 = self._make_group(
            channels[2], channels[3], blocks_per_group, 2, groups,
            use_skip_init, use_skip_paths
        )
        self.final_norm = _normalization(channels[3], groups)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(channels[3], num_classes)
        nn.init.normal_(
            self.classifier.weight, mean=0.0, std=math.sqrt(1.0 / channels[3])
        )
        nn.init.zeros_(self.classifier.bias)

        self.depth = depth
        self.width = width
        self.num_classes = num_classes

    @staticmethod
    def _make_group(
        in_channels: int,
        out_channels: int,
        count: int,
        first_stride: int,
        groups: int,
        use_skip_init: bool,
        use_skip_paths: bool,
    ) -> nn.Sequential:
        units = []
        for index in range(count):
            units.append(
                WideResidualUnit(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    first_stride if index == 0 else 1,
                    groups,
                    use_skip_init,
                    use_skip_paths,
                    project_skip=index == 0,
                )
            )
        return nn.Sequential(*units)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.stem(inputs)
        outputs = self.block1(outputs)
        outputs = self.block2(outputs)
        outputs = self.block3(outputs)
        outputs = self.final_norm(F.relu(outputs))
        outputs = outputs.mean(dim=(2, 3))
        return self.classifier(self.dropout(outputs))


def wrn_16_4(num_classes: int = 10, **kwargs) -> WideResNet:
    """Build the reference model's default WRN-16-4 configuration."""
    return WideResNet(num_classes=num_classes, depth=16, width=4, **kwargs)


if __name__ == "__main__":
    model = wrn_16_4()
    sample = torch.randn(2, 3, 32, 32)
    logits = model(sample)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"output shape: {tuple(logits.shape)}")
    print(f"parameters: {parameters:,}")
