from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    return min(8, channels)


class PlainBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(out_channels), out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x), inplace=False)


@dataclass(frozen=True)
class ArchitectureSpec:
    kind: str
    widths: tuple[int, ...] = (32, 64, 128, 128)


class SplitConvNet(nn.Module):
    """Depth-matched plain/residual CNN with explicit representation cuts."""

    def __init__(self, spec: ArchitectureSpec, num_classes: int = 10) -> None:
        super().__init__()
        if spec.kind not in {"plain", "residual"}:
            raise ValueError(f"Unknown architecture: {spec.kind}")
        block_cls = PlainBlock if spec.kind == "plain" else ResidualBlock
        blocks: list[nn.Module] = []
        in_channels = 3
        for index, width in enumerate(spec.widths):
            stride = 1 if index == 0 else 2
            blocks.append(block_cls(in_channels, width, stride))
            in_channels = width
        self.blocks = nn.ModuleList(blocks)
        self.classifier = nn.Linear(spec.widths[-1], num_classes)
        self.spec = spec

    @property
    def num_cuts(self) -> int:
        return len(self.blocks) + 1

    def encode_to(self, x: torch.Tensor, cut: int) -> torch.Tensor:
        if not 0 <= cut <= len(self.blocks):
            raise ValueError(f"cut must be in [0, {len(self.blocks)}]")
        for block in self.blocks[:cut]:
            x = block(x)
        return x

    def forward_from(self, representation: torch.Tensor, cut: int) -> torch.Tensor:
        x = representation
        for block in self.blocks[cut:]:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_from(self.encode_to(x, len(self.blocks)), len(self.blocks))

    def prefix_parameters(self, cut: int) -> list[nn.Parameter]:
        return [p for block in self.blocks[:cut] for p in block.parameters()]

    def suffix_parameters(self, cut: int) -> list[nn.Parameter]:
        params = [p for block in self.blocks[cut:] for p in block.parameters()]
        return params + list(self.classifier.parameters())

