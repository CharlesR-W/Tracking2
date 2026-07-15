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


class VGGConvModule(nn.Module):
    """One atomic VGG intervention: convolution plus optional normalization."""

    def __init__(self, in_channels: int, out_channels: int, batch_norm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.conv(x)), inplace=False)


class InstrumentedVGG19(nn.Module):
    """CIFAR-sized VGG-19 with 19 addressable parametric modules.

    Pooling and ReLU operations are deliberately not interventions: a transplant
    replaces one learned affine map while leaving the computation graph fixed.
    """

    _CONVS_PER_STAGE = (2, 2, 4, 4, 4)
    _CHANNELS = (64, 128, 256, 512, 512)

    def __init__(
        self, num_classes: int = 10, classifier_width: int = 512,
        width_multiplier: float = 1.0, batch_norm: bool = False,
    ) -> None:
        super().__init__()
        if not 0 < width_multiplier <= 1:
            raise ValueError("width_multiplier must be in (0, 1]")
        stages: list[nn.Module] = []
        in_channels = 3
        scaled_channels = tuple(max(4, int(channels * width_multiplier)) for channels in self._CHANNELS)
        for count, channels in zip(self._CONVS_PER_STAGE, scaled_channels):
            layers: list[nn.Module] = []
            for _ in range(count):
                layers.append(VGGConvModule(in_channels, channels, batch_norm))
                in_channels = channels
            layers.append(nn.MaxPool2d(2))
            stages.append(nn.Sequential(*layers))
        self.stages = nn.ModuleList(stages)
        self.fc1 = nn.Linear(scaled_channels[-1], classifier_width)
        self.fc2 = nn.Linear(classifier_width, classifier_width)
        self.final_linear = nn.Linear(classifier_width, num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for stage in self.stages:
            x = stage(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.relu(self.fc1(x), inplace=False)
        x = F.relu(self.fc2(x), inplace=False)
        return self.final_linear(x)

    def intervention_modules(self) -> list[tuple[str, nn.Module]]:
        result: list[tuple[str, nn.Module]] = []
        for stage_index, stage in enumerate(self.stages, start=1):
            block_index = 0
            for module in stage:
                if isinstance(module, VGGConvModule):
                    block_index += 1
                    result.append((f"stage{stage_index}.conv{block_index}", module))
        result.extend([
            ("fc1", self.fc1),
            ("fc2", self.fc2),
            ("final_linear", self.final_linear),
        ])
        return result

    def parameters_after(self, module_index: int) -> list[nn.Parameter]:
        modules = self.intervention_modules()
        return [parameter for _, module in modules[module_index + 1 :] for parameter in module.parameters()]

    def _ordered_operations(self) -> list[tuple[int | None, nn.Module]]:
        """Forward operations annotated by parametric-module index."""
        operations: list[tuple[int | None, nn.Module]] = []
        module_index = 0
        for stage in self.stages:
            for operation in stage:
                if isinstance(operation, VGGConvModule):
                    operations.append((module_index, operation))
                    module_index += 1
                else:
                    operations.append((None, operation))
        return operations

    def encode_to_module(self, x: torch.Tensor, module_index: int) -> torch.Tensor:
        if not 0 <= module_index < 16:
            raise ValueError("activation cuts currently select one of the 16 convolutional modules")
        for index, operation in self._ordered_operations():
            x = operation(x)
            if index == module_index:
                return x
        raise AssertionError("module index not reached")

    def forward_from_module(self, representation: torch.Tensor, module_index: int) -> torch.Tensor:
        if not 0 <= module_index < 16:
            raise ValueError("activation cuts currently select one of the 16 convolutional modules")
        x = representation
        passed_cut = False
        for index, operation in self._ordered_operations():
            if passed_cut:
                x = operation(x)
            elif index == module_index:
                passed_cut = True
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        x = F.relu(self.fc1(x), inplace=False)
        x = F.relu(self.fc2(x), inplace=False)
        return self.final_linear(x)

    def parameters_through(self, module_index: int) -> list[nn.Parameter]:
        modules = self.intervention_modules()
        return [parameter for _, module in modules[: module_index + 1] for parameter in module.parameters()]
