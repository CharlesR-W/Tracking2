from __future__ import annotations

import argparse
import copy
import io
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import CIFAR10, FakeData
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, ToTensor

from .models import InstrumentedVGG19


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class CriticalityConfig:
    output: str = "artifacts/criticality/seed0"
    data_root: str = "data"
    fake_data: bool = False
    train_size: int = 50000
    test_size: int = 10000
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.05
    weight_decay: float = 5e-4
    checkpoint_epochs: tuple[int, ...] = (0, 1, 2, 5, 10, 20, 40, 100)
    lr_milestones: tuple[int, ...] = (30, 60, 90)
    classifier_width: int = 512
    width_multiplier: float = 1.0
    batch_norm: bool = False
    recovery_modules: tuple[int, ...] = (0, 4, 8, 12, 18)
    recovery_sources: tuple[str, ...] = ("random", "0")
    recovery_steps: int = 200
    recovery_eval_steps: tuple[int, ...] = (0, 10, 50, 200)
    recovery_lr: float = 0.01
    amp: bool = True
    seed: int = 0
    device: str = "auto"


class TensorTransformDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform: Any) -> None:
        self.images, self.labels, self.transform = images, labels, transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return self.transform(self.images[index]), self.labels[index]


def _parquet_tensors(path: Path, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    rows = pq.read_table(path, columns=["img", "label"]).slice(0, limit).to_pylist()
    images = np.empty((len(rows), 3, 32, 32), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        image = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
        images[index] = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        labels[index] = row["label"]
    return torch.from_numpy(images), torch.from_numpy(labels)


def datasets(config: CriticalityConfig) -> tuple[Dataset, Dataset]:
    train_transform = Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), Normalize(CIFAR_MEAN, CIFAR_STD)])
    test_transform = Normalize(CIFAR_MEAN, CIFAR_STD)
    root = Path(config.data_root)
    if config.fake_data:
        base_train = FakeData(max(256, config.train_size), image_size=(3, 32, 32), num_classes=10, transform=ToTensor())
        base_test = FakeData(max(256, config.test_size), image_size=(3, 32, 32), num_classes=10, transform=ToTensor(), random_offset=10000)
        train = Subset(base_train, range(min(config.train_size, len(base_train))))
        test = Subset(base_test, range(min(config.test_size, len(base_test))))
        return train, test
    train_parquet, test_parquet = root / "cifar10-train.parquet", root / "cifar10-test.parquet"
    if train_parquet.exists() and test_parquet.exists():
        train_x, train_y = _parquet_tensors(train_parquet, config.train_size)
        test_x, test_y = _parquet_tensors(test_parquet, config.test_size)
        return TensorTransformDataset(train_x, train_y, train_transform), TensorTransformDataset(test_x, test_y, test_transform)
    train = CIFAR10(root, train=True, download=True, transform=Compose([ToTensor(), train_transform]))
    test = CIFAR10(root, train=False, download=True, transform=Compose([ToTensor(), test_transform]))
    return Subset(train, range(min(config.train_size, len(train)))), Subset(test, range(min(config.test_size, len(test))))


def make_loader(data: Dataset, config: CriticalityConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    # Keep workers at zero: this is portable to restricted local smoke-test
    # environments and CIFAR is small enough to remain GPU-fed from memory.
    return DataLoader(
        data, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
        pin_memory=torch.cuda.is_available(), generator=generator,
    )


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, data: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = correct = count = 0
    for x, y in data:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count, "error": 1 - correct / count}


def transplant_module(target: InstrumentedVGG19, source: InstrumentedVGG19, module_index: int) -> None:
    target.intervention_modules()[module_index][1].load_state_dict(
        source.intervention_modules()[module_index][1].state_dict()
    )


def randomized_source(config: CriticalityConfig, module_index: int, device: torch.device) -> InstrumentedVGG19:
    state = torch.random.get_rng_state()
    torch.manual_seed(config.seed + 10000 + module_index)
    source = InstrumentedVGG19(
        classifier_width=config.classifier_width, width_multiplier=config.width_multiplier,
        batch_norm=config.batch_norm,
    ).to(device)
    torch.random.set_rng_state(state)
    return source


def recovery_curve(
    hybrid: InstrumentedVGG19,
    module_index: int,
    train_data: DataLoader,
    test_data: DataLoader,
    config: CriticalityConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    for parameter in hybrid.parameters():
        parameter.requires_grad_(False)
    suffix = hybrid.parameters_after(module_index)
    for parameter in suffix:
        parameter.requires_grad_(True)
    if not suffix:
        return [{"step": 0, **evaluate(hybrid, test_data, device)}]
    optimizer = torch.optim.SGD(suffix, lr=config.recovery_lr, momentum=0.9)
    use_amp = device.type == "cuda" and config.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    wanted = set(config.recovery_eval_steps) | {0, config.recovery_steps}
    result: list[dict[str, float]] = []
    iterator = iter(train_data)
    for step in range(config.recovery_steps + 1):
        if step in wanted:
            result.append({"step": step, **evaluate(hybrid, test_data, device)})
        if step == config.recovery_steps:
            break
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_data)
            x, y = next(iterator)
        hybrid.train()
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss = F.cross_entropy(hybrid(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return result


def run(config: CriticalityConfig) -> Path:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    train_set, test_set = datasets(config)
    train_data = make_loader(train_set, config, True)
    test_data = make_loader(test_set, config, False)
    model = InstrumentedVGG19(
        classifier_width=config.classifier_width, width_multiplier=config.width_multiplier,
        batch_norm=config.batch_norm,
    ).to(device)
    checkpoints = sorted(set(epoch for epoch in config.checkpoint_epochs if 0 <= epoch <= config.epochs) | {0, config.epochs})
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(config.lr_milestones), gamma=0.2)
    use_amp = device.type == "cuda" and config.amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    started = time.time()
    training: list[dict[str, Any]] = []
    for epoch in range(config.epochs + 1):
        if epoch in checkpoints:
            snapshots[epoch] = cpu_state(model)
            torch.save(snapshots[epoch], output / f"checkpoint_epoch{epoch}.pt")
            training.append({"epoch": epoch, **evaluate(model, test_data, device)})
            print(f"[checkpoint] epoch={epoch} accuracy={training[-1]['accuracy']:.4f}", flush=True)
        if epoch == config.epochs:
            break
        model.train()
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    final = InstrumentedVGG19(
        classifier_width=config.classifier_width, width_multiplier=config.width_multiplier,
        batch_norm=config.batch_norm,
    ).to(device)
    final.load_state_dict(snapshots[config.epochs])
    baseline = evaluate(final, test_data, device)
    module_names = [name for name, _ in final.intervention_modules()]
    source_models: dict[str, InstrumentedVGG19] = {}
    for epoch, snapshot in snapshots.items():
        source = InstrumentedVGG19(
            classifier_width=config.classifier_width, width_multiplier=config.width_multiplier,
            batch_norm=config.batch_norm,
        ).to(device)
        source.load_state_dict(snapshot)
        source_models[str(epoch)] = source
    interventions: list[dict[str, Any]] = []
    recoveries: list[dict[str, Any]] = []
    for module_index, module_name in enumerate(module_names):
        sources = {**source_models, "random": randomized_source(config, module_index, device)}
        for source_name, source in sources.items():
            hybrid = copy.deepcopy(final)
            transplant_module(hybrid, source, module_index)
            metrics = evaluate(hybrid, test_data, device)
            interventions.append({
                "module_index": module_index, "module": module_name, "source": source_name,
                **metrics, "delta_loss": metrics["loss"] - baseline["loss"],
                "delta_error": metrics["error"] - baseline["error"],
            })
            if module_index in config.recovery_modules and source_name in config.recovery_sources:
                for point in recovery_curve(hybrid, module_index, train_data, test_data, config, device):
                    recoveries.append({"module_index": module_index, "module": module_name, "source": source_name, **point})
    artifact = {
        "schema_version": 1, "experiment": "critical_module_tracking", "config": asdict(config),
        "device": str(device), "module_names": module_names, "baseline": baseline,
        "training": training, "interventions": interventions, "recoveries": recoveries,
        "runtime_seconds": time.time() - started,
    }
    path = output / "criticality.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] artifact={path} runtime_seconds={artifact['runtime_seconds']:.1f}", flush=True)
    return path


def parse_args() -> CriticalityConfig:
    parser = argparse.ArgumentParser(description="VGG-19 critical-module checkpoint transplant experiment")
    parser.add_argument("--output", default="artifacts/criticality/seed0")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 40, 100])
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[30, 60, 90])
    parser.add_argument("--classifier-width", type=int, default=512)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--recovery-modules", type=int, nargs="+", default=[0, 4, 8, 12, 18])
    parser.add_argument("--recovery-sources", nargs="+", default=["random", "0"])
    parser.add_argument("--recovery-steps", type=int, default=200)
    parser.add_argument("--recovery-eval-steps", type=int, nargs="+", default=[0, 10, 50, 200])
    parser.add_argument("--recovery-lr", type=float, default=0.01)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return CriticalityConfig(
        output=args.output, data_root=args.data_root, fake_data=args.fake_data,
        train_size=args.train_size, test_size=args.test_size, epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        checkpoint_epochs=tuple(args.checkpoints), lr_milestones=tuple(args.lr_milestones),
        classifier_width=args.classifier_width, width_multiplier=args.width_multiplier,
        batch_norm=args.batch_norm,
        recovery_modules=tuple(args.recovery_modules),
        recovery_sources=tuple(args.recovery_sources), recovery_steps=args.recovery_steps,
        recovery_eval_steps=tuple(args.recovery_eval_steps), recovery_lr=args.recovery_lr,
        amp=not args.no_amp, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
