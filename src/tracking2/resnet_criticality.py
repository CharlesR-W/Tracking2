from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .cnn_checkpoints import sha256
from .criticality import (
    CriticalityConfig,
    cpu_state,
    datasets,
    evaluate,
    make_loader,
    ordered_dataset_fingerprints,
    resolved_data_backend,
)
from .models import InstrumentedResNet18V2
from .provenance import runtime_provenance


RESNET_ARCHITECTURE_NAME = "InstrumentedResNet18V2"
SMOKE_STATUS = "MOCKUP / PIPELINE SMOKE TEST"


@dataclass
class ResNetCriticalityConfig:
    output: str = "artifacts/resnet_criticality/seed0"
    data_root: str = "data"
    data_backend: str = "torchvision"
    fake_data: bool = False
    train_size: int = 50000
    test_size: int = 10000
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 0.05
    weight_decay: float = 0.0
    checkpoint_epochs: tuple[int, ...] = (0, 1, 5, 20, 100)
    lr_milestones: tuple[int, ...] = (30, 60, 90)
    width: int = 64
    seed: int = 0
    device: str = "auto"
    amp: bool = True
    measure_criticality: bool = True


def transplant_module(
    target: InstrumentedResNet18V2, source: InstrumentedResNet18V2, module_index: int
) -> None:
    target.intervention_modules()[module_index][1].load_state_dict(
        source.intervention_modules()[module_index][1].state_dict()
    )


def model_from_state(config: ResNetCriticalityConfig, state: dict, device: torch.device) -> InstrumentedResNet18V2:
    model = InstrumentedResNet18V2(width=config.width).to(device)
    model.load_state_dict(state)
    return model


def run(config: ResNetCriticalityConfig) -> Path:
    torch.manual_seed(config.seed)
    if not config.fake_data and config.data_backend != "torchvision":
        raise ValueError(
            "Measured ResNet training requires data_backend='torchvision'"
        )
    device = torch.device(config.device if config.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    data_config = CriticalityConfig(
        data_root=config.data_root, data_backend=config.data_backend,
        train_size=config.train_size, test_size=config.test_size,
        batch_size=config.batch_size, seed=config.seed, fake_data=config.fake_data,
    )
    train_set, test_set = datasets(data_config)
    dataset_identity = ordered_dataset_fingerprints(
        train_set,
        test_set,
        backend=resolved_data_backend(data_config),
    )
    train_data = make_loader(train_set, data_config, True)
    test_data = make_loader(test_set, data_config, False)
    model = InstrumentedResNet18V2(width=config.width).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(config.lr_milestones), gamma=0.2
    )
    use_amp = device.type == "cuda" and config.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    checkpoints = sorted(set(config.checkpoint_epochs) | {0, config.epochs})
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    training: list[dict] = []
    checkpoint_manifest: list[dict] = []
    started = time.time()
    for epoch in range(config.epochs + 1):
        if epoch in checkpoints:
            snapshots[epoch] = cpu_state(model)
            checkpoint_path = output / f"checkpoint_epoch{epoch}.pt"
            torch.save(snapshots[epoch], checkpoint_path)
            metrics = evaluate(model, test_data, device)
            checkpoint_record = {
                "epoch": epoch,
                "path": checkpoint_path.name,
                "sha256": sha256(checkpoint_path),
            }
            checkpoint_manifest.append(checkpoint_record)
            training.append({**checkpoint_record, **metrics})
            print(f"[checkpoint] epoch={epoch} accuracy={training[-1]['accuracy']:.4f}", flush=True)
        if epoch == config.epochs:
            break
        model.train()
        for x, y in train_data:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

    final = model_from_state(config, snapshots[config.epochs], device)
    baseline = evaluate(final, test_data, device)
    module_names = [name for name, _ in final.intervention_modules()]
    interventions: list[dict] = []
    if config.measure_criticality:
        sources = {
            str(epoch): model_from_state(config, snapshot, device)
            for epoch, snapshot in snapshots.items()
        }
        for module_index, module_name in enumerate(module_names):
            torch.manual_seed(config.seed + 10000 + module_index)
            random_source = InstrumentedResNet18V2(width=config.width).to(device)
            for source_name, source in {"random": random_source, **sources}.items():
                hybrid = copy.deepcopy(final)
                transplant_module(hybrid, source, module_index)
                metrics = evaluate(hybrid, test_data, device)
                interventions.append({
                    "module_index": module_index, "module": module_name, "source": source_name,
                    **metrics, "delta_loss": metrics["loss"] - baseline["loss"],
                    "delta_error": metrics["error"] - baseline["error"],
                })
    artifact = {
        "schema_version": 1,
        "experiment": ("resnet18_block_criticality" if config.measure_criticality
                       else "resnet18_training_checkpoints"),
        "status": SMOKE_STATUS if config.fake_data else "MEASURED",
        "architecture": {
            "name": RESNET_ARCHITECTURE_NAME,
            "width": config.width,
            "block_names": list(model.block_names),
        },
        "config": asdict(config), "device": str(device),
        "provenance": runtime_provenance(),
        "dataset": dataset_identity,
        "checkpoints": checkpoint_manifest,
        "module_names": module_names, "baseline": baseline, "training": training,
        "interventions": interventions, "runtime_seconds": time.time() - started,
    }
    path = output / ("resnet_criticality.json" if config.measure_criticality else "resnet_training.json")
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] {path}", flush=True)
    return path


def parse_args() -> ResNetCriticalityConfig:
    parser = argparse.ArgumentParser(description="Paper-matched CIFAR ResNet-18 block criticality")
    parser.add_argument("--output", default=ResNetCriticalityConfig.output)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--data-backend",
        choices=("auto", "torchvision", "parquet"),
        default="torchvision",
    )
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 1, 5, 20, 100])
    parser.add_argument("--lr-milestones", type=int, nargs="+", default=[30, 60, 90])
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--training-only", action="store_true",
                        help="train and save checkpoints without block-transplant interventions")
    args = parser.parse_args()
    return ResNetCriticalityConfig(
        output=args.output, data_root=args.data_root,
        data_backend=args.data_backend, fake_data=args.fake_data,
        train_size=args.train_size,
        test_size=args.test_size, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        checkpoint_epochs=tuple(args.checkpoints), lr_milestones=tuple(args.lr_milestones),
        width=args.width, seed=args.seed, device=args.device, amp=not args.no_amp,
        measure_criticality=not args.training_only,
    )


if __name__ == "__main__":
    run(parse_args())
