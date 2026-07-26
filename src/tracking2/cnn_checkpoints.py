from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .experiment import (
    Config,
    choose_device,
    evaluate,
    loader,
    ordered_dataset_fingerprints,
    prepare_true_datasets,
    resolve_data_backend,
    seed_everything,
    train_one_epoch,
)
from .models import ArchitectureSpec, SplitConvNet
from .provenance import runtime_provenance


@dataclass
class CNNCheckpointConfig:
    """One uninterrupted CIFAR-10 run used by the post's checkpoint prefixes."""

    output: str = "artifacts/lw_post/cnn_seed0"
    data_root: str = "data"
    data_backend: str = "torchvision"
    fake_data: bool = False
    train_size: int = 50000
    test_size: int = 10000
    epochs: int = 30
    checkpoint_epochs: tuple[int, ...] = (0, 1, 5, 10, 20, 30)
    widths: tuple[int, ...] = (32, 64, 128, 128)
    batch_size: int = 256
    learning_rate: float = 0.05
    weight_decay: float = 5e-4
    seed: int = 0
    device: str = "auto"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(config: CNNCheckpointConfig) -> Path:
    seed_everything(config.seed)
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    checkpoints = tuple(sorted(set(config.checkpoint_epochs)))
    if not checkpoints or checkpoints[0] < 0 or checkpoints[-1] > config.epochs:
        raise ValueError("checkpoint_epochs must lie between 0 and epochs")

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    data_config = Config(
        data_root=config.data_root,
        data_backend=config.data_backend,
        fake_data=config.fake_data,
        train_size=config.train_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        device=config.device,
    )
    datasets = prepare_true_datasets(data_config)
    dataset_fingerprints = ordered_dataset_fingerprints(
        datasets, resolve_data_backend(data_config)
    )
    train_data = loader(datasets["train"], data_config, True)
    test_data = loader(datasets["test"], data_config, False)

    model = SplitConvNet(
        ArchitectureSpec("residual", config.widths)
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    started = time.time()
    records: list[dict[str, object]] = []

    def save_checkpoint(epoch: int) -> None:
        path = output / f"checkpoint_epoch{epoch}.pt"
        torch.save(model.state_dict(), path)
        records.append(
            {
                "epoch": epoch,
                "path": path.name,
                "sha256": sha256(path),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                **evaluate(model, test_data, device),
            }
        )
        print(
            f"[checkpoint] epoch={epoch} accuracy={records[-1]['accuracy']:.4f}",
            flush=True,
        )

    if 0 in checkpoints:
        save_checkpoint(0)
    for epoch in range(1, config.epochs + 1):
        print(f"[train] epoch={epoch}/{config.epochs}", flush=True)
        train_one_epoch(model, train_data, optimizer, device)
        scheduler.step()
        if epoch in checkpoints:
            save_checkpoint(epoch)

    artifact = {
        "schema_version": 1,
        "experiment": "lw_post_cnn_checkpoint_trajectory",
        "status": "MOCKUP / PIPELINE SMOKE TEST" if config.fake_data else "MEASURED",
        "architecture": "four-block residual CNN with GroupNorm",
        "trajectory_semantics": (
            "Every checkpoint is from this one uninterrupted end-to-end "
            "training run."
        ),
        "config": asdict(config),
        "dataset": dataset_fingerprints,
        "device": str(device),
        "provenance": runtime_provenance(),
        "checkpoints": records,
        "runtime_seconds": time.time() - started,
    }
    path = output / "training.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] {path}", flush=True)
    return path


def parse_args() -> CNNCheckpointConfig:
    parser = argparse.ArgumentParser(
        description="Train one uninterrupted residual-CNN trajectory for the LW post."
    )
    parser.add_argument("--output", default=CNNCheckpointConfig.output)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--data-backend",
        choices=("torchvision", "parquet"),
        default=CNNCheckpointConfig.data_backend,
    )
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=list(CNNCheckpointConfig.checkpoint_epochs),
    )
    parser.add_argument(
        "--widths", type=int, nargs="+", default=list(CNNCheckpointConfig.widths)
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return CNNCheckpointConfig(
        output=args.output,
        data_root=args.data_root,
        data_backend=args.data_backend,
        fake_data=args.fake_data,
        train_size=args.train_size,
        test_size=args.test_size,
        epochs=args.epochs,
        checkpoint_epochs=tuple(args.checkpoints),
        widths=tuple(args.widths),
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
