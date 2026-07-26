from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .experiment import choose_device, loader, prepare_true_datasets, seed_everything, train_one_epoch
from .models import ArchitectureSpec, SplitConvNet
from .representation_surrogates import (
    fit_representation_surrogate,
    moment_diagnostics,
    sample_representation_surrogate,
)


@dataclass
class SuffixStatisticsConfig:
    output: str = "artifacts/suffix_statistics/t5_cut3_seed0"
    data_root: str = "data"
    fake_data: bool = False
    train_size: int = 10000
    test_size: int = 2000
    checkpoint_epoch: int = 5
    checkpoint_batches: int | None = None
    cut: int = 3
    widths: tuple[int, ...] = (32, 64, 128, 128)
    batch_size: int = 256
    learning_rate: float = 0.05
    weight_decay: float = 5e-4
    pca_fit_size: int = 5000
    pca_components: int = 128
    relax_epochs: int = 10
    relax_batch_zoom: bool = False
    relax_learning_rate: float = 0.01
    seed: int = 0
    device: str = "auto"


@torch.no_grad()
def encode_dataset(
    model: SplitConvNet, data: DataLoader, cut: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    representations, labels = [], []
    for x, y in data:
        representations.append(model.encode_to(x.to(device), cut).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(representations), np.concatenate(labels)


@torch.no_grad()
def evaluate_suffix(
    model: SplitConvNet, cut: int, data: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = total_correct = total = 0
    for representation, labels in data:
        representation, labels = representation.to(device), labels.to(device)
        logits = model.forward_from(representation, cut)
        total_loss += float(F.cross_entropy(logits, labels, reduction="sum"))
        total_correct += int((logits.argmax(-1) == labels).sum())
        total += len(labels)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def representation_loader(
    representations: np.ndarray, labels: np.ndarray, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(representations), torch.from_numpy(labels.astype(np.int64)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def run(config: SuffixStatisticsConfig) -> Path:
    seed_everything(config.seed)
    if not 1 <= config.cut <= len(config.widths):
        raise ValueError("cut must select a nonempty prefix")
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    started = time.time()

    # Reuse the real-CIFAR loading path, but construct activation-space surrogates below.
    from .experiment import Config
    data_config = Config(
        data_root=config.data_root, fake_data=config.fake_data,
        train_size=config.train_size, test_size=config.test_size,
        batch_size=config.batch_size, seed=config.seed, device=config.device,
    )
    true_datasets = prepare_true_datasets(data_config)
    train_data = loader(true_datasets["train"], data_config, True)
    test_data = loader(true_datasets["test"], data_config, False)

    model = SplitConvNet(ArchitectureSpec("residual", config.widths)).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.checkpoint_epoch))
    for epoch in range(config.checkpoint_epoch):
        print(f"[prefix training] epoch={epoch + 1}/{config.checkpoint_epoch}", flush=True)
        train_one_epoch(model, train_data, optimizer, device)
        scheduler.step()
    if config.checkpoint_batches is not None:
        if config.checkpoint_epoch != 0:
            raise ValueError("checkpoint_batches is an epoch-0 zoom and requires checkpoint_epoch=0")
        if not 0 <= config.checkpoint_batches <= len(train_data):
            raise ValueError(f"checkpoint_batches must be between 0 and {len(train_data)}")
        model.train()
        for batch_index, (images, labels) in enumerate(train_data):
            if batch_index >= config.checkpoint_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
        print(
            f"[prefix training] epoch=0 batches={config.checkpoint_batches}/{len(train_data)}",
            flush=True,
        )
    torch.save(model.state_dict(), output / "checkpoint.pt")

    train_rep, train_labels = encode_dataset(model, loader(true_datasets["train"], data_config, False), config.cut, device)
    test_rep, test_labels = encode_dataset(model, test_data, config.cut, device)
    fit_count = min(config.pca_fit_size, len(train_rep))
    surrogate = fit_representation_surrogate(
        train_rep[:fit_count], train_labels[:fit_count],
        min(config.pca_components, fit_count - 1, train_rep[0].size), config.seed,
    )
    train_sets = {"true": train_rep}
    test_sets = {"true": test_rep}
    diagnostics: dict[str, dict[str, float]] = {}
    for offset, kind in enumerate(("mean", "gaussian"), start=1):
        train_sets[kind] = sample_representation_surrogate(surrogate, train_labels, kind, config.seed + offset)
        test_sets[kind] = sample_representation_surrogate(surrogate, test_labels, kind, config.seed + 100 + offset)
        diagnostics[kind] = moment_diagnostics(test_rep, test_sets[kind], test_labels, surrogate)

    eval_loaders = {
        name: representation_loader(values, test_labels, config.batch_size, config.seed, False)
        for name, values in test_sets.items()
    }
    records = []
    for train_distribution in ("mean", "gaussian", "true"):
        candidate = copy.deepcopy(model)
        for parameter in candidate.prefix_parameters(config.cut):
            parameter.requires_grad_(False)
        suffix_optimizer = torch.optim.SGD(
            candidate.suffix_parameters(config.cut), lr=config.relax_learning_rate, momentum=0.9
        )
        train_loader = representation_loader(
            train_sets[train_distribution], train_labels, config.batch_size,
            config.seed + 1000, True,
        )
        batches_per_relax_epoch = len(train_loader)
        batch_checkpoints = {
            epoch * batches_per_relax_epoch for epoch in range(config.relax_epochs + 1)
        }
        if config.relax_batch_zoom:
            batch_checkpoints.update(
                batch for batch in (0, 1, 2, 5, 10, 20, 50, 100, batches_per_relax_epoch)
                if batch <= batches_per_relax_epoch
            )

        def evaluate_checkpoint(relax_batch: int) -> None:
            checkpoint_loaders = (
                eval_loaders if relax_batch % batches_per_relax_epoch == 0
                else {"true": eval_loaders["true"]}
            )
            for eval_distribution, eval_loader in checkpoint_loaders.items():
                metrics = evaluate_suffix(candidate, config.cut, eval_loader, device)
                records.append({
                    "train_distribution": train_distribution,
                    "eval_distribution": eval_distribution,
                    "relax_epoch": relax_batch / batches_per_relax_epoch,
                    "relax_batch": relax_batch,
                    **metrics,
                })

        evaluate_checkpoint(0)
        relax_batch = 0
        for _ in range(config.relax_epochs):
            candidate.train()
            for representation, labels in train_loader:
                representation, labels = representation.to(device), labels.to(device)
                suffix_optimizer.zero_grad(set_to_none=True)
                loss = F.cross_entropy(candidate.forward_from(representation, config.cut), labels)
                loss.backward()
                suffix_optimizer.step()
                relax_batch += 1
                if relax_batch in batch_checkpoints:
                    evaluate_checkpoint(relax_batch)
                    candidate.train()

    artifact = {
        "status": "MOCKUP / PIPELINE SMOKE TEST" if config.fake_data else "MEASURED",
        "config": asdict(config),
        "device": str(device),
        "representation_shape": list(train_rep.shape[1:]),
        "explained_variance_fraction": float(
            np.var((train_rep[:fit_count].reshape(fit_count, -1) - surrogate.pca_mean) @ surrogate.components.T, axis=0).sum()
            / np.var(train_rep[:fit_count].reshape(fit_count, -1), axis=0).sum()
        ),
        "moment_diagnostics": diagnostics,
        "records": records,
        "runtime_seconds": time.time() - started,
    }
    artifact_path = output / "suffix_statistics.json"
    artifact_path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] {artifact_path}", flush=True)
    return artifact_path


def parse_args() -> SuffixStatisticsConfig:
    parser = argparse.ArgumentParser(description="Frozen-prefix activation-statistics suffix pilot")
    parser.add_argument("--output", default=SuffixStatisticsConfig.output)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--checkpoint-epoch", type=int, default=5)
    parser.add_argument(
        "--checkpoint-batches", type=int,
        help="Stop within epoch 0 after this many optimizer batches (0 is exact random initialization).",
    )
    parser.add_argument("--cut", type=int, default=3)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 128])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pca-fit-size", type=int, default=5000)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--relax-epochs", type=int, default=10)
    parser.add_argument(
        "--relax-batch-zoom", action="store_true",
        help="Also evaluate at log-spaced optimizer batches within the first suffix-relaxation epoch.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return SuffixStatisticsConfig(
        output=args.output, data_root=args.data_root, fake_data=args.fake_data,
        train_size=args.train_size, test_size=args.test_size,
        checkpoint_epoch=args.checkpoint_epoch, checkpoint_batches=args.checkpoint_batches,
        cut=args.cut, widths=tuple(args.widths),
        batch_size=args.batch_size, pca_fit_size=args.pca_fit_size,
        pca_components=args.pca_components, relax_epochs=args.relax_epochs,
        relax_batch_zoom=args.relax_batch_zoom,
        seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
