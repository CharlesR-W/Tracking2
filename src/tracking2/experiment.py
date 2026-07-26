from __future__ import annotations

import argparse
import hashlib
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
from torchvision.transforms import ToTensor

from .measurements import (
    cross_representation_loss,
    empirical_fisher_schur_trace,
    refit_suffix,
    relative_representation_drift,
    suffix_update_decomposition,
)
from .models import ArchitectureSpec, SplitConvNet
from .surrogates import ArrayDataset, fit_pca_surrogate, sample_surrogate


@dataclass
class Config:
    output: str = "artifacts/pilot"
    data_root: str = "data"
    data_backend: str = "auto"
    fake_data: bool = False
    train_size: int = 50000
    test_size: int = 10000
    pca_fit_size: int = 20000
    pca_components: int = 128
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 0.05
    weight_decay: float = 5e-4
    widths: tuple[int, ...] = (32, 64, 128, 128)
    architectures: tuple[str, ...] = ("residual",)
    train_distributions: tuple[str, ...] = ("mean", "covariance", "true")
    checkpoint_epochs: tuple[int, ...] = (0, 1, 2, 5, 10, 20)
    refit_steps: int = 100
    refit_batches: int = 8
    refit_lr: float = 0.02
    fisher_samples: int = 16
    seed: int = 0
    device: str = "auto"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataset_arrays(dataset: Dataset, limit: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(limit, len(dataset))
    images = np.empty((count, 3, 32, 32), dtype=np.float32)
    labels = np.empty(count, dtype=np.int64)
    for index in range(count):
        image, label = dataset[index]
        images[index] = np.asarray(image, dtype=np.float32)
        labels[index] = int(label)
    return images, labels


def parquet_arrays(path: Path, limit: int) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["img", "label"]).slice(0, limit)
    rows = table.to_pylist()
    images = np.empty((len(rows), 3, 32, 32), dtype=np.float32)
    labels = np.empty(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        image = Image.open(io.BytesIO(row["img"]["bytes"])).convert("RGB")
        images[index] = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        labels[index] = row["label"]
    return images, labels


def resolve_data_backend(config: Config) -> str:
    if config.fake_data:
        return "fake_data"
    if config.data_backend not in {"auto", "torchvision", "parquet"}:
        raise ValueError("data_backend must be auto, torchvision, or parquet")
    if config.data_backend != "auto":
        return config.data_backend
    root = Path(config.data_root)
    if (
        (root / "cifar10-train.parquet").exists()
        and (root / "cifar10-test.parquet").exists()
    ):
        return "parquet"
    return "torchvision"


def prepare_true_datasets(config: Config) -> dict[str, Dataset]:
    backend = resolve_data_backend(config)
    if backend == "fake_data":
        train_base = FakeData(max(config.train_size, 200), image_size=(3, 32, 32), num_classes=10, transform=ToTensor(), random_offset=0)
        test_base = FakeData(max(config.test_size, 100), image_size=(3, 32, 32), num_classes=10, transform=ToTensor(), random_offset=10000)
        train_images, train_labels = dataset_arrays(train_base, config.train_size)
        test_images, test_labels = dataset_arrays(test_base, config.test_size)
    elif backend == "parquet":
        train_path = Path(config.data_root) / "cifar10-train.parquet"
        test_path = Path(config.data_root) / "cifar10-test.parquet"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(
                "The parquet backend requires cifar10-train.parquet and "
                "cifar10-test.parquet under data_root"
            )
        train_images, train_labels = parquet_arrays(
            train_path, config.train_size
        )
        test_images, test_labels = parquet_arrays(test_path, config.test_size)
    else:
        train_base = CIFAR10(config.data_root, train=True, download=True, transform=ToTensor())
        test_base = CIFAR10(config.data_root, train=False, download=True, transform=ToTensor())
        train_images, train_labels = dataset_arrays(train_base, config.train_size)
        test_images, test_labels = dataset_arrays(test_base, config.test_size)
    return {"train": ArrayDataset(train_images, train_labels), "test": ArrayDataset(test_images, test_labels)}


def ordered_dataset_fingerprints(
    datasets: dict[str, Dataset], backend: str
) -> dict[str, object]:
    """Hash the exact ordered image and label tensors consumed by a run."""

    result: dict[str, object] = {"backend": backend, "splits": {}}
    splits: dict[str, object] = {}
    for split in ("train", "test"):
        dataset = datasets[split]
        if not isinstance(dataset, ArrayDataset):
            raise TypeError("fingerprinting expects materialized ArrayDataset splits")
        images = np.ascontiguousarray(dataset.images.numpy())
        labels = np.ascontiguousarray(dataset.labels.numpy())

        def digest(array: np.ndarray) -> str:
            value = hashlib.sha256()
            value.update(str(array.dtype).encode())
            value.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            value.update(memoryview(array).cast("B"))
            return value.hexdigest()

        splits[split] = {
            "count": len(dataset),
            "ordered_images_sha256": digest(images),
            "ordered_labels_sha256": digest(labels),
        }
    result["splits"] = splits
    return result


def prepare_datasets(config: Config) -> dict[str, dict[str, Dataset]]:
    true_datasets = prepare_true_datasets(config)
    train_images = true_datasets["train"].images.numpy()
    train_labels = true_datasets["train"].labels.numpy()
    test_images = true_datasets["test"].images.numpy()
    test_labels = true_datasets["test"].labels.numpy()
    fit_count = min(config.pca_fit_size, len(train_images))
    surrogate = fit_pca_surrogate(
        train_images[:fit_count], train_labels[:fit_count],
        min(config.pca_components, fit_count - 1, train_images.shape[1] * 32 * 32), config.seed,
    )
    result: dict[str, dict[str, Dataset]] = {
        "true": true_datasets
    }
    for offset, kind in enumerate(("mean", "covariance"), start=1):
        result[kind] = {
            "train": ArrayDataset(sample_surrogate(surrogate, train_labels, kind, config.seed + offset), train_labels),
            "test": ArrayDataset(sample_surrogate(surrogate, test_labels, kind, config.seed + 100 + offset), test_labels),
        }
    return result


def loader(dataset: Dataset, config: Config, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0, generator=generator)


@torch.no_grad()
def evaluate(model: SplitConvNet, data: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = total_correct = total = 0
    for x, y in data:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(F.cross_entropy(logits, y, reduction="sum"))
        total_correct += int((logits.argmax(-1) == y).sum())
        total += len(y)
    return {"loss": total_loss / total, "accuracy": total_correct / total}


def train_one_epoch(model: SplitConvNet, data: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    model.train()
    for x, y in data:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()


def cpu_state(model: SplitConvNet) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def take_batches(data: DataLoader, count: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for x, y in data:
        batches.append((x.to(device), y.to(device)))
        if len(batches) >= count:
            break
    return batches


def run_tracking(
    snapshots: dict[int, dict[str, torch.Tensor]],
    spec: ArchitectureSpec,
    datasets: dict[str, dict[str, Dataset]],
    config: Config,
    device: torch.device,
) -> list[dict[str, Any]]:
    train_batches = take_batches(loader(datasets["true"]["train"], config, True), config.refit_batches, device)
    eval_x, eval_y = next(iter(loader(datasets["true"]["test"], config, False)))
    eval_batch = (eval_x.to(device), eval_y.to(device))
    fisher_batch = (eval_batch[0][: config.fisher_samples], eval_batch[1][: config.fisher_samples])
    records: list[dict[str, Any]] = []
    previous_refits: dict[int, SplitConvNet] = {}
    previous_refit_losses: dict[int, float] = {}
    previous_model: SplitConvNet | None = None
    previous_epoch: int | None = None
    for epoch in sorted(snapshots):
        model = SplitConvNet(spec).to(device)
        model.load_state_dict(snapshots[epoch])
        print(f"[tracking] checkpoint={epoch} cuts={model.num_cuts - 1}", flush=True)
        for cut in range(1, model.num_cuts):
            print(f"[tracking] checkpoint={epoch} cut={cut}/{model.num_cuts - 1}", flush=True)
            refit = refit_suffix(model, cut, train_batches, eval_batch, config.refit_steps, config.refit_lr)
            record: dict[str, Any] = {
                "epoch": epoch,
                "previous_epoch": previous_epoch,
                "cut": cut,
                "head_regret": refit.loss_before - refit.loss_after,
                "loss_before_refit": refit.loss_before,
                "loss_after_refit": refit.loss_after,
                "prediction_kl": refit.prediction_kl,
                "refit_path_length": refit.path_length,
            }
            record.update(empirical_fisher_schur_trace(model, cut, fisher_batch))
            if cut in previous_refits:
                old_head_new_rep = cross_representation_loss(previous_refits[cut], model, cut, eval_batch)
                record["old_optimum_on_new_rep_loss"] = old_head_new_rep
                record["tracking_demand"] = old_head_new_rep - refit.loss_after
                record["interface_drift_cost"] = old_head_new_rep - previous_refit_losses[cut]
                record["relative_representation_drift"] = relative_representation_drift(
                    previous_model, model, cut, eval_batch[0]
                )
                record.update(suffix_update_decomposition(
                    previous_model, model, previous_refits[cut], refit.refit_model, cut
                ))
            else:
                record["old_optimum_on_new_rep_loss"] = None
                record["tracking_demand"] = None
                record["interface_drift_cost"] = None
                record["relative_representation_drift"] = None
                record.update({
                    "actual_suffix_update_norm": None,
                    "tracking_direction_norm": None,
                    "resolving_direction_norm": None,
                    "tracking_alignment": None,
                    "resolving_alignment": None,
                    "tracking_coefficient": None,
                    "resolving_coefficient": None,
                    "two_direction_explained_fraction": None,
                })
            previous_refits[cut] = refit.refit_model
            previous_refit_losses[cut] = refit.loss_after
            records.append(record)
        previous_model = model
        previous_epoch = epoch
    return records


def run(config: Config) -> Path:
    seed_everything(config.seed)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    device = choose_device(config.device)
    started = time.time()
    print(
        f"[setup] seed={config.seed} device={device} train={config.train_size} "
        f"test={config.test_size} epochs={config.epochs}",
        flush=True,
    )
    datasets = prepare_datasets(config)
    print("[setup] CIFAR and cumulant-surrogate datasets ready", flush=True)
    records: dict[str, Any] = {"config": asdict(config), "device": str(device), "experiment_a": [], "experiment_b": []}
    checkpoints = sorted(set(e for e in config.checkpoint_epochs if 0 <= e <= config.epochs) | {0, config.epochs})
    test_loaders = {name: loader(parts["test"], config, False) for name, parts in datasets.items()}
    for architecture in config.architectures:
        spec = ArchitectureSpec(architecture, config.widths)
        for train_distribution in config.train_distributions:
            print(f"[train] architecture={architecture} distribution={train_distribution}", flush=True)
            model = SplitConvNet(spec).to(device)
            optimizer = torch.optim.SGD(
                model.parameters(), lr=config.learning_rate, momentum=0.9, weight_decay=config.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
            snapshots: dict[int, dict[str, torch.Tensor]] = {}
            for epoch in range(config.epochs + 1):
                if epoch in checkpoints:
                    print(f"[measure] distribution={train_distribution} epoch={epoch}", flush=True)
                    snapshots[epoch] = cpu_state(model)
                    for test_distribution, test_data in test_loaders.items():
                        metrics = evaluate(model, test_data, device)
                        records["experiment_a"].append({
                            "architecture": architecture,
                            "train_distribution": train_distribution,
                            "test_distribution": test_distribution,
                            "epoch": epoch,
                            **metrics,
                        })
                if epoch < config.epochs:
                    train_one_epoch(model, loader(datasets[train_distribution]["train"], config, True), optimizer, device)
                    scheduler.step()
            if train_distribution == "true":
                records["experiment_b"].extend(
                    {"architecture": architecture, **row}
                    for row in run_tracking(snapshots, spec, datasets, config, device)
                )
    records["runtime_seconds"] = time.time() - started
    artifact = output / "results.json"
    artifact.write_text(json.dumps(records, indent=2, allow_nan=False))
    print(f"[done] seed={config.seed} runtime_seconds={records['runtime_seconds']:.1f} artifact={artifact}", flush=True)
    return artifact


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/pilot")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--pca-fit-size", type=int, default=20000)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128, 128])
    parser.add_argument("--architectures", nargs="+", default=["residual"])
    parser.add_argument("--train-distributions", nargs="+", default=["mean", "covariance", "true"])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20])
    parser.add_argument("--refit-steps", type=int, default=100)
    parser.add_argument("--refit-batches", type=int, default=8)
    parser.add_argument("--fisher-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return Config(
        output=args.output, fake_data=args.fake_data, train_size=args.train_size,
        test_size=args.test_size, pca_fit_size=args.pca_fit_size, pca_components=args.pca_components,
        epochs=args.epochs, batch_size=args.batch_size, widths=tuple(args.widths),
        architectures=tuple(args.architectures), train_distributions=tuple(args.train_distributions),
        checkpoint_epochs=tuple(args.checkpoints), refit_steps=args.refit_steps,
        refit_batches=args.refit_batches, fisher_samples=args.fisher_samples,
        seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    print(run(parse_args()))
