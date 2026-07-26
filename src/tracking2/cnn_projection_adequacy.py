from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.utils.data import Subset

from .cnn_checkpoints import sha256
from .experiment import (
    Config,
    choose_device,
    loader,
    ordered_dataset_fingerprints,
    prepare_true_datasets,
    resolve_data_backend,
    seed_everything,
)
from .models import ArchitectureSpec, SplitConvNet
from .provenance import runtime_provenance
from .suffix_statistics import encode_dataset
from .surrogates import ArrayDataset


@dataclass
class CNNProjectionAdequacyConfig:
    """PCA-only gate run before fitting any activation surrogate moments."""

    output: str = "artifacts/lw_post/cnn_projection_adequacy"
    checkpoint: str = ""
    training_manifest: str = ""
    checkpoint_epoch: int = 30
    data_root: str = "data"
    data_backend: str = "torchvision"
    fake_data: bool = False
    train_size: int = 50000
    pca_fit_size: int = 10000
    test_size: int = 10000
    cuts: tuple[int, ...] = (1, 2, 3, 4)
    widths: tuple[int, ...] = (32, 64, 128, 128)
    batch_size: int = 256
    pca_ranks: tuple[int, ...] = (128, 512, 1024)
    seed: int = 0
    device: str = "auto"


def _array_sha256(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _load_checkpoint_lineage(
    config: CNNProjectionAdequacyConfig,
    checkpoint: Path,
) -> dict[str, object]:
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    checkpoint_hash = sha256(checkpoint)
    lineage: dict[str, object] = {
        "checkpoint": {
            "epoch": config.checkpoint_epoch,
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        }
    }
    if not config.training_manifest:
        if not config.fake_data:
            raise ValueError(
                "training_manifest is required for measured projection adequacy"
            )
        return lineage

    manifest_path = Path(config.training_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"training manifest not found: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text())
    expected_status = (
        "MOCKUP / PIPELINE SMOKE TEST" if config.fake_data else "MEASURED"
    )
    if manifest.get("status") != expected_status:
        raise ValueError(
            f"training manifest status must be {expected_status!r}"
        )
    if manifest.get("experiment") != "lw_post_cnn_checkpoint_trajectory":
        raise ValueError("training manifest has the wrong experiment")
    training_config = manifest.get("config")
    if not isinstance(training_config, dict):
        raise ValueError("training manifest config must be an object")
    expected_config = {
        "fake_data": config.fake_data,
        "widths": list(config.widths),
        "seed": config.seed,
        "train_size": config.train_size,
        "test_size": config.test_size,
    }
    mismatches = {
        key: {"expected": expected, "found": training_config.get(key)}
        for key, expected in expected_config.items()
        if training_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "training manifest config does not match projection analysis: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )
    matches = [
        row
        for row in manifest.get("checkpoints", [])
        if isinstance(row, dict)
        and row.get("epoch") == config.checkpoint_epoch
    ]
    if len(matches) != 1:
        raise ValueError(
            "training manifest must contain exactly one requested checkpoint"
        )
    record = matches[0]
    recorded_path = Path(str(record.get("path", "")))
    if not recorded_path.is_absolute():
        recorded_path = manifest_path.parent / recorded_path
    if recorded_path.resolve() != checkpoint.resolve():
        raise ValueError("checkpoint path does not match training manifest")
    if record.get("sha256") != checkpoint_hash:
        raise ValueError("checkpoint SHA-256 does not match training manifest")
    lineage["training_manifest"] = {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "status": manifest["status"],
        "experiment": manifest["experiment"],
        "source_provenance": manifest.get("provenance"),
        "dataset": manifest.get("dataset"),
    }
    return lineage


def _validate_dataset_lineage(
    config: CNNProjectionAdequacyConfig,
    lineage: dict[str, object],
    consumed_dataset: dict[str, object],
) -> None:
    manifest = lineage.get("training_manifest")
    if not isinstance(manifest, dict):
        return
    training_dataset = manifest.get("dataset")
    if not isinstance(training_dataset, dict):
        raise ValueError("training manifest lacks dataset provenance")
    if training_dataset.get("backend") != consumed_dataset.get("backend"):
        raise ValueError("training and projection dataset backends do not match")
    training_splits = training_dataset.get("splits")
    consumed_splits = consumed_dataset.get("splits")
    if not isinstance(training_splits, dict) or not isinstance(
        consumed_splits, dict
    ):
        raise ValueError("dataset fingerprints must contain split records")
    if training_splits != consumed_splits:
        raise ValueError(
            "training and projection full dataset fingerprints do not match"
        )
    if not config.fake_data and consumed_dataset.get("backend") == "fake_data":
        raise ValueError("measured projection analysis cannot use fake data")


def _fit_class_counts(labels: np.ndarray) -> dict[int, int]:
    return {
        int(class_id): int(np.sum(labels == class_id))
        for class_id in np.unique(labels)
    }


def _ordered_prefix_fingerprint(
    dataset: ArrayDataset,
    count: int,
) -> dict[str, object]:
    if not isinstance(dataset, ArrayDataset):
        raise TypeError("PCA prefix fingerprint expects an ArrayDataset")
    if not 0 < count <= len(dataset):
        raise ValueError("PCA prefix count exceeds the training dataset")
    images = np.ascontiguousarray(dataset.images[:count].numpy())
    labels = np.ascontiguousarray(dataset.labels[:count].numpy())

    def digest(array: np.ndarray) -> str:
        value = hashlib.sha256()
        value.update(str(array.dtype).encode())
        value.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        value.update(memoryview(array).cast("B"))
        return value.hexdigest()

    return {
        "count": count,
        "ordered_images_sha256": digest(images),
        "ordered_labels_sha256": digest(labels),
        "selection": f"range(0, {count})",
    }


def _held_out_reference(
    flat: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float, np.ndarray, float]:
    class_count = int(labels.max()) + 1
    class_sums = np.zeros(
        (class_count, flat.shape[1]), dtype=np.float64
    )
    class_counts = np.bincount(labels, minlength=class_count).astype(
        np.int64
    )
    for class_id in np.unique(labels):
        class_sums[class_id] = np.sum(
            flat[labels == class_id], axis=0, dtype=np.float64
        )
    class_means = np.zeros_like(class_sums, dtype=np.float32)
    present = class_counts > 0
    class_means[present] = (
        class_sums[present] / class_counts[present, None]
    ).astype(np.float32)
    global_mean = (
        np.sum(class_sums, axis=0) / len(labels)
    ).astype(np.float32)

    total_denominator = within_denominator = 0.0
    for start in range(0, len(flat), 512):
        stop = min(start + 512, len(flat))
        values = flat[start:stop]
        centered = values - global_mean
        total_denominator += float(
            np.sum(centered * centered, dtype=np.float64)
        )
        centered = values - class_means[labels[start:stop]]
        within_denominator += float(
            np.sum(centered * centered, dtype=np.float64)
        )
    between = class_means[present] - global_mean
    weights = class_counts[present].astype(np.float64)[:, None]
    between_denominator = float(
        np.sum(weights * between**2, dtype=np.float64)
    )
    return (
        class_means,
        global_mean,
        total_denominator,
        within_denominator,
        between,
        between_denominator,
    )


@torch.no_grad()
def _evaluate_rank(
    model: SplitConvNet,
    cut: int,
    test_representation: np.ndarray,
    test_labels: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    coverage_reference: tuple[
        np.ndarray, np.ndarray, float, float, np.ndarray, float
    ],
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    flat = test_representation.reshape(
        len(test_representation), -1
    ).astype(np.float32, copy=False)
    (
        class_means,
        global_mean,
        total_denominator,
        within_denominator,
        between,
        between_denominator,
    ) = coverage_reference
    total_numerator = within_numerator = 0.0
    between_coordinates = between @ components.T
    present_counts = np.bincount(test_labels)[
        np.bincount(test_labels) > 0
    ].astype(np.float64)[:, None]
    between_numerator = float(
        np.sum(
            present_counts * between_coordinates**2,
            dtype=np.float64,
        )
    )
    true_loss = projected_loss = predictive_kl = 0.0
    true_correct = projected_correct = count = 0
    representation_shape = test_representation.shape[1:]

    for start in range(0, len(flat), batch_size):
        stop = min(start + batch_size, len(flat))
        values = flat[start:stop]
        coordinates = (values - pca_mean) @ components.T
        reconstructed = coordinates @ components + pca_mean

        centered = values - global_mean
        projected_centered = centered @ components.T
        total_numerator += float(
            np.sum(
                projected_centered * projected_centered,
                dtype=np.float64,
            )
        )
        centered = values - class_means[test_labels[start:stop]]
        projected_centered = centered @ components.T
        within_numerator += float(
            np.sum(
                projected_centered * projected_centered,
                dtype=np.float64,
            )
        )

        labels = torch.from_numpy(
            test_labels[start:stop].astype(np.int64)
        ).to(device)
        true_values = torch.from_numpy(
            values.reshape((stop - start,) + representation_shape)
        ).to(device)
        projected_values = torch.from_numpy(
            reconstructed.reshape(
                (stop - start,) + representation_shape
            ).astype(np.float32, copy=False)
        ).to(device)
        true_logits = model.forward_from(true_values, cut)
        projected_logits = model.forward_from(projected_values, cut)
        true_loss += float(
            F.cross_entropy(true_logits, labels, reduction="sum")
        )
        projected_loss += float(
            F.cross_entropy(projected_logits, labels, reduction="sum")
        )
        true_correct += int((true_logits.argmax(1) == labels).sum())
        projected_correct += int(
            (projected_logits.argmax(1) == labels).sum()
        )
        predictive_kl += float(
            F.kl_div(
                F.log_softmax(projected_logits, dim=1),
                F.softmax(true_logits, dim=1),
                reduction="sum",
            )
        )
        count += len(labels)

    def fraction(numerator: float, denominator: float) -> float:
        return float(numerator / max(denominator, 1e-12))

    coverage = {
        "total_variance_fraction": fraction(
            total_numerator, total_denominator
        ),
        "within_class_variance_fraction": fraction(
            within_numerator, within_denominator
        ),
        "between_class_mean_variance_fraction": fraction(
            between_numerator, between_denominator
        ),
    }
    functional = {
        "true_loss": true_loss / count,
        "true_accuracy": true_correct / count,
        "projected_true_loss": projected_loss / count,
        "projected_true_accuracy": projected_correct / count,
        "true_to_projected_predictive_kl": max(
            0.0, predictive_kl / count
        ),
    }
    return coverage, functional


def run(config: CNNProjectionAdequacyConfig) -> Path:
    started = time.time()
    seed_everything(config.seed)
    if not config.checkpoint:
        raise ValueError("checkpoint is required")
    if config.pca_fit_size < 2:
        raise ValueError("pca_fit_size must be at least 2")
    if config.train_size < config.pca_fit_size:
        raise ValueError("train_size must be at least pca_fit_size")
    if config.test_size < 1:
        raise ValueError("test_size must be positive")
    ranks = tuple(sorted(set(config.pca_ranks)))
    if not ranks or ranks[0] < 1:
        raise ValueError("pca_ranks must contain positive ranks")
    if not config.cuts:
        raise ValueError("cuts must be nonempty")

    checkpoint = Path(config.checkpoint)
    lineage = _load_checkpoint_lineage(config, checkpoint)
    device = choose_device(config.device)
    data_config = Config(
        data_root=config.data_root,
        data_backend=config.data_backend,
        fake_data=config.fake_data,
        # Materialize the full ordered split to prove checkpoint-dataset
        # lineage, but encode only the prefix Subset constructed below.
        train_size=config.train_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        device=config.device,
    )
    datasets = prepare_true_datasets(data_config)
    if len(datasets["train"]) != config.train_size:
        raise ValueError("data source cannot supply the requested training split")
    if len(datasets["test"]) != config.test_size:
        raise ValueError("data source cannot supply the requested test bank")
    dataset_fingerprints = ordered_dataset_fingerprints(
        datasets, resolve_data_backend(data_config)
    )
    _validate_dataset_lineage(config, lineage, dataset_fingerprints)
    if not isinstance(datasets["train"], ArrayDataset):
        raise TypeError("projection analysis requires materialized training data")
    prefix_fingerprint = _ordered_prefix_fingerprint(
        datasets["train"], config.pca_fit_size
    )
    fit_dataset = Subset(
        datasets["train"], range(config.pca_fit_size)
    )
    train_data = loader(fit_dataset, data_config, False)
    test_data = loader(datasets["test"], data_config, False)

    model = SplitConvNet(
        ArchitectureSpec("residual", config.widths)
    ).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    slices: list[dict[str, object]] = []
    for cut in config.cuts:
        if not 1 <= cut <= len(config.widths):
            raise ValueError(
                f"cut must lie in [1, {len(config.widths)}]"
            )
        print(f"[projection gate] cut={cut}", flush=True)
        fit_representation, fit_labels = encode_dataset(
            model, train_data, cut, device
        )
        test_representation, test_labels = encode_dataset(
            model, test_data, cut, device
        )
        flat_fit = fit_representation.reshape(
            len(fit_representation), -1
        ).astype(np.float32, copy=False)
        native_dimension = flat_fit.shape[1]
        sample_rank_ceiling = len(flat_fit) - 1
        maximal_feasible_rank = min(
            sample_rank_ceiling, native_dimension
        )
        if ranks[-1] > maximal_feasible_rank:
            raise ValueError(
                f"requested PCA rank {ranks[-1]} exceeds cut-{cut} ceiling "
                f"{maximal_feasible_rank}"
            )
        pca = PCA(
            n_components=ranks[-1],
            svd_solver="randomized",
            random_state=config.seed + 10_000 * cut,
        )
        pca.fit(flat_fit)
        pca_mean = pca.mean_.astype(np.float32)
        maximal_components = pca.components_.astype(np.float32)
        basis_sha256 = _array_sha256(pca_mean, maximal_components)
        flat_test = test_representation.reshape(
            len(test_representation), -1
        ).astype(np.float32, copy=False)
        coverage_reference = _held_out_reference(
            flat_test, test_labels
        )
        rank_results: list[dict[str, object]] = []
        for rank in ranks:
            components = maximal_components[:rank]
            coverage, functional = _evaluate_rank(
                model,
                cut,
                test_representation,
                test_labels,
                pca_mean,
                components,
                batch_size=config.batch_size,
                device=device,
                coverage_reference=coverage_reference,
            )
            rank_results.append(
                {
                    "pca_rank": rank,
                    "pca_basis_prefix_sha256": _array_sha256(
                        pca_mean, components
                    ),
                    "held_out_coverage": coverage,
                    "step_zero_true_vs_projected": functional,
                }
            )
        slices.append(
            {
                "cut": cut,
                "representation_shape": list(
                    fit_representation.shape[1:]
                ),
                "activation_flattened_dimension": native_dimension,
                "pca_fit_count": len(fit_representation),
                "pca_fit_class_counts": _fit_class_counts(fit_labels),
                "sample_rank_ceiling": sample_rank_ceiling,
                "activation_dimension_rank_ceiling": native_dimension,
                "maximal_feasible_rank": maximal_feasible_rank,
                "maximal_requested_rank": ranks[-1],
                "pca_solver": "randomized",
                "pca_random_state": config.seed + 10_000 * cut,
                "maximal_pca_basis_sha256": basis_sha256,
                "rank_results": rank_results,
            }
        )

    artifact = {
        "schema_version": 1,
        "experiment": "lw_post_cnn_projection_adequacy",
        "status": (
            "MOCKUP / PIPELINE SMOKE TEST"
            if config.fake_data
            else "MEASURED"
        ),
        "purpose": (
            "PCA-only adequacy gate. It fits no class-conditional surrogate "
            "moments or covariances, Gaussian surrogate, or relaxation "
            "trajectory. Held-out class means are used only to decompose "
            "reported variance coverage."
        ),
        "fit_protocol": (
            "For each requested cut, encode exactly the first pca_fit_size "
            "ordered training examples, fit one randomized PCA basis at the "
            "maximum requested rank, and evaluate nested leading-coordinate "
            "prefixes on the held-out ordered test activation bank. "
            "Projection metrics are streamed in batches without materializing "
            "a projected test bank."
        ),
        "config": asdict(config),
        "architecture": {
            "name": "four-block residual CNN with GroupNorm",
            "widths": list(config.widths),
        },
        "lineage": lineage,
        "dataset": dataset_fingerprints,
        "dataset_roles": {
            "train": (
                "full ordered split fingerprinted for training lineage; only "
                "the declared prefix is activation-encoded"
            ),
            "test": "held-out projection adequacy evaluation bank",
        },
        "pca_fit_prefix": prefix_fingerprint,
        "device": str(device),
        "provenance": runtime_provenance(),
        "slices": slices,
        "runtime_seconds": time.time() - started,
    }
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "cnn_projection_adequacy.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] {path}", flush=True)
    return path


def parse_args() -> CNNProjectionAdequacyConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run a PCA-only held-out projection adequacy gate for a CNN "
            "checkpoint."
        )
    )
    parser.add_argument(
        "--output", default=CNNProjectionAdequacyConfig.output
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-manifest", default="")
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--data-backend",
        choices=("torchvision", "parquet"),
        default=CNNProjectionAdequacyConfig.data_backend,
    )
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--pca-fit-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument(
        "--cuts", type=int, nargs="+", default=[1, 2, 3, 4]
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=list(CNNProjectionAdequacyConfig.widths),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--pca-ranks", type=int, nargs="+", default=[128, 512, 1024]
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return CNNProjectionAdequacyConfig(
        output=args.output,
        checkpoint=args.checkpoint,
        training_manifest=args.training_manifest,
        checkpoint_epoch=args.checkpoint_epoch,
        data_root=args.data_root,
        data_backend=args.data_backend,
        fake_data=args.fake_data,
        train_size=args.train_size,
        pca_fit_size=args.pca_fit_size,
        test_size=args.test_size,
        cuts=tuple(args.cuts),
        widths=tuple(args.widths),
        batch_size=args.batch_size,
        pca_ranks=tuple(args.pca_ranks),
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
