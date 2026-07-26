from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .cnn_checkpoints import sha256
from .criticality import (
    CriticalityConfig,
    datasets,
    make_loader,
    ordered_dataset_fingerprints,
    resolved_data_backend,
)
from .models import InstrumentedResNet18V2
from .provenance import runtime_provenance
from .resnet_criticality import RESNET_ARCHITECTURE_NAME, SMOKE_STATUS
from .representation_surrogates import (
    fit_representation_surrogate,
    moment_diagnostics,
    pca_coverage_diagnostics,
    project_representation,
    sample_representation_surrogate,
    truncate_representation_surrogate,
)
from .vgg_suffix_statistics import representation_loader


@dataclass
class ResNetSuffixStatisticsConfig:
    output: str = "artifacts/resnet_suffix_statistics/pilot"
    checkpoint: str = ""
    training_manifest: str = ""
    checkpoint_epoch: int = 100
    data_root: str = "data"
    data_backend: str = "torchvision"
    fake_data: bool = False
    train_size: int = 10000
    test_size: int = 2000
    batch_size: int = 128
    width: int = 64
    cuts: tuple[int, ...] = tuple(range(8))
    pca_fit_size: int = 5000
    pca_ranks: tuple[int, ...] = (512,)
    surrogate_draws: int = 3
    mean_noise_radii: tuple[float, ...] = (1.0,)
    include_projected_true: bool = False
    true_eval_only: bool = False
    relax_epochs: int = 5
    relax_learning_rate: float = 0.01
    seed: int = 0
    device: str = "auto"


def _has_reproducible_source_identity(provenance: object) -> bool:
    if not isinstance(provenance, dict):
        return False
    revision = provenance.get("source_revision")
    archive_hash = provenance.get("source_archive_sha256")
    clean_revision = (
        isinstance(revision, str)
        and re.fullmatch(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})", revision)
        is not None
    )
    archived_source = (
        isinstance(archive_hash, str)
        and re.fullmatch(r"[0-9a-fA-F]{64}", archive_hash) is not None
    )
    return clean_revision or archived_source


@torch.no_grad()
def encode(model: InstrumentedResNet18V2, data, cut: int, device: torch.device):
    model.eval()
    representations, labels = [], []
    for x, y in data:
        representations.append(model.encode_to_block(x.to(device), cut).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(representations), np.concatenate(labels)


def _materialize_bank(data: DataLoader) -> TensorDataset:
    images, labels = [], []
    for batch_images, batch_labels in data:
        images.append(batch_images.detach().cpu())
        labels.append(batch_labels.detach().cpu())
    return TensorDataset(torch.cat(images), torch.cat(labels))


def _bank_fingerprint(data: TensorDataset, seed: int) -> dict[str, object]:
    images, labels = data.tensors
    digest = hashlib.sha256()
    image_array = np.ascontiguousarray(images.numpy())
    label_array = np.ascontiguousarray(labels.numpy(), dtype=np.int64)
    digest.update(str(image_array.shape).encode())
    digest.update(str(image_array.dtype).encode())
    digest.update(memoryview(image_array))
    digest.update(str(label_array.shape).encode())
    digest.update(memoryview(label_array))
    return {
        "seed": seed,
        "count": len(labels),
        "sha256": digest.hexdigest(),
    }


def _pca_basis_sha256(pca_mean: np.ndarray, components: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (pca_mean, components):
        values = np.ascontiguousarray(array)
        digest.update(str(values.shape).encode())
        digest.update(str(values.dtype).encode())
        digest.update(memoryview(values))
    return digest.hexdigest()


@torch.no_grad()
def evaluate_suffix(model: InstrumentedResNet18V2, cut: int, data, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sum = correct = count = 0
    for z, y in data:
        z, y = z.to(device), y.to(device)
        logits = model.forward_from_block(z, cut)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def _expected_architecture(config: ResNetSuffixStatisticsConfig) -> dict[str, object]:
    model = InstrumentedResNet18V2(width=config.width)
    return {
        "name": RESNET_ARCHITECTURE_NAME,
        "width": config.width,
        "block_names": list(model.block_names),
    }


def _load_checkpoint_lineage(
    config: ResNetSuffixStatisticsConfig,
    checkpoint: Path,
) -> tuple[dict, dict[str, object]]:
    if not config.training_manifest:
        raise ValueError("training_manifest is required for checkpoint lineage")
    manifest_path = Path(config.training_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_status = SMOKE_STATUS if config.fake_data else "MEASURED"
    if manifest.get("status") != expected_status:
        raise ValueError(
            f"training manifest status must be {expected_status!r}, found "
            f"{manifest.get('status')!r}"
        )
    if manifest.get("experiment") not in {
        "resnet18_training_checkpoints",
        "resnet18_block_criticality",
    }:
        raise ValueError("training manifest has the wrong experiment")
    if not config.fake_data and not _has_reproducible_source_identity(
        manifest.get("provenance")
    ):
        raise ValueError(
            "measured training manifest must record a clean full source revision "
            "or source archive SHA-256"
        )
    expected_architecture = _expected_architecture(config)
    if manifest.get("architecture") != expected_architecture:
        raise ValueError("training manifest architecture does not match analysis")

    training_config = manifest.get("config")
    if not isinstance(training_config, dict):
        raise ValueError("training manifest config must be an object")
    expected_config = {
        "fake_data": config.fake_data,
        "seed": config.seed,
        "width": config.width,
        "data_backend": config.data_backend,
    }
    mismatches = {
        key: {"expected": expected, "found": training_config.get(key)}
        for key, expected in expected_config.items()
        if training_config.get(key) != expected
    }
    if mismatches:
        raise ValueError(
            "training manifest config does not match analysis: "
            f"{json.dumps(mismatches, sort_keys=True)}"
        )

    checkpoint_records = manifest.get("checkpoints")
    if not isinstance(checkpoint_records, list):
        raise ValueError("training manifest checkpoints must be an array")
    matches = [
        row
        for row in checkpoint_records
        if isinstance(row, dict) and row.get("epoch") == config.checkpoint_epoch
    ]
    if len(matches) != 1:
        raise ValueError(
            "training manifest must contain exactly one checkpoint for analysis "
            f"epoch {config.checkpoint_epoch}"
        )
    record = matches[0]
    recorded_path = Path(str(record.get("path", "")))
    if not recorded_path.is_absolute():
        recorded_path = manifest_path.parent / recorded_path
    if recorded_path.resolve() != checkpoint.resolve():
        raise ValueError(
            f"checkpoint path does not match training manifest: "
            f"{checkpoint.resolve()} != {recorded_path.resolve()}"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    checkpoint_hash = sha256(checkpoint)
    if record.get("sha256") != checkpoint_hash:
        raise ValueError("checkpoint SHA-256 does not match training manifest")

    lineage = {
        "training_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
            "status": manifest["status"],
            "experiment": manifest["experiment"],
        },
        "checkpoint": {
            "epoch": config.checkpoint_epoch,
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "model_seed": config.seed,
        "architecture": expected_architecture,
        "training_source": manifest.get("provenance"),
    }
    return manifest, lineage


def _validate_dataset_lineage(
    training_manifest: Mapping[str, object],
    analysis_dataset: Mapping[str, object],
) -> None:
    training_dataset = training_manifest.get("dataset")
    if not isinstance(training_dataset, dict):
        raise ValueError("training manifest does not record dataset fingerprints")
    if training_dataset.get("backend") != analysis_dataset.get("backend"):
        raise ValueError("training and analysis dataset backends do not match")
    for split in ("train", "test"):
        training_split = training_dataset.get(split)
        analysis_split = analysis_dataset.get(split)
        if not isinstance(training_split, dict) or not isinstance(analysis_split, dict):
            raise ValueError(f"missing {split} dataset fingerprint")
        for key in ("source_count", "source_sha256"):
            if training_split.get(key) != analysis_split.get(key):
                raise ValueError(
                    f"training and analysis {split} dataset {key} do not match"
                )


def _relax_suffix(
    model: InstrumentedResNet18V2,
    cut: int,
    train_representation: np.ndarray,
    train_labels: np.ndarray,
    evaluation_sets: Mapping[str, np.ndarray],
    test_labels: np.ndarray,
    *,
    distribution: str,
    draw: int,
    batch_size: int,
    relax_epochs: int,
    relax_learning_rate: float,
    shuffle_seed: int,
    device: torch.device,
) -> list[dict[str, object]]:
    candidate = copy.deepcopy(model)
    for parameter in candidate.parameters_through_block(cut):
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(
        candidate.parameters_after_block(cut),
        lr=relax_learning_rate,
        momentum=0.9,
    )
    relax_data = representation_loader(
        train_representation,
        train_labels,
        batch_size,
        shuffle_seed,
        True,
    )
    evaluation_loaders = {
        name: representation_loader(
            values,
            test_labels,
            batch_size,
            shuffle_seed,
            False,
        )
        for name, values in evaluation_sets.items()
    }
    records: list[dict[str, object]] = []
    for relax_epoch in range(relax_epochs + 1):
        for evaluation_distribution, evaluation_data in evaluation_loaders.items():
            records.append(
                {
                    "draw": draw,
                    "train_distribution": distribution,
                    "eval_distribution": evaluation_distribution,
                    "relax_epoch": relax_epoch,
                    **evaluate_suffix(candidate, cut, evaluation_data, device),
                }
            )
        if relax_epoch == relax_epochs:
            break
        candidate.train()
        for representation, labels in relax_data:
            representation, labels = representation.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(
                candidate.forward_from_block(representation, cut), labels
            )
            loss.backward()
            optimizer.step()
    return records


def run(config: ResNetSuffixStatisticsConfig) -> Path:
    if not config.checkpoint:
        raise ValueError("checkpoint is required")
    if not config.training_manifest:
        raise ValueError("training_manifest is required")
    if config.surrogate_draws < 1:
        raise ValueError("surrogate_draws must be positive")
    if not config.pca_ranks or any(rank < 1 for rank in config.pca_ranks):
        raise ValueError("pca_ranks must contain positive ranks")
    if any(radius < 0 for radius in config.mean_noise_radii):
        raise ValueError("mean_noise_radii must be non-negative")
    if not config.fake_data and config.data_backend != "torchvision":
        raise ValueError(
            "Measured ResNet suffix analysis requires data_backend='torchvision'"
        )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    analysis_provenance = runtime_provenance()
    if not config.fake_data and not _has_reproducible_source_identity(
        analysis_provenance
    ):
        raise ValueError(
            "measured analysis requires a clean full source revision or source "
            "archive SHA-256"
        )
    checkpoint = Path(config.checkpoint)
    training_manifest, lineage = _load_checkpoint_lineage(config, checkpoint)
    model = InstrumentedResNet18V2(width=config.width).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    data_config = CriticalityConfig(
        data_root=config.data_root,
        data_backend=config.data_backend,
        train_size=config.train_size,
        test_size=config.test_size,
        batch_size=config.batch_size,
        seed=config.seed,
        fake_data=config.fake_data,
    )
    train_set, test_set = datasets(data_config)
    dataset_identity = ordered_dataset_fingerprints(
        train_set,
        test_set,
        backend=resolved_data_backend(data_config),
    )
    _validate_dataset_lineage(training_manifest, dataset_identity)
    lineage["dataset_source"] = {
        "backend": dataset_identity["backend"],
        "train": dataset_identity["train"],
        "test": dataset_identity["test"],
    }

    train_bank_seed = config.seed + 2_000_000
    test_bank_seed = config.seed + 2_000_001
    torch.manual_seed(train_bank_seed)
    train_bank = _materialize_bank(make_loader(train_set, data_config, False))
    torch.manual_seed(test_bank_seed)
    test_bank = _materialize_bank(make_loader(test_set, data_config, False))
    analysis_banks = {
        "definition": (
            "One ordered augmented-and-normalized train image bank and one "
            "ordered deterministic test image bank, materialized once and reused "
            "for every cut and PCA rank."
        ),
        "train": _bank_fingerprint(train_bank, train_bank_seed),
        "test": _bank_fingerprint(test_bank, test_bank_seed),
    }
    train_data = make_loader(train_bank, data_config, False)
    test_data = make_loader(test_bank, data_config, False)

    started = time.time()
    slices: list[dict] = []
    for cut in config.cuts:
        if not 0 <= cut < 8:
            raise ValueError("cuts must select residual blocks 0..7")
        module = model.block_names[cut]
        print(f"[slice] cut={cut}:{module}", flush=True)
        train_rep, train_labels = encode(model, train_data, cut, device)
        test_rep, test_labels = encode(model, test_data, cut, device)
        fit_count = min(config.pca_fit_size, len(train_rep))
        max_rank = min(fit_count - 1, train_rep[0].size)
        ranks = sorted({min(rank, max_rank) for rank in config.pca_ranks})
        if not ranks or ranks[0] < 1:
            raise ValueError("pca_ranks are infeasible for the activation bank")
        pca_fit_class_counts = {
            int(class_id): int(np.sum(train_labels[:fit_count] == class_id))
            for class_id in np.unique(train_labels[:fit_count])
        }
        moment_fit_class_counts = {
            int(class_id): int(np.sum(train_labels == class_id))
            for class_id in np.unique(train_labels)
        }
        empirical_covariance_rank_ceiling = min(moment_fit_class_counts.values()) - 1
        maximal_surrogate = fit_representation_surrogate(
            train_rep[:fit_count],
            train_labels[:fit_count],
            ranks[-1],
            config.seed + cut,
        )
        maximal_pca_basis_sha256 = _pca_basis_sha256(
            maximal_surrogate.pca_mean,
            maximal_surrogate.components,
        )

        reference_records: list[dict[str, object]] = []
        for draw in range(config.surrogate_draws):
            condition_shuffle_seed = (
                config.seed + 100_000 * draw + 1_000 * cut
            )
            reference_records.extend(
                _relax_suffix(
                    model,
                    cut,
                    train_rep,
                    train_labels,
                    {"true": test_rep},
                    test_labels,
                    distribution="true",
                    draw=draw,
                    batch_size=config.batch_size,
                    relax_epochs=config.relax_epochs,
                    relax_learning_rate=config.relax_learning_rate,
                    shuffle_seed=condition_shuffle_seed,
                    device=device,
                )
            )

        rank_results: list[dict[str, object]] = []
        for rank in ranks:
            surrogate = truncate_representation_surrogate(
                maximal_surrogate,
                train_rep,
                train_labels,
                rank,
            )
            coverage = pca_coverage_diagnostics(
                surrogate, test_rep, test_labels
            )
            records: list[dict[str, object]] = []
            diagnostics: list[dict[str, object]] = []
            for draw in range(config.surrogate_draws):
                condition_shuffle_seed = (
                    config.seed + 100_000 * draw + 1_000 * cut
                )
                train_sets: dict[str, np.ndarray] = {}
                test_sets: dict[str, np.ndarray] = {"true": test_rep}

                noise_seed = (
                    config.seed
                    + 1_000_000 * draw
                    + 10_000 * cut
                )
                if config.include_projected_true:
                    train_sets["projected_true"] = project_representation(
                        surrogate, train_rep
                    )
                    projected_test = project_representation(surrogate, test_rep)
                    if not config.true_eval_only:
                        test_sets["projected_true"] = projected_test

                gaussian_train = sample_representation_surrogate(
                    surrogate,
                    train_labels,
                    "gaussian",
                    noise_seed + 1,
                    paired_noise_rank=ranks[-1],
                )
                gaussian_test = sample_representation_surrogate(
                    surrogate,
                    test_labels,
                    "gaussian",
                    noise_seed + 2,
                    paired_noise_rank=ranks[-1],
                )
                train_sets["gaussian"] = gaussian_train
                if not config.true_eval_only:
                    test_sets["gaussian"] = gaussian_test
                diagnostics.append({
                    "draw": draw,
                    "distribution": "gaussian",
                    **moment_diagnostics(
                        test_rep, gaussian_test, test_labels, surrogate
                    ),
                })

                mean_names: list[str] = []
                for radius_index, radius in enumerate(config.mean_noise_radii):
                    name = (
                        "mean"
                        if config.mean_noise_radii == (1.0,)
                        else f"mean_r{radius:g}"
                    )
                    mean_names.append(name)
                    mean_train = sample_representation_surrogate(
                        surrogate,
                        train_labels,
                        "mean",
                        noise_seed + 10,
                        mean_noise_radius=radius,
                        paired_noise_rank=ranks[-1],
                    )
                    mean_test = sample_representation_surrogate(
                        surrogate,
                        test_labels,
                        "mean",
                        noise_seed + 100,
                        mean_noise_radius=radius,
                        paired_noise_rank=ranks[-1],
                    )
                    train_sets[name] = mean_train
                    if not config.true_eval_only:
                        test_sets[name] = mean_test
                    diagnostics.append({
                        "draw": draw,
                        "distribution": name,
                        "mean_noise_radius": radius,
                        "isotropic_covariance_trace_ratio": radius**2,
                        **moment_diagnostics(
                            test_rep, mean_test, test_labels, surrogate
                        ),
                    })

                train_distributions = [
                    *mean_names,
                    "gaussian",
                    *(
                        ["projected_true"]
                        if config.include_projected_true
                        else []
                    ),
                ]
                for train_distribution in train_distributions:
                    records.extend(
                        _relax_suffix(
                            model,
                            cut,
                            train_sets[train_distribution],
                            train_labels,
                            test_sets,
                            test_labels,
                            distribution=train_distribution,
                            draw=draw,
                            batch_size=config.batch_size,
                            relax_epochs=config.relax_epochs,
                            relax_learning_rate=config.relax_learning_rate,
                            shuffle_seed=condition_shuffle_seed,
                            device=device,
                        )
                    )

            rank_results.append({
                "pca_rank": rank,
                "maximal_pca_basis_sha256": maximal_pca_basis_sha256,
                "pca_basis_prefix_sha256": _pca_basis_sha256(
                    surrogate.pca_mean,
                    surrogate.components,
                ),
                "pca_basis_fit_count": fit_count,
                "pca_basis_fit_class_counts": pca_fit_class_counts,
                "moment_fit_count": len(train_rep),
                "moment_fit_class_counts": moment_fit_class_counts,
                "empirical_class_covariance_rank_ceiling": (
                    empirical_covariance_rank_ceiling
                ),
                "rank_exceeds_empirical_class_covariance_ceiling": (
                    rank > empirical_covariance_rank_ceiling
                ),
                "covariance_shrinkage": 0.05,
                "pooled_within_class_variance_per_pca_coordinate": (
                    surrogate.pooled_variance
                ),
                "trace_matched_isotropic_covariance_trace": (
                    rank * surrogate.pooled_variance
                ),
                "held_out_explained_variance_fraction": coverage[
                    "total_variance_fraction"
                ],
                "held_out_coverage": coverage,
                "coverage_evaluation": "held-out CIFAR-10 test activation bank",
                "moment_diagnostics": diagnostics,
                "records": records,
            })

        slices.append({
            "cut": cut,
            "module": module,
            "condition": "native",
            "representation_shape": list(train_rep.shape[1:]),
            "native_dimension": int(train_rep[0].size),
            "maximal_pca_rank": ranks[-1],
            "maximal_pca_basis_sha256": maximal_pca_basis_sha256,
            "pca_basis_fit_count": fit_count,
            "pca_basis_fit_class_counts": pca_fit_class_counts,
            "moment_fit_count": len(train_rep),
            "moment_fit_class_counts": moment_fit_class_counts,
            "reference_records": reference_records,
            "rank_results": rank_results,
        })
    artifact = {
        "schema_version": 3,
        "experiment": "resnet18_suffix_statistics_sweep",
        "status": SMOKE_STATUS if config.fake_data else "MEASURED",
        "mean_noise_definition": (
            "At radius 1, isotropic covariance trace matches the pooled average "
            "within-class covariance trace in the fitted PCA space."
        ),
        "pca_protocol": (
            "Each cut fits one maximal PCA basis on the configured prefix of the "
            "fixed analysis bank. Lower ranks truncate leading coordinates of "
            "that basis. Class means and covariances are re-estimated from all "
            "training activations after projection. The true suffix reference is "
            "trained once per draw and shared across ranks. Nested ranks use "
            "leading coordinates from the same maximal standard-normal banks."
        ),
        "checkpoint": lineage["checkpoint"],
        "lineage": lineage,
        "dataset": dataset_identity,
        "analysis_banks": analysis_banks,
        "config": asdict(config),
        "device": str(device),
        "provenance": analysis_provenance,
        "architecture": _expected_architecture(config),
        "module_names": list(model.block_names),
        "slices": slices,
        "runtime_seconds": time.time() - started,
    }
    path = output / "resnet_suffix_statistics.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] {path}", flush=True)
    return path


def parse_args() -> ResNetSuffixStatisticsConfig:
    parser = argparse.ArgumentParser(description="Part-B suffix-statistics sweep on ResNet-18 blocks")
    parser.add_argument("--output", default=ResNetSuffixStatisticsConfig.output)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--training-manifest", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--data-backend",
        choices=("auto", "torchvision", "parquet"),
        default="torchvision",
    )
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--cuts", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--pca-fit-size", type=int, default=5000)
    parser.add_argument("--pca-ranks", type=int, nargs="+", default=[512])
    parser.add_argument("--surrogate-draws", type=int, default=3)
    parser.add_argument(
        "--mean-noise-radii", type=float, nargs="+", default=[1.0]
    )
    parser.add_argument("--include-projected-true", action="store_true")
    parser.add_argument("--true-eval-only", action="store_true")
    parser.add_argument("--relax-epochs", type=int, default=5)
    parser.add_argument("--relax-learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return ResNetSuffixStatisticsConfig(
        output=args.output, checkpoint=args.checkpoint,
        training_manifest=args.training_manifest,
        checkpoint_epoch=args.checkpoint_epoch,
        data_root=args.data_root, data_backend=args.data_backend,
        fake_data=args.fake_data,
        train_size=args.train_size, test_size=args.test_size,
        batch_size=args.batch_size, width=args.width, cuts=tuple(args.cuts),
        pca_fit_size=args.pca_fit_size, pca_ranks=tuple(args.pca_ranks),
        surrogate_draws=args.surrogate_draws,
        mean_noise_radii=tuple(args.mean_noise_radii),
        include_projected_true=args.include_projected_true,
        true_eval_only=args.true_eval_only,
        relax_epochs=args.relax_epochs,
        relax_learning_rate=args.relax_learning_rate, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
