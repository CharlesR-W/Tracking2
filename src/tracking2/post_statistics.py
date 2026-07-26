from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

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
from .representation_surrogates import (
    RepresentationSurrogate,
    fit_representation_surrogate,
    moment_diagnostics,
    pca_coverage_diagnostics,
    project_representation,
    sample_representation_surrogate,
    truncate_representation_surrogate,
)
from .suffix_statistics import encode_dataset, evaluate_suffix, representation_loader


@dataclass
class PostStatisticsConfig:
    """Measured suffix-relaxation controls used by the LW post and dashboard."""

    output: str = "artifacts/lw_post/cnn_statistics"
    checkpoint: str = ""
    checkpoint_epoch: int = 30
    data_root: str = "data"
    data_backend: str = "torchvision"
    fake_data: bool = False
    train_size: int = 50000
    test_size: int = 10000
    cuts: tuple[int, ...] = (1, 2, 3, 4)
    widths: tuple[int, ...] = (32, 64, 128, 128)
    batch_size: int = 256
    pca_fit_size: int = 10000
    pca_ranks: tuple[int, ...] = (512,)
    gaussian_covariance_shrinkages: tuple[float, ...] = (0.0,)
    mean_noise_radii: tuple[float, ...] = (1.0,)
    surrogate_draws: int = 1
    true_eval_only: bool = False
    suffix_initialization: str = "warm"
    learning_rate_regime: str = "fixed_lr"
    relax_epochs: int = 10
    relax_learning_rate: float = 0.01
    seed: int = 0
    device: str = "auto"


def gaussian_distribution_name(shrinkage: float) -> str:
    if shrinkage == 0:
        return "gaussian_empirical"
    return f"gaussian_shrunk_s{shrinkage:g}"


def condition_shuffle_seed(seed: int, cut: int, draw: int) -> int:
    """Match minibatch order across every distribution in one comparison draw."""

    return seed + 100_000 * draw + 1_000 * cut


def _shuffle_indices(count: int, seed: int) -> np.ndarray:
    """Return the exact first-epoch order used by the relaxation loader."""

    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(count, generator=generator).numpy()


def _shuffled_representation_loader(
    representations: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    seed: int,
) -> DataLoader:
    """Shuffle from a sampler-owned RNG so its first permutation is auditable."""

    dataset = TensorDataset(
        torch.from_numpy(representations),
        torch.from_numpy(labels.astype(np.int64)),
    )
    sampler = RandomSampler(
        dataset,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def _index_sha256(indices: np.ndarray) -> str:
    values = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(memoryview(values)).hexdigest()


def _initial_step_diagnostics(
    candidate: SplitConvNet,
    optimizer: torch.optim.Optimizer,
    cut: int,
    train_representation: np.ndarray,
    train_labels: np.ndarray,
    *,
    batch_size: int,
    shuffle_seed: int,
    device: torch.device,
) -> dict[str, object]:
    """Measure the actual first SGD step, then restore the unrelaxed suffix."""

    indices = _shuffle_indices(len(train_representation), shuffle_seed)[:batch_size]
    representation = torch.from_numpy(train_representation[indices]).to(device)
    labels = torch.from_numpy(
        train_labels[indices].astype(np.int64)
    ).to(device)
    suffix_parameters = [
        parameter
        for parameter in candidate.suffix_parameters(cut)
        if parameter.requires_grad
    ]
    parameter_count = sum(parameter.numel() for parameter in suffix_parameters)
    if parameter_count == 0:
        raise RuntimeError("suffix has no trainable parameters")

    candidate.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.cross_entropy(candidate.forward_from(representation, cut), labels)
    loss.backward()
    squared_gradient = sum(
        torch.sum(parameter.grad.detach() ** 2)
        for parameter in suffix_parameters
        if parameter.grad is not None
    )
    total_gradient_norm = float(torch.sqrt(squared_gradient))
    rms_gradient = total_gradient_norm / parameter_count**0.5
    squared_weight = sum(
        torch.sum(parameter.detach() ** 2)
        for parameter in suffix_parameters
    )
    weight_norm = float(torch.sqrt(squared_weight))
    if weight_norm == 0:
        raise RuntimeError("suffix has zero total weight norm")

    initial_parameters = [
        parameter.detach().clone() for parameter in suffix_parameters
    ]
    optimizer.step()
    squared_update = sum(
        torch.sum((parameter.detach() - initial) ** 2)
        for parameter, initial in zip(suffix_parameters, initial_parameters)
    )
    update_norm = float(torch.sqrt(squared_update))
    with torch.no_grad():
        for parameter, initial in zip(suffix_parameters, initial_parameters):
            parameter.copy_(initial)
    optimizer.state.clear()
    optimizer.zero_grad(set_to_none=True)

    return {
        "initial_training_loss": float(loss.detach()),
        # Compatibility alias retained for existing verifiers and reports.
        "initial_gradient_norm": total_gradient_norm,
        "initial_gradient_total_norm": total_gradient_norm,
        "initial_gradient_rms": rms_gradient,
        "initial_suffix_parameter_count": parameter_count,
        "initial_suffix_weight_norm": weight_norm,
        "first_step_update_norm": update_norm,
        "first_step_update_to_weight_ratio": update_norm / weight_norm,
        "initial_batch_shuffle_seed": shuffle_seed,
        "initial_batch_size": len(indices),
        "initial_batch_index_sha256": _index_sha256(indices),
    }


@torch.no_grad()
def _step_zero_projection_diagnostics(
    model: SplitConvNet,
    cut: int,
    true_representation: np.ndarray,
    projected_representation: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Quantify the suffix's update-0 functional change under PCA projection."""

    model.eval()
    true_loss = projected_loss = 0.0
    true_correct = projected_correct = count = 0
    predictive_kl = 0.0
    for start in range(0, len(labels), batch_size):
        stop = min(start + batch_size, len(labels))
        batch_labels = torch.from_numpy(
            labels[start:stop].astype(np.int64)
        ).to(device)
        true_values = torch.from_numpy(
            true_representation[start:stop]
        ).to(device)
        projected_values = torch.from_numpy(
            projected_representation[start:stop]
        ).to(device)
        true_logits = model.forward_from(true_values, cut)
        projected_logits = model.forward_from(projected_values, cut)
        true_loss += float(
            F.cross_entropy(true_logits, batch_labels, reduction="sum")
        )
        projected_loss += float(
            F.cross_entropy(projected_logits, batch_labels, reduction="sum")
        )
        true_correct += int((true_logits.argmax(1) == batch_labels).sum())
        projected_correct += int(
            (projected_logits.argmax(1) == batch_labels).sum()
        )
        predictive_kl += float(
            F.kl_div(
                F.log_softmax(projected_logits, dim=1),
                F.softmax(true_logits, dim=1),
                reduction="sum",
            )
        )
        count += len(batch_labels)
    return {
        "true_loss": true_loss / count,
        "true_accuracy": true_correct / count,
        "projected_true_loss": projected_loss / count,
        "projected_true_accuracy": projected_correct / count,
        "true_to_projected_predictive_kl": max(0.0, predictive_kl / count),
    }


def _relax_suffix(
    model: SplitConvNet,
    cut: int,
    train_rep: np.ndarray,
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
    suffix_initialization: str = "warm",
    initialization_seed: int = 0,
    learning_rate_regime: str = "fixed_lr",
    target_first_step_update_norm: float | None = None,
) -> list[dict[str, object]]:
    candidate = copy.deepcopy(model)
    if suffix_initialization not in {"warm", "reinitialized"}:
        raise ValueError("suffix_initialization must be 'warm' or 'reinitialized'")
    if suffix_initialization == "reinitialized":
        cuda_devices = (
            [device.index if device.index is not None else torch.cuda.current_device()]
            if device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(initialization_seed)
            for block in candidate.blocks[cut:]:
                block.apply(
                    lambda module: (
                        module.reset_parameters()
                        if hasattr(module, "reset_parameters")
                        else None
                    )
                )
            candidate.classifier.reset_parameters()
    for parameter in candidate.prefix_parameters(cut):
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(
        candidate.suffix_parameters(cut),
        lr=relax_learning_rate,
        momentum=0.9,
    )
    train_data = _shuffled_representation_loader(
        train_rep, train_labels, batch_size, shuffle_seed
    )
    evaluation_loaders = {
        name: representation_loader(
            values, test_labels, batch_size, shuffle_seed, False
        )
        for name, values in evaluation_sets.items()
    }
    initial_diagnostics = _initial_step_diagnostics(
        candidate,
        optimizer,
        cut,
        train_rep,
        train_labels,
        batch_size=batch_size,
        shuffle_seed=shuffle_seed,
        device=device,
    )
    if learning_rate_regime not in {
        "fixed_lr",
        "match_true_initial_update",
    }:
        raise ValueError(
            "learning_rate_regime must be 'fixed_lr' or "
            "'match_true_initial_update'"
        )
    learning_rate_multiplier = 1.0
    if learning_rate_regime == "match_true_initial_update":
        if target_first_step_update_norm is None:
            if distribution != "true":
                raise ValueError(
                    "matched-update relaxation requires a true-condition target"
                )
            target_first_step_update_norm = float(
                initial_diagnostics["first_step_update_norm"]
            )
        observed_update = float(initial_diagnostics["first_step_update_norm"])
        learning_rate_multiplier = (
            target_first_step_update_norm / max(observed_update, 1e-12)
        )
        # The clamp is deliberately broad: it guards degenerate zero-gradient
        # batches without preventing the orders-of-magnitude correction that
        # motivated this control.
        learning_rate_multiplier = float(
            np.clip(learning_rate_multiplier, 1e-3, 1e3)
        )
        for group in optimizer.param_groups:
            group["lr"] = relax_learning_rate * learning_rate_multiplier
    effective_learning_rate = relax_learning_rate * learning_rate_multiplier
    matched_update_norm = (
        float(initial_diagnostics["first_step_update_norm"])
        * learning_rate_multiplier
    )
    records: list[dict[str, object]] = []
    for relax_epoch in range(relax_epochs + 1):
        for evaluation_distribution, evaluation_data in evaluation_loaders.items():
            records.append(
                {
                    "draw": draw,
                    "train_distribution": distribution,
                    "eval_distribution": evaluation_distribution,
                    "relax_epoch": relax_epoch,
                    "suffix_initialization": suffix_initialization,
                    "learning_rate_regime": learning_rate_regime,
                    "base_relax_learning_rate": relax_learning_rate,
                    "learning_rate_multiplier": learning_rate_multiplier,
                    "effective_relax_learning_rate": effective_learning_rate,
                    "matched_first_step_update_norm": matched_update_norm,
                    **initial_diagnostics,
                    **evaluate_suffix(candidate, cut, evaluation_data, device),
                }
            )
        if relax_epoch == relax_epochs:
            break
        candidate.train()
        for representation, labels in train_data:
            representation, labels = representation.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(candidate.forward_from(representation, cut), labels)
            loss.backward()
            optimizer.step()
    return records


def run(config: PostStatisticsConfig) -> Path:
    seed_everything(config.seed)
    if not config.checkpoint:
        raise ValueError("checkpoint is required")
    if config.surrogate_draws < 1:
        raise ValueError("surrogate_draws must be positive")
    if any(radius < 0 for radius in config.mean_noise_radii):
        raise ValueError("mean_noise_radii must be non-negative")
    if (
        not config.gaussian_covariance_shrinkages
        or any(
            not 0 <= shrinkage <= 1
            for shrinkage in config.gaussian_covariance_shrinkages
        )
    ):
        raise ValueError(
            "gaussian_covariance_shrinkages must contain values in [0, 1]"
        )
    if config.suffix_initialization not in {"warm", "reinitialized"}:
        raise ValueError("suffix_initialization must be 'warm' or 'reinitialized'")
    if config.learning_rate_regime not in {
        "fixed_lr",
        "match_true_initial_update",
    }:
        raise ValueError(
            "learning_rate_regime must be 'fixed_lr' or "
            "'match_true_initial_update'"
        )

    checkpoint = Path(config.checkpoint)
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
    train_data = loader(datasets["train"], data_config, False)
    test_data = loader(datasets["test"], data_config, False)

    model = SplitConvNet(ArchitectureSpec("residual", config.widths)).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    started = time.time()
    slices: list[dict[str, object]] = []

    for cut in config.cuts:
        if not 1 <= cut <= len(config.widths):
            raise ValueError(f"cut must lie in [1, {len(config.widths)}]")
        print(f"[cut] {cut}/{len(config.widths)}", flush=True)
        train_rep, train_labels = encode_dataset(model, train_data, cut, device)
        test_rep, test_labels = encode_dataset(model, test_data, cut, device)
        fit_count = min(config.pca_fit_size, len(train_rep))
        max_rank = min(fit_count - 1, train_rep[0].size)
        ranks = sorted({min(rank, max_rank) for rank in config.pca_ranks})
        if not ranks or ranks[0] < 1:
            raise ValueError("pca_ranks must contain at least one positive rank")
        pca_fit_class_counts = {
            int(class_id): int(np.sum(train_labels[:fit_count] == class_id))
            for class_id in np.unique(train_labels[:fit_count])
        }
        moment_fit_class_counts = {
            int(class_id): int(np.sum(train_labels == class_id))
            for class_id in np.unique(train_labels)
        }
        covariance_rank_ceiling = min(moment_fit_class_counts.values()) - 1
        maximal_surrogate = fit_representation_surrogate(
            train_rep[:fit_count],
            train_labels[:fit_count],
            ranks[-1],
            config.seed + 10_000 * cut,
            covariance_shrinkage=0.0,
        )
        reference_records: list[dict[str, object]] = []
        rank_results: list[dict[str, object]] = []
        for rank in ranks:
            print(f"[cut={cut}] truncate maximal PCA to rank={rank}", flush=True)
            surrogate = truncate_representation_surrogate(
                maximal_surrogate,
                train_rep,
                train_labels,
                rank,
                covariance_shrinkage=0.0,
            )
            coverage = pca_coverage_diagnostics(
                surrogate, test_rep, test_labels
            )
            records: list[dict[str, object]] = []
            diagnostics: list[dict[str, object]] = []
            gaussian_surrogates: list[
                tuple[str, RepresentationSurrogate]
            ] = []
            gaussian_provenance: list[dict[str, object]] = []
            for shrinkage in config.gaussian_covariance_shrinkages:
                gaussian_surrogate = (
                    surrogate
                    if shrinkage == 0
                    else truncate_representation_surrogate(
                        maximal_surrogate,
                        train_rep,
                        train_labels,
                        rank,
                        covariance_shrinkage=shrinkage,
                    )
                )
                gaussian_name = gaussian_distribution_name(shrinkage)
                gaussian_surrogates.append(
                    (gaussian_name, gaussian_surrogate)
                )
                gaussian_provenance.append(
                    {
                        "distribution": gaussian_name,
                        **gaussian_surrogate.covariance_provenance(),
                    }
                )
            projected_test = project_representation(surrogate, test_rep)
            step_zero_projection = _step_zero_projection_diagnostics(
                model,
                cut,
                test_rep,
                projected_test,
                test_labels,
                batch_size=config.batch_size,
                device=device,
            )
            for draw in range(config.surrogate_draws):
                common_shuffle_seed = condition_shuffle_seed(
                    config.seed, cut, draw
                )
                evaluation_sets: dict[str, np.ndarray] = {"true": test_rep}
                if not config.true_eval_only:
                    evaluation_sets["projected_true"] = projected_test

                for gaussian_name, gaussian_surrogate in gaussian_surrogates:
                    gaussian_test = sample_representation_surrogate(
                        gaussian_surrogate,
                        test_labels,
                        "gaussian",
                        config.seed + 100_000 * draw + 1_000 * cut + 3,
                        paired_noise_rank=ranks[-1],
                    )
                    diagnostics.append(
                        {
                            "draw": draw,
                            "distribution": gaussian_name,
                            **moment_diagnostics(
                                test_rep,
                                gaussian_test,
                                test_labels,
                                gaussian_surrogate,
                            ),
                        }
                    )
                    if not config.true_eval_only:
                        evaluation_sets[gaussian_name] = gaussian_test

                mean_names: list[tuple[str, float]] = []
                for radius in config.mean_noise_radii:
                    name = f"mean_r{radius:g}"
                    mean_names.append((name, radius))
                    mean_test = sample_representation_surrogate(
                        surrogate,
                        test_labels,
                        "mean",
                        config.seed
                        + 100_000 * draw
                        + 1_000 * cut
                        + 3,
                        mean_noise_radius=radius,
                        paired_noise_rank=ranks[-1],
                    )
                    diagnostics.append(
                        {
                            "draw": draw,
                            "distribution": name,
                            "mean_noise_radius": radius,
                            "isotropic_covariance_trace_ratio": radius**2,
                            **moment_diagnostics(
                                test_rep, mean_test, test_labels, surrogate
                            ),
                        }
                    )
                    if not config.true_eval_only:
                        evaluation_sets[name] = mean_test

                # The true-trained row belongs to each rank because its
                # deployment evaluations are rank- and draw-specific.
                true_records = _relax_suffix(
                    model,
                    cut,
                    train_rep,
                    train_labels,
                    evaluation_sets,
                    test_labels,
                    distribution="true",
                    draw=draw,
                    batch_size=config.batch_size,
                    relax_epochs=config.relax_epochs,
                    relax_learning_rate=config.relax_learning_rate,
                    shuffle_seed=common_shuffle_seed,
                    device=device,
                    suffix_initialization=config.suffix_initialization,
                    initialization_seed=(
                        config.seed + 9_000_000 + 1_000 * cut + draw
                    ),
                    learning_rate_regime=config.learning_rate_regime,
                )
                records.extend(true_records)
                true_update_target = float(
                    true_records[0]["matched_first_step_update_norm"]
                )

                # Shallow-cut banks are several gigabytes. Materialise and
                # release one training distribution at a time while retaining
                # the smaller evaluation banks needed for the cross-evaluation.
                projected_train = project_representation(surrogate, train_rep)
                records.extend(
                    _relax_suffix(
                        model,
                        cut,
                        projected_train,
                        train_labels,
                        evaluation_sets,
                        test_labels,
                        distribution="projected_true",
                        draw=draw,
                        batch_size=config.batch_size,
                        relax_epochs=config.relax_epochs,
                        relax_learning_rate=config.relax_learning_rate,
                        shuffle_seed=common_shuffle_seed,
                        device=device,
                        suffix_initialization=config.suffix_initialization,
                        initialization_seed=(
                            config.seed + 9_000_000 + 1_000 * cut + draw
                        ),
                        learning_rate_regime=config.learning_rate_regime,
                        target_first_step_update_norm=true_update_target,
                    )
                )
                del projected_train

                for gaussian_name, gaussian_surrogate in gaussian_surrogates:
                    gaussian_train = sample_representation_surrogate(
                        gaussian_surrogate,
                        train_labels,
                        "gaussian",
                        config.seed + 100_000 * draw + 1_000 * cut + 2,
                        paired_noise_rank=ranks[-1],
                    )
                    records.extend(
                        _relax_suffix(
                            model,
                            cut,
                            gaussian_train,
                            train_labels,
                            evaluation_sets,
                            test_labels,
                            distribution=gaussian_name,
                            draw=draw,
                            batch_size=config.batch_size,
                            relax_epochs=config.relax_epochs,
                            relax_learning_rate=config.relax_learning_rate,
                            shuffle_seed=common_shuffle_seed,
                            device=device,
                            suffix_initialization=config.suffix_initialization,
                            initialization_seed=(
                                config.seed + 9_000_000 + 1_000 * cut + draw
                            ),
                            learning_rate_regime=config.learning_rate_regime,
                            target_first_step_update_norm=true_update_target,
                        )
                    )
                    del gaussian_train

                for name, radius in mean_names:
                    mean_train = sample_representation_surrogate(
                        surrogate,
                        train_labels,
                        "mean",
                        config.seed
                        + 100_000 * draw
                        + 1_000 * cut
                        + 2,
                        mean_noise_radius=radius,
                        paired_noise_rank=ranks[-1],
                    )
                    records.extend(
                        _relax_suffix(
                            model,
                            cut,
                            mean_train,
                            train_labels,
                            evaluation_sets,
                            test_labels,
                            distribution=name,
                            draw=draw,
                            batch_size=config.batch_size,
                            relax_epochs=config.relax_epochs,
                            relax_learning_rate=config.relax_learning_rate,
                            shuffle_seed=common_shuffle_seed,
                            device=device,
                            suffix_initialization=config.suffix_initialization,
                            initialization_seed=(
                                config.seed + 9_000_000 + 1_000 * cut + draw
                            ),
                            learning_rate_regime=config.learning_rate_regime,
                            target_first_step_update_norm=true_update_target,
                        )
                    )
                    del mean_train

            if not reference_records:
                reference_records = [
                    dict(row)
                    for row in records
                    if row["train_distribution"] == "true"
                    and row["eval_distribution"] == "true"
                ]

            rank_results.append(
                {
                    "pca_rank": rank,
                    "held_out_explained_variance_fraction": coverage[
                        "total_variance_fraction"
                    ],
                    "held_out_coverage": coverage,
                    "pooled_within_class_variance_per_pca_coordinate": (
                        surrogate.pooled_variance
                    ),
                    "trace_matched_isotropic_covariance_trace": (
                        rank * surrogate.pooled_variance
                    ),
                    "gaussian_covariance_estimators": gaussian_provenance,
                    "empirical_class_covariance_rank_ceiling": (
                        covariance_rank_ceiling
                    ),
                    "rank_exceeds_empirical_class_covariance_ceiling": (
                        rank > covariance_rank_ceiling
                    ),
                    "evaluation_protocol": (
                        "true-eval-only targeted sensitivity"
                        if config.true_eval_only
                        else "full train-by-evaluation distribution matrix"
                    ),
                    "step_zero_true_vs_projected": step_zero_projection,
                    "records": records,
                    "moment_diagnostics": diagnostics,
                }
            )
            del projected_test
        slices.append(
            {
                "cut": cut,
                "representation_shape": list(train_rep.shape[1:]),
                "native_dimension": int(train_rep[0].size),
                "pca_fit_count": fit_count,
                "pca_fit_class_counts": pca_fit_class_counts,
                "moment_fit_count": len(train_rep),
                "moment_fit_class_counts": moment_fit_class_counts,
                "reference_records": reference_records,
                "rank_results": rank_results,
            }
        )

    artifact = {
        "schema_version": 2,
        "experiment": "lw_post_cnn_suffix_statistics",
        "status": "MOCKUP / PIPELINE SMOKE TEST" if config.fake_data else "MEASURED",
        "architecture": "four-block residual CNN with GroupNorm",
        "checkpoint": {
            "epoch": config.checkpoint_epoch,
            "path": str(checkpoint),
            "sha256": sha256(checkpoint),
        },
        "mean_noise_definition": (
            "At radius 1, the isotropic covariance has the same total trace as "
            "the pooled average within-class covariance in the fitted PCA space. "
            "Radius 0 is exact class-centroid replay; radius r multiplies the "
            "reference RMS radius by r and its covariance trace by r^2."
        ),
        "cross_evaluation_definition": (
            "The primary records form a rank- and draw-matched Cartesian matrix: "
            "each suffix trained on true, projected-true, Gaussian, or "
            "mean-plus-noise activations is evaluated with cross-entropy and "
            "accuracy on every corresponding held-out activation bank. The "
            "true-eval-only switch is a targeted sensitivity shortcut."
        ),
        "pca_protocol": (
            "Each cell fits one maximal PCA basis and lower-rank controls use "
            "nested leading-coordinate truncations of that same basis. "
            "Within a cut and draw, every training distribution uses the same "
            "explicit minibatch-label permutation, and the recorded gradient "
            "diagnostic uses exactly the first optimizer batch from that "
            "permutation. Every mean-noise radius scales the same sampled "
            "standard-normal cloud. Nested PCA ranks also use leading coordinates "
            "from the same maximal standard-normal banks. Projected-true is a "
            "matched in-subspace train/deploy control: it measures the combined "
            "effect of projection on the representation presented to the fixed "
            "suffix, rather than a pure reconstruction-loss quantity. Exact "
            "empirical-covariance Gaussian replay is primary; declared spherical "
            "shrinkage values are separate sensitivity conditions. Covariance "
            "factorization and any numerical PSD adjustment are recorded. "
            "The PCA basis is fitted on the declared PCA-fit subset, but class "
            "means and covariances are estimated from the entire training "
            "activation bank after projection into that fixed basis. "
            "Variance coverage is evaluated on the held-out CIFAR-10 test "
            "activations, not on the rows used to fit PCA. By default every "
            "trained suffix is evaluated on every rank- and draw-matched test "
            "distribution; true-eval-only is reserved for targeted sensitivity "
            "runs. Warm-started suffix relaxation is primary; any reinitialized "
            "suffix run is an explicitly labelled retained-knowledge control, "
            "with initialization held fixed across distributions within a draw. "
            "Fixed learning rate is primary. A separately labelled matched-update "
            "regime rescales each condition's learning rate so its measured first "
            "SGD update norm matches the paired true-replay update; the multiplier "
            "and effective learning rate are recorded on every row."
        ),
        "config": asdict(config),
        "dataset": dataset_fingerprints,
        "device": str(device),
        "provenance": runtime_provenance(),
        "slices": slices,
        "runtime_seconds": time.time() - started,
    }
    path = output / "post_statistics.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] {path}", flush=True)
    return path


def parse_args() -> PostStatisticsConfig:
    parser = argparse.ArgumentParser(
        description="Run post-facing CNN PCA and mean-noise controls from one checkpoint."
    )
    parser.add_argument("--output", default=PostStatisticsConfig.output)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-epoch", type=int, required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--data-backend",
        choices=("torchvision", "parquet"),
        default=PostStatisticsConfig.data_backend,
    )
    parser.add_argument(
        "--suffix-initialization",
        choices=("warm", "reinitialized"),
        default="warm",
    )
    parser.add_argument(
        "--learning-rate-regime",
        choices=("fixed_lr", "match_true_initial_update"),
        default="fixed_lr",
    )
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=50000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument("--cuts", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument(
        "--widths", type=int, nargs="+", default=list(PostStatisticsConfig.widths)
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--pca-fit-size", type=int, default=10000)
    parser.add_argument("--pca-ranks", type=int, nargs="+", default=[512])
    parser.add_argument(
        "--gaussian-covariance-shrinkages",
        type=float,
        nargs="+",
        default=[0.0],
    )
    parser.add_argument(
        "--mean-noise-radii", type=float, nargs="+", default=[1.0]
    )
    parser.add_argument("--surrogate-draws", type=int, default=1)
    parser.add_argument(
        "--true-eval-only",
        action="store_true",
        help=(
            "Evaluate only on true activations; intended for targeted "
            "sensitivity runs, not the primary cross-evaluation matrix."
        ),
    )
    parser.add_argument("--relax-epochs", type=int, default=10)
    parser.add_argument("--relax-learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return PostStatisticsConfig(
        output=args.output,
        checkpoint=args.checkpoint,
        checkpoint_epoch=args.checkpoint_epoch,
        data_root=args.data_root,
        data_backend=args.data_backend,
        fake_data=args.fake_data,
        train_size=args.train_size,
        test_size=args.test_size,
        cuts=tuple(args.cuts),
        widths=tuple(args.widths),
        batch_size=args.batch_size,
        pca_fit_size=args.pca_fit_size,
        pca_ranks=tuple(args.pca_ranks),
        gaussian_covariance_shrinkages=tuple(
            args.gaussian_covariance_shrinkages
        ),
        mean_noise_radii=tuple(args.mean_noise_radii),
        surrogate_draws=args.surrogate_draws,
        true_eval_only=args.true_eval_only,
        suffix_initialization=args.suffix_initialization,
        learning_rate_regime=args.learning_rate_regime,
        relax_epochs=args.relax_epochs,
        relax_learning_rate=args.relax_learning_rate,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
