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
    mean_noise_radii: tuple[float, ...] = (1.0,)
    surrogate_draws: int = 1
    relax_epochs: int = 10
    relax_learning_rate: float = 0.01
    seed: int = 0
    device: str = "auto"


def condition_shuffle_seed(seed: int, cut: int, draw: int) -> int:
    """Match minibatch order across every distribution in one comparison draw."""

    return seed + 100_000 * draw + 1_000 * cut


def _relax_suffix(
    model: SplitConvNet,
    cut: int,
    train_rep: np.ndarray,
    train_labels: np.ndarray,
    test_rep: np.ndarray,
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
    for parameter in candidate.prefix_parameters(cut):
        parameter.requires_grad_(False)
    optimizer = torch.optim.SGD(
        candidate.suffix_parameters(cut),
        lr=relax_learning_rate,
        momentum=0.9,
    )
    train_data = representation_loader(
        train_rep, train_labels, batch_size, shuffle_seed, True
    )
    test_data = representation_loader(
        test_rep, test_labels, batch_size, shuffle_seed, False
    )
    diagnostic_representation = torch.from_numpy(train_rep[:batch_size]).to(device)
    diagnostic_labels = torch.from_numpy(
        train_labels[:batch_size].astype(np.int64)
    ).to(device)
    candidate.train()
    diagnostic_loss = F.cross_entropy(
        candidate.forward_from(diagnostic_representation, cut),
        diagnostic_labels,
    )
    diagnostic_loss.backward()
    initial_gradient_norm = float(
        torch.sqrt(
            sum(
                torch.sum(parameter.grad.detach() ** 2)
                for parameter in candidate.suffix_parameters(cut)
                if parameter.grad is not None
            )
        )
    )
    optimizer.zero_grad(set_to_none=True)
    records: list[dict[str, object]] = []
    for relax_epoch in range(relax_epochs + 1):
        records.append(
            {
                "draw": draw,
                "train_distribution": distribution,
                "eval_distribution": "true",
                "relax_epoch": relax_epoch,
                "initial_training_loss": float(diagnostic_loss.detach()),
                "initial_gradient_norm": initial_gradient_norm,
                **evaluate_suffix(candidate, cut, test_data, device),
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
        reference_records: list[dict[str, object]] = []
        for draw in range(config.surrogate_draws):
            reference_records.extend(
                _relax_suffix(
                    model,
                    cut,
                    train_rep,
                    train_labels,
                    test_rep,
                    test_labels,
                    distribution="true",
                    draw=draw,
                    batch_size=config.batch_size,
                    relax_epochs=config.relax_epochs,
                    relax_learning_rate=config.relax_learning_rate,
                    shuffle_seed=condition_shuffle_seed(
                        config.seed, cut, draw
                    ),
                    device=device,
                )
            )

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
        )
        rank_results: list[dict[str, object]] = []
        for rank in ranks:
            print(f"[cut={cut}] truncate maximal PCA to rank={rank}", flush=True)
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

            projected_train = project_representation(surrogate, train_rep)
            for draw in range(config.surrogate_draws):
                common_shuffle_seed = condition_shuffle_seed(
                    config.seed, cut, draw
                )
                records.extend(
                    _relax_suffix(
                        model,
                        cut,
                        projected_train,
                        train_labels,
                        test_rep,
                        test_labels,
                        distribution="projected_true",
                        draw=draw,
                        batch_size=config.batch_size,
                        relax_epochs=config.relax_epochs,
                        relax_learning_rate=config.relax_learning_rate,
                        shuffle_seed=common_shuffle_seed,
                        device=device,
                    )
                )
            # Shallow-cut banks are several gigabytes. Release each replay bank
            # before materialising the next one so the measured grid fits on a
            # normal high-memory GPU host.
            del projected_train

            for draw in range(config.surrogate_draws):
                common_shuffle_seed = condition_shuffle_seed(
                    config.seed, cut, draw
                )
                gaussian_train = sample_representation_surrogate(
                    surrogate,
                    train_labels,
                    "gaussian",
                    config.seed + 100_000 * draw + 1_000 * cut + 2,
                    paired_noise_rank=ranks[-1],
                )
                gaussian_test = sample_representation_surrogate(
                    surrogate,
                    test_labels,
                    "gaussian",
                    config.seed + 100_000 * draw + 1_000 * cut + 3,
                    paired_noise_rank=ranks[-1],
                )
                diagnostics.append(
                    {
                        "draw": draw,
                        "distribution": "gaussian",
                        **moment_diagnostics(
                            test_rep, gaussian_test, test_labels, surrogate
                        ),
                    }
                )
                records.extend(
                    _relax_suffix(
                        model,
                        cut,
                        gaussian_train,
                        train_labels,
                        test_rep,
                        test_labels,
                        distribution="gaussian",
                        draw=draw,
                        batch_size=config.batch_size,
                        relax_epochs=config.relax_epochs,
                        relax_learning_rate=config.relax_learning_rate,
                        shuffle_seed=common_shuffle_seed,
                        device=device,
                    )
                )
                del gaussian_train, gaussian_test

                for radius in config.mean_noise_radii:
                    name = f"mean_r{radius:g}"
                    mean_train = sample_representation_surrogate(
                        surrogate,
                        train_labels,
                        "mean",
                        config.seed
                        + 100_000 * draw
                        + 1_000 * cut
                        + 10,
                        mean_noise_radius=radius,
                        paired_noise_rank=ranks[-1],
                    )
                    mean_test = sample_representation_surrogate(
                        surrogate,
                        test_labels,
                        "mean",
                        config.seed
                        + 100_000 * draw
                        + 1_000 * cut
                        + 100,
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
                    records.extend(
                        _relax_suffix(
                            model,
                            cut,
                            mean_train,
                            train_labels,
                            test_rep,
                            test_labels,
                            distribution=name,
                            draw=draw,
                            batch_size=config.batch_size,
                            relax_epochs=config.relax_epochs,
                            relax_learning_rate=config.relax_learning_rate,
                            shuffle_seed=common_shuffle_seed,
                            device=device,
                        )
                    )
                    del mean_train, mean_test

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
                    "covariance_shrinkage": 0.05,
                    "empirical_class_covariance_rank_ceiling": (
                        covariance_rank_ceiling
                    ),
                    "rank_exceeds_empirical_class_covariance_ceiling": (
                        rank > covariance_rank_ceiling
                    ),
                    "records": records,
                    "moment_diagnostics": diagnostics,
                }
            )
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
        "schema_version": 1,
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
        "pca_protocol": (
            "Each cell fits one maximal PCA basis and lower-rank controls use "
            "nested leading-coordinate truncations of that same basis. "
            "Within a cut and draw, every training distribution uses the same "
            "minibatch-label order; every mean-noise radius scales the same "
            "sampled standard-normal cloud. Nested PCA ranks also use leading "
            "coordinates from the same maximal standard-normal banks. "
            "Projected-true replay isolates reconstruction loss. Gaussian replay "
            "then replaces the retained empirical distribution by a "
            "class-conditional Gaussian with 5% spherical covariance shrinkage. "
            "The PCA basis is fitted on the declared PCA-fit subset, but class "
            "means and covariances are estimated from the entire training "
            "activation bank after projection into that fixed basis. "
            "Variance coverage is evaluated on the held-out CIFAR-10 test "
            "activations, not on the rows used to fit PCA."
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
        "--mean-noise-radii", type=float, nargs="+", default=[1.0]
    )
    parser.add_argument("--surrogate-draws", type=int, default=1)
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
        mean_noise_radii=tuple(args.mean_noise_radii),
        surrogate_draws=args.surrogate_draws,
        relax_epochs=args.relax_epochs,
        relax_learning_rate=args.relax_learning_rate,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
