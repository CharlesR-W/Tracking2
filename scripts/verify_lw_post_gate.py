from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from verify_lw_post_artifacts import (
    load,
    require_complete_statistics,
)
from tracking2.cnn_checkpoints import sha256


def endpoint_loss(
    rank_result: dict,
    *,
    train_distribution: str,
    relax_epoch: int,
    draw: int = 0,
) -> float:
    matches = [
        row
        for row in rank_result["records"]
        if row["draw"] == draw
        and row["train_distribution"] == train_distribution
        and row["eval_distribution"] == "true"
        and row["relax_epoch"] == relax_epoch
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one endpoint for {train_distribution}, found {len(matches)}"
        )
    return float(matches[0]["loss"])


def source_identity(training: dict) -> tuple[str | None, str | None]:
    provenance = training.get("provenance", {})
    revision = provenance.get("source_revision")
    archive = provenance.get("source_archive_sha256")
    return (
        revision
        if isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
        else None,
        archive
        if isinstance(archive, str)
        and re.fullmatch(r"[0-9a-f]{64}", archive)
        else None,
    )


def rank_summary(rank_result: dict, relax_epoch: int) -> dict[str, float]:
    true_loss = endpoint_loss(
        rank_result,
        train_distribution="true",
        relax_epoch=relax_epoch,
    )
    projected_loss = endpoint_loss(
        rank_result,
        train_distribution="projected_true",
        relax_epoch=relax_epoch,
    )
    gaussian_loss = endpoint_loss(
        rank_result,
        train_distribution="gaussian_empirical",
        relax_epoch=relax_epoch,
    )
    mean_loss = endpoint_loss(
        rank_result,
        train_distribution="mean_r1",
        relax_epoch=relax_epoch,
    )
    step_zero = rank_result["step_zero_true_vs_projected"]
    return {
        "rank": int(rank_result["pca_rank"]),
        "held_out_total_coverage": float(
            rank_result["held_out_coverage"]["total_variance_fraction"]
        ),
        "held_out_within_class_coverage": float(
            rank_result["held_out_coverage"][
                "within_class_variance_fraction"
            ]
        ),
        "step_zero_projection_loss_excess": float(
            step_zero["projected_true_loss"] - step_zero["true_loss"]
        ),
        "step_zero_predictive_kl": float(
            step_zero["true_to_projected_predictive_kl"]
        ),
        "projected_relaxation_excess": projected_loss - true_loss,
        "gaussian_vs_projected_excess": gaussian_loss - projected_loss,
        "gaussian_vs_true_excess": gaussian_loss - true_loss,
        "mean_vs_gaussian_excess": mean_loss - gaussian_loss,
    }


def evaluate_gate(rows: list[dict[str, float]]) -> list[str]:
    failures: list[str] = []
    highest = rows[-1]
    if abs(highest["step_zero_projection_loss_excess"]) > 0.05:
        failures.append("highest-rank update-0 projection CE excess exceeds 0.05 nat")
    if highest["step_zero_predictive_kl"] > 0.02:
        failures.append("highest-rank update-0 projection KL exceeds 0.02 nat")
    if abs(highest["projected_relaxation_excess"]) > 0.05:
        failures.append(
            "highest-rank projected-vs-true relaxation excess exceeds 0.05 nat"
        )
    if len(rows) >= 2:
        left = rows[-2]["gaussian_vs_projected_excess"]
        right = rows[-1]["gaussian_vs_projected_excess"]
        tolerance = max(0.05, 0.25 * max(abs(left), abs(right)))
        if abs(right - left) > tolerance:
            failures.append(
                "Gaussian-vs-projected contrast is not stable at the top two ranks"
            )
        if left * right < 0 and max(abs(left), abs(right)) > 0.02:
            failures.append(
                "Gaussian-vs-projected contrast reverses sign at the top two ranks"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the predeclared CNN PCA/covariance gate."
    )
    parser.add_argument("training_root", type=Path)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--relax-epochs", type=int, default=5)
    args = parser.parse_args()

    training_path = args.training_root / "training.json"
    training = load(training_path)
    dataset = training["dataset"]
    revision, archive = source_identity(training)
    checkpoint = args.training_root / "checkpoint_epoch30.pt"
    checkpoint_hash = sha256(checkpoint)

    summaries: dict[str, list[dict[str, float]]] = {}
    failures: list[str] = []
    for cut, ranks in ((1, [128, 512, 1024]), (4, [128, 512, 1024, 2048])):
        path = args.result_root / f"rank_gate_cut{cut}" / "post_statistics.json"
        artifact = load(path)
        require_complete_statistics(
            path,
            artifact,
            expected_epoch=30,
            expected_cuts=[cut],
            expected_ranks=ranks,
            expected_radii=[1.0],
            expected_shrinkages=[0.0],
            expected_draws=1,
            expected_true_eval_only=False,
            expected_relax_epochs=args.relax_epochs,
            checkpoint_hash=checkpoint_hash,
            dataset_fingerprints=dataset,
            source_revision=revision,
            source_archive_sha256=archive,
            expected_seed=args.seed,
        )
        rows = [
            rank_summary(rank_result, args.relax_epochs)
            for rank_result in artifact["slices"][0]["rank_results"]
        ]
        summaries[f"cut{cut}"] = rows
        failures.extend(f"cut{cut}: {failure}" for failure in evaluate_gate(rows))

    print(json.dumps({"summaries": summaries, "failures": failures}, indent=2))
    if failures:
        print(
            "PCA adequacy gate failed. Extend the rank/control battery or narrow "
            "the claim before launching the confirmatory grid."
        )
        return 2
    print("PCA adequacy gate passed for the declared practical-equivalence margins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
