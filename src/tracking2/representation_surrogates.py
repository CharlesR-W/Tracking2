from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA


@dataclass
class RepresentationSurrogate:
    """Class-conditional statistics in a declared PCA subspace of phi(x)."""

    pca_mean: np.ndarray
    components: np.ndarray
    class_means: np.ndarray
    class_covariances: np.ndarray
    class_factors: np.ndarray
    pooled_variance: float
    representation_shape: tuple[int, ...]


def _surrogate_from_coordinates(
    *,
    z: np.ndarray,
    labels: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
    covariance_shrinkage: float,
    representation_shape: tuple[int, ...],
) -> RepresentationSurrogate:
    classes = np.arange(int(labels.max()) + 1)
    global_mean = z.mean(0)
    global_covariance = np.atleast_2d(np.cov(z, rowvar=False)).astype(np.float32)
    means, covariances, within_class_variances = [], [], []
    for class_id in classes:
        z_class = z[labels == class_id]
        means.append(z_class.mean(0) if len(z_class) else global_mean)
        covariance = (
            np.atleast_2d(np.cov(z_class, rowvar=False)).astype(np.float32)
            if len(z_class) >= 2
            else global_covariance.copy()
        )
        scale = float(np.trace(covariance) / covariance.shape[0])
        if len(z_class):
            # np.cov uses the unbiased n-1 denominator. Reusing its per-coordinate
            # trace makes radius-1 isotropic replay exactly trace-matched to the
            # average fitted within-class covariance (shrinkage preserves trace).
            within_class_variances.append(scale)
        covariance = (1 - covariance_shrinkage) * covariance
        covariance += (
            covariance_shrinkage
            * scale
            * np.eye(covariance.shape[0], dtype=np.float32)
        )
        covariances.append(covariance)
    covariance_array = np.asarray(covariances, dtype=np.float32)
    factors = np.asarray(
        [
            np.linalg.cholesky(
                covariance
                + 1e-6 * np.eye(covariance.shape[0], dtype=np.float32)
            )
            for covariance in covariance_array
        ],
        dtype=np.float32,
    )
    return RepresentationSurrogate(
        pca_mean=pca_mean.astype(np.float32),
        components=components.astype(np.float32),
        class_means=np.asarray(means, dtype=np.float32),
        class_covariances=covariance_array,
        class_factors=factors,
        pooled_variance=float(np.mean(within_class_variances)),
        representation_shape=representation_shape,
    )


def fit_representation_surrogate(
    representations: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    seed: int,
    covariance_shrinkage: float = 0.05,
) -> RepresentationSurrogate:
    flat = representations.reshape(len(representations), -1).astype(
        np.float32, copy=False
    )
    if not 1 <= n_components < len(flat):
        raise ValueError("n_components must be positive and smaller than the sample count")
    pca = PCA(n_components=min(n_components, flat.shape[1]), svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(flat)
    return _surrogate_from_coordinates(
        z=z,
        labels=labels,
        pca_mean=pca.mean_,
        components=pca.components_,
        covariance_shrinkage=covariance_shrinkage,
        representation_shape=tuple(representations.shape[1:]),
    )


def truncate_representation_surrogate(
    model: RepresentationSurrogate,
    representations: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    *,
    covariance_shrinkage: float = 0.05,
) -> RepresentationSurrogate:
    """Reuse the leading coordinates of one maximal PCA fit at a smaller rank."""
    if not 1 <= n_components <= model.components.shape[0]:
        raise ValueError("n_components must select a nonempty prefix of the PCA basis")
    flat = representations.reshape(len(representations), -1).astype(
        np.float32, copy=False
    )
    components = model.components[:n_components]
    z = np.empty((len(flat), n_components), dtype=np.float32)
    for start in range(0, len(flat), 512):
        stop = min(start + 512, len(flat))
        z[start:stop] = (
            flat[start:stop] - model.pca_mean
        ) @ components.T
    return _surrogate_from_coordinates(
        z=z,
        labels=labels,
        pca_mean=model.pca_mean,
        components=components,
        covariance_shrinkage=covariance_shrinkage,
        representation_shape=tuple(representations.shape[1:]),
    )


def sample_representation_surrogate(
    model: RepresentationSurrogate,
    labels: np.ndarray,
    kind: str,
    seed: int,
    *,
    mean_noise_radius: float = 1.0,
    paired_noise_rank: int | None = None,
) -> np.ndarray:
    if kind not in {"mean", "gaussian"}:
        raise ValueError(f"Unknown representation surrogate: {kind}")
    if mean_noise_radius < 0:
        raise ValueError("mean_noise_radius must be non-negative")
    rng = np.random.default_rng(seed)
    dimension = model.components.shape[0]
    noise_rank = paired_noise_rank if paired_noise_rank is not None else dimension
    if noise_rank < dimension:
        raise ValueError("paired_noise_rank must cover the fitted PCA rank")
    # Generating one row-aligned maximal bank makes leading coordinates
    # identical across a nested PCA-rank sweep. Lower ranks take a prefix.
    noise_bank = rng.normal(size=(len(labels), noise_rank)).astype(np.float32)
    z = np.empty((len(labels), dimension), dtype=np.float32)
    for class_id in np.unique(labels):
        mask = labels == class_id
        noise = noise_bank[mask, :dimension]
        if kind == "mean":
            z[mask] = (
                model.class_means[class_id]
                + mean_noise_radius * np.sqrt(model.pooled_variance) * noise
            )
        else:
            z[mask] = model.class_means[class_id] + noise @ model.class_factors[class_id].T
    flat = z @ model.components
    flat += model.pca_mean
    return flat.reshape((len(labels),) + model.representation_shape).astype(np.float32)


def project_representation(
    model: RepresentationSurrogate,
    representations: np.ndarray,
) -> np.ndarray:
    """Project real activations through the fitted PCA reconstruction map."""
    flat = representations.reshape(len(representations), -1).astype(
        np.float32, copy=False
    )
    reconstructed = np.empty_like(flat)
    for start in range(0, len(flat), 512):
        stop = min(start + 512, len(flat))
        z = (flat[start:stop] - model.pca_mean) @ model.components.T
        reconstructed[start:stop] = z @ model.components
        reconstructed[start:stop] += model.pca_mean
    return reconstructed.reshape(
        (len(representations),) + model.representation_shape
    ).astype(np.float32)


def pca_coverage_diagnostics(
    model: RepresentationSurrogate,
    representations: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """Variance fractions retained by PCA on an evaluation bank.

    The shallow CNN bank is several gigabytes, so the total and within-class
    sums are accumulated in chunks instead of materialising float64 centered
    copies of the entire bank.
    """
    flat = representations.reshape(len(representations), -1).astype(
        np.float32, copy=False
    )
    components = model.components.astype(np.float32, copy=False)
    classes = np.unique(labels)
    class_means: dict[int, np.ndarray] = {}
    between_weights: list[int] = []
    for class_id in classes:
        mask = labels == class_id
        between_weights.append(int(mask.sum()))
        class_means[int(class_id)] = flat[mask].mean(
            axis=0, dtype=np.float64
        ).astype(np.float32)
    weights = np.asarray(between_weights, dtype=np.float64)
    global_mean = np.average(
        np.stack([class_means[int(class_id)] for class_id in classes]),
        axis=0,
        weights=weights,
    ).astype(np.float32)

    total_numerator = total_denominator = 0.0
    within_numerator = within_denominator = 0.0
    chunk_size = 512
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        values = flat[start:stop]
        global_centered = values - global_mean
        projected_global = global_centered @ components.T
        total_numerator += float(
            np.sum(projected_global * projected_global, dtype=np.float64)
        )
        total_denominator += float(
            np.sum(global_centered * global_centered, dtype=np.float64)
        )

        means = np.stack(
            [class_means[int(class_id)] for class_id in labels[start:stop]]
        )
        within_centered = values - means
        projected_within = within_centered @ components.T
        within_numerator += float(
            np.sum(projected_within * projected_within, dtype=np.float64)
        )
        within_denominator += float(
            np.sum(within_centered * within_centered, dtype=np.float64)
        )

    between = np.stack(
        [class_means[int(class_id)] - global_mean for class_id in classes]
    )
    projected_between = between @ components.T
    column_weights = weights[:, None]

    def fraction(numerator: float, denominator: float) -> float:
        return float(numerator / max(denominator, 1e-12))

    return {
        "total_variance_fraction": fraction(
            total_numerator, total_denominator
        ),
        "within_class_variance_fraction": fraction(
            within_numerator, within_denominator
        ),
        "between_class_mean_variance_fraction": fraction(
            np.sum(
                column_weights * projected_between**2, dtype=np.float64
            ),
            np.sum(column_weights * between**2, dtype=np.float64),
        ),
    }


def moment_diagnostics(
    real: np.ndarray,
    synthetic: np.ndarray,
    labels: np.ndarray,
    model: RepresentationSurrogate | None = None,
) -> dict[str, float | str]:
    """Held-out errors for the moments the surrogate actually targets.

    When a fitted surrogate is supplied, diagnostics are computed in its PCA
    coordinates.  Comparing full native-space covariances would incorrectly
    count every deliberately discarded PCA direction as a matching failure.
    The mean error is normalized by within-class RMS variation rather than by
    the norm of the class mean, which is unstable when class signal is weak.
    """
    real_flat = real.reshape(len(real), -1).astype(np.float32, copy=False)
    synthetic_flat = synthetic.reshape(len(synthetic), -1).astype(
        np.float32, copy=False
    )
    diagnostic_space = "native"
    if model is not None:
        center = model.pca_mean.astype(np.float64)
        components = model.components.astype(np.float64)

        def to_coordinates(values: np.ndarray) -> np.ndarray:
            coordinates = np.empty(
                (len(values), len(components)), dtype=np.float64
            )
            for start in range(0, len(values), 512):
                stop = min(start + 512, len(values))
                coordinates[start:stop] = (
                    values[start:stop].astype(np.float64) - center
                ) @ components.T
            return coordinates

        real_flat = to_coordinates(real_flat)
        synthetic_flat = to_coordinates(synthetic_flat)
        diagnostic_space = "fitted PCA subspace"
    mean_errors, covariance_errors = [], []
    for class_id in np.unique(labels):
        left, right = real_flat[labels == class_id], synthetic_flat[labels == class_id]
        left_mean, right_mean = left.mean(0), right.mean(0)
        left_centered, right_centered = left - left_mean, right - right_mean
        within_scale = np.sqrt(np.mean(np.sum(left_centered**2, axis=1)))
        mean_errors.append(np.linalg.norm(left_mean - right_mean) / max(within_scale, 1e-12))
        # Frobenius covariance error without materialising a D x D matrix.
        cross = left_centered @ right_centered.T
        left_gram, right_gram = left_centered @ left_centered.T, right_centered @ right_centered.T
        n_left, n_right = max(len(left) - 1, 1), max(len(right) - 1, 1)
        error_sq = (
            np.sum(left_gram * left_gram, dtype=np.float64) / n_left**2
            + np.sum(right_gram * right_gram, dtype=np.float64) / n_right**2
            - 2
            * np.sum(cross * cross, dtype=np.float64)
            / (n_left * n_right)
        )
        baseline_sq = (
            np.sum(left_gram * left_gram, dtype=np.float64) / n_left**2
        )
        covariance_errors.append(np.sqrt(max(error_sq, 0)) / max(np.sqrt(baseline_sq), 1e-12))
    return {
        "class_mean_relative_error": float(np.mean(mean_errors)),
        "class_covariance_relative_error": float(np.mean(covariance_errors)),
        "diagnostic_space": diagnostic_space,
        "mean_normalization": "within-class RMS radius",
    }
