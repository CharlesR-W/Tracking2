from __future__ import annotations

from dataclasses import dataclass, field

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
    covariance_shrinkage: float = 0.05
    class_factorization_methods: tuple[str, ...] = ()
    class_factor_jitters: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    class_clipped_negative_eigenvalue_counts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )
    class_max_negative_eigenvalue_magnitudes: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )

    def covariance_provenance(self) -> dict[str, object]:
        """Describe the covariance target actually used by Gaussian replay."""
        class_count = len(self.class_covariances)
        metadata_complete = (
            len(self.class_factorization_methods) == class_count
            and len(self.class_factor_jitters) == class_count
            and len(
                self.class_clipped_negative_eigenvalue_counts
            ) == class_count
            and len(
                self.class_max_negative_eigenvalue_magnitudes
            ) == class_count
        )
        clipped_count = int(
            np.sum(
                self.class_clipped_negative_eigenvalue_counts,
                dtype=np.int64,
            )
        )
        maximum_jitter = float(
            np.max(self.class_factor_jitters, initial=0.0)
        )
        maximum_clip = float(
            np.max(
                self.class_max_negative_eigenvalue_magnitudes,
                initial=0.0,
            )
        )
        exact_empirical = (
            metadata_complete
            and self.covariance_shrinkage == 0.0
            and maximum_jitter == 0.0
            and clipped_count == 0
        )
        numerical_adjustment = maximum_jitter > 0 or clipped_count > 0
        if not metadata_complete:
            target = "covariance_provenance_unavailable"
        elif self.covariance_shrinkage > 0 and numerical_adjustment:
            target = "shrunk_empirical_covariance_with_numerical_adjustment"
        elif self.covariance_shrinkage > 0:
            target = "shrunk_empirical_covariance"
        elif maximum_jitter > 0:
            target = "empirical_covariance_with_diagonal_jitter"
        elif clipped_count:
            target = "empirical_covariance_after_psd_projection"
        else:
            target = "exact_empirical_covariance"
        return {
            "covariance_target": target,
            "requested_covariance_shrinkage": self.covariance_shrinkage,
            "exact_empirical_covariance": exact_empirical,
            "covariance_metadata_complete": metadata_complete,
            "factorization_methods": list(
                self.class_factorization_methods
            ),
            "per_class_factor_jitter": self.class_factor_jitters.tolist(),
            "maximum_factor_jitter": maximum_jitter,
            "per_class_clipped_negative_eigenvalue_count": (
                self.class_clipped_negative_eigenvalue_counts.tolist()
            ),
            "total_clipped_negative_eigenvalue_count": clipped_count,
            "maximum_clipped_negative_eigenvalue_magnitude": maximum_clip,
        }


def _factor_covariance(
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float, int, float]:
    """Factor a covariance without silently changing its sampling target.

    Positive-definite matrices use a zero-jitter Cholesky factor. Empirical
    covariances can be positive semidefinite (for example when rank exceeds
    the per-class sample count), so those use an eigendecomposition. Any
    negative eigenvalues introduced by finite-precision arithmetic are
    explicitly clipped and reported instead of being hidden behind an
    unconditional diagonal jitter.
    """
    symmetric = np.asarray(
        (covariance + covariance.T) / 2, dtype=np.float64
    )
    try:
        factor = np.linalg.cholesky(symmetric)
        return factor, symmetric, "cholesky", 0.0, 0, 0.0
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        negative = eigenvalues < 0
        clipped_count = int(np.count_nonzero(negative))
        maximum_clip = (
            float(-np.min(eigenvalues[negative]))
            if clipped_count
            else 0.0
        )
        clipped = np.maximum(eigenvalues, 0.0)
        factor = eigenvectors * np.sqrt(clipped)[None, :]
        if clipped_count:
            actual_covariance = factor @ factor.T
            method = "eigh_psd_projection"
        else:
            actual_covariance = symmetric
            method = "eigh_semidefinite"
        return (
            factor,
            actual_covariance,
            method,
            0.0,
            clipped_count,
            maximum_clip,
        )


def _surrogate_from_coordinates(
    *,
    z: np.ndarray,
    labels: np.ndarray,
    pca_mean: np.ndarray,
    components: np.ndarray,
    covariance_shrinkage: float,
    representation_shape: tuple[int, ...],
) -> RepresentationSurrogate:
    if not 0 <= covariance_shrinkage <= 1:
        raise ValueError("covariance_shrinkage must be in [0, 1]")
    classes = np.arange(int(labels.max()) + 1)
    global_mean = z.mean(0)
    global_covariance = np.atleast_2d(
        np.cov(z, rowvar=False)
    ).astype(np.float64)
    means, covariances, factors, within_class_variances = [], [], [], []
    factorization_methods: list[str] = []
    factor_jitters: list[float] = []
    clipped_eigenvalue_counts: list[int] = []
    maximum_clip_magnitudes: list[float] = []
    for class_id in classes:
        z_class = z[labels == class_id]
        means.append(z_class.mean(0) if len(z_class) else global_mean)
        covariance = (
            np.atleast_2d(np.cov(z_class, rowvar=False)).astype(np.float64)
            if len(z_class) >= 2
            else global_covariance.copy()
        )
        scale = float(np.trace(covariance) / covariance.shape[0])
        covariance = (1 - covariance_shrinkage) * covariance
        covariance += (
            covariance_shrinkage
            * scale
            * np.eye(covariance.shape[0], dtype=np.float64)
        )
        (
            factor,
            actual_covariance,
            factorization_method,
            factor_jitter,
            clipped_eigenvalue_count,
            maximum_clip_magnitude,
        ) = _factor_covariance(covariance)
        if len(z_class):
            # np.cov uses the unbiased n-1 denominator. Reusing the trace of
            # the actual Gaussian sampling target makes radius-1 isotropic
            # replay trace-matched even if a PSD projection was required.
            within_class_variances.append(
                float(
                    np.trace(actual_covariance)
                    / actual_covariance.shape[0]
                )
            )
        covariances.append(actual_covariance)
        factors.append(factor)
        factorization_methods.append(factorization_method)
        factor_jitters.append(factor_jitter)
        clipped_eigenvalue_counts.append(clipped_eigenvalue_count)
        maximum_clip_magnitudes.append(maximum_clip_magnitude)
    covariance_array = np.asarray(covariances, dtype=np.float32)
    factor_array = np.asarray(factors, dtype=np.float32)
    return RepresentationSurrogate(
        pca_mean=pca_mean.astype(np.float32),
        components=components.astype(np.float32),
        class_means=np.asarray(means, dtype=np.float32),
        class_covariances=covariance_array,
        class_factors=factor_array,
        pooled_variance=float(np.mean(within_class_variances)),
        representation_shape=representation_shape,
        covariance_shrinkage=float(covariance_shrinkage),
        class_factorization_methods=tuple(factorization_methods),
        class_factor_jitters=np.asarray(
            factor_jitters, dtype=np.float64
        ),
        class_clipped_negative_eigenvalue_counts=np.asarray(
            clipped_eigenvalue_counts, dtype=np.int64
        ),
        class_max_negative_eigenvalue_magnitudes=np.asarray(
            maximum_clip_magnitudes, dtype=np.float64
        )
    )


def fit_representation_surrogate(
    representations: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    seed: int,
    covariance_shrinkage: float = 0.05,
    *,
    moment_representations: np.ndarray | None = None,
    moment_labels: np.ndarray | None = None,
) -> RepresentationSurrogate:
    if (moment_representations is None) != (moment_labels is None):
        raise ValueError(
            "moment_representations and moment_labels must be supplied together"
        )
    flat = representations.reshape(len(representations), -1).astype(
        np.float32, copy=False
    )
    if not 1 <= n_components < len(flat):
        raise ValueError("n_components must be positive and smaller than the sample count")
    pca = PCA(n_components=min(n_components, flat.shape[1]), svd_solver="randomized", random_state=seed)
    if moment_representations is None:
        coordinate_labels = labels
        z = pca.fit_transform(flat)
        representation_shape = tuple(representations.shape[1:])
    else:
        pca.fit(flat)
        moment_flat = moment_representations.reshape(
            len(moment_representations), -1
        ).astype(np.float32, copy=False)
        if moment_flat.shape[1] != flat.shape[1]:
            raise ValueError(
                "PCA-fit and moment-fit representations must have the same "
                "flattened dimension"
            )
        if len(moment_flat) != len(moment_labels):
            raise ValueError(
                "moment_representations and moment_labels must have equal length"
            )
        z = np.empty(
            (len(moment_flat), len(pca.components_)), dtype=np.float32
        )
        for start in range(0, len(moment_flat), 512):
            stop = min(start + 512, len(moment_flat))
            z[start:stop] = (
                moment_flat[start:stop] - pca.mean_
            ) @ pca.components_.T
        coordinate_labels = moment_labels
        representation_shape = tuple(moment_representations.shape[1:])
    return _surrogate_from_coordinates(
        z=z,
        labels=coordinate_labels,
        pca_mean=pca.mean_,
        components=pca.components_,
        covariance_shrinkage=covariance_shrinkage,
        representation_shape=representation_shape,
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
        n_left, n_right = max(len(left) - 1, 1), max(len(right) - 1, 1)
        feature_count = left_centered.shape[1]
        if feature_count <= min(len(left), len(right)):
            # In the bounded PCA controls, r is usually much smaller than the
            # per-class held-out count. Materialising r x r covariances is then
            # exactly equivalent to the Gram identity and far cheaper than
            # constructing three n x n matrices.
            left_covariance = left_centered.T @ left_centered / n_left
            right_covariance = right_centered.T @ right_centered / n_right
            difference = left_covariance - right_covariance
            error_sq = np.sum(difference * difference, dtype=np.float64)
            baseline_sq = np.sum(
                left_covariance * left_covariance, dtype=np.float64
            )
        else:
            # For a very wide native representation, use the dual Gram identity
            # so the diagnostic never materialises a D x D covariance.
            cross = left_centered @ right_centered.T
            left_gram = left_centered @ left_centered.T
            right_gram = right_centered @ right_centered.T
            error_sq = (
                np.sum(left_gram * left_gram, dtype=np.float64) / n_left**2
                + np.sum(right_gram * right_gram, dtype=np.float64)
                / n_right**2
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
