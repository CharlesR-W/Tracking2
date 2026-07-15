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
    pooled_variance: float
    representation_shape: tuple[int, ...]


def fit_representation_surrogate(
    representations: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    seed: int,
    covariance_shrinkage: float = 0.05,
) -> RepresentationSurrogate:
    flat = representations.reshape(len(representations), -1).astype(np.float32)
    if not 1 <= n_components < len(flat):
        raise ValueError("n_components must be positive and smaller than the sample count")
    pca = PCA(n_components=min(n_components, flat.shape[1]), svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(flat)
    classes = np.arange(int(labels.max()) + 1)
    global_mean = z.mean(0)
    global_covariance = np.atleast_2d(np.cov(z, rowvar=False)).astype(np.float32)
    means, covariances, within_class_variances = [], [], []
    for class_id in classes:
        z_class = z[labels == class_id]
        means.append(z_class.mean(0) if len(z_class) else global_mean)
        covariance = (
            np.atleast_2d(np.cov(z_class, rowvar=False)).astype(np.float32)
            if len(z_class) >= 2 else global_covariance.copy()
        )
        scale = float(np.trace(covariance) / covariance.shape[0])
        covariance = (1 - covariance_shrinkage) * covariance
        covariance += covariance_shrinkage * scale * np.eye(covariance.shape[0], dtype=np.float32)
        covariances.append(covariance)
        if len(z_class):
            within_class_variances.append(float(np.mean((z_class - z_class.mean(0)) ** 2)))
    return RepresentationSurrogate(
        pca_mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        class_means=np.asarray(means, dtype=np.float32),
        class_covariances=np.asarray(covariances, dtype=np.float32),
        pooled_variance=float(np.mean(within_class_variances)),
        representation_shape=tuple(representations.shape[1:]),
    )


def sample_representation_surrogate(
    model: RepresentationSurrogate, labels: np.ndarray, kind: str, seed: int
) -> np.ndarray:
    if kind not in {"mean", "gaussian"}:
        raise ValueError(f"Unknown representation surrogate: {kind}")
    rng = np.random.default_rng(seed)
    dimension = model.components.shape[0]
    z = np.empty((len(labels), dimension), dtype=np.float32)
    for class_id in np.unique(labels):
        mask = labels == class_id
        count = int(mask.sum())
        if kind == "mean":
            noise = rng.normal(size=(count, dimension)).astype(np.float32)
            z[mask] = model.class_means[class_id] + np.sqrt(model.pooled_variance) * noise
        else:
            z[mask] = rng.multivariate_normal(
                model.class_means[class_id], model.class_covariances[class_id], size=count
            ).astype(np.float32)
    flat = z @ model.components + model.pca_mean
    return flat.reshape((len(labels),) + model.representation_shape).astype(np.float32)


def moment_diagnostics(
    real: np.ndarray, synthetic: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Held-out relative errors for the moments the pilot claims to match."""
    real_flat = real.reshape(len(real), -1).astype(np.float64)
    synthetic_flat = synthetic.reshape(len(synthetic), -1).astype(np.float64)
    mean_errors, covariance_errors = [], []
    for class_id in np.unique(labels):
        left, right = real_flat[labels == class_id], synthetic_flat[labels == class_id]
        left_mean, right_mean = left.mean(0), right.mean(0)
        mean_errors.append(np.linalg.norm(left_mean - right_mean) / max(np.linalg.norm(left_mean), 1e-12))
        left_centered, right_centered = left - left_mean, right - right_mean
        # Frobenius covariance error without materialising a D x D matrix.
        cross = left_centered @ right_centered.T
        left_gram, right_gram = left_centered @ left_centered.T, right_centered @ right_centered.T
        n_left, n_right = max(len(left) - 1, 1), max(len(right) - 1, 1)
        error_sq = (
            np.sum(left_gram**2) / n_left**2
            + np.sum(right_gram**2) / n_right**2
            - 2 * np.sum(cross**2) / (n_left * n_right)
        )
        baseline_sq = np.sum(left_gram**2) / n_left**2
        covariance_errors.append(np.sqrt(max(error_sq, 0)) / max(np.sqrt(baseline_sq), 1e-12))
    return {
        "class_mean_relative_error": float(np.mean(mean_errors)),
        "class_covariance_relative_error": float(np.mean(covariance_errors)),
    }
