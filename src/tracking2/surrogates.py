from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch.utils.data import Dataset


@dataclass
class PCASurrogateModel:
    mean: np.ndarray
    components: np.ndarray
    class_means: np.ndarray
    class_covariances: np.ndarray
    pooled_variance: float
    image_shape: tuple[int, int, int]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            mean=self.mean,
            components=self.components,
            class_means=self.class_means,
            class_covariances=self.class_covariances,
            pooled_variance=np.asarray(self.pooled_variance),
            image_shape=np.asarray(self.image_shape),
        )

    @classmethod
    def load(cls, path: Path) -> "PCASurrogateModel":
        data = np.load(path)
        return cls(
            mean=data["mean"],
            components=data["components"],
            class_means=data["class_means"],
            class_covariances=data["class_covariances"],
            pooled_variance=float(data["pooled_variance"]),
            image_shape=tuple(int(x) for x in data["image_shape"]),
        )


def fit_pca_surrogate(
    images: np.ndarray,
    labels: np.ndarray,
    n_components: int,
    seed: int,
    covariance_shrinkage: float = 0.05,
) -> PCASurrogateModel:
    """Fit class statistics in a fixed PCA coordinate system.

    Images must be float arrays in [0, 1], shaped [N,C,H,W].
    """
    flat = images.reshape(len(images), -1).astype(np.float32)
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    z = pca.fit_transform(flat)
    classes = np.arange(max(10, int(labels.max()) + 1))
    global_mean = z.mean(0)
    global_covariance = np.cov(z, rowvar=False).astype(np.float32)
    means = np.stack([
        z[labels == c].mean(0) if np.any(labels == c) else global_mean for c in classes
    ])
    covariances = []
    residual_variances = []
    for c in classes:
        zc = z[labels == c]
        covariance = (
            np.cov(zc, rowvar=False).astype(np.float32)
            if len(zc) >= 2
            else global_covariance.copy()
        )
        scale = float(np.trace(covariance) / n_components)
        covariance = (1 - covariance_shrinkage) * covariance
        covariance += covariance_shrinkage * scale * np.eye(n_components, dtype=np.float32)
        covariances.append(covariance)
        if len(zc):
            residual_variances.append(np.mean((zc - zc.mean(0)) ** 2))
    return PCASurrogateModel(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        class_means=means.astype(np.float32),
        class_covariances=np.stack(covariances),
        pooled_variance=float(np.mean(residual_variances)),
        image_shape=tuple(images.shape[1:]),
    )


def sample_surrogate(
    model: PCASurrogateModel,
    labels: np.ndarray,
    kind: str,
    seed: int,
) -> np.ndarray:
    """Sample a selected-statistics CIFAR proxy and reconstruct pixel images.

    `mean` retains class means and adds class-independent isotropic nuisance.
    `covariance` matches class means/covariances under a Gaussian model.
    """
    if kind not in {"mean", "covariance"}:
        raise ValueError(f"Unknown surrogate kind: {kind}")
    rng = np.random.default_rng(seed)
    d = model.components.shape[0]
    z = np.empty((len(labels), d), dtype=np.float32)
    for c in np.unique(labels):
        mask = labels == c
        count = int(mask.sum())
        if kind == "mean":
            noise = rng.normal(size=(count, d)).astype(np.float32)
            z[mask] = model.class_means[c] + np.sqrt(model.pooled_variance) * noise
        else:
            z[mask] = rng.multivariate_normal(
                model.class_means[c], model.class_covariances[c], size=count
            ).astype(np.float32)
    flat = z @ model.components + model.mean
    return np.clip(flat.reshape((len(labels),) + model.image_shape), 0.0, 1.0).astype(np.float32)


class ArrayDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, images: np.ndarray, labels: np.ndarray) -> None:
        self.images = torch.from_numpy(images.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.images[index], self.labels[index]
