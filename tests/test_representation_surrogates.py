import numpy as np

from tracking2.representation_surrogates import (
    fit_representation_surrogate,
    moment_diagnostics,
    sample_representation_surrogate,
)


def test_representation_surrogates_preserve_shape_and_are_finite():
    rng = np.random.default_rng(4)
    representations = rng.normal(size=(80, 6, 3, 3)).astype(np.float32)
    labels = np.repeat(np.arange(4), 20)
    fitted = fit_representation_surrogate(representations, labels, 8, seed=2)
    for kind in ("mean", "gaussian"):
        sample = sample_representation_surrogate(fitted, labels, kind, seed=3)
        assert sample.shape == representations.shape
        assert np.isfinite(sample).all()
        diagnostics = moment_diagnostics(representations, sample, labels)
        assert set(diagnostics) == {"class_mean_relative_error", "class_covariance_relative_error"}
        assert all(np.isfinite(value) for value in diagnostics.values())


def test_gaussian_sampling_matches_fitted_class_means():
    rng = np.random.default_rng(0)
    labels = np.repeat(np.arange(2), 1000)
    x = rng.normal(size=(2000, 5)).astype(np.float32) + labels[:, None] * 2
    fitted = fit_representation_surrogate(x, labels, 5, seed=0)
    sample = sample_representation_surrogate(fitted, labels, "gaussian", seed=1)
    for class_id in (0, 1):
        np.testing.assert_allclose(
            sample[labels == class_id].mean(0), x[labels == class_id].mean(0), atol=0.15
        )
