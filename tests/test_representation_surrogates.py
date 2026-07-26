import numpy as np

from tracking2.representation_surrogates import (
    fit_representation_surrogate,
    moment_diagnostics,
    pca_coverage_diagnostics,
    project_representation,
    sample_representation_surrogate,
    truncate_representation_surrogate,
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
        diagnostics = moment_diagnostics(representations, sample, labels, fitted)
        assert diagnostics["diagnostic_space"] == "fitted PCA subspace"
        assert diagnostics["mean_normalization"] == "within-class RMS radius"
        assert np.isfinite(diagnostics["class_mean_relative_error"])
        assert np.isfinite(diagnostics["class_covariance_relative_error"])


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


def test_mean_noise_radius_controls_within_class_spread():
    rng = np.random.default_rng(4)
    labels = np.repeat(np.arange(2), 200)
    representations = rng.normal(size=(400, 5)).astype(np.float32)
    fitted = fit_representation_surrogate(representations, labels, 4, seed=0)
    centroids = sample_representation_surrogate(
        fitted, labels, "mean", seed=3, mean_noise_radius=0.0
    )
    trace_matched = sample_representation_surrogate(
        fitted, labels, "mean", seed=3, mean_noise_radius=1.0
    )
    half_radius = sample_representation_surrogate(
        fitted, labels, "mean", seed=3, mean_noise_radius=0.5
    )
    for class_id in range(2):
        selected = labels == class_id
        assert np.max(np.std(centroids[selected], axis=0)) < 1e-6
        assert np.mean(np.var(trace_matched[selected], axis=0)) > 0
    trace_coordinates = (
        trace_matched.reshape(len(labels), -1) - fitted.pca_mean
    ) @ fitted.components.T
    half_coordinates = (
        half_radius.reshape(len(labels), -1) - fitted.pca_mean
    ) @ fitted.components.T
    class_centres = fitted.class_means[labels]
    np.testing.assert_allclose(
        2 * (half_coordinates - class_centres),
        trace_coordinates - class_centres,
        atol=1e-5,
    )


def test_radius_one_target_is_exactly_trace_matched():
    rng = np.random.default_rng(14)
    labels = np.repeat(np.arange(3), 40)
    representations = rng.normal(size=(120, 6)).astype(np.float32)
    fitted = fit_representation_surrogate(representations, labels, 5, seed=0)
    fitted_mean_trace = float(
        np.mean([np.trace(covariance) for covariance in fitted.class_covariances])
    )
    isotropic_trace = fitted.components.shape[0] * fitted.pooled_variance
    np.testing.assert_allclose(isotropic_trace, fitted_mean_trace, rtol=1e-6)


def test_projected_representation_is_pca_idempotent():
    rng = np.random.default_rng(5)
    representations = rng.normal(size=(80, 7)).astype(np.float32)
    labels = np.repeat(np.arange(2), 40)
    fitted = fit_representation_surrogate(representations, labels, 3, seed=0)
    once = project_representation(fitted, representations)
    twice = project_representation(fitted, once)
    np.testing.assert_allclose(once, twice, atol=1e-5)


def test_truncation_reuses_leading_maximal_pca_basis():
    rng = np.random.default_rng(8)
    representations = rng.normal(size=(100, 9)).astype(np.float32)
    labels = np.repeat(np.arange(2), 50)
    maximal = fit_representation_surrogate(representations, labels, 6, seed=0)
    truncated = truncate_representation_surrogate(
        maximal, representations, labels, 3
    )
    np.testing.assert_allclose(truncated.pca_mean, maximal.pca_mean)
    np.testing.assert_allclose(truncated.components, maximal.components[:3])


def test_nested_ranks_share_leading_standard_normal_coordinates():
    rng = np.random.default_rng(18)
    representations = rng.normal(size=(120, 7)).astype(np.float32)
    labels = np.repeat(np.arange(3), 40)
    maximal = fit_representation_surrogate(representations, labels, 4, seed=0)
    truncated = truncate_representation_surrogate(
        maximal, representations, labels, 2
    )
    maximal_sample = sample_representation_surrogate(
        maximal,
        labels,
        "mean",
        seed=9,
        paired_noise_rank=4,
    )
    truncated_sample = sample_representation_surrogate(
        truncated,
        labels,
        "mean",
        seed=9,
        paired_noise_rank=4,
    )
    maximal_coordinates = (
        maximal_sample - maximal.pca_mean
    ) @ maximal.components.T
    truncated_coordinates = (
        truncated_sample - truncated.pca_mean
    ) @ truncated.components.T
    maximal_noise = (
        maximal_coordinates - maximal.class_means[labels]
    ) / np.sqrt(maximal.pooled_variance)
    truncated_noise = (
        truncated_coordinates - truncated.class_means[labels]
    ) / np.sqrt(truncated.pooled_variance)
    np.testing.assert_allclose(
        truncated_noise, maximal_noise[:, :2], atol=2e-5
    )


def test_coverage_diagnostics_are_bounded_on_held_out_data():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(100, 8)).astype(np.float32)
    held_out = rng.normal(size=(60, 8)).astype(np.float32)
    train_labels = np.repeat(np.arange(2), 50)
    held_out_labels = np.repeat(np.arange(2), 30)
    fitted = fit_representation_surrogate(train, train_labels, 4, seed=0)
    coverage = pca_coverage_diagnostics(fitted, held_out, held_out_labels)
    assert set(coverage) == {
        "total_variance_fraction",
        "within_class_variance_fraction",
        "between_class_mean_variance_fraction",
    }
    assert all(0 <= value <= 1.00001 for value in coverage.values())


def test_pca_diagnostics_ignore_declared_discarded_directions():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(2), 100)
    real = rng.normal(size=(200, 6)).astype(np.float32)
    fitted = fit_representation_surrogate(real, labels, 2, seed=0)
    # Construct a direction orthogonal to the fitted PCA subspace and perturb
    # only that deliberately discarded component.
    candidate = rng.normal(size=6)
    candidate -= fitted.components.T @ (fitted.components @ candidate)
    candidate /= np.linalg.norm(candidate)
    synthetic = real + rng.normal(size=(200, 1)).astype(np.float32) * candidate.astype(np.float32)
    native = moment_diagnostics(real, synthetic, labels)
    target = moment_diagnostics(real, synthetic, labels, fitted)
    assert native["class_covariance_relative_error"] > 0.01
    assert target["class_mean_relative_error"] < 1e-6
    assert target["class_covariance_relative_error"] < 1e-6
