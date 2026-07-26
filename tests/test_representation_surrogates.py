import numpy as np

from tracking2.representation_surrogates import (
    RepresentationSurrogate,
    _factor_covariance,
    fit_representation_surrogate,
    moment_diagnostics,
    pca_coverage_diagnostics,
    project_representation,
    sample_representation_surrogate,
    truncate_representation_surrogate,
)


def _coordinates(model, representations):
    return (
        representations.reshape(len(representations), -1) - model.pca_mean
    ) @ model.components.T


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


def test_exact_empirical_covariance_has_no_hidden_jitter():
    rng = np.random.default_rng(31)
    labels = np.repeat(np.arange(2), 400)
    mixing = np.array(
        [[1.0, 0.8, -0.2], [0.1, 1.7, 0.4], [0.3, -0.6, 0.9]],
        dtype=np.float32,
    )
    representations = (
        rng.normal(size=(800, 3)).astype(np.float32) @ mixing
        + labels[:, None]
    )
    fitted = fit_representation_surrogate(
        representations,
        labels,
        3,
        seed=0,
        covariance_shrinkage=0.0,
    )
    coordinates = _coordinates(fitted, representations)
    for class_id in range(2):
        expected = np.cov(
            coordinates[labels == class_id], rowvar=False
        )
        np.testing.assert_allclose(
            fitted.class_covariances[class_id],
            expected,
            rtol=2e-6,
            atol=2e-6,
        )
        np.testing.assert_allclose(
            fitted.class_factors[class_id]
            @ fitted.class_factors[class_id].T,
            fitted.class_covariances[class_id],
            rtol=2e-6,
            atol=2e-6,
        )
    provenance = fitted.covariance_provenance()
    assert provenance["covariance_target"] == "exact_empirical_covariance"
    assert provenance["exact_empirical_covariance"] is True
    assert provenance["requested_covariance_shrinkage"] == 0.0
    assert provenance["per_class_factor_jitter"] == [0.0, 0.0]
    assert provenance["per_class_clipped_negative_eigenvalue_count"] == [0, 0]


def test_gaussian_samples_match_exact_and_five_percent_targets():
    rng = np.random.default_rng(32)
    labels = np.repeat(np.arange(2), 500)
    mixing = np.array(
        [[1.4, 0.7, 0.2], [-0.3, 0.8, 0.6], [0.5, -0.4, 1.1]],
        dtype=np.float32,
    )
    representations = (
        rng.normal(size=(1000, 3)).astype(np.float32) @ mixing
        + 0.5 * labels[:, None]
    )
    exact = fit_representation_surrogate(
        representations,
        labels,
        3,
        seed=0,
        covariance_shrinkage=0.0,
    )
    sensitivity = fit_representation_surrogate(
        representations,
        labels,
        3,
        seed=0,
        covariance_shrinkage=0.05,
    )
    for class_id in range(2):
        scale = np.trace(exact.class_covariances[class_id]) / 3
        expected = (
            0.95 * exact.class_covariances[class_id]
            + 0.05 * scale * np.eye(3)
        )
        np.testing.assert_allclose(
            sensitivity.class_covariances[class_id],
            expected,
            rtol=3e-6,
            atol=3e-6,
        )

    sample_labels = np.repeat(np.arange(2), 20_000)
    for model in (exact, sensitivity):
        sample = sample_representation_surrogate(
            model, sample_labels, "gaussian", seed=7
        )
        sample_coordinates = _coordinates(model, sample)
        for class_id in range(2):
            sampled_covariance = np.cov(
                sample_coordinates[sample_labels == class_id],
                rowvar=False,
            )
            relative_error = np.linalg.norm(
                sampled_covariance - model.class_covariances[class_id]
            ) / np.linalg.norm(model.class_covariances[class_id])
            assert relative_error < 0.03
    provenance = sensitivity.covariance_provenance()
    assert provenance["covariance_target"] == "shrunk_empirical_covariance"
    assert provenance["exact_empirical_covariance"] is False
    assert provenance["requested_covariance_shrinkage"] == 0.05


def test_psd_projection_is_reported_instead_of_called_exact():
    (
        singular_factor,
        singular_target,
        singular_method,
        singular_jitter,
        singular_clipped_count,
        _,
    ) = _factor_covariance(np.array([[1.0, 1.0], [1.0, 1.0]]))
    assert singular_method == "eigh_semidefinite"
    assert singular_jitter == 0.0
    assert singular_clipped_count == 0
    np.testing.assert_allclose(
        singular_factor @ singular_factor.T,
        singular_target,
        atol=1e-12,
    )

    factor, target, method, jitter, clipped_count, maximum_clip = (
        _factor_covariance(
            np.array([[1.0, 1.000001], [1.000001, 1.0]])
        )
    )
    assert method == "eigh_psd_projection"
    assert jitter == 0.0
    assert clipped_count == 1
    assert maximum_clip > 0
    np.testing.assert_allclose(factor @ factor.T, target, atol=1e-12)

    model = RepresentationSurrogate(
        pca_mean=np.zeros(2),
        components=np.eye(2),
        class_means=np.zeros((1, 2)),
        class_covariances=target[None],
        class_factors=factor[None],
        pooled_variance=float(np.trace(target) / 2),
        representation_shape=(2,),
        covariance_shrinkage=0.0,
        class_factorization_methods=(method,),
        class_factor_jitters=np.array([jitter]),
        class_clipped_negative_eigenvalue_counts=np.array([clipped_count]),
        class_max_negative_eigenvalue_magnitudes=np.array([maximum_clip]),
    )
    provenance = model.covariance_provenance()
    assert (
        provenance["covariance_target"]
        == "empirical_covariance_after_psd_projection"
    )
    assert provenance["exact_empirical_covariance"] is False
    assert provenance["per_class_clipped_negative_eigenvalue_count"] == [1]

    legacy_model = RepresentationSurrogate(
        pca_mean=np.zeros(2),
        components=np.eye(2),
        class_means=np.zeros((1, 2)),
        class_covariances=np.eye(2)[None],
        class_factors=np.eye(2)[None],
        pooled_variance=1.0,
        representation_shape=(2,),
        covariance_shrinkage=0.0,
    )
    legacy_provenance = legacy_model.covariance_provenance()
    assert (
        legacy_provenance["covariance_target"]
        == "covariance_provenance_unavailable"
    )
    assert legacy_provenance["exact_empirical_covariance"] is False


def test_covariance_shrinkage_must_be_a_fraction():
    rng = np.random.default_rng(33)
    representations = rng.normal(size=(40, 3)).astype(np.float32)
    labels = np.repeat(np.arange(2), 20)
    for invalid in (-0.01, 1.01):
        try:
            fit_representation_surrogate(
                representations,
                labels,
                3,
                seed=0,
                covariance_shrinkage=invalid,
            )
        except ValueError as error:
            assert "covariance_shrinkage" in str(error)
        else:
            raise AssertionError("invalid shrinkage was accepted")


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


def test_maximal_fit_with_full_moment_bank_matches_previous_truncation():
    rng = np.random.default_rng(81)
    representations = rng.normal(size=(120, 7)).astype(np.float32)
    labels = np.tile(np.arange(3), 40)
    fit_count = 45
    basis_only = fit_representation_surrogate(
        representations[:fit_count],
        labels[:fit_count],
        5,
        seed=4,
        covariance_shrinkage=0.0,
    )
    previous = truncate_representation_surrogate(
        basis_only,
        representations,
        labels,
        5,
        covariance_shrinkage=0.0,
    )
    reused = fit_representation_surrogate(
        representations[:fit_count],
        labels[:fit_count],
        5,
        seed=4,
        covariance_shrinkage=0.0,
        moment_representations=representations,
        moment_labels=labels,
    )
    np.testing.assert_allclose(reused.pca_mean, previous.pca_mean)
    np.testing.assert_allclose(reused.components, previous.components)
    np.testing.assert_allclose(reused.class_means, previous.class_means)
    np.testing.assert_allclose(
        reused.class_covariances, previous.class_covariances
    )
    np.testing.assert_allclose(reused.class_factors, previous.class_factors)
    assert reused.pooled_variance == previous.pooled_variance
    assert reused.covariance_provenance() == previous.covariance_provenance()


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
