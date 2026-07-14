import numpy as np

from tracking2.surrogates import fit_pca_surrogate, sample_surrogate


def test_surrogate_shapes_and_range():
    rng = np.random.default_rng(0)
    images = rng.uniform(size=(40, 3, 4, 4)).astype(np.float32)
    labels = np.repeat(np.arange(2), 20)
    model = fit_pca_surrogate(images, labels, n_components=6, seed=0)
    for kind in ("mean", "covariance"):
        samples = sample_surrogate(model, labels, kind, seed=1)
        assert samples.shape == images.shape
        assert samples.min() >= 0
        assert samples.max() <= 1

