import torch

from tracking2.measurements import empirical_fisher_schur_trace
from tracking2.models import ArchitectureSpec, SplitConvNet


def test_fisher_schur_is_bounded():
    model = SplitConvNet(ArchitectureSpec("residual", (4, 8)))
    batch = (torch.randn(4, 3, 32, 32), torch.tensor([0, 1, 2, 3]))
    result = empirical_fisher_schur_trace(model, 1, batch)
    assert 0 <= result["effective_prefix_trace"] <= result["prefix_trace"] + 1e-5
    assert 0 <= result["compensable_fraction"] <= 1 + 1e-5

