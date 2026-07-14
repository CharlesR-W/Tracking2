import torch

from tracking2.models import ArchitectureSpec, SplitConvNet


def test_split_forward_matches_full_forward():
    x = torch.randn(3, 3, 32, 32)
    for kind in ("plain", "residual"):
        model = SplitConvNet(ArchitectureSpec(kind, (8, 16, 16)))
        model.eval()
        expected = model(x)
        for cut in range(model.num_cuts):
            actual = model.forward_from(model.encode_to(x, cut), cut)
            torch.testing.assert_close(actual, expected)

