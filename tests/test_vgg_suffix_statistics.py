import torch

from tracking2.models import InstrumentedVGG19
from tracking2.report import (
    vgg_statistics_sweep_figures,
    vgg_suffix_statistics_detail_figures,
    vgg_suffix_statistics_figure,
)
from tracking2.vgg_suffix_statistics import (
    VGGSuffixStatisticsConfig,
    evaluate_suffix,
    representation_loader,
    run,
)


def test_vgg_suffix_evaluation_matches_full_model():
    model = InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625, batch_norm=True).eval()
    x = torch.randn(8, 3, 32, 32)
    y = torch.arange(8) % 10
    cut = 8
    with torch.no_grad():
        representations = model.encode_to_module(x, cut).numpy()
        expected = model(x)
    loader = representation_loader(representations, y.numpy(), 4, seed=0, shuffle=False)
    result = evaluate_suffix(model, cut, loader, torch.device("cpu"))
    expected_loss = torch.nn.functional.cross_entropy(expected, y).item()
    assert abs(result["loss"] - expected_loss) < 1e-5


def test_dashboard_bound_default_uses_three_surrogate_draws():
    assert VGGSuffixStatisticsConfig().surrogate_draws == 3


def test_vgg_suffix_statistics_smoke(tmp_path):
    final = InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625, batch_norm=True)
    initial = InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625, batch_norm=True)
    final_path, initial_path = tmp_path / "final.pt", tmp_path / "initial.pt"
    torch.save(final.state_dict(), final_path)
    torch.save(initial.state_dict(), initial_path)
    artifact_path = run(VGGSuffixStatisticsConfig(
        output=str(tmp_path / "artifact"), checkpoint=str(final_path),
        initialization_checkpoint=str(initial_path), fake_data=True,
        train_size=8, test_size=4, batch_size=4, classifier_width=16,
        width_multiplier=0.0625, cuts=(8,), conditions=("native", "reset0"),
        pca_fit_size=8, pca_components=3, surrogate_draws=1, relax_epochs=1,
        relax_batch_zoom=True, seed=0, device="cpu",
    ))
    payload = __import__("json").loads(artifact_path.read_text())
    assert payload["status"].startswith("MOCKUP")
    assert {(item["cut"], item["condition"]) for item in payload["slices"]} == {
        (8, "native"), (8, "reset0")}
    assert all(len(item["records"]) == 21 for item in payload["slices"])
    assert all(any(row["relax_epoch"] == 0.5 for row in item["records"])
               for item in payload["slices"])
    figure = vgg_suffix_statistics_figure(payload)
    assert len(figure.data) == 6
    accuracies, decompositions, diagnostics = vgg_suffix_statistics_detail_figures(payload)
    assert len(accuracies.data) == 6
    assert "end-of-protocol accuracy effects" in decompositions
    assert "plot-vgg-suffix-accuracy-decomposition-cut8" in decompositions
    assert "class-covariance relative error" in diagnostics

    native_payload = __import__("copy").deepcopy(payload)
    native_payload["config"]["checkpoint_epoch"] = 0
    native_payload["slices"] = [item for item in native_payload["slices"]
                                if item["condition"] == "native"]
    gap_figure, coverage_figure, sweep_diagnostics = vgg_statistics_sweep_figures([native_payload])
    assert len(gap_figure.data) == 2
    assert len(coverage_figure.data) == 1
    assert "stage4.conv1" in sweep_diagnostics
