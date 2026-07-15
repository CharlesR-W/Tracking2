import copy

import torch

from tracking2.criticality import transplant_module
from tracking2.report import criticality_heatmap
from tracking2.models import InstrumentedVGG19


def tiny_vgg() -> InstrumentedVGG19:
    return InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625)


def test_vgg_has_nineteen_ordered_parametric_modules():
    names = [name for name, _ in tiny_vgg().intervention_modules()]
    assert len(names) == 19
    assert names[:3] == ["stage1.conv1", "stage1.conv2", "stage2.conv1"]
    assert names[-3:] == ["fc1", "fc2", "final_linear"]


def test_final_to_final_transplant_is_logit_identity():
    model = tiny_vgg().eval()
    x = torch.randn(2, 3, 32, 32)
    expected = model(x)
    for index in (0, 8, 18):
        hybrid = copy.deepcopy(model)
        transplant_module(hybrid, model, index)
        torch.testing.assert_close(hybrid(x), expected)


def test_transplant_changes_only_selected_module():
    target, source = tiny_vgg(), tiny_vgg()
    before = {key: value.clone() for key, value in target.state_dict().items()}
    transplant_module(target, source, 4)
    changed = [key for key, value in target.state_dict().items() if not torch.equal(value, before[key])]
    assert changed
    assert all(key.startswith("stages.2.0.") for key in changed)


def test_batch_norm_state_is_part_of_atomic_transplant():
    target = InstrumentedVGG19(classifier_width=16, width_multiplier=0.0625, batch_norm=True)
    source = copy.deepcopy(target)
    source.intervention_modules()[0][1].norm.running_mean.fill_(3.0)
    transplant_module(target, source, 0)
    torch.testing.assert_close(target.intervention_modules()[0][1].norm.running_mean, torch.full_like(target.intervention_modules()[0][1].norm.running_mean, 3.0))


def test_suffix_parameters_exclude_transplanted_and_prefix_modules():
    model = tiny_vgg()
    modules = model.intervention_modules()
    suffix_ids = {id(parameter) for parameter in model.parameters_after(8)}
    for _, module in modules[:9]:
        assert suffix_ids.isdisjoint(id(parameter) for parameter in module.parameters())
    assert {id(parameter) for parameter in modules[9][1].parameters()} <= suffix_ids


def test_vgg_split_forward_matches_full_forward_at_conv_cuts():
    model = tiny_vgg().eval()
    x = torch.randn(2, 3, 32, 32)
    expected = model(x)
    for cut in (0, 7, 8, 9, 13, 15):
        actual = model.forward_from_module(model.encode_to_module(x, cut), cut)
        torch.testing.assert_close(actual, expected)


def test_criticality_heatmap_matches_paper_source_order():
    payload = {
        "module_names": ["layer1"],
        "config": {"checkpoint_epochs": [0, 1]},
        "interventions": [
            {"source": source, "module": "layer1", "delta_error": value}
            for source, value in (("random", 0.8), ("0", 0.6), ("1", 0.0))
        ],
    }
    figure = criticality_heatmap(payload)
    assert list(figure.data[0].y) == ["fresh random draw", "checkpoint 0", "epoch 1 (intact final)"]
    assert figure.layout.yaxis.autorange == "reversed"
