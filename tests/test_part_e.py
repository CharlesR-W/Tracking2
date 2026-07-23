from __future__ import annotations

import torch

from transformers import GPT2Config, GPT2LMHeadModel

from tracking2.part_e import (
    GateConfig,
    GPT2Suffix,
    _make_eval_banks,
    _future_leakage_check,
    cache_cut_activations,
    fit_sequence_surrogate,
    ntp_cross_entropy,
)


def tiny_model() -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=97,
        n_positions=16,
        n_ctx=16,
        n_embd=32,
        n_layer=3,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    return model


def test_cached_cut_suffix_reproduces_full_model_logits():
    torch.manual_seed(0)
    model = tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (4, 16))
    mask = torch.ones_like(ids)
    caches, logits = cache_cut_activations(
        model, ids, mask, cuts=(0, 1, 2), batch_size=2,
        device=torch.device("cpu"), dtype=torch.float32, logit_limit=len(ids),
    )
    for cut in (0, 1, 2):
        suffix = GPT2Suffix(model, cut).eval()
        replay = suffix(caches[cut].float(), mask)
        assert torch.allclose(replay, logits.float(), atol=2e-3, rtol=2e-3)
        assert torch.isfinite(ntp_cross_entropy(replay, ids))
        assert suffix.lm_head.weight.data_ptr() != model.transformer.wte.weight.data_ptr()


def test_sequence_surrogate_shapes_and_future_target_guard(tmp_path):
    torch.manual_seed(1)
    model = tiny_model()
    ids = torch.randint(0, model.config.vocab_size, (12, 16))
    mask = torch.ones_like(ids)
    caches, _ = cache_cut_activations(
        model, ids, mask, cuts=(1,), batch_size=4,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    config = GateConfig(
        output=str(tmp_path), context_length=16, pca_components=8,
        pca_fit_rows=128, token_groups=8, position_bins=4,
    )
    surrogate = fit_sequence_surrogate(caches[1], ids, config, torch.device("cpu"))
    for distribution in ("mean", "token_gaussian", "seq_gaussian"):
        generator = torch.Generator().manual_seed(4)
        sample = surrogate.sample(ids, distribution, generator)
        assert sample.shape == caches[1].shape
        assert torch.isfinite(sample).all()
        assert _future_leakage_check(
            surrogate, ids, torch.device("cpu"), distribution, seed=9,
        ) == 0.0
    projected = surrogate.project(caches[1].float())
    assert projected.shape == caches[1].shape
    assert 0.0 < surrogate.coverage <= 1.00001
    assert min(config.pca_min_components, config.pca_components) <= surrogate.components
    assert surrogate.components <= config.pca_components
    assert surrogate.diagnostics["covariance_estimator"].startswith("one_step_flip_flop")


def test_adaptive_pca_uses_smallest_dimension_that_meets_target(tmp_path):
    torch.manual_seed(2)
    hidden = torch.randn(20, 9, 12)
    hidden[..., 4:] *= 0.01
    ids = torch.randint(0, 97, (20, 9))
    config = GateConfig(
        output=str(tmp_path), context_length=9, pca_components=10,
        pca_min_components=2, pca_target_coverage=0.90, pca_fit_rows=160,
        token_groups=4, position_bins=3, covariance_fit_stories=10,
    )
    surrogate = fit_sequence_surrogate(hidden, ids, config, torch.device("cpu"))
    assert 2 <= surrogate.components < 10
    assert surrogate.coverage >= 0.90
    assert surrogate.diagnostics["pca_target_met"] is True


def test_five_condition_eval_banks_are_complete(tmp_path):
    torch.manual_seed(3)
    ids = torch.randint(0, 97, (10, 9))
    hidden = torch.randn(10, 9, 12)
    config = GateConfig(
        output=str(tmp_path), context_length=9, pca_components=8,
        pca_min_components=2, pca_fit_rows=80, token_groups=4,
        position_bins=3, covariance_fit_stories=8,
    )
    surrogate = fit_sequence_surrogate(hidden, ids, config, torch.device("cpu"))
    banks = _make_eval_banks(surrogate, hidden, ids, torch.device("cpu"), seed=11)
    assert set(banks) == {
        "mean", "token_gaussian", "seq_gaussian", "projected_true", "true",
    }
    assert all(bank.shape == hidden.shape for bank in banks.values())
    assert torch.allclose(banks["projected_true"], surrogate.project(hidden))
