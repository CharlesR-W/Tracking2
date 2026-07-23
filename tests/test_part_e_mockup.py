import json

from tracking2.part_e_mockup import part_e_dashboard_html, part_e_mockup_html


def test_part_e_mockup_is_unambiguously_unrun_and_compact():
    html = part_e_mockup_html()

    assert '<section id="e" class="panel part-e-mockup"' in html
    assert 'data-epistemic-status="MOCKUP-PLANNED-UNRUN"' in html
    assert "Does Part B's suffix-statistics result survive next-token prediction?" in html
    assert "no measured, simulated, or illustrative outcome values" in html
    assert html.count('data-mockup-watermark="true"') == 3
    assert html.count("MOCKUP · PLANNED · UNRUN") >= 3


def test_part_e_is_a_direct_three_distribution_part_b_replication():
    html = part_e_mockup_html()

    assert "Freeze once, relax three matched suffixes, evaluate all nine pairs" in html
    assert "mean-only, sequence-Gaussian, and empirical true" in html
    assert "Complete three-by-three relax-by-evaluate matrix" in html
    assert html.count('data-matrix-cell="planned"') == 9
    assert r"\Delta_{\mathrm{true}\mid r}" in html
    assert "Near-zero sequence-Gaussian excess loss" in html
    assert "larger mean-only excess loss" in html
    assert "every matrix entry is a label, not a value" in html


def test_part_e_retains_only_necessary_ntp_adaptations():
    html = part_e_mockup_html()

    assert "Complete residual-stream sequence" in html
    assert "attention mask and positions" in html
    assert "Next-token target" in html
    assert "matrix-normal proxy" in html
    assert "story-level resampling" in html
    assert "cloned, untied LM head" in html
    assert "future targets must leave all earlier generated rows unchanged" in html
    assert "projected-true versus full-true gap" in html
    assert "the untouched cached suffix reproduces" in html


def test_part_e_secondary_controls_do_not_expand_headline_matrix():
    html = part_e_mockup_html()

    assert "IID-token Gaussian" in html
    assert "projected-true replay" in html
    assert "Neither needs to enlarge the headline $3\\times3$ matrix" in html
    assert "Part D retains ownership" in html


def test_part_e_defers_separate_adaptation_studies():
    html = part_e_mockup_html()

    assert "Instruction continuation, response-only SFT, LoRA" in html
    assert "separate adaptation studies" in html
    assert "not be prerequisites" in html
    assert "P/C/I/S" not in html
    assert "finite tracking" not in html.lower()


def test_part_e_has_small_gate_and_artifact_contract():
    html = part_e_mockup_html()

    assert "Small first gate" in html
    assert "cuts 0/5/11" in html
    assert "64 suffix updates" in html
    assert "three model seeds and three surrogate draws" in html
    for artifact in (
        "manifest.json",
        "replay_manifest.json",
        "sequence_surrogates.npz",
        "surrogate_diagnostics.parquet",
        "suffix_statistics.json",
    ):
        assert artifact in html


def test_part_e_measured_dashboard_replaces_mockup_with_saved_evidence(tmp_path):
    distributions = ["mean", "token_gaussian", "seq_gaussian", "projected_true", "true"]
    records = []
    for cut in (0, 5, 11):
        for update in (0, 32):
            for train_index, train in enumerate(distributions):
                for eval_index, evaluate in enumerate(distributions):
                    records.append({
                        "cut": cut, "update": update,
                        "train_distribution": train, "eval_distribution": evaluate,
                        "loss_nats_per_token": 1.0 + 0.01 * train_index + 0.02 * eval_index,
                    })
    diagnostics = [{
        "cut": cut, "pca_components": 128 + cut, "pca_coverage": 0.96,
        "projected_true_excess_loss": 0.01,
        "parity": {"max_abs_logit_error": 1e-6},
        "initial_gradient_norms": {name: 1.0 + i for i, name in enumerate(distributions)},
        "learning_rate_scales": {name: 1.0 for name in distributions},
    } for cut in (0, 5, 11)]
    payload = {
        "schema_version": 2, "status": "MEASURED", "records": records,
        "diagnostics": diagnostics, "config": {"seed": 0}, "runtime_seconds": 12.0,
    }
    matched = tmp_path / "matched.json"
    equal = tmp_path / "equal.json"
    matched.write_text(json.dumps(payload))
    equal.write_text(json.dumps(payload))

    dashboard = part_e_dashboard_html(matched, equal)

    assert 'class="panel part-e-measured"' in dashboard
    assert 'data-epistemic-status="MEASURED-DIAGNOSTIC-ONE-SEED"' in dashboard
    assert "What structure in a residual-stream sequence" in dashboard
    assert r"H_\ell=\phi_\ell(X)\in\mathbb R^{T\times d}" in dashboard
    assert r"\mathcal L_{\mathrm{NTP}}" in dashboard
    assert r"\Sigma_{\mathrm{chan}}\otimes K_{\mathrm{pos}}" in dashboard
    assert "supervised distributional test, not a text generator" in dashboard
    assert "Train on one hidden distribution; evaluate on every hidden distribution" in dashboard
    assert "Full five-by-five relax-by-evaluate matrix" in dashboard
    assert "MOCKUP · PLANNED · UNRUN" not in dashboard
    assert "old gap" not in dashboard.lower()
    assert "earlier result" not in dashboard.lower()
    assert "window.PART_E_DATA=" in dashboard
