from tracking2.part_d_moment_mockup import part_d_moment_mockup_html


def test_part_d_owns_tab_and_is_unambiguously_theory_not_result():
    html = part_d_moment_mockup_html()

    assert html.startswith('<section id="d" class="panel part-d-tangent"')
    assert 'data-epistemic-status="THEORY"' in html
    assert 'data-run-status="PLANNED-UNRUN"' in html
    assert "No result is shown here" in html
    panel_count = html.count('data-panel-status="THEORY-PLANNED-UNRUN"')
    watermark_count = html.count('data-theory-watermark="true"')
    assert panel_count == 6  # hero plus five independently screenshot-able panels
    assert watermark_count == panel_count


def test_part_d_defines_distributional_transport_chart_and_optimal_suffix():
    html = part_d_moment_mockup_html()

    assert "The perturbation is a movement of a distribution, not one activation vector" in html
    assert r"Q_{a,\ell}=\operatorname{Law}(h^\ell,y)" in html
    assert "b^*(a)" in html
    assert "h_i(u)" in html
    assert "M_u=\\mathbb E_i[V_i^\\top V_i]" in html
    assert "class-conditional translations" in html
    assert "higher-order Hermite transports" in html
    assert "Label it by a moment only after reporting that overlap" in html


def test_part_d_compares_full_and_suffix_update_eigenspectra():
    html = part_d_moment_mockup_html()

    assert "A_{\\rm full}" in html
    assert "A_{\\rm suf}=I-\\eta_bH_{bb}" in html
    assert "Compare eigenvalues of update maps—not raw Fisher eigenvalues" in html
    assert "\\tau_j=-1/\\log|\\lambda_j|" in html
    assert "prefix/suffix participation" in html
    assert "\\mathcal S_a(s)" in html
    assert "static Schur complement is <strong>not</strong> the full-system spectrum" in html
    assert "\\rho(A)&lt;1" in html
    assert "pseudospectral checks" in html


def test_part_d_derives_frequency_response_and_tracking_modes():
    html = part_d_moment_mockup_html()

    assert "The suffix is a driven linear system" in html
    assert "B_u=-\\eta_bH_{bu}" in html
    assert "\\mathcal R_\\ell(z)" in html
    assert "K_H=" in html
    assert "\\Gamma_j(e^{i\\omega})" in html
    assert "\\widehat F_{\\rm eff}(\\omega)" in html
    assert "which distributional perturbations?" in html
    assert "c(v,\\omega)" in html
    assert "adaptive amplification" in html
    assert "small residual predictive response and small moving-optimum error" in html


def test_part_d_places_fisher_schur_as_static_geometric_limit():
    html = part_d_moment_mockup_html()

    assert "Fisher measures visible residuals; it does not set the poles" in html
    assert "G_{uu}-G_{ub}G_{bb}^{\\dagger}G_{bu}" in html
    assert "static Fisher/GGN Schur complement" in html
    assert "K_G=-G_{bb}^{\\dagger}G_{bu}" in html
    assert "K_H=-H_{bb}^{\\dagger}H_{bu}" in html
    assert "coincide only when" in html
    assert "exact update Jacobian" in html


def test_part_d_has_concrete_dashboard_and_defers_lyapunov_products():
    html = part_d_moment_mockup_html()

    for view in (
        "Complex eigenvalue plane",
        "Relaxation spectrum",
        "Response singular-value map",
        "Mode portraits",
        "DC triangle",
    ):
        assert view in html
    assert "Finite-difference JVP" in html
    assert "Amplitude sweep" in html
    assert "Finite nonlinear sinusoidal rollouts" in html
    assert "Products $A_{k+K-1}\\cdots A_k$" in html
    assert "finite-time Lyapunov exponents" in html
    assert "should follow only after the one-step operator is validated" in html


def test_part_d_contains_no_fake_result_language():
    lower = part_d_moment_mockup_html().lower()
    for result_claim in (
        "we observe", "we observed", "we find", "we found", "results show", "the data show"
    ):
        assert result_claim not in lower
