import hashlib
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FIGURES = {
    "closing_cycle.svg",
    "cnn_over_training_time.gif",
    "cnn_relaxation_time.gif",
    "cnn_resnet_architectures.svg",
    "fbd_materials_nn_analogy.svg",
    "method_make_datasets.svg",
    "method_train_compare.svg",
    "resnet_over_training_time.gif",
    "tracking_resolving_only.png",
}
TALK_EXPORT_DIGESTS = {
    "Tracking2-talk-slides.html": (
        "989f8cb41ad7f096b3f34edb6dc3b54b0f2f586ed6d4b3a3a041ade2c42e2090"
    ),
    "Tracking2-talk-slides.pdf": (
        "f41ce72ac5b00e3b46f27c85b4f961d261e2c973c039d8e85be0bcdcef29dfcb"
    ),
    "slides.html": (
        "3e3ad9bd69232a37732b139e3d5f9f97bc0d4fc2aea27be52ddbcf331863afc8"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_current_publication_figure_directory_has_exact_allowlist():
    figure_dir = PROJECT_ROOT / "LW post" / "figures"
    assert {path.name for path in figure_dir.iterdir()} == CANONICAL_FIGURES

    manifest = json.loads(
        (PROJECT_ROOT / "LW post" / "figure_manifest.json").read_text()
    )
    assert manifest["schema_version"] == 1
    assert manifest["prose_restoration_commit"].startswith("9218f5d")
    assert manifest["asset_git_import_commit"].startswith("68d8ecb")
    assert "not verified" in manifest["pre_import_asset_provenance"]
    assert "restoration_commit" not in manifest
    assert {asset["filename"] for asset in manifest["assets"]} == CANONICAL_FIGURES
    for asset in manifest["assets"]:
        assert asset["source_family"] == "delivered talk"
        assert _sha256(figure_dir / asset["filename"]) == asset["sha256"]
        assert (figure_dir / asset["filename"]).read_bytes() == (
            PROJECT_ROOT / "LW post" / "deprecated" / "figures" / asset["filename"]
        ).read_bytes()


def test_post_uses_every_active_figure_and_no_rejected_redesign_asset():
    post = (
        PROJECT_ROOT / "LW post" / "free-body-diagrams-for-neural-networks.md"
    ).read_text()
    image_targets = set(re.findall(r"!\[[^]]*\]\(([^)]+)\)", post))
    assert image_targets == {
        f"https://charlesr-w.github.io/Tracking2/figures/{name}"
        for name in CANONICAL_FIGURES
    }
    assert "conceptual_internal_cut.svg" not in post
    assert "class_conditioned_pushforward_pca.svg" not in post
    assert "cnn_measured_controls.png" not in post
    assert "cnn_measured_horizon.png" not in post
    assert "tracking_resolving.svg" not in post


def test_delivered_talk_is_byte_identical_and_local_assets_resolve():
    talk_dir = PROJECT_ROOT / "LW post" / "deprecated" / "talk"
    for name, expected_digest in TALK_EXPORT_DIGESTS.items():
        assert _sha256(talk_dir / name) == expected_digest

    references = re.findall(
        r'(?:src|href)="([^"]+)"', (talk_dir / "slides.html").read_text()
    )
    local_references = [
        reference
        for reference in references
        if not reference.startswith(("http://", "https://", "data:", "#"))
    ]
    assert len(local_references) == 19
    assert len(set(local_references)) == 18
    assert all(
        (talk_dir / reference).resolve().is_file()
        for reference in local_references
    )


def test_omnibus_archive_has_only_the_seven_tracked_artifact_trees():
    archive = PROJECT_ROOT / "deprecated" / "2026-07-omnibus"
    artifact_root = archive / "artifacts"
    assert {path.name for path in artifact_root.iterdir()} == {
        "cifar_pilot_seed0",
        "confirmatory",
        "criticality",
        "suffix_statistics",
        "resnet_suffix_statistics",
        "vgg_batchzoom",
        "vgg_suffix_statistics",
    }
    assert len([path for path in artifact_root.rglob("*") if path.is_file()]) == 72
    assert _sha256(archive / "report.html") == (
        "698d809448a54be3dec57c1aae1263aa2516f30f8ec7c6ef1b9e1473b8196eb3"
    )


def test_pages_surface_is_appendix_plus_manifest_assets_only():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "fetch-depth: 0" in workflow
    assert "verify_lw_post_publication.py --profile committed" in workflow
    assert "omnibus-report" not in workflow
    assert "report.html" not in workflow
    for basename in CANONICAL_FIGURES:
        assert workflow.count(f'LW post/figures/{basename}') == 1
