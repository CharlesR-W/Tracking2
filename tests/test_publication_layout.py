import hashlib
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_FIGURES = {
    "class_conditioned_pushforward_pca.svg",
    "conceptual_internal_cut.svg",
    "cnn_measured_controls.png",
    "cnn_measured_horizon.png",
    "tracking_resolving.svg",
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


def test_pages_surface_is_appendix_plus_five_assets_only():
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "fetch-depth: 0" in workflow
    assert "verify_lw_post_publication.py --profile committed" in workflow
    assert "omnibus-report" not in workflow
    assert "report.html" not in workflow
    for basename in CANONICAL_FIGURES:
        assert workflow.count(f'LW post/figures/{basename}') == 1
