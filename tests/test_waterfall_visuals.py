from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from tracking2.waterfall_visuals import (
    CNN_DURATIONS_MS,
    RELAXATION_DURATIONS_MS,
    RESNET_DURATIONS_MS,
    build_payload,
    build_waterfall_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "artifacts" / "waterfall_visuals" / "manifest.json"


def test_payload_pins_complete_resnet_first_pilot() -> None:
    payload = build_payload(PROJECT_ROOT, MANIFEST)

    assert payload["canonical_example"] == "resnet18"
    assert payload["evidence_status"] == (
        "historical pilots; not independent-seed confirmation"
    )
    assert payload["input_count"] == 29
    assert payload["resnet"]["epochs"] == [0, 1, 5, 20, 100]
    assert payload["resnet"]["cuts"] == list(range(1, 9))
    assert payload["resnet"]["model_units"] == 1
    assert payload["resnet"]["redraws_per_cell"] == 3
    assert payload["cnn"]["epochs"] == [0, 1, 5, 10, 20, 30]
    assert payload["cnn"]["checkpoint_semantics"] == "separately scheduled models"

    assert payload["resnet"]["series"]["mean"][4][0] == pytest.approx(
        -51.3666666667
    )
    assert payload["resnet"]["series"]["gaussian"][4][0] == pytest.approx(-12.7)
    assert payload["resnet"]["series"]["true"][4][0] == pytest.approx(
        -0.1166666667
    )
    assert payload["cnn"]["series"]["mean"][5] == pytest.approx(
        [-33.83, -19.85, -8.0, 0.17]
    )


def test_relaxation_curve_is_one_fixed_checkpoint() -> None:
    payload = build_payload(PROJECT_ROOT, MANIFEST)
    relaxation = payload["relaxation"]

    assert relaxation["checkpoint_epoch"] == 1
    assert relaxation["cut"] == 3
    assert relaxation["series"]["true"]["steps"] == [float(i) for i in range(11)]
    assert relaxation["series"]["true"]["accuracy"][-1] == pytest.approx(50.26)


def test_tracked_outputs_are_byte_identical_to_rebuild(tmp_path: Path) -> None:
    figure_dir = tmp_path / "figures"
    html_path = tmp_path / "waterfalls.html"
    outputs = build_waterfall_outputs(
        PROJECT_ROOT,
        MANIFEST,
        figure_dir,
        html_path,
    )

    assert html_path.read_bytes() == (PROJECT_ROOT / "LW post" / "waterfalls.html").read_bytes()
    for output in outputs.figures:
        tracked = PROJECT_ROOT / "LW post" / "figures" / output.name
        assert output.read_bytes() == tracked.read_bytes(), output.name

    expected = {
        "resnet_over_training_time.gif": RESNET_DURATIONS_MS,
        "cnn_over_training_time.gif": CNN_DURATIONS_MS,
        "cnn_relaxation_time.gif": RELAXATION_DURATIONS_MS,
    }
    for name, durations in expected.items():
        with Image.open(figure_dir / name) as image:
            actual = []
            for frame_index in range(image.n_frames):
                image.seek(frame_index)
                actual.append(int(image.info["duration"]))
            assert image.size == (1440, 810)
            assert actual == durations
