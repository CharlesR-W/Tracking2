import json

from tracking2.cnn_checkpoints import CNNCheckpointConfig, run
from tracking2.experiment import Config, resolve_data_backend


def test_explicit_torchvision_backend_does_not_silently_switch(tmp_path):
    (tmp_path / "cifar10-train.parquet").touch()
    (tmp_path / "cifar10-test.parquet").touch()
    assert resolve_data_backend(
        Config(data_root=str(tmp_path), data_backend="torchvision")
    ) == "torchvision"
    assert resolve_data_backend(
        Config(data_root=str(tmp_path), data_backend="auto")
    ) == "parquet"


def test_cnn_checkpoint_smoke_is_one_trajectory(tmp_path):
    path = run(
        CNNCheckpointConfig(
            output=str(tmp_path / "cnn"),
            fake_data=True,
            train_size=8,
            test_size=4,
            epochs=1,
            checkpoint_epochs=(0, 1),
            widths=(4, 4),
            batch_size=4,
            seed=0,
            device="cpu",
        )
    )
    artifact = json.loads(path.read_text())
    assert artifact["status"] == "MOCKUP / PIPELINE SMOKE TEST"
    assert artifact["dataset"]["backend"] == "fake_data"
    assert artifact["dataset"]["splits"]["train"]["count"] == 8
    assert len(
        artifact["dataset"]["splits"]["train"]["ordered_images_sha256"]
    ) == 64
    assert [row["epoch"] for row in artifact["checkpoints"]] == [0, 1]
    assert len({row["sha256"] for row in artifact["checkpoints"]}) == 2
    assert "one uninterrupted" in artifact["trajectory_semantics"]
