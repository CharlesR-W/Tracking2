import copy
import json
from collections import defaultdict

import pytest
import torch

import tracking2.resnet_suffix_statistics as resnet_statistics
from scripts import validate_architecture_checkpoints as checkpoint_validator
from scripts.verify_lw_post_resnet_ablations import verify_distribution_grid
from tracking2.cnn_checkpoints import sha256
from tracking2.models import InstrumentedResNet18V2
from tracking2.resnet_criticality import ResNetCriticalityConfig, run as run_criticality
from tracking2.resnet_suffix_statistics import (
    ResNetSuffixStatisticsConfig,
    evaluate_suffix,
    run as run_suffix,
)
from tracking2.vgg_suffix_statistics import representation_loader


def smoke_training(tmp_path):
    manifest = run_criticality(ResNetCriticalityConfig(
        output=str(tmp_path / "criticality"), fake_data=True,
        train_size=8, test_size=4, epochs=0, batch_size=4,
        checkpoint_epochs=(0,), width=4, seed=0, device="cpu", amp=False,
        measure_criticality=False,
    ))
    return manifest, manifest.parent / "checkpoint_epoch0.pt"


def test_resnet_block_split_matches_full_forward():
    model = InstrumentedResNet18V2(width=4).eval()
    x = torch.randn(4, 3, 32, 32)
    for cut in range(8):
        with torch.no_grad():
            representation = model.encode_to_block(x, cut)
            actual = model.forward_from_block(representation, cut)
            expected = model(x)
        assert torch.allclose(actual, expected, atol=1e-6)


def test_resnet_suffix_evaluation_matches_full_model():
    model = InstrumentedResNet18V2(width=4).eval()
    x = torch.randn(4, 3, 32, 32)
    y = torch.arange(4) % 10
    cut = 3
    with torch.no_grad():
        representation = model.encode_to_block(x, cut).numpy()
        expected = model(x)
    loader = representation_loader(representation, y.numpy(), 2, seed=0, shuffle=False)
    result = evaluate_suffix(model, cut, loader, torch.device("cpu"))
    assert abs(result["loss"] - torch.nn.functional.cross_entropy(expected, y).item()) < 1e-5


def test_resnet_criticality_and_suffix_smokes(tmp_path):
    criticality_path, checkpoint = smoke_training(tmp_path)
    criticality = json.loads(criticality_path.read_text())
    assert criticality["status"] == "MOCKUP / PIPELINE SMOKE TEST"
    assert criticality["experiment"] == "resnet18_training_checkpoints"
    assert criticality["architecture"]["name"] == "InstrumentedResNet18V2"
    assert criticality["module_names"] == [
        "stage0", "stage1.resblk1", "stage1.resblk2", "stage2.resblk1",
        "stage2.resblk2", "stage3.resblk1", "stage3.resblk2",
        "stage4.resblk1", "stage4.resblk2", "final_linear",
    ]
    assert criticality["checkpoints"] == [{
        "epoch": 0,
        "path": "checkpoint_epoch0.pt",
        "sha256": sha256(checkpoint),
    }]
    assert criticality["dataset"]["backend"] == "fake"
    suffix_path = run_suffix(ResNetSuffixStatisticsConfig(
        output=str(tmp_path / "suffix"), checkpoint=str(checkpoint),
        training_manifest=str(criticality_path), checkpoint_epoch=0,
        fake_data=True, train_size=8, test_size=4, batch_size=4, width=4,
        cuts=(3,), pca_fit_size=8, pca_ranks=(3,), surrogate_draws=1,
        relax_epochs=1, seed=0, device="cpu",
    ))
    suffix = json.loads(suffix_path.read_text())
    assert suffix["status"] == "MOCKUP / PIPELINE SMOKE TEST"
    assert suffix["schema_version"] == 3
    assert suffix["checkpoint"]["epoch"] == 0
    assert len(suffix["checkpoint"]["sha256"]) == 64
    assert suffix["lineage"]["model_seed"] == 0
    assert suffix["lineage"]["architecture"] == criticality["architecture"]
    assert suffix["lineage"]["training_manifest"]["sha256"] == sha256(
        criticality_path
    )
    assert suffix["analysis_banks"]["train"]["seed"] == 2_000_000
    assert suffix["analysis_banks"]["test"]["seed"] == 2_000_001
    assert suffix["analysis_banks"]["train"]["count"] == 8
    assert suffix["analysis_banks"]["test"]["count"] == 4
    assert suffix["slices"][0]["module"] == "stage2.resblk2"
    assert len(suffix["slices"][0]["reference_records"]) == 2
    rank_result = suffix["slices"][0]["rank_results"][0]
    assert rank_result["pca_rank"] == 3
    assert "held_out_explained_variance_fraction" in rank_result
    assert len(rank_result["records"]) == 12


def test_resnet_suffix_smoke_supports_projected_and_noise_controls(tmp_path):
    criticality_path, checkpoint = smoke_training(tmp_path)
    suffix_path = run_suffix(ResNetSuffixStatisticsConfig(
        output=str(tmp_path / "suffix"), checkpoint=str(checkpoint),
        training_manifest=str(criticality_path), checkpoint_epoch=0,
        fake_data=True, train_size=8, test_size=4, batch_size=4, width=4,
        cuts=(3,), pca_fit_size=6, pca_ranks=(2, 3), surrogate_draws=1,
        mean_noise_radii=(0.0, 1.0), include_projected_true=True,
        true_eval_only=True, relax_epochs=1, seed=0, device="cpu",
    ))
    suffix = json.loads(suffix_path.read_text())
    slice_result = suffix["slices"][0]
    assert len(slice_result["reference_records"]) == 2
    assert [row["pca_rank"] for row in slice_result["rank_results"]] == [2, 3]
    for rank_result in slice_result["rank_results"]:
        records = rank_result["records"]
        assert len(records) == 8
        assert {row["train_distribution"] for row in records} == {
            "mean_r0", "mean_r1", "gaussian", "projected_true",
        }
        assert rank_result["pca_basis_fit_count"] == 6
        assert rank_result["moment_fit_count"] == 8
        assert sum(rank_result["pca_basis_fit_class_counts"].values()) == 6
        assert sum(rank_result["moment_fit_class_counts"].values()) == 8
        assert rank_result["covariance_shrinkage"] == 0.05
        assert rank_result[
            "trace_matched_isotropic_covariance_trace"
        ] == pytest.approx(
            rank_result["pca_rank"]
            * rank_result["pooled_within_class_variance_per_pca_coordinate"]
        )
        assert (
            rank_result["maximal_pca_basis_sha256"]
            == slice_result["maximal_pca_basis_sha256"]
        )


def test_resnet_suffix_rejects_checkpoint_lineage_mismatches(tmp_path):
    training_path, checkpoint = smoke_training(tmp_path)
    original = json.loads(training_path.read_text())
    cases = {}

    wrong_status = copy.deepcopy(original)
    wrong_status["status"] = "MEASURED"
    cases["status"] = wrong_status
    wrong_seed = copy.deepcopy(original)
    wrong_seed["config"]["seed"] = 1
    cases["seed"] = wrong_seed
    wrong_architecture = copy.deepcopy(original)
    wrong_architecture["architecture"]["width"] = 8
    cases["architecture"] = wrong_architecture
    wrong_epoch = copy.deepcopy(original)
    wrong_epoch["checkpoints"][0]["epoch"] = 1
    cases["epoch"] = wrong_epoch
    wrong_hash = copy.deepcopy(original)
    wrong_hash["checkpoints"][0]["sha256"] = "0" * 64
    cases["hash"] = wrong_hash

    for name, payload in cases.items():
        manifest = training_path.parent / f"{name}.json"
        manifest.write_text(json.dumps(payload))
        with pytest.raises((ValueError, FileNotFoundError)):
            run_suffix(ResNetSuffixStatisticsConfig(
                output=str(tmp_path / f"suffix-{name}"),
                checkpoint=str(checkpoint),
                training_manifest=str(manifest),
                checkpoint_epoch=0,
                fake_data=True,
                train_size=8,
                test_size=4,
                batch_size=4,
                width=4,
                cuts=(3,),
                pca_fit_size=6,
                pca_ranks=(2,),
                surrogate_draws=1,
                relax_epochs=0,
                seed=0,
                device="cpu",
            ))


def test_nested_sweep_pairs_shuffle_and_mean_noise_seeds(
    tmp_path, monkeypatch
):
    training_path, checkpoint = smoke_training(tmp_path)
    relax_calls = []
    sample_calls = []
    fit_lengths = []
    moment_fit_lengths = []
    bank_dataset_ids = defaultdict(set)
    current_rank = None

    original_encode = resnet_statistics.encode
    original_fit = resnet_statistics.fit_representation_surrogate
    original_truncate = resnet_statistics.truncate_representation_surrogate
    original_sample = resnet_statistics.sample_representation_surrogate

    def fit_spy(representations, *args, **kwargs):
        fit_lengths.append(len(representations))
        return original_fit(representations, *args, **kwargs)

    def truncate_spy(model, representations, labels, *args, **kwargs):
        nonlocal current_rank
        current_rank = args[0]
        moment_fit_lengths.append(len(representations))
        return original_truncate(
            model, representations, labels, *args, **kwargs
        )

    def sample_spy(model, labels, kind, seed, **kwargs):
        sample_calls.append(
            (
                current_rank,
                kind,
                len(labels),
                seed,
                kwargs.get("mean_noise_radius", 1.0),
                kwargs.get("paired_noise_rank"),
            )
        )
        return original_sample(model, labels, kind, seed, **kwargs)

    def encode_spy(model, data, cut, device):
        bank_dataset_ids[len(data.dataset)].add(id(data.dataset))
        return original_encode(model, data, cut, device)

    def relax_spy(
        *args,
        distribution,
        draw,
        shuffle_seed,
        **_kwargs,
    ):
        relax_calls.append((args[1], distribution, draw, shuffle_seed))
        return [{
            "draw": draw,
            "train_distribution": distribution,
            "eval_distribution": "true",
            "relax_epoch": 0,
            "loss": 1.0,
            "accuracy": 0.5,
        }]

    monkeypatch.setattr(
        resnet_statistics, "fit_representation_surrogate", fit_spy
    )
    monkeypatch.setattr(
        resnet_statistics, "truncate_representation_surrogate", truncate_spy
    )
    monkeypatch.setattr(
        resnet_statistics, "sample_representation_surrogate", sample_spy
    )
    monkeypatch.setattr(resnet_statistics, "encode", encode_spy)
    monkeypatch.setattr(resnet_statistics, "_relax_suffix", relax_spy)

    run_suffix(ResNetSuffixStatisticsConfig(
        output=str(tmp_path / "suffix-pairing"),
        checkpoint=str(checkpoint),
        training_manifest=str(training_path),
        checkpoint_epoch=0,
        fake_data=True,
        train_size=8,
        test_size=4,
        batch_size=4,
        width=4,
        cuts=(2, 3),
        pca_fit_size=6,
        pca_ranks=(2, 3),
        surrogate_draws=1,
        mean_noise_radii=(0.0, 1.0),
        include_projected_true=True,
        true_eval_only=True,
        relax_epochs=0,
        seed=0,
        device="cpu",
    ))

    assert fit_lengths == [6, 6]
    assert moment_fit_lengths == [8, 8, 8, 8]
    assert sum(
        distribution == "true"
        for _, distribution, _, _ in relax_calls
    ) == 2
    for cut in (2, 3):
        assert {
            shuffle_seed
            for call_cut, _, _, shuffle_seed in relax_calls
            if call_cut == cut
        } == {1000 * cut}
    assert all(len(dataset_ids) == 1 for dataset_ids in bank_dataset_ids.values())

    mean_seed_groups = defaultdict(set)
    seeds_by_rank_and_condition = defaultdict(set)
    for rank, kind, count, seed, radius, paired_noise_rank in sample_calls:
        assert paired_noise_rank == 3
        seeds_by_rank_and_condition[(rank, kind, count, radius)].add(seed)
        if kind == "mean":
            mean_seed_groups[(count, seed)].add(radius)
    conditions = {
        (kind, count, radius)
        for _, kind, count, _, radius, _ in sample_calls
    }
    for kind, count, radius in conditions:
        assert (
            seeds_by_rank_and_condition[(2, kind, count, radius)]
            == seeds_by_rank_and_condition[(3, kind, count, radius)]
        )
    assert len(mean_seed_groups) == 4
    assert all(radii == {0.0, 1.0} for radii in mean_seed_groups.values())


def test_measured_resnet_analysis_requires_torchvision_backend():
    with pytest.raises(ValueError, match="data_backend='torchvision'"):
        run_suffix(ResNetSuffixStatisticsConfig(
            checkpoint="missing.pt",
            training_manifest="missing.json",
            fake_data=False,
            data_backend="auto",
        ))


def test_resnet_verifier_rejects_duplicate_cartesian_cells():
    records = [
        {
            "draw": draw,
            "train_distribution": distribution,
            "eval_distribution": "true",
            "relax_epoch": epoch,
        }
        for draw in (0, 1)
        for distribution in ("true", "gaussian")
        for epoch in (0, 1)
    ]
    verify_distribution_grid(
        records,
        distributions={"true", "gaussian"},
        draws={0, 1},
        epochs={0, 1},
        context="fixture",
    )
    duplicate = [*records[:-1], records[0]]
    with pytest.raises(RuntimeError, match="duplicate"):
        verify_distribution_grid(
            duplicate,
            distributions={"true", "gaussian"},
            draws={0, 1},
            epochs={0, 1},
            context="fixture",
        )


def test_resnet_manifest_validator_rejects_reused_checkpoint_weights(tmp_path):
    checkpoints = []
    for epoch in checkpoint_validator.RESNET_EPOCHS:
        path = tmp_path / f"checkpoint_epoch{epoch}.pt"
        path.write_bytes(b"identical checkpoint")
        checkpoints.append(
            {
                "epoch": epoch,
                "path": path.name,
                "sha256": checkpoint_validator.sha256(path),
            }
        )
    payload = {
        "status": "MEASURED",
        "experiment": "resnet18_training_checkpoints",
        "architecture": {
            "name": "InstrumentedResNet18V2",
            "width": 64,
            "block_names": checkpoint_validator.RESNET_BLOCKS,
        },
        "provenance": {"source_revision": "a" * 40},
        "config": {
            "data_backend": "torchvision",
            "fake_data": False,
            "train_size": 50000,
            "test_size": 10000,
            "epochs": 100,
            "batch_size": 128,
            "learning_rate": 0.05,
            "checkpoint_epochs": list(checkpoint_validator.RESNET_EPOCHS),
            "lr_milestones": [30, 60, 90],
            "seed": 0,
            "weight_decay": 0.0,
            "width": 64,
            "measure_criticality": False,
        },
        "dataset": {
            "backend": "torchvision",
            "train": {"source_count": 50000, "source_sha256": "b" * 64},
            "test": {"source_count": 10000, "source_sha256": "c" * 64},
        },
        "checkpoints": checkpoints,
    }

    errors = checkpoint_validator.validate_resnet(
        tmp_path / "resnet_training.json", payload, seed=0
    )
    assert errors == [
        "checkpoint SHA-256 digests must be distinct across epochs"
    ]
