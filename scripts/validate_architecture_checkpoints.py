from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


RESNET_EPOCHS = (0, 1, 5, 20, 100)
RESNET_BLOCKS = [
    "stage1.resblk1",
    "stage1.resblk2",
    "stage2.resblk1",
    "stage2.resblk2",
    "stage3.resblk1",
    "stage3.resblk2",
    "stage4.resblk1",
    "stage4.resblk2",
]
SOURCE_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_resnet(manifest_path: Path, payload: dict, seed: int) -> list[str]:
    errors: list[str] = []
    if payload.get("status") != "MEASURED":
        errors.append("status must be exactly MEASURED")
    if payload.get("experiment") != "resnet18_training_checkpoints":
        errors.append("experiment must be resnet18_training_checkpoints")
    expected_architecture = {
        "name": "InstrumentedResNet18V2",
        "width": 64,
        "block_names": RESNET_BLOCKS,
    }
    if payload.get("architecture") != expected_architecture:
        errors.append("architecture metadata mismatch")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or not (
        isinstance(provenance.get("source_revision"), str)
        and SOURCE_REVISION.fullmatch(provenance["source_revision"]) is not None
        or isinstance(provenance.get("source_archive_sha256"), str)
        and SHA256.fullmatch(provenance["source_archive_sha256"]) is not None
    ):
        errors.append(
            "provenance must record a clean full source revision or source archive SHA-256"
        )

    config = payload.get("config")
    if not isinstance(config, dict):
        return [*errors, "config must be an object"]
    expected_config = {
        "data_backend": "torchvision",
        "fake_data": False,
        "train_size": 50000,
        "test_size": 10000,
        "epochs": 100,
        "batch_size": 128,
        "learning_rate": 0.05,
        "checkpoint_epochs": list(RESNET_EPOCHS),
        "lr_milestones": [30, 60, 90],
        "seed": seed,
        "weight_decay": 0.0,
        "width": 64,
        "measure_criticality": False,
    }
    for key, expected in expected_config.items():
        if config.get(key) != expected:
            errors.append(
                f"config.{key}: expected {expected!r}, found {config.get(key)!r}"
            )

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("backend") != "torchvision":
        errors.append("dataset backend/fingerprints must record torchvision")
    else:
        for split, expected_count in (("train", 50000), ("test", 10000)):
            split_data = dataset.get(split)
            if not isinstance(split_data, dict):
                errors.append(f"dataset.{split} must be an object")
                continue
            if split_data.get("source_count") != expected_count:
                errors.append(
                    f"dataset.{split}.source_count must be {expected_count}"
                )
            digest = split_data.get("source_sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                errors.append(f"dataset.{split}.source_sha256 is invalid")

    records = payload.get("checkpoints")
    if not isinstance(records, list):
        return [*errors, "checkpoints must be an array"]
    epochs = [record.get("epoch") for record in records if isinstance(record, dict)]
    if epochs != list(RESNET_EPOCHS):
        errors.append(
            f"checkpoint epochs: expected {list(RESNET_EPOCHS)!r}, found {epochs!r}"
        )
    verified_hashes: list[str] = []
    for epoch in RESNET_EPOCHS:
        matches = [
            record
            for record in records
            if isinstance(record, dict) and record.get("epoch") == epoch
        ]
        if len(matches) != 1:
            errors.append(f"checkpoint epoch {epoch} must appear exactly once")
            continue
        record = matches[0]
        expected_name = f"checkpoint_epoch{epoch}.pt"
        if record.get("path") != expected_name:
            errors.append(
                f"checkpoint epoch {epoch} path must be {expected_name!r}"
            )
            continue
        checkpoint = manifest_path.parent / expected_name
        if not checkpoint.is_file():
            errors.append(f"checkpoint file missing: {checkpoint}")
            continue
        observed_hash = sha256(checkpoint)
        if record.get("sha256") != observed_hash:
            errors.append(f"checkpoint epoch {epoch} SHA-256 mismatch")
            continue
        verified_hashes.append(observed_hash)
    if (
        len(verified_hashes) == len(RESNET_EPOCHS)
        and len(set(verified_hashes)) != len(verified_hashes)
    ):
        errors.append("checkpoint SHA-256 digests must be distinct across epochs")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the training manifest before reusing architecture checkpoints."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("architecture", choices=("resnet", "vgg"))
    parser.add_argument("seed", type=int)
    args = parser.parse_args()

    if not args.manifest.exists():
        print(f"[checkpoint manifest missing] {args.manifest}")
        return 1
    payload = json.loads(args.manifest.read_text())
    if args.architecture == "resnet":
        errors = validate_resnet(args.manifest, payload, args.seed)
        if errors:
            print(
                "[checkpoint manifest mismatch] "
                + json.dumps(errors, sort_keys=True)
            )
            return 1
        print(f"[checkpoint manifest valid] {args.manifest}")
        return 0

    config = payload.get("config", {})
    expected = {
        "fake_data": False,
        "train_size": 50000,
        "test_size": 10000,
        "epochs": 100,
        "batch_size": 128,
        "checkpoint_epochs": [0, 1, 5, 20, 100],
        "lr_milestones": [30, 60, 90],
        "seed": args.seed,
    }
    expected.update({
        "batch_norm": True,
        "classifier_width": 512,
        "width_multiplier": 1.0,
    })
    mismatches = {
        key: {"expected": value, "found": config.get(key)}
        for key, value in expected.items() if config.get(key) != value
    }
    if mismatches:
        print(f"[checkpoint manifest mismatch] {json.dumps(mismatches, sort_keys=True)}")
        return 1
    print(f"[checkpoint manifest valid] {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
