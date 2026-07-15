from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .criticality import CriticalityConfig, datasets, make_loader, transplant_module
from .models import InstrumentedVGG19
from .representation_surrogates import (
    fit_representation_surrogate,
    moment_diagnostics,
    sample_representation_surrogate,
)


@dataclass
class VGGSuffixStatisticsConfig:
    output: str = "artifacts/vgg_suffix_statistics/pilot"
    checkpoint: str = "artifacts/criticality/seed2/checkpoint_epoch100.pt"
    initialization_checkpoint: str | None = "artifacts/criticality/seed2/checkpoint_epoch0.pt"
    data_root: str = "data"
    fake_data: bool = False
    train_size: int = 10000
    test_size: int = 2000
    batch_size: int = 128
    classifier_width: int = 512
    width_multiplier: float = 1.0
    batch_norm: bool = True
    cuts: tuple[int, ...] = (7, 8, 9, 13)
    conditions: tuple[str, ...] = ("native", "reset0")
    pca_fit_size: int = 5000
    pca_components: int = 128
    # The report's validation gate requires enough independent surrogate
    # samples to distinguish a stable contrast from one lucky draw.
    surrogate_draws: int = 3
    relax_epochs: int = 5
    relax_learning_rate: float = 0.01
    seed: int = 0
    device: str = "auto"


def representation_loader(
    representations: np.ndarray, labels: np.ndarray, batch_size: int, seed: int, shuffle: bool
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(torch.from_numpy(representations), torch.from_numpy(labels.astype(np.int64))),
        batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0,
    )


@torch.no_grad()
def encode(
    model: InstrumentedVGG19, data: DataLoader, cut: int, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    representations, labels = [], []
    for x, y in data:
        representations.append(model.encode_to_module(x.to(device), cut).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(representations), np.concatenate(labels)


@torch.no_grad()
def evaluate_suffix(
    model: InstrumentedVGG19, cut: int, data: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    loss_sum = correct = count = 0
    for z, y in data:
        z, y = z.to(device), y.to(device)
        logits = model.forward_from_module(z, cut)
        loss_sum += float(F.cross_entropy(logits, y, reduction="sum"))
        correct += int((logits.argmax(1) == y).sum())
        count += len(y)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def run(config: VGGSuffixStatisticsConfig) -> Path:
    if not set(config.conditions) <= {"native", "reset0"}:
        raise ValueError("conditions must be native and/or reset0")
    if "reset0" in config.conditions and not config.initialization_checkpoint:
        raise ValueError("reset0 requires --initialization-checkpoint")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device if config.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()

    data_config = CriticalityConfig(
        data_root=config.data_root, fake_data=config.fake_data, train_size=config.train_size,
        test_size=config.test_size, batch_size=config.batch_size, seed=config.seed,
    )
    train_set, test_set = datasets(data_config)
    train_data = make_loader(train_set, data_config, False)
    test_data = make_loader(test_set, data_config, False)

    final = InstrumentedVGG19(
        classifier_width=config.classifier_width, width_multiplier=config.width_multiplier,
        batch_norm=config.batch_norm,
    ).to(device)
    final.load_state_dict(torch.load(config.checkpoint, map_location=device))
    initial = None
    if config.initialization_checkpoint:
        initial = copy.deepcopy(final)
        initial.load_state_dict(torch.load(config.initialization_checkpoint, map_location=device))

    module_names = [name for name, _ in final.intervention_modules()]
    slices: list[dict] = []
    for cut in config.cuts:
        if not 0 <= cut < 16:
            raise ValueError("cuts must select convolutional modules 0..15")
        for condition in config.conditions:
            model = copy.deepcopy(final)
            if condition == "reset0":
                assert initial is not None
                transplant_module(model, initial, cut)
            print(f"[slice] cut={cut}:{module_names[cut]} condition={condition}", flush=True)
            train_rep, train_labels = encode(model, train_data, cut, device)
            test_rep, test_labels = encode(model, test_data, cut, device)
            fit_count = min(config.pca_fit_size, len(train_rep))
            surrogate = fit_representation_surrogate(
                train_rep[:fit_count], train_labels[:fit_count],
                min(config.pca_components, fit_count - 1, train_rep[0].size),
                config.seed + cut,
            )
            projected = (train_rep[:fit_count].reshape(fit_count, -1) - surrogate.pca_mean) @ surrogate.components.T
            explained = float(np.var(projected, axis=0).sum() / np.var(train_rep[:fit_count].reshape(fit_count, -1), axis=0).sum())
            records: list[dict] = []
            diagnostics: list[dict] = []
            for draw in range(config.surrogate_draws):
                train_sets = {"true": train_rep}
                test_sets = {"true": test_rep}
                for offset, kind in enumerate(("mean", "gaussian"), start=1):
                    draw_seed = config.seed + 10000 * draw + 100 * cut + offset
                    train_sets[kind] = sample_representation_surrogate(surrogate, train_labels, kind, draw_seed)
                    test_sets[kind] = sample_representation_surrogate(surrogate, test_labels, kind, draw_seed + 5000)
                    diagnostics.append({"draw": draw, "distribution": kind, **moment_diagnostics(test_rep, test_sets[kind], test_labels)})
                eval_loaders = {
                    name: representation_loader(values, test_labels, config.batch_size, config.seed, False)
                    for name, values in test_sets.items()
                }
                for train_distribution in ("mean", "gaussian", "true"):
                    candidate = copy.deepcopy(model)
                    for parameter in candidate.parameters_through(cut):
                        parameter.requires_grad_(False)
                    optimizer = torch.optim.SGD(
                        candidate.parameters_after(cut), lr=config.relax_learning_rate, momentum=0.9
                    )
                    relax_data = representation_loader(
                        train_sets[train_distribution], train_labels, config.batch_size,
                        config.seed + draw, True,
                    )
                    for relax_epoch in range(config.relax_epochs + 1):
                        for eval_distribution, eval_data in eval_loaders.items():
                            records.append({
                                "draw": draw, "train_distribution": train_distribution,
                                "eval_distribution": eval_distribution, "relax_epoch": relax_epoch,
                                **evaluate_suffix(candidate, cut, eval_data, device),
                            })
                        if relax_epoch == config.relax_epochs:
                            break
                        candidate.train()
                        for z, y in relax_data:
                            z, y = z.to(device), y.to(device)
                            optimizer.zero_grad(set_to_none=True)
                            F.cross_entropy(candidate.forward_from_module(z, cut), y).backward()
                            optimizer.step()
            slices.append({
                "cut": cut, "module": module_names[cut], "condition": condition,
                "representation_shape": list(train_rep.shape[1:]),
                "explained_variance_fraction": explained,
                "moment_diagnostics": diagnostics, "records": records,
            })
    artifact = {
        "schema_version": 1, "experiment": "vgg_suffix_statistics_bridge",
        "status": "MOCKUP / PIPELINE SMOKE TEST" if config.fake_data else "MEASURED",
        "config": asdict(config), "device": str(device), "module_names": module_names,
        "slices": slices, "runtime_seconds": time.time() - started,
    }
    path = output / "vgg_suffix_statistics.json"
    path.write_text(json.dumps(artifact, indent=2, allow_nan=False))
    print(f"[done] {path}", flush=True)
    return path


def parse_args() -> VGGSuffixStatisticsConfig:
    parser = argparse.ArgumentParser(description="VGG native/transplanted activation-statistics bridge")
    parser.add_argument("--output", default=VGGSuffixStatisticsConfig.output)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initialization-checkpoint")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--fake-data", action="store_true")
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--classifier-width", type=int, default=512)
    parser.add_argument("--width-multiplier", type=float, default=1.0)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--cuts", type=int, nargs="+", default=[7, 8, 9, 13])
    parser.add_argument("--conditions", nargs="+", default=["native", "reset0"])
    parser.add_argument("--pca-fit-size", type=int, default=5000)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--surrogate-draws", type=int, default=1)
    parser.add_argument("--relax-epochs", type=int, default=5)
    parser.add_argument("--relax-learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    return VGGSuffixStatisticsConfig(
        output=args.output, checkpoint=args.checkpoint,
        initialization_checkpoint=args.initialization_checkpoint, data_root=args.data_root,
        fake_data=args.fake_data, train_size=args.train_size, test_size=args.test_size,
        batch_size=args.batch_size, classifier_width=args.classifier_width,
        width_multiplier=args.width_multiplier, batch_norm=args.batch_norm,
        cuts=tuple(args.cuts), conditions=tuple(args.conditions),
        pca_fit_size=args.pca_fit_size, pca_components=args.pca_components,
        surrogate_draws=args.surrogate_draws, relax_epochs=args.relax_epochs,
        relax_learning_rate=args.relax_learning_rate, seed=args.seed, device=args.device,
    )


if __name__ == "__main__":
    run(parse_args())
