from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA

from tracking2.criticality import CriticalityConfig, datasets, make_loader, transplant_module
from tracking2.models import InstrumentedVGG19
from tracking2.vgg_suffix_statistics import encode


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate VGG cut PCA coverage before a C2/C3 run")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--initialization-checkpoint", required=True)
    parser.add_argument("--cut", type=int, default=7)
    parser.add_argument("--condition", choices=("native", "reset0"), default="reset0")
    parser.add_argument("--components", type=int, default=1024)
    parser.add_argument("--fit-size", type=int, default=5000)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    config = CriticalityConfig(
        data_root=args.data_root, train_size=args.train_size, test_size=1,
        batch_size=args.batch_size, seed=args.seed,
    )
    train_set, _ = datasets(config)
    train_data = make_loader(train_set, config, False)
    model = InstrumentedVGG19(batch_norm=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    if args.condition == "reset0":
        initial = InstrumentedVGG19(batch_norm=True).to(device)
        initial.load_state_dict(torch.load(args.initialization_checkpoint, map_location=device))
        transplant_module(model, initial, args.cut)

    representations, _ = encode(model, train_data, args.cut, device)
    flat = representations[: args.fit_size].reshape(args.fit_size, -1).astype(np.float32)
    count = min(args.components, len(flat) - 1, flat.shape[1])
    pca = PCA(n_components=count, svd_solver="randomized", random_state=args.seed + args.cut)
    pca.fit(flat)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    for target in (0.8, 0.85, 0.9):
        indices = np.flatnonzero(cumulative >= target)
        print(f"target={target:.0%} components={int(indices[0] + 1) if len(indices) else f'>{count}'}")
    for components in (128, 256, 384, 512, 768, 1024):
        if components <= count:
            print(f"components={components} coverage={cumulative[components - 1]:.6f}")


if __name__ == "__main__":
    main()
