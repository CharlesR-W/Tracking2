from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .models import SplitConvNet


@dataclass
class RefitResult:
    loss_before: float
    loss_after: float
    prediction_kl: float
    path_length: float
    refit_model: SplitConvNet


@torch.no_grad()
def batch_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float(F.cross_entropy(model(x), y))


def refit_suffix(
    model: SplitConvNet,
    cut: int,
    train_batches: list[tuple[torch.Tensor, torch.Tensor]],
    eval_batch: tuple[torch.Tensor, torch.Tensor],
    steps: int,
    learning_rate: float,
) -> RefitResult:
    """Warm-start the downstream optimum with a fixed cached representation."""
    candidate = copy.deepcopy(model)
    candidate.train()
    for parameter in candidate.prefix_parameters(cut):
        parameter.requires_grad_(False)
    cached = []
    with torch.no_grad():
        for x, y in train_batches:
            cached.append((candidate.encode_to(x, cut).detach(), y))
    optimizer = torch.optim.SGD(candidate.suffix_parameters(cut), lr=learning_rate, momentum=0.9)
    initial = [p.detach().clone() for p in candidate.suffix_parameters(cut)]
    for step in range(steps):
        representation, y = cached[step % len(cached)]
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(candidate.forward_from(representation, cut), y)
        loss.backward()
        optimizer.step()
    x_eval, y_eval = eval_batch
    model.eval()
    candidate.eval()
    with torch.no_grad():
        logits_before = model(x_eval)
        logits_after = candidate(x_eval)
        loss_before = float(F.cross_entropy(logits_before, y_eval))
        loss_after = float(F.cross_entropy(logits_after, y_eval))
        prediction_kl = float(
            F.kl_div(logits_before.log_softmax(-1), logits_after.softmax(-1), reduction="batchmean")
        )
    path_sq = sum(float((p.detach() - p0).pow(2).sum()) for p, p0 in zip(candidate.suffix_parameters(cut), initial))
    return RefitResult(loss_before, loss_after, prediction_kl, path_sq**0.5, candidate)


def cross_representation_loss(
    head_model: SplitConvNet,
    representation_model: SplitConvNet,
    cut: int,
    batch: tuple[torch.Tensor, torch.Tensor],
) -> float:
    x, y = batch
    head_model.eval()
    representation_model.eval()
    with torch.no_grad():
        representation = representation_model.encode_to(x, cut)
        logits = head_model.forward_from(representation, cut)
        return float(F.cross_entropy(logits, y))


def relative_representation_drift(
    previous_model: SplitConvNet,
    current_model: SplitConvNet,
    cut: int,
    x: torch.Tensor,
) -> float:
    previous_model.eval()
    current_model.eval()
    with torch.no_grad():
        previous = previous_model.encode_to(x, cut)
        current = current_model.encode_to(x, cut)
        denominator = previous.pow(2).sum().sqrt().clamp_min(1e-12)
        return float((current - previous).pow(2).sum().sqrt() / denominator)


def suffix_update_decomposition(
    previous_model: SplitConvNet,
    current_model: SplitConvNet,
    previous_optimum: SplitConvNet,
    current_optimum: SplitConvNet,
    cut: int,
) -> dict[str, float]:
    """Project the actual suffix update onto tracking and resolving directions.

    Tracking is movement of the refitted optimum. Resolving points from the
    previous actual suffix toward its refitted optimum. The two directions need
    not be orthogonal, so we report cosines and a joint two-vector regression.
    """
    def vector(model: SplitConvNet) -> torch.Tensor:
        return torch.cat([parameter.detach().flatten() for parameter in model.suffix_parameters(cut)])

    previous = vector(previous_model)
    current = vector(current_model)
    previous_star = vector(previous_optimum)
    current_star = vector(current_optimum)
    actual = current - previous
    tracking = current_star - previous_star
    resolving = previous_star - previous

    def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
        denominator = left.norm() * right.norm()
        return float(left.dot(right) / denominator) if denominator > 0 else 0.0

    basis = torch.stack([tracking, resolving], dim=1)
    gram = basis.T @ basis
    ridge = 1e-8 * torch.trace(gram).clamp_min(1.0)
    coefficients = torch.linalg.solve(gram + ridge * torch.eye(2, device=gram.device), basis.T @ actual)
    fitted = basis @ coefficients
    explained = 1.0 - float((actual - fitted).pow(2).sum() / actual.pow(2).sum().clamp_min(1e-12))
    return {
        "actual_suffix_update_norm": float(actual.norm()),
        "tracking_direction_norm": float(tracking.norm()),
        "resolving_direction_norm": float(resolving.norm()),
        "tracking_alignment": cosine(actual, tracking),
        "resolving_alignment": cosine(actual, resolving),
        "tracking_coefficient": float(coefficients[0]),
        "resolving_coefficient": float(coefficients[1]),
        "two_direction_explained_fraction": explained,
    }


def _flatten_grads(grads: tuple[torch.Tensor | None, ...], params: list[nn.Parameter]) -> torch.Tensor:
    pieces = []
    for grad, parameter in zip(grads, params):
        pieces.append(torch.zeros_like(parameter).flatten() if grad is None else grad.flatten())
    return torch.cat(pieces) if pieces else torch.zeros(0)


def empirical_fisher_schur_trace(
    model: SplitConvNet,
    cut: int,
    batch: tuple[torch.Tensor, torch.Tensor],
    damping_ratio: float = 0.1,
) -> dict[str, float]:
    """Small-sample empirical-Fisher traces and suffix-compensated prefix trace.

    If rows of Ga/Gb are per-example score gradients, the Schur correction is
    computed entirely through m-by-m sample Gram matrices.
    """
    x, y = batch
    prefix = model.prefix_parameters(cut)
    suffix = model.suffix_parameters(cut)
    if not prefix or not suffix:
        return {"prefix_trace": 0.0, "suffix_trace": 0.0, "effective_prefix_trace": 0.0, "compensable_fraction": 0.0}
    rows_a, rows_b = [], []
    model.eval()
    for index in range(len(x)):
        logits = model(x[index : index + 1])
        loss = F.cross_entropy(logits, y[index : index + 1])
        grads = torch.autograd.grad(loss, prefix + suffix, allow_unused=True)
        rows_a.append(_flatten_grads(grads[: len(prefix)], prefix).detach())
        rows_b.append(_flatten_grads(grads[len(prefix) :], suffix).detach())
    ga, gb = torch.stack(rows_a), torch.stack(rows_b)
    m = len(x)
    ka = ga @ ga.T
    kb = gb @ gb.T
    prefix_trace = float(torch.trace(ka) / m)
    suffix_trace = float(torch.trace(kb) / m)
    eye = torch.eye(m, device=kb.device, dtype=kb.dtype)
    # The empirical Fisher has rank at most m. An absolute near-zero ridge makes
    # a sufficiently expressive suffix absorb essentially the whole sample span.
    # Scale the ridge to the mean nonzero sample-kernel eigenvalue so values are
    # comparable across layers and checkpoints.
    ridge = damping_ratio * float(torch.trace(kb) / m)
    absorption = kb @ torch.linalg.solve(kb + ridge * eye, eye)
    correction = float(torch.trace(ka @ absorption) / m)
    effective = max(0.0, prefix_trace - correction)
    fraction = correction / prefix_trace if prefix_trace > 0 else 0.0
    return {
        "prefix_trace": prefix_trace,
        "suffix_trace": suffix_trace,
        "effective_prefix_trace": effective,
        "compensable_fraction": fraction,
        "fisher_ridge_ratio": damping_ratio,
        "fisher_sample_rank_cap": float(m),
    }
