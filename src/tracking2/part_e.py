"""Part E: Part-B-style suffix relaxation for GPT-2 next-token prediction.

The module is deliberately artifact-first.  ``prepare`` creates deterministic
token/replay banks, ``train`` writes resumable GPT-2-small checkpoints, and
``analyze`` rebuilds every suffix measurement from those saved objects.  Raw
activation caches are transient; model checkpoints and scientific artifacts are
retained.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import create_causal_mask


SCHEMA_VERSION = 2
DIAGNOSTIC_DISTRIBUTIONS = (
    "mean", "token_gaussian", "seq_gaussian", "projected_true", "true",
)


@dataclass(frozen=True)
class GateConfig:
    output: str
    source_run: str | None = None
    seed: int = 0
    context_length: int = 256
    train_loss_tokens: int = 50_000_000
    validation_tokens: int = 2_000_000
    replay_fit_stories: int = 512
    replay_eval_stories: int = 128
    train_batch_size: int = 32
    grad_accumulation: int = 1
    learning_rate: float = 3e-4
    warmup_steps: int = 200
    checkpoint_tokens: tuple[int, ...] = (0, 4_000_000, 16_000_000, 50_000_000)
    analysis_checkpoint_tokens: tuple[int, ...] = (4_000_000, 16_000_000, 50_000_000)
    cuts: tuple[int, ...] = (0, 5, 11)
    pca_components: int = 512
    pca_min_components: int = 128
    pca_target_coverage: float = 0.95
    pca_fit_rows: int = 32_768
    token_groups: int = 256
    position_bins: int = 16
    covariance_shrinkage: float = 0.05
    position_shrinkage: float = 0.10
    covariance_fit_stories: int = 128
    gradient_matched_lr: bool = True
    suffix_updates: int = 64
    suffix_batch_size: int = 8
    suffix_learning_rate: float = 5e-5
    eval_stories: int = 64
    eval_batch_size: int = 8
    cache_batch_size: int = 8
    dtype: str = "bfloat16"
    dataset_id: str = "roneneldan/TinyStories"
    dataset_revision: str = "main"
    tokenizer_id: str = "gpt2"
    tokenizer_revision: str = "main"
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def checkpoint_name(tokens: int) -> str:
    return f"tokens_{tokens:09d}"


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


def autocast_context(device: torch.device, dtype: torch.dtype):
    enabled = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def batched(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _tokenize_batch(tokenizer: GPT2TokenizerFast, texts: list[str]) -> list[list[int]]:
    encoded = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)["input_ids"]
    eos = tokenizer.eos_token_id
    return [list(tokens) + [eos] for tokens in encoded]


def _stream_to_memmap(
    stream: Iterable[dict],
    tokenizer: GPT2TokenizerFast,
    path: Path,
    token_count: int,
    batch_size: int = 256,
) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(path, mode="w+", dtype=np.int32, shape=(token_count,))
    cursor = 0
    stories = 0
    pending: list[str] = []
    for row in stream:
        pending.append(row["text"])
        if len(pending) < batch_size:
            continue
        for ids in _tokenize_batch(tokenizer, pending):
            take = min(len(ids), token_count - cursor)
            output[cursor : cursor + take] = ids[:take]
            cursor += take
            stories += 1
            if cursor >= token_count:
                output.flush()
                return {"tokens": cursor, "stories": stories, "sha256": sha256_file(path)}
        pending.clear()
    if pending and cursor < token_count:
        for ids in _tokenize_batch(tokenizer, pending):
            take = min(len(ids), token_count - cursor)
            output[cursor : cursor + take] = ids[:take]
            cursor += take
            stories += 1
            if cursor >= token_count:
                break
    output.flush()
    if cursor != token_count:
        raise RuntimeError(f"Dataset ended after {cursor:,} tokens; requested {token_count:,}")
    return {"tokens": cursor, "stories": stories, "sha256": sha256_file(path)}


def _build_replay_banks(
    stream: Iterable[dict],
    tokenizer: GPT2TokenizerFast,
    context_length: int,
    fit_count: int,
    eval_count: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    required = fit_count + eval_count
    sequences: list[list[int]] = []
    story_hashes: list[str] = []
    pending: list[str] = []
    for row in stream:
        pending.append(row["text"])
        if len(pending) < 256:
            continue
        tokenized = _tokenize_batch(tokenizer, pending)
        for text, ids in zip(pending, tokenized, strict=True):
            if len(ids) >= context_length:
                sequences.append(ids[:context_length])
                story_hashes.append(sha256_bytes(text.encode()))
                if len(sequences) >= required:
                    array = np.asarray(sequences, dtype=np.int32)
                    return array[:fit_count], array[fit_count:], story_hashes
        pending.clear()
    raise RuntimeError(f"Only found {len(sequences)} stories with at least {context_length} tokens")


def prepare_data(config: GateConfig) -> None:
    output = Path(config.output)
    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained(
        config.tokenizer_id, revision=config.tokenizer_revision,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_dir = data_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    train_path = data_dir / "train_tokens.bin"
    validation_path = data_dir / "validation_tokens.bin"
    if not train_path.exists():
        train_stream = load_dataset(
            config.dataset_id, split="train", streaming=True, revision=config.dataset_revision,
        )
        train_meta = _stream_to_memmap(
            train_stream, tokenizer, train_path,
            config.train_loss_tokens + config.context_length + 1,
        )
    else:
        train_meta = {"tokens": train_path.stat().st_size // 4, "sha256": sha256_file(train_path)}

    if not validation_path.exists():
        validation_stream = load_dataset(
            config.dataset_id, split="validation", streaming=True, revision=config.dataset_revision,
        )
        validation_meta = _stream_to_memmap(
            validation_stream, tokenizer, validation_path,
            config.validation_tokens + config.context_length + 1,
        )
    else:
        validation_meta = {
            "tokens": validation_path.stat().st_size // 4,
            "sha256": sha256_file(validation_path),
        }

    replay_fit_path = data_dir / "replay_fit.npz"
    replay_eval_path = data_dir / "replay_eval.npz"
    if not replay_fit_path.exists() or not replay_eval_path.exists():
        replay_stream = load_dataset(
            config.dataset_id, split="validation", streaming=True, revision=config.dataset_revision,
        )
        fit_ids, eval_ids, story_hashes = _build_replay_banks(
            replay_stream, tokenizer, config.context_length,
            config.replay_fit_stories, config.replay_eval_stories,
        )
        attention_fit = np.ones_like(fit_ids, dtype=np.int8)
        attention_eval = np.ones_like(eval_ids, dtype=np.int8)
        np.savez_compressed(replay_fit_path, input_ids=fit_ids, attention_mask=attention_fit)
        np.savez_compressed(replay_eval_path, input_ids=eval_ids, attention_mask=attention_eval)
    else:
        story_hashes = []

    tokenizer_hashes = {
        path.name: sha256_file(path) for path in sorted(tokenizer_dir.iterdir()) if path.is_file()
    }
    replay_manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "tokenizer_id": config.tokenizer_id,
        "tokenizer_revision": config.tokenizer_revision,
        "tokenizer_hashes": tokenizer_hashes,
        "context_length": config.context_length,
        "train": train_meta,
        "validation": validation_meta,
        "replay_fit": {
            "stories": config.replay_fit_stories,
            "path": str(replay_fit_path.relative_to(output)),
            "sha256": sha256_file(replay_fit_path),
        },
        "replay_eval": {
            "stories": config.replay_eval_stories,
            "path": str(replay_eval_path.relative_to(output)),
            "sha256": sha256_file(replay_eval_path),
        },
        "story_hashes": story_hashes,
    }
    write_json(output / "replay_manifest.json", replay_manifest)


class TokenBlocks(Dataset):
    def __init__(self, path: Path, context_length: int):
        self.tokens = np.memmap(path, mode="r", dtype=np.int32)
        self.context_length = context_length
        self.blocks = (len(self.tokens) - 1) // context_length

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int) -> torch.Tensor:
        start = index * self.context_length
        values = np.asarray(self.tokens[start : start + self.context_length], dtype=np.int64)
        return torch.from_numpy(values.copy())


def infinite_loader(loader: DataLoader) -> Iterator[torch.Tensor]:
    while True:
        yield from loader


@torch.no_grad()
def evaluate_lm(
    model: GPT2LMHeadModel,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    batches: int = 32,
) -> float:
    model.eval()
    losses = []
    for index, input_ids in enumerate(loader):
        if index >= batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        with autocast_context(device, dtype):
            losses.append(float(model(input_ids=input_ids, labels=input_ids, use_cache=False).loss))
    model.train()
    return float(np.mean(losses))


def _save_checkpoint(
    model: GPT2LMHeadModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
    step: int,
    loss_tokens: int,
    config: GateConfig,
    data_rng: torch.Generator | None = None,
) -> dict:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    state = {
        "step": step,
        "loss_tokens": loss_tokens,
        "optimizer": optimizer.state_dict(),
        "gate_config": asdict(config),
        "torch_rng_state": torch.get_rng_state(),
    }
    if data_rng is not None:
        state["data_rng_state"] = data_rng.get_state()
    if torch.cuda.is_available():
        state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(state, checkpoint_dir / "training_state.pt")
    return {
        "path": str(checkpoint_dir),
        "model_sha256": sha256_file(checkpoint_dir / "model.safetensors"),
        "state_sha256": sha256_file(checkpoint_dir / "training_state.pt"),
        "step": step,
        "loss_tokens": loss_tokens,
    }


def train_model(config: GateConfig) -> None:
    seed_everything(config.seed)
    output = Path(config.output)
    data_dir = output / "data"
    if not (data_dir / "train_tokens.bin").exists():
        raise FileNotFoundError("Run Part E prepare before train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(config.dtype)
    tokenizer = GPT2TokenizerFast.from_pretrained(data_dir / "tokenizer")

    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=config.context_length,
        n_ctx=config.context_length,
        n_embd=config.n_embd,
        n_layer=config.n_layer,
        n_head=config.n_head,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    train_dataset = TokenBlocks(data_dir / "train_tokens.bin", config.context_length)
    validation_dataset = TokenBlocks(data_dir / "validation_tokens.bin", config.context_length)
    validation_loader = DataLoader(
        validation_dataset, batch_size=config.train_batch_size, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=True,
    )
    metrics_path = output / "training_metrics.jsonl"

    tokens_per_update = (
        config.train_batch_size * (config.context_length - 1) * config.grad_accumulation
    )
    total_steps = math.ceil(config.train_loss_tokens / tokens_per_update)
    checkpoint_root = output / "checkpoints"
    existing: list[tuple[int, Path]] = []
    for tokens in config.checkpoint_tokens:
        path = checkpoint_root / checkpoint_name(tokens)
        if (path / "model.safetensors").exists() and (path / "training_state.pt").exists():
            existing.append((tokens, path))

    data_rng = torch.Generator().manual_seed(config.seed + 17_003)
    if existing:
        _, resume_path = max(existing)
        state = torch.load(resume_path / "training_state.pt", map_location="cpu", weights_only=False)
        saved_config = state.get("gate_config", {})
        compatibility_keys = (
            "seed", "context_length", "train_loss_tokens", "train_batch_size",
            "grad_accumulation", "learning_rate", "warmup_steps", "n_embd", "n_layer", "n_head",
        )
        mismatches = {
            key: (saved_config.get(key), getattr(config, key))
            for key in compatibility_keys if saved_config.get(key) != getattr(config, key)
        }
        if mismatches:
            raise RuntimeError(f"Checkpoint/config mismatch; use a new output directory: {mismatches}")
        model = GPT2LMHeadModel.from_pretrained(resume_path).to(device)
        model.config.use_cache = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
        if "data_rng_state" not in state:
            raise RuntimeError(
                "This checkpoint predates exact data-stream resumption; use a new output directory."
            )
        data_rng.set_state(state["data_rng_state"])
        start_step = int(state["step"])
        resume_loss_tokens = int(state["loss_tokens"])
        checkpoints = []
        for _, path in sorted(existing):
            prior_state = torch.load(
                path / "training_state.pt", map_location="cpu", weights_only=False,
            )
            checkpoints.append({
                "path": str(path), "model_sha256": sha256_file(path / "model.safetensors"),
                "state_sha256": sha256_file(path / "training_state.pt"),
                "step": int(prior_state["step"]),
                "loss_tokens": int(prior_state["loss_tokens"]),
            })
        print(json.dumps({"event": "resume", "checkpoint": str(resume_path), "step": start_step}), flush=True)
    else:
        model = GPT2LMHeadModel(model_config).to(device)
        model.tie_weights()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.1)
        start_step = 0
        resume_loss_tokens = 0
        initial = _save_checkpoint(
            model, optimizer, checkpoint_root / checkpoint_name(0), 0, 0, config, data_rng,
        )
        checkpoints = [initial]

    checkpoint_targets = [target for target in config.checkpoint_tokens if target > resume_loss_tokens]
    start_time = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)

    for step in range(start_step + 1, total_steps + 1):
        step_loss = 0.0
        for _ in range(config.grad_accumulation):
            indices = torch.randint(0, len(train_dataset), (config.train_batch_size,), generator=data_rng)
            input_ids = torch.stack([train_dataset[int(index)] for index in indices]).to(
                device, non_blocking=True,
            )
            with autocast_context(device, dtype):
                loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
                loss = loss / config.grad_accumulation
            loss.backward()
            step_loss += float(loss.detach())
        progress = min(1.0, step / total_steps)
        if step <= config.warmup_steps:
            lr_scale = step / max(1, config.warmup_steps)
        else:
            decay_progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
            lr_scale = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        for group in optimizer.param_groups:
            group["lr"] = config.learning_rate * lr_scale
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        loss_tokens = min(config.train_loss_tokens, step * tokens_per_update)
        if step == 1 or step % 50 == 0 or loss_tokens >= config.train_loss_tokens:
            record = {
                "step": step,
                "loss_tokens": loss_tokens,
                "train_loss": step_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "elapsed_seconds": time.time() - start_time,
                "tokens_per_second": loss_tokens / max(time.time() - start_time, 1e-9),
                "device": str(device),
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record), flush=True)

        while checkpoint_targets and loss_tokens >= checkpoint_targets[0]:
            target = checkpoint_targets.pop(0)
            validation_loss = evaluate_lm(model, validation_loader, device, dtype)
            checkpoint_dir = output / "checkpoints" / checkpoint_name(target)
            saved = _save_checkpoint(model, optimizer, checkpoint_dir, step, loss_tokens, config, data_rng)
            saved["validation_loss"] = validation_loss
            checkpoints.append(saved)
            append_jsonl(metrics_path, {
                "event": "checkpoint", "target_loss_tokens": target,
                "validation_loss": validation_loss, **saved,
            })
            print(json.dumps({"checkpoint": saved}), flush=True)

    write_json(output / "checkpoint_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "checkpoints": checkpoints,
        "retention": "model and optimizer state retained for every declared checkpoint",
    })


def _load_replay(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = np.load(path)
    return (
        torch.from_numpy(payload["input_ids"].astype(np.int64)),
        torch.from_numpy(payload["attention_mask"].astype(np.int64)),
    )


@torch.no_grad()
def cache_cut_activations(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cuts: tuple[int, ...],
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    logit_limit: int = 0,
    storage_dtype: torch.dtype = torch.float16,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    model.eval()
    cached: dict[int, list[torch.Tensor]] = {cut: [] for cut in cuts}
    logits_batches: list[torch.Tensor] = []
    current: dict[int, torch.Tensor] = {}
    hooks = []
    for cut in cuts:
        def capture(_module, _inputs, output, cut_index=cut):
            current[cut_index] = output[0] if isinstance(output, tuple) else output
        hooks.append(model.transformer.h[cut].register_forward_hook(capture))
    try:
        for start in range(0, len(input_ids), batch_size):
            ids = input_ids[start : start + batch_size].to(device)
            mask = attention_mask[start : start + batch_size].to(device)
            current.clear()
            with autocast_context(device, dtype):
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            retained = max(0, logit_limit - sum(len(part) for part in logits_batches))
            if retained:
                logits_batches.append(logits[:retained].detach().to("cpu", dtype=storage_dtype))
            for cut in cuts:
                cached[cut].append(current[cut].detach().to("cpu", dtype=storage_dtype))
    finally:
        for hook in hooks:
            hook.remove()
    retained_logits = torch.cat(logits_batches) if logits_batches else torch.empty(0)
    return {cut: torch.cat(parts) for cut, parts in cached.items()}, retained_logits


class GPT2Suffix(nn.Module):
    """Independent GPT-2 suffix with a deliberately untied LM head."""

    def __init__(self, model: GPT2LMHeadModel, cut: int):
        super().__init__()
        self.config = copy.deepcopy(model.config)
        self.blocks = nn.ModuleList(copy.deepcopy(model.transformer.h[cut + 1 :]))
        self.ln_f = copy.deepcopy(model.transformer.ln_f)
        self.lm_head = nn.Linear(model.config.n_embd, model.config.vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(model.lm_head.weight.detach().clone())
        self.tie_policy = "cloned_untied_for_suffix_relaxation"

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length = hidden_states.shape[:2]
        cache_position = torch.arange(sequence_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0)
        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=hidden_states,
            attention_mask=attention_mask.view(batch_size, -1),
            cache_position=cache_position,
            past_key_values=None,
            position_ids=position_ids,
        )
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                past_key_values=None,
                cache_position=cache_position,
                attention_mask=causal_mask,
                head_mask=None,
                use_cache=False,
                output_attentions=False,
            )[0]
        return self.lm_head(self.ln_f(hidden_states))


def ntp_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )


@dataclass
class SequenceSurrogate:
    hidden_mean: torch.Tensor
    basis: torch.Tensor
    group_lookup: torch.Tensor
    group_means: torch.Tensor
    position_means: torch.Tensor
    position_bins: torch.Tensor
    channel_cholesky: torch.Tensor
    position_cholesky: torch.Tensor
    isotropic_scale: float
    last_position_mean: torch.Tensor
    coverage: float
    diagnostics: dict

    @property
    def components(self) -> int:
        return self.basis.shape[1]

    def conditional_scores(self, targets: torch.Tensor) -> torch.Tensor:
        groups = self.group_lookup[targets]
        return self.group_means[groups] + self.position_means[self.position_bins][None, :, :]

    def sample(
        self,
        input_ids: torch.Tensor,
        distribution: str,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if distribution == "true":
            raise ValueError("True activations are supplied directly")
        targets = input_ids[:, 1:]
        mean_scores = self.conditional_scores(targets)
        shape = mean_scores.shape
        noise = torch.randn(shape, generator=generator, device=mean_scores.device)
        if distribution == "mean":
            scores = mean_scores + noise * self.isotropic_scale
        elif distribution == "token_gaussian":
            noise = torch.einsum("ntk,jk->ntj", noise, self.channel_cholesky)
            scores = mean_scores + noise
        elif distribution == "seq_gaussian":
            noise = torch.einsum("ts,nsk->ntk", self.position_cholesky, noise)
            noise = torch.einsum("ntk,jk->ntj", noise, self.channel_cholesky)
            scores = mean_scores + noise
        else:
            raise KeyError(distribution)
        hidden = self.hidden_mean + torch.einsum("ntk,dk->ntd", scores, self.basis)
        last = self.last_position_mean.expand(len(input_ids), 1, -1)
        return torch.cat([hidden, last], dim=1)

    def project(self, hidden: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("ntd,dk->ntk", hidden[:, :-1] - self.hidden_mean, self.basis)
        reconstructed = self.hidden_mean + torch.einsum("ntk,dk->ntd", scores, self.basis)
        return torch.cat([reconstructed, hidden[:, -1:]], dim=1)


def _group_statistics(
    scores: torch.Tensor,
    targets: torch.Tensor,
    vocab_size: int,
    group_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat_targets = targets.reshape(-1)
    frequencies = torch.bincount(flat_targets, minlength=vocab_size)
    top = torch.topk(frequencies, k=min(group_count, int((frequencies > 0).sum()))).indices
    lookup = torch.full((vocab_size,), len(top), dtype=torch.long, device=scores.device)
    lookup[top] = torch.arange(len(top), device=scores.device)
    groups = lookup[flat_targets]
    flat_scores = scores.reshape(-1, scores.shape[-1])
    sums = torch.zeros(len(top) + 1, scores.shape[-1], device=scores.device)
    counts = torch.zeros(len(top) + 1, device=scores.device)
    sums.index_add_(0, groups, flat_scores)
    counts.index_add_(0, groups, torch.ones_like(groups, dtype=torch.float32))
    means = sums / counts.clamp_min(1)[:, None]
    global_mean = flat_scores.mean(0)
    means[counts == 0] = global_mean
    return lookup, means


@torch.no_grad()
def fit_sequence_surrogate(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    config: GateConfig,
    device: torch.device,
) -> SequenceSurrogate:
    # Work in float32; randomly cap the rows used for PCA while using all rows
    # for conditional means/covariances after projection.
    hidden = hidden.to(device=device, dtype=torch.float32)
    input_ids = input_ids.to(device)
    rows = hidden[:, :-1].reshape(-1, hidden.shape[-1])
    row_generator = torch.Generator(device=device).manual_seed(config.seed + 910)
    if len(rows) > config.pca_fit_rows:
        indices = torch.randperm(len(rows), generator=row_generator, device=device)[: config.pca_fit_rows]
        pca_rows = rows[indices]
    else:
        pca_rows = rows
    hidden_mean = pca_rows.mean(0)
    centered = pca_rows - hidden_mean
    max_q = min(config.pca_components, centered.shape[0] - 1, centered.shape[1])
    _, singular_values, full_basis = torch.pca_lowrank(
        centered, q=max_q, center=False, niter=4,
    )
    total_variance = centered.square().sum() / max(1, len(centered) - 1)
    cumulative = singular_values.square().cumsum(0) / max(1, len(centered) - 1)
    coverage_curve = cumulative / total_variance.clamp_min(1e-12)
    target_indices = torch.nonzero(coverage_curve >= config.pca_target_coverage)
    target_q = int(target_indices[0]) + 1 if len(target_indices) else max_q
    q = min(max_q, max(config.pca_min_components, target_q))
    basis = full_basis[:, :q]
    captured_variance = cumulative[q - 1]
    coverage = float((captured_variance / total_variance.clamp_min(1e-12)).cpu())

    scores = torch.einsum("ntd,dk->ntk", hidden[:, :-1] - hidden_mean, basis)
    targets = input_ids[:, 1:]
    lookup, group_means = _group_statistics(
        scores, targets, config.vocab_size if hasattr(config, "vocab_size") else 50_257,
        config.token_groups,
    )
    position_bins = torch.div(
        torch.arange(scores.shape[1], device=device) * config.position_bins,
        scores.shape[1], rounding_mode="floor",
    ).clamp_max(config.position_bins - 1)
    base = group_means[lookup[targets]]
    residual_for_position = scores - base
    position_means = torch.zeros(config.position_bins, q, device=device)
    position_counts = torch.zeros(config.position_bins, device=device)
    expanded_bins = position_bins[None, :].expand(len(scores), -1).reshape(-1)
    position_means.index_add_(0, expanded_bins, residual_for_position.reshape(-1, q))
    position_counts.index_add_(0, expanded_bins, torch.ones_like(expanded_bins, dtype=torch.float32))
    position_means /= position_counts.clamp_min(1)[:, None]
    mean_scores = base + position_means[position_bins][None, :, :]
    residual = scores - mean_scores

    covariance_residual = residual[: config.covariance_fit_stories]
    flat_residual = covariance_residual.reshape(-1, q)
    channel_identity = torch.eye(q, device=device)
    channel_cov = flat_residual.T @ flat_residual / max(1, len(flat_residual) - 1)
    channel_diagonal = torch.diag(torch.diag(channel_cov))
    channel_cov = (
        (1 - config.covariance_shrinkage) * channel_cov
        + config.covariance_shrinkage * channel_diagonal
        + 1e-5 * channel_identity
    )

    # One stable flip-flop refinement for a separable matrix-normal covariance:
    # estimate rows after channel whitening, then channels after row whitening.
    initial_channel_cholesky = torch.linalg.cholesky(channel_cov)
    channel_whitened = torch.linalg.solve_triangular(
        initial_channel_cholesky,
        covariance_residual.transpose(1, 2),
        upper=False,
    )
    position_cov = torch.einsum(
        "nkt,nks->ts", channel_whitened, channel_whitened,
    ) / max(1, len(covariance_residual) * q)
    position_cov /= torch.diag(position_cov).mean().clamp_min(1e-8)
    position_identity = torch.eye(position_cov.shape[0], device=device)
    position_cov = (
        (1 - config.position_shrinkage) * position_cov
        + config.position_shrinkage * position_identity
        + 1e-5 * position_identity
    )
    position_cholesky = torch.linalg.cholesky(position_cov)
    position_whitened = torch.linalg.solve_triangular(
        position_cholesky, covariance_residual, upper=False,
    )
    flat_whitened = position_whitened.reshape(-1, q)
    channel_cov = flat_whitened.T @ flat_whitened / max(1, len(flat_whitened) - 1)
    channel_diagonal = torch.diag(torch.diag(channel_cov))
    channel_cov = (
        (1 - config.covariance_shrinkage) * channel_cov
        + config.covariance_shrinkage * channel_diagonal
        + 1e-5 * channel_identity
    )
    channel_cholesky = torch.linalg.cholesky(channel_cov)
    isotropic_scale = float(torch.sqrt(torch.trace(channel_cov) / q).cpu())

    fitted_channel = torch.cov(flat_residual.T)
    diagnostics = {
        "pca_coverage": coverage,
        "pca_components": q,
        "pca_max_components": max_q,
        "pca_target_coverage": config.pca_target_coverage,
        "pca_target_met": coverage >= config.pca_target_coverage,
        "fit_rows": len(pca_rows),
        "conditional_mean_rmse": float(torch.sqrt(residual.square().mean()).cpu()),
        "channel_covariance_trace": float(torch.trace(fitted_channel).cpu()),
        "position_covariance_offdiag_abs_mean": float(
            (position_cov - torch.diag(torch.diag(position_cov))).abs().mean().cpu()
        ),
        "token_groups": int(group_means.shape[0]),
        "covariance_fit_stories": len(covariance_residual),
        "covariance_estimator": "one_step_flip_flop_channel_then_position_then_channel",
    }
    return SequenceSurrogate(
        hidden_mean=hidden_mean,
        basis=basis,
        group_lookup=lookup,
        group_means=group_means,
        position_means=position_means,
        position_bins=position_bins,
        channel_cholesky=channel_cholesky,
        position_cholesky=position_cholesky,
        isotropic_scale=isotropic_scale,
        last_position_mean=hidden[:, -1:].mean(0),
        coverage=coverage,
        diagnostics=diagnostics,
    )


def _surrogate_to_npz(surrogate: SequenceSurrogate, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        hidden_mean=surrogate.hidden_mean.detach().cpu().numpy(),
        basis=surrogate.basis.detach().cpu().numpy(),
        group_lookup=surrogate.group_lookup.detach().cpu().numpy(),
        group_means=surrogate.group_means.detach().cpu().numpy(),
        position_means=surrogate.position_means.detach().cpu().numpy(),
        position_bins=surrogate.position_bins.detach().cpu().numpy(),
        channel_cholesky=surrogate.channel_cholesky.detach().cpu().numpy(),
        position_cholesky=surrogate.position_cholesky.detach().cpu().numpy(),
        isotropic_scale=np.asarray(surrogate.isotropic_scale),
        last_position_mean=surrogate.last_position_mean.detach().cpu().numpy(),
        coverage=np.asarray(surrogate.coverage),
    )


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


@torch.no_grad()
def evaluate_suffix_distribution(
    suffix: GPT2Suffix,
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    suffix.eval()
    losses = []
    parameter_dtype = next(suffix.parameters()).dtype
    for start in range(0, len(input_ids), batch_size):
        h = hidden[start : start + batch_size].to(device=device, dtype=parameter_dtype)
        ids = input_ids[start : start + batch_size].to(device)
        mask = attention_mask[start : start + batch_size].to(device)
        with autocast_context(device, dtype):
            loss = ntp_cross_entropy(suffix(h, mask), ids)
        losses.append((float(loss), len(ids)))
    return sum(loss * count for loss, count in losses) / sum(count for _, count in losses)


def _future_leakage_check(
    surrogate: SequenceSurrogate,
    input_ids: torch.Tensor,
    device: torch.device,
    distribution: str,
    seed: int,
) -> float:
    boundary = input_ids.shape[1] // 2
    original = input_ids[:4].to(device).clone()
    permuted = original.clone()
    permuted[:, boundary + 1 :] = torch.flip(permuted[:, boundary + 1 :], dims=[1])
    first = surrogate.sample(original, distribution, _device_generator(device, seed))
    second = surrogate.sample(permuted, distribution, _device_generator(device, seed))
    return float((first[:, :boundary] - second[:, :boundary]).abs().max().cpu())


def _make_eval_banks(
    surrogate: SequenceSurrogate,
    true_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    device: torch.device,
    seed: int,
) -> dict[str, torch.Tensor]:
    ids = input_ids.to(device)
    return {
        "true": true_hidden.to(device),
        "projected_true": surrogate.project(true_hidden.to(device=device, dtype=torch.float32)),
        "mean": surrogate.sample(ids, "mean", _device_generator(device, seed + 1)),
        "token_gaussian": surrogate.sample(
            ids, "token_gaussian", _device_generator(device, seed + 2),
        ),
        "seq_gaussian": surrogate.sample(
            ids, "seq_gaussian", _device_generator(device, seed + 3),
        ),
    }


def _parity_metrics(
    suffix: GPT2Suffix,
    hidden: torch.Tensor,
    full_logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    suffix.eval()
    parameter_dtype = next(suffix.parameters()).dtype
    with torch.no_grad(), autocast_context(device, torch.float32):
        replay_logits = suffix(
            hidden[:2].to(device=device, dtype=parameter_dtype),
            attention_mask[:2].to(device),
        )
    reference = full_logits[:2].to(device=device, dtype=replay_logits.dtype)
    max_abs = float((replay_logits - reference).abs().max().cpu())
    replay_loss = float(ntp_cross_entropy(replay_logits, input_ids[:2].to(device)).cpu())
    reference_loss = float(ntp_cross_entropy(reference, input_ids[:2].to(device)).cpu())
    return {
        "max_abs_logit_error": max_abs,
        "replay_loss": replay_loss,
        "reference_loss": reference_loss,
        "absolute_loss_error": abs(replay_loss - reference_loss),
    }


def _training_hidden(
    distribution: str,
    surrogate: SequenceSurrogate,
    true_hidden: torch.Tensor,
    ids: torch.Tensor,
    indices: torch.Tensor,
    generator: torch.Generator,
    dtype: torch.dtype,
) -> torch.Tensor:
    if distribution == "true":
        return true_hidden.to(device=ids.device, dtype=dtype)[indices]
    if distribution == "projected_true":
        return surrogate.project(true_hidden.to(device=ids.device, dtype=torch.float32)[indices])
    return surrogate.sample(ids[indices], distribution, generator)


def initial_gradient_norm(
    model: GPT2LMHeadModel,
    cut: int,
    distribution: str,
    surrogate: SequenceSurrogate,
    fit_hidden_true: torch.Tensor,
    fit_ids: torch.Tensor,
    fit_mask: torch.Tensor,
    config: GateConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> float:
    suffix = GPT2Suffix(model, cut).to(device).eval()
    generator = _device_generator(device, seed)
    indices = torch.randint(
        0, len(fit_ids), (config.suffix_batch_size,), generator=generator, device=device,
    )
    ids = fit_ids.to(device)
    mask = fit_mask.to(device)
    hidden = _training_hidden(
        distribution, surrogate, fit_hidden_true, ids, indices, generator,
        next(suffix.parameters()).dtype,
    )
    with autocast_context(device, dtype):
        loss = ntp_cross_entropy(suffix(hidden, mask[indices]), ids[indices])
    loss.backward()
    norm = float(torch.nn.utils.clip_grad_norm_(suffix.parameters(), float("inf")).cpu())
    del suffix
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return norm


def relax_one_suffix(
    model: GPT2LMHeadModel,
    cut: int,
    train_distribution: str,
    surrogate: SequenceSurrogate,
    fit_hidden_true: torch.Tensor,
    fit_ids: torch.Tensor,
    fit_mask: torch.Tensor,
    eval_banks: dict[str, torch.Tensor],
    eval_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    config: GateConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    initial_grad_norm: float,
    learning_rate_scale: float,
) -> tuple[list[dict], GPT2Suffix]:
    suffix = GPT2Suffix(model, cut).to(device)
    suffix.eval()  # keep dropout disabled while retaining gradients
    effective_learning_rate = config.suffix_learning_rate * learning_rate_scale
    optimizer = torch.optim.AdamW(suffix.parameters(), lr=effective_learning_rate, weight_decay=0.0)
    generator = _device_generator(device, seed)
    eval_updates = {0, 1, 2, 4, 8, 16, 32, config.suffix_updates}
    records: list[dict] = []
    latest_train_loss: float | None = None
    latest_gradient_norm: float | None = None

    def evaluate(update: int) -> None:
        for eval_distribution in DIAGNOSTIC_DISTRIBUTIONS:
            loss = evaluate_suffix_distribution(
                suffix, eval_banks[eval_distribution], eval_ids, eval_mask,
                config.eval_batch_size, device, dtype,
            )
            records.append({
                "train_distribution": train_distribution,
                "eval_distribution": eval_distribution,
                "update": update,
                "loss_nats_per_token": loss,
                "train_loss_nats_per_token": latest_train_loss,
                "gradient_norm": latest_gradient_norm,
                "initial_gradient_norm": initial_grad_norm,
                "learning_rate_scale": learning_rate_scale,
                "effective_learning_rate": effective_learning_rate,
            })

    evaluate(0)
    for update in range(1, config.suffix_updates + 1):
        indices = torch.randint(
            0, len(fit_ids), (config.suffix_batch_size,), generator=generator, device=device,
        )
        batch_ids = fit_ids.to(device)[indices]
        batch_mask = fit_mask.to(device)[indices]
        batch_hidden = _training_hidden(
            train_distribution, surrogate, fit_hidden_true, fit_ids.to(device),
            indices, generator, next(suffix.parameters()).dtype,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, dtype):
            loss = ntp_cross_entropy(suffix(batch_hidden, batch_mask), batch_ids)
        loss.backward()
        latest_train_loss = float(loss.detach().cpu())
        latest_gradient_norm = float(torch.nn.utils.clip_grad_norm_(suffix.parameters(), 1.0).cpu())
        optimizer.step()
        if update in eval_updates:
            evaluate(update)
    return records, suffix


def _analyze_cell(
    model: GPT2LMHeadModel,
    checkpoint_tokens: int,
    cut: int,
    fit_hidden: torch.Tensor,
    eval_hidden: torch.Tensor,
    full_eval_logits: torch.Tensor,
    fit_ids: torch.Tensor,
    fit_mask: torch.Tensor,
    eval_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    config: GateConfig,
    output: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[dict], dict]:
    cell_seed = config.seed * 10_000 + checkpoint_tokens // 1_000_000 * 100 + cut
    surrogate = fit_sequence_surrogate(fit_hidden, fit_ids, config, device)
    cell_dir = output / "analysis" / checkpoint_name(checkpoint_tokens) / f"cut_{cut:02d}"
    _surrogate_to_npz(surrogate, cell_dir / "sequence_surrogates.npz")

    eval_count = min(config.eval_stories, len(eval_ids))
    eval_ids_small = eval_ids[:eval_count]
    eval_mask_small = eval_mask[:eval_count]
    eval_hidden_small = eval_hidden[:eval_count]
    eval_banks = _make_eval_banks(
        surrogate, eval_hidden_small, eval_ids_small, device, cell_seed + 500,
    )

    base_suffix = GPT2Suffix(model, cut).to(device)
    parity = _parity_metrics(
        base_suffix, eval_hidden_small, full_eval_logits[:eval_count],
        eval_ids_small, eval_mask_small, device, dtype,
    )
    projected_hidden = surrogate.project(eval_hidden_small.to(device))
    projected_loss = evaluate_suffix_distribution(
        base_suffix, projected_hidden, eval_ids_small, eval_mask_small,
        config.eval_batch_size, device, dtype,
    )
    full_true_loss = evaluate_suffix_distribution(
        base_suffix, eval_hidden_small, eval_ids_small, eval_mask_small,
        config.eval_batch_size, device, dtype,
    )
    del base_suffix
    torch.cuda.empty_cache() if device.type == "cuda" else None

    leakage = {
        distribution: _future_leakage_check(
            surrogate, eval_ids_small, device, distribution, cell_seed + 800,
        )
        for distribution in ("mean", "token_gaussian", "seq_gaussian")
    }
    gradient_norms = {
        distribution: initial_gradient_norm(
            model, cut, distribution, surrogate, fit_hidden, fit_ids, fit_mask,
            config, device, dtype, cell_seed + 900,
        )
        for distribution in DIAGNOSTIC_DISTRIBUTIONS
    }
    target_gradient_norm = gradient_norms["true"]
    learning_rate_scales = {
        distribution: (
            max(0.25, min(4.0, target_gradient_norm / max(norm, 1e-12)))
            if config.gradient_matched_lr else 1.0
        )
        for distribution, norm in gradient_norms.items()
    }
    diagnostics = {
        "checkpoint_tokens": checkpoint_tokens,
        "cut": cut,
        **surrogate.diagnostics,
        "projected_true_loss": projected_loss,
        "full_true_loss": full_true_loss,
        "projected_true_excess_loss": projected_loss - full_true_loss,
        "future_leakage_max_abs": max(leakage.values()),
        "future_leakage_by_distribution": leakage,
        "initial_gradient_norms": gradient_norms,
        "learning_rate_scales": learning_rate_scales,
        "gradient_matched_lr": config.gradient_matched_lr,
        "parity": parity,
        "tie_policy": "cloned_untied_for_suffix_relaxation",
    }

    records: list[dict] = []
    for distribution_index, train_distribution in enumerate(DIAGNOSTIC_DISTRIBUTIONS):
        dist_records, relaxed_suffix = relax_one_suffix(
            model, cut, train_distribution, surrogate,
            fit_hidden, fit_ids, fit_mask, eval_banks,
            eval_ids_small, eval_mask_small, config, device, dtype,
            cell_seed + 1000 + distribution_index,
            gradient_norms[train_distribution], learning_rate_scales[train_distribution],
        )
        for record in dist_records:
            record.update({"checkpoint_tokens": checkpoint_tokens, "cut": cut, "draw": 0})
        records.extend(dist_records)
        del relaxed_suffix
        torch.cuda.empty_cache() if device.type == "cuda" else None

    write_json(cell_dir / "diagnostics.json", diagnostics)
    write_json(cell_dir / "suffix_statistics.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "MEASURED",
        "checkpoint_tokens": checkpoint_tokens,
        "cut": cut,
        "records": records,
        "diagnostics": diagnostics,
    })
    return records, diagnostics


def analyze(config: GateConfig) -> None:
    seed_everything(config.seed)
    output = Path(config.output)
    source = Path(config.source_run) if config.source_run else output
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(config.dtype)
    fit_ids, fit_mask = _load_replay(source / "data" / "replay_fit.npz")
    eval_ids, eval_mask = _load_replay(source / "data" / "replay_eval.npz")
    all_records: list[dict] = []
    all_diagnostics: list[dict] = []
    analysis_start = time.time()

    for checkpoint_tokens in config.analysis_checkpoint_tokens:
        checkpoint_dir = source / "checkpoints" / checkpoint_name(checkpoint_tokens)
        model = GPT2LMHeadModel.from_pretrained(checkpoint_dir).to(device)
        model.config.use_cache = False
        model.eval()
        fit_cache, _ = cache_cut_activations(
            model, fit_ids, fit_mask, config.cuts, config.cache_batch_size, device, dtype,
            logit_limit=0,
        )
        eval_cache, eval_logits = cache_cut_activations(
            model, eval_ids, eval_mask, config.cuts, config.cache_batch_size, device,
            torch.float32, logit_limit=2, storage_dtype=torch.float32,
        )
        for cut in config.cuts:
            cell_path = (
                output / "analysis" / checkpoint_name(checkpoint_tokens)
                / f"cut_{cut:02d}" / "suffix_statistics.json"
            )
            if cell_path.exists():
                saved_cell = json.loads(cell_path.read_text())
                if saved_cell.get("status") != "MEASURED":
                    raise RuntimeError(f"Refusing to reuse incomplete analysis cell: {cell_path}")
                cell_records = saved_cell["records"]
                diagnostics = saved_cell["diagnostics"]
                print(json.dumps({
                    "event": "analysis_cell_reused",
                    "checkpoint_tokens": checkpoint_tokens,
                    "cut": cut,
                }), flush=True)
            else:
                cell_records, diagnostics = _analyze_cell(
                    model, checkpoint_tokens, cut,
                    fit_cache[cut], eval_cache[cut], eval_logits,
                    fit_ids, fit_mask, eval_ids, eval_mask,
                    config, output, device, dtype,
                )
            all_records.extend(cell_records)
            all_diagnostics.append(diagnostics)
            print(json.dumps({
                "event": "analysis_cell_complete",
                "checkpoint_tokens": checkpoint_tokens,
                "cut": cut,
                "elapsed_seconds": time.time() - analysis_start,
            }), flush=True)
        del model, fit_cache, eval_cache, eval_logits
        torch.cuda.empty_cache() if device.type == "cuda" else None

    write_json(output / "suffix_statistics.json", {
        "schema_version": SCHEMA_VERSION,
        "experiment": "part_e_gpt2_suffix_statistics",
        "status": "MEASURED",
        "config": asdict(config),
        "records": all_records,
        "diagnostics": all_diagnostics,
        "runtime_seconds": time.time() - analysis_start,
        "device": str(device),
    })
    pd.json_normalize(all_diagnostics).to_parquet(
        output / "surrogate_diagnostics.parquet", index=False,
    )
    write_json(output / "manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "experiment": "part_e_gpt2_suffix_statistics",
        "status": "MEASURED",
        "config": asdict(config),
        "source_run": str(source.resolve()),
        "checkpoint_manifest_sha256": sha256_file(source / "checkpoint_manifest.json"),
        "replay_manifest_sha256": sha256_file(source / "replay_manifest.json"),
        "suffix_statistics_sha256": sha256_file(output / "suffix_statistics.json"),
        "surrogate_diagnostics_sha256": sha256_file(output / "surrogate_diagnostics.parquet"),
        "checkpoints_retained": True,
    })


def make_config(args: argparse.Namespace) -> GateConfig:
    return GateConfig(
        output=str(args.output),
        source_run=None if args.source_run is None else str(args.source_run),
        seed=args.seed,
        context_length=args.context_length,
        train_loss_tokens=args.train_loss_tokens,
        validation_tokens=args.validation_tokens,
        replay_fit_stories=args.replay_fit_stories,
        replay_eval_stories=args.replay_eval_stories,
        train_batch_size=args.train_batch_size,
        checkpoint_tokens=tuple(args.checkpoint_tokens),
        analysis_checkpoint_tokens=tuple(args.analysis_checkpoint_tokens),
        cuts=tuple(args.cuts),
        pca_components=args.pca_components,
        pca_min_components=args.pca_min_components,
        pca_target_coverage=args.pca_target_coverage,
        pca_fit_rows=args.pca_fit_rows,
        token_groups=args.token_groups,
        position_bins=args.position_bins,
        covariance_fit_stories=args.covariance_fit_stories,
        gradient_matched_lr=args.gradient_matched_lr,
        suffix_updates=args.suffix_updates,
        suffix_batch_size=args.suffix_batch_size,
        eval_stories=args.eval_stories,
        eval_batch_size=args.eval_batch_size,
        cache_batch_size=args.cache_batch_size,
        dtype=args.dtype,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "analyze", "all"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/part_e/gate_seed0"))
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--train-loss-tokens", type=int, default=50_000_000)
    parser.add_argument("--validation-tokens", type=int, default=2_000_000)
    parser.add_argument("--replay-fit-stories", type=int, default=512)
    parser.add_argument("--replay-eval-stories", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--checkpoint-tokens", type=int, nargs="+", default=[0, 4_000_000, 16_000_000, 50_000_000])
    parser.add_argument("--analysis-checkpoint-tokens", type=int, nargs="+", default=[4_000_000, 16_000_000, 50_000_000])
    parser.add_argument("--cuts", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--pca-components", type=int, default=512)
    parser.add_argument("--pca-min-components", type=int, default=128)
    parser.add_argument("--pca-target-coverage", type=float, default=0.95)
    parser.add_argument("--pca-fit-rows", type=int, default=32_768)
    parser.add_argument("--token-groups", type=int, default=256)
    parser.add_argument("--position-bins", type=int, default=16)
    parser.add_argument("--covariance-fit-stories", type=int, default=128)
    parser.add_argument(
        "--gradient-matched-lr", action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument("--suffix-updates", type=int, default=64)
    parser.add_argument("--suffix-batch-size", type=int, default=8)
    parser.add_argument("--eval-stories", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-head", type=int, default=12)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = make_config(args)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "gate_config.json", asdict(config))
    if args.action in ("prepare", "all"):
        prepare_data(config)
    if args.action in ("train", "all"):
        train_model(config)
    if args.action in ("analyze", "all"):
        analyze(config)


if __name__ == "__main__":
    main()
