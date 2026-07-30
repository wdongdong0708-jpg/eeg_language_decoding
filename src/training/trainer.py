"""Minimal, provenance-friendly EEG–speech training loops."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[dict[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    max_batches: int | None = None,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float | int]:
    model.train()
    total_loss = 0.0
    example_count = 0
    batch_count = 0
    for batch in batches:
        if max_batches is not None and batch_count >= max_batches:
            break
        eeg = batch["eeg"].to(device, non_blocking=True)
        speech = batch["speech"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(eeg, speech)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
        optimizer.step()
        count = int(eeg.shape[0])
        total_loss += float(loss.detach()) * count
        example_count += count
        batch_count += 1
    if example_count == 0:
        raise ValueError("Training loader produced no examples")
    return {
        "loss": total_loss / example_count,
        "example_count": example_count,
        "batch_count": batch_count,
    }


@torch.inference_mode()
def evaluate_loss(
    model: nn.Module,
    batches: Iterable[dict[str, object]],
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    model.eval()
    total_loss = 0.0
    example_count = 0
    batch_count = 0
    for batch in batches:
        if max_batches is not None and batch_count >= max_batches:
            break
        eeg = batch["eeg"].to(device, non_blocking=True)
        speech = batch["speech"].to(device, non_blocking=True)
        loss = model.compute_loss(eeg, speech)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite validation loss: {loss.item()}")
        count = int(eeg.shape[0])
        total_loss += float(loss) * count
        example_count += count
        batch_count += 1
    if example_count == 0:
        raise ValueError("Validation loader produced no examples")
    return {
        "loss": total_loss / example_count,
        "example_count": example_count,
        "batch_count": batch_count,
    }
