"""Greedy / sampling generation helpers for ``KimiK3Model``."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from kimi_k3 import KimiK3Model


@torch.no_grad()
def generate_text_basic(
    model: KimiK3Model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_token_id: Optional[int] = None,
) -> torch.Tensor:
    """Autoregressive decode (no KV cache — simple and clear)."""
    model.eval()
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    ctx = model.cfg["context_length"]

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -ctx:]
        logits, _ = model(ids_cond)
        next_logits = logits[:, -1, :]

        if temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            logits_t = next_logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits_t, min(top_k, logits_t.size(-1)))
                logits_t = logits_t.masked_fill(logits_t < v[:, [-1]], float("-inf"))
            probs = F.softmax(logits_t, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

        ids = torch.cat([ids, next_id], dim=1)
        if eos_token_id is not None and (next_id == eos_token_id).all():
            break
    return ids


@torch.no_grad()
def generate_text_basic_stream(
    model: KimiK3Model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 0.0,
) -> Iterable[torch.Tensor]:
    """Yield one new token id tensor ``[B, 1]`` at a time."""
    model.eval()
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    ctx = model.cfg["context_length"]

    for _ in range(max_new_tokens):
        ids_cond = ids[:, -ctx:]
        logits, _ = model(ids_cond)
        next_logits = logits[:, -1, :]
        if temperature <= 0:
            next_id = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        ids = torch.cat([ids, next_id], dim=1)
        yield next_id
