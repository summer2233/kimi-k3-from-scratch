"""Low-level ops for Kimi K3: SiTU, norms, short conv, KDA recurrence, RoPE.

KDA recurrence (channel-wise forget α, write gate β)::

    S_t = (I - β_t k_t k_t^T) Diag(α_t) S_{t-1} + β_t k_t v_t^T
    o_t = S_t^T q_t

See Kimi Linear / KDA (arXiv:2510.26692) and fla.ops.kda.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Activations — SiTU (Sigmoid Tanh Unit) for SiTU-GLU
# ---------------------------------------------------------------------------

def soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Soft-cap: cap * tanh(x / cap). Used on SiTU tanh branch and up-proj."""
    if cap is None or cap <= 0:
        return x
    return cap * torch.tanh(x / cap)


def situ(
    x: torch.Tensor,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    """Sigmoid Tanh Unit (SiTU).

    Unlike SiLU ``x * sigmoid(x)``::

        situ(x) = soft_cap(x, linear_beta) * sigmoid(beta * x)

    Config: ``activation_situ_beta`` (4.0), ``activation_situ_linear_beta`` (25.0).
    """
    return soft_cap(x, linear_beta) * torch.sigmoid(beta * x)


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    inv = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.eps)
        return (self.weight * x_f).to(dtype)


class RMSNormGated(nn.Module):
    """RMSNorm then sigmoid gate — used on KDA outputs (``o_norm``)."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        x_f = x_f * torch.rsqrt(var + self.eps)
        x_f = self.weight * x_f
        return (x_f * torch.sigmoid(gate.float())).to(dtype)


# ---------------------------------------------------------------------------
# Short causal convolution (depthwise) — KDA q/k/v path
# ---------------------------------------------------------------------------

class ShortConvolution(nn.Module):
    """Depthwise causal conv1d + SiLU, as in KDA / Kimi Linear."""

    def __init__(self, hidden_size: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=kernel_size,
            groups=hidden_size,
            bias=bias,
            padding=kernel_size - 1,
        )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [B, T, C]
        b, t, c = x.shape
        x_t = x.transpose(1, 2)
        if cache is not None:
            x_t = torch.cat([cache, x_t], dim=-1)
            new_cache = x_t[:, :, -(self.kernel_size - 1) :] if output_final_state else None
            y = self.conv(x_t)[:, :, -t:]
        else:
            y = self.conv(x_t)[:, :, :t]
            new_cache = (
                F.pad(x_t, (self.kernel_size - 1, 0))[:, :, -(self.kernel_size - 1) :]
                if output_final_state
                else None
            )
        y = F.silu(y).transpose(1, 2)
        return y, new_cache


# ---------------------------------------------------------------------------
# KDA pure-torch kernel (recurrent — clear, not production-fast)
# ---------------------------------------------------------------------------

def recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Token-by-token Kimi Delta Attention recurrence.

    Args shapes:
      q, k: ``[B, T, H, D_k]``
      v:    ``[B, T, H, D_v]``
      alpha:``[B, T, H, D_k]``  channel-wise forget in (0, 1)
      beta: ``[B, T, H]``       write gate after sigmoid
      state:``[B, H, D_k, D_v]``
    """
    if use_qk_l2norm:
        q = l2norm(q, dim=-1)
        k = l2norm(k, dim=-1)

    b, t, h, dk = k.shape
    dv = v.shape[-1]
    dtype = q.dtype

    q = q.float()
    k = k.float()
    v = v.float()
    alpha = alpha.float()
    beta = beta.float()

    scale = dk ** -0.5
    q = q * scale

    state = (
        torch.zeros(b, h, dk, dv, device=q.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.float()
    )
    outs = []
    for i in range(t):
        q_t = q[:, i]
        k_t = k[:, i]
        v_t = v[:, i]
        a_t = alpha[:, i]
        b_t = beta[:, i].unsqueeze(-1)

        # Diag(α) S
        state = state * a_t.unsqueeze(-1)
        # delta-rule erase + write
        kv_mem = torch.einsum("bhkd,bhk->bhd", state, k_t)
        delta = (v_t - kv_mem) * b_t
        state = state + torch.einsum("bhk,bhd->bhkd", k_t, delta)
        o_t = torch.einsum("bhkd,bhk->bhd", state, q_t)
        outs.append(o_t)

    out = torch.stack(outs, dim=1).to(dtype)
    final = state.to(dtype) if output_final_state else None
    return out, final


def chunk_kda_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """KDA entrypoint (clear recurrent form). Production stacks use FLA chunk kernels."""
    return recurrent_kda(
        q, k, v, alpha, beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm=use_qk_l2norm,
    )


# ---------------------------------------------------------------------------
# RoPE (used on MLA rope slices)
# ---------------------------------------------------------------------------

def precompute_rope_cos_sin(
    head_dim: int,
    context_length: int,
    theta_base: float = 10000.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert head_dim % 2 == 0, "RoPE head dim must be even"
    inv_freq = 1.0 / (
        theta_base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    t = torch.arange(context_length, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply RoPE. ``x`` layout: [B, H, T, D]."""
    d = x.shape[-1]
    if position_ids is None:
        cos = cos[: x.shape[-2], :d].view(1, 1, -1, d)
        sin = sin[: x.shape[-2], :d].view(1, 1, -1, d)
    else:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)

    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    return x * cos + rotated * sin
