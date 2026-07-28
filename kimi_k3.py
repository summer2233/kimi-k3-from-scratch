"""Kimi K3 text architecture (pure PyTorch, structure-level).

Readable reimplementation of the Kimi K3 language stack:

  hybrid KDA + Gated MLA · LatentMoE · SiTU-GLU · Block AttnRes

Default config matches ``inference-optimization/Kimi-K3-0.18B`` (4-layer
miniature of full moonshotai/Kimi-K3). Not bit-exact with production kernels.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kimi_k3_ops import (
    RMSNorm,
    RMSNormGated,
    ShortConvolution,
    apply_rope,
    chunk_kda_simple,
    precompute_rope_cos_sin,
    situ,
    soft_cap,
)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

# Reference numbers for the full open model (do not instantiate — too large).
KIMI_K3_FULL_REFERENCE: Dict = {
    "vocab_size": 160_000,  # model card ~160K
    "context_length": 1_048_576,
    "emb_dim": 7168,
    "n_layers": 93,
    "n_heads": 96,
    "n_kv_heads": 96,
    "intermediate_size": 33792,
    "moe_intermediate_size": 3072,
    "routed_expert_hidden_size": 3584,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "num_shared_experts": 2,
    "first_k_dense_replace": 1,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "linear_num_heads": 96,  # scales with model; card: linear head_dim 128
    "linear_head_dim": 128,
    "linear_conv_kernel_size": 4,
    "attn_res_block_size": 12,  # full-scale Block AttnRes
    "n_kda_layers": 69,
    "n_mla_layers": 24,
    "activation": "situ",
    "total_params": "2.8T",
    "activated_params": "104B",
}

# inference-optimization/Kimi-K3-0.18B — same layer *types*, shrunk dims.
# Extra fields (router, SiTU betas, rope, AttnRes, …) aligned with the
# published 0.40B tiny config / full K3 design.
KIMI_K3_CONFIG_0_18B: Dict = {
    "vocab_size": 163_840,
    "context_length": 4096,
    "emb_dim": 512,
    "n_layers": 4,
    "n_heads": 4,
    "n_kv_heads": 4,
    "intermediate_size": 1024,
    "moe_intermediate_size": 256,
    "routed_expert_hidden_size": 256,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "first_k_dense_replace": 1,
    "kv_lora_rank": 64,
    "q_lora_rank": 128,
    "qk_nope_head_dim": 64,
    "qk_rope_head_dim": 32,
    "v_head_dim": 64,
    "linear_num_heads": 4,
    "linear_head_dim": 64,
    "linear_conv_kernel_size": 4,
    "linear_use_full_rank_gate": True,
    # HF uses 1-based indices in linear_attn_config
    "kda_layers": [1, 2, 3],
    "full_attn_layers": [4],
    "attn_res_block_size": 4,
    "rms_norm_eps": 1e-5,
    "rope_base": 10_000.0,
    "mla_use_output_gate": True,
    "mla_use_nope": True,
    "latent_moe_use_norm": True,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "topk_method": "noaux_tc",
    "routed_scaling_factor": 1.0,
    "hidden_act": "situ",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "initializer_range": 0.02,
    "dtype": "float32",
}

# Same topology, smaller dims (CPU demos and unit tests).
KIMI_K3_CONFIG_MICRO: Dict = {
    **KIMI_K3_CONFIG_0_18B,
    "vocab_size": 256,
    "context_length": 128,
    "emb_dim": 64,
    "n_layers": 4,
    "n_heads": 2,
    "n_kv_heads": 2,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "routed_expert_hidden_size": 32,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "num_shared_experts": 1,
    "kv_lora_rank": 16,
    "q_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "linear_num_heads": 2,
    "linear_head_dim": 16,
    "kda_layers": [1, 2, 3],
    "full_attn_layers": [4],
    "attn_res_block_size": 4,
}


def layer_type(cfg: Dict, layer_idx_0: int) -> str:
    """Return ``'kda'`` or ``'full'`` (MLA) for a 0-based layer index."""
    one_based = layer_idx_0 + 1
    if one_based in cfg["full_attn_layers"]:
        return "full"
    if one_based in cfg["kda_layers"]:
        return "kda"
    # fallback: 3:1 KDA:MLA
    return "full" if one_based % 4 == 0 else "kda"


# ---------------------------------------------------------------------------
# SiTU-GLU feed-forward (dense) + LatentMoE
# ---------------------------------------------------------------------------

class DenseFFN(nn.Module):
    """SiTU-GLU MLP on the first ``first_k_dense_replace`` layers (layer 0)."""

    def __init__(self, cfg: Dict):
        super().__init__()
        d, h = cfg["emb_dim"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(d, h, bias=False)
        self.up_proj = nn.Linear(d, h, bias=False)
        self.down_proj = nn.Linear(h, d, bias=False)
        self.situ_beta = cfg["activation_situ_beta"]
        self.situ_linear_beta = cfg["activation_situ_linear_beta"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = situ(self.gate_proj(x), self.situ_beta, self.situ_linear_beta)
        # soft-cap on the up branch (K3 SiTU-GLU stability trick)
        u = soft_cap(self.up_proj(x), self.situ_linear_beta)
        return self.down_proj(g * u)


class ExpertMLP(nn.Module):
    """Single routed expert in latent space: SiTU-GLU."""

    def __init__(self, latent_dim: int, intermediate: int, cfg: Dict):
        super().__init__()
        self.w1 = nn.Linear(latent_dim, intermediate, bias=False)  # gate
        self.w3 = nn.Linear(latent_dim, intermediate, bias=False)  # up
        self.w2 = nn.Linear(intermediate, latent_dim, bias=False)  # down
        self.situ_beta = cfg["activation_situ_beta"]
        self.situ_linear_beta = cfg["activation_situ_linear_beta"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = situ(self.w1(x), self.situ_beta, self.situ_linear_beta)
        u = soft_cap(self.w3(x), self.situ_linear_beta)
        return self.w2(g * u)


class LatentMoE(nn.Module):
    """Stable LatentMoE.

    Routed experts run in a compressed latent space::

        hidden → down → RMSNorm → top-k experts → up → hidden
        (+ shared experts on full hidden)

    RMSNorm sits after the latent down-projection. Router: sigmoid scores with
    a ``noaux_tc`` correction bias.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        d = cfg["emb_dim"]
        latent = cfg["routed_expert_hidden_size"]
        inter = cfg["moe_intermediate_size"]
        n_exp = cfg["num_experts"]
        self.top_k = cfg["num_experts_per_token"]
        self.n_shared = cfg["num_shared_experts"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]
        self.renormalize = cfg["moe_renormalize"]
        self.use_norm = cfg["latent_moe_use_norm"]

        # Latent path: down → (norm on latent) → experts → up
        self.routed_expert_down_proj = nn.Linear(d, latent, bias=False)
        self.routed_expert_norm = (
            RMSNorm(latent, cfg["rms_norm_eps"]) if self.use_norm else nn.Identity()
        )
        self.routed_expert_up_proj = nn.Linear(latent, d, bias=False)

        self.gate = nn.Linear(d, n_exp, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(n_exp))

        self.experts = nn.ModuleList(
            [ExpertMLP(latent, inter, cfg) for _ in range(n_exp)]
        )

        # Shared experts fused as one wide SiTU-GLU (width × num_shared)
        shared_inter = inter * max(self.n_shared, 1)
        self.shared_gate = nn.Linear(d, shared_inter, bias=False)
        self.shared_up = nn.Linear(d, shared_inter, bias=False)
        self.shared_down = nn.Linear(shared_inter, d, bias=False)
        self.situ_beta = cfg["activation_situ_beta"]
        self.situ_linear_beta = cfg["activation_situ_linear_beta"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        # shared experts on full hidden
        sg = situ(self.shared_gate(flat), self.situ_beta, self.situ_linear_beta)
        su = soft_cap(self.shared_up(flat), self.situ_linear_beta)
        shared_out = self.shared_down(sg * su)

        # sigmoid router + noaux_tc bias for top-k selection
        logits = self.gate(flat)
        scores = torch.sigmoid(logits)
        scores_for_choice = scores + self.e_score_correction_bias
        _, topk_idx = torch.topk(scores_for_choice, self.top_k, dim=-1)
        topk_weights = scores.gather(-1, topk_idx)
        if self.renormalize:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)
        topk_weights = topk_weights * self.routed_scaling_factor

        # latent down → norm → experts (K3 LatentMoE placement)
        z = self.routed_expert_norm(self.routed_expert_down_proj(flat))

        routed = torch.zeros_like(z)
        for e_id, expert in enumerate(self.experts):
            mask = topk_idx == e_id
            if not mask.any():
                continue
            token_mask = mask.any(dim=-1)
            tok = z[token_mask]
            out_e = expert(tok)
            w = (topk_weights * mask.float()).sum(dim=-1)[token_mask].unsqueeze(-1)
            routed[token_mask] = routed[token_mask] + out_e * w

        routed = self.routed_expert_up_proj(routed)
        return (shared_out + routed).view(b, t, d)


# ---------------------------------------------------------------------------
# KDA + Gated MLA
# ---------------------------------------------------------------------------

class KimiDeltaAttention(nn.Module):
    """Kimi Delta Attention — linear / delta-rule mixer with channel-wise gates.

    Distinct from scalar-gated Gated DeltaNet: forget gate α is per key-dim
    (shape ``[B,T,H,Dk]``), from low-rank ``f_a → f_b`` + ``A_log`` / ``dt_bias``.
    """

    def __init__(self, cfg: Dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = cfg["emb_dim"]
        self.num_heads = cfg["linear_num_heads"]
        self.head_dim = cfg["linear_head_dim"]
        self.key_dim = self.num_heads * self.head_dim
        self.value_dim = self.key_dim
        self.conv_kernel = cfg["linear_conv_kernel_size"]

        self.q_proj = nn.Linear(d, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d, self.value_dim, bias=False)

        self.q_conv1d = ShortConvolution(self.key_dim, self.conv_kernel)
        self.k_conv1d = ShortConvolution(self.key_dim, self.conv_kernel)
        self.v_conv1d = ShortConvolution(self.value_dim, self.conv_kernel)

        self.f_a_proj = nn.Linear(d, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.key_dim, bias=False)
        self.b_proj = nn.Linear(d, self.num_heads, bias=False)

        self.A_log = nn.Parameter(
            torch.log(torch.empty(self.num_heads).uniform_(1, 16))
        )
        dt = torch.exp(
            torch.rand(self.key_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        ).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)

        self.g_proj = nn.Linear(d, self.value_dim, bias=True)
        self.o_norm = RMSNormGated(self.head_dim, cfg["rms_norm_eps"])
        self.o_proj = nn.Linear(self.value_dim, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        b, t, _ = x.shape
        q, _ = self.q_conv1d(self.q_proj(x))
        k, _ = self.k_conv1d(self.k_proj(x))
        v, _ = self.v_conv1d(self.v_proj(x))

        # low-rank forget path (no mid activation; matches KDA / FLA layout)
        f = self.f_b_proj(self.f_a_proj(x))
        f = f.view(b, t, self.num_heads, self.head_dim)
        dt = self.dt_bias.view(1, 1, self.num_heads, self.head_dim)
        a_log = self.A_log.view(1, 1, self.num_heads, 1)
        g = -a_log.float().exp() * F.softplus(f.float() + dt)
        alpha = g.exp()

        beta = torch.sigmoid(self.b_proj(x))

        q = q.view(b, t, self.num_heads, self.head_dim)
        k = k.view(b, t, self.num_heads, self.head_dim)
        v = v.view(b, t, self.num_heads, self.head_dim)

        init_state = None if cache is None else cache.get("recurrent_state")
        o, final_state = chunk_kda_simple(
            q, k, v, alpha, beta,
            initial_state=init_state,
            output_final_state=use_cache,
            use_qk_l2norm=True,
        )

        gate = self.g_proj(x).view(b, t, self.num_heads, self.head_dim)
        o = self.o_norm(o, gate)
        o = self.o_proj(o.reshape(b, t, self.value_dim))

        new_cache = {"recurrent_state": final_state} if use_cache else None
        return o, new_cache


class GatedMLA(nn.Module):
    """Gated Multi-head Latent Attention (full-attention layers).

    - Low-rank Q (q_a → norm → q_b) and KV (kv_a_mqa → norm → kv_b)
    - Q/K split into NOPE + RoPE parts (``mla_use_nope``)
    - Output gate ``g_proj`` (``mla_use_output_gate``)
    """

    def __init__(self, cfg: Dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.n_kv_heads = cfg["n_kv_heads"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.qk_nope = cfg["qk_nope_head_dim"]
        self.qk_rope = cfg["qk_rope_head_dim"]
        self.v_head_dim = cfg["v_head_dim"]
        self.qk_head_dim = self.qk_nope + self.qk_rope
        self.use_output_gate = cfg["mla_use_output_gate"]
        self.use_nope = cfg.get("mla_use_nope", True)
        eps = cfg["rms_norm_eps"]

        self.q_a_proj = nn.Linear(d, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
        )

        self.kv_a_proj_with_mqa = nn.Linear(
            d, self.kv_lora_rank + self.qk_rope, bias=False
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_kv_heads * (self.qk_nope + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, d, bias=False)
        if self.use_output_gate:
            self.g_proj = nn.Linear(d, self.n_heads * self.v_head_dim, bias=False)

        self.scale = self.qk_head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        b, t, _ = x.shape

        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(x)))
        q = q.view(b, t, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = q.split([self.qk_nope, self.qk_rope], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        kv_latent, k_rope = kv.split([self.kv_lora_rank, self.qk_rope], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)
        kv_b = self.kv_b_proj(kv_latent).view(
            b, t, self.n_kv_heads, self.qk_nope + self.v_head_dim
        )
        k_nope, v = kv_b.split([self.qk_nope, self.v_head_dim], dim=-1)

        # RoPE only on rope slices; NOPE half stays position-free
        q_rope = apply_rope(q_rope.transpose(1, 2), cos, sin)
        k_rope = k_rope.unsqueeze(1).expand(-1, self.n_kv_heads, -1, -1)
        k_rope = apply_rope(k_rope, cos, sin)

        q_nope = q_nope.transpose(1, 2)
        k_nope = k_nope.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_nope:
            q_full = torch.cat([q_nope, q_rope], dim=-1)
            k_full = torch.cat([k_nope, k_rope], dim=-1)
        else:
            q_full, k_full = q_rope, k_rope

        if cache is not None:
            k_full = torch.cat([cache["k"], k_full], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        new_cache = {"k": k_full, "v": v} if use_cache else None

        if self.n_heads != self.n_kv_heads:
            rep = self.n_heads // self.n_kv_heads
            k_full = k_full.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        attn = torch.matmul(q_full, k_full.transpose(-1, -2)) * self.scale
        q_len, k_len = q_full.shape[2], k_full.shape[2]
        causal = torch.triu(
            torch.ones(q_len, k_len, device=x.device, dtype=torch.bool),
            diagonal=1 + (k_len - q_len),
        )
        attn = attn.masked_fill(causal, torch.finfo(attn.dtype).min)
        if attention_mask is not None:
            am = attention_mask[:, None, None, :k_len]
            attn = attn.masked_fill(am == 0, torch.finfo(attn.dtype).min)

        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, -1)

        if self.use_output_gate:
            out = out * torch.sigmoid(self.g_proj(x))

        return self.o_proj(out), new_cache


# ---------------------------------------------------------------------------
# Block Attention Residuals (AttnRes)
# ---------------------------------------------------------------------------

class BlockAttnRes(nn.Module):
    """Block Attention Residuals — learned mix over residual history.

    Standard PreNorm residual is a fixed sum. AttnRes replaces that with
    softmax attention over preceding residual sources (embedding / prior
    layer outputs in the current block). Each module stores:

      - ``norm``: RMSNorm over bank keys
      - ``proj``: Linear producing a pseudo-query from the current stream
                 (zero-init → near-uniform weights at start)

    Full K3 uses ``attn_res_block_size≈12``; the 0.18B tiny model uses 4.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        d = cfg["emb_dim"]
        self.norm = RMSNorm(d, cfg["rms_norm_eps"])
        self.proj = nn.Linear(d, d, bias=False)
        # zero-init query so early training ≈ uniform residual mix
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    current residual stream [B, T, D]
            bank: history stack [B, T, N, D] (N ≥ 1 sources)
        """
        b, t, n, d = bank.shape
        q = self.proj(self.norm(x))  # [B, T, D]
        k = self.norm(bank.reshape(b * t * n, d)).view(b, t, n, d)
        scores = torch.einsum("btd,btnd->btn", q, k) * (d ** -0.5)
        weights = F.softmax(scores, dim=-1)
        return torch.einsum("btn,btnd->btd", weights, bank)


# ---------------------------------------------------------------------------
# Transformer block + model
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Dict, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_kind = layer_type(cfg, layer_idx)
        self.input_layernorm = RMSNorm(cfg["emb_dim"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["emb_dim"], cfg["rms_norm_eps"])

        if self.layer_kind == "kda":
            self.self_attn = KimiDeltaAttention(cfg, layer_idx)
        else:
            self.self_attn = GatedMLA(cfg, layer_idx)

        if layer_idx < cfg["first_k_dense_replace"]:
            self.mlp = DenseFFN(cfg)
            self.is_moe = False
        else:
            self.mlp = LatentMoE(cfg)
            self.is_moe = True

        # separate AttnRes mixers for attention path and MLP path
        self.self_attention_res = BlockAttnRes(cfg)
        self.mlp_res = BlockAttnRes(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bank: torch.Tensor,
        mlp_bank: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        cache: Optional[dict] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict]]:
        # AttnRes mix over attention residual history → input to attn sublayer
        residual = self.self_attention_res(x, attn_bank)
        h = self.input_layernorm(residual)
        if self.layer_kind == "kda":
            attn_out, new_cache = self.self_attn(h, cache=cache, use_cache=use_cache)
        else:
            attn_out, new_cache = self.self_attn(
                h, cos, sin, attention_mask=attention_mask, cache=cache, use_cache=use_cache
            )
        x = residual + attn_out
        # append this layer's attention contribution for later mixers in block
        new_attn_bank = torch.cat([attn_bank, attn_out.unsqueeze(2)], dim=2)

        residual = self.mlp_res(x, mlp_bank)
        h = self.post_attention_layernorm(residual)
        mlp_out = self.mlp(h)
        x = residual + mlp_out
        new_mlp_bank = torch.cat([mlp_bank, mlp_out.unsqueeze(2)], dim=2)

        return x, new_attn_bank, new_mlp_bank, new_cache


class KimiK3Model(nn.Module):
    """Kimi K3 text tower (no vision) — embedding → hybrid blocks → LM head."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        d = cfg["emb_dim"]
        self.tok_emb = nn.Embedding(cfg["vocab_size"], d)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg, i) for i in range(cfg["n_layers"])]
        )
        self.final_norm = RMSNorm(d, cfg["rms_norm_eps"])
        self.out_head = nn.Linear(d, cfg["vocab_size"], bias=False)

        # final AttnRes over accumulated attention residual sources
        self.output_attn_res_norm = RMSNorm(d, cfg["rms_norm_eps"])
        self.output_attn_res_proj = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.output_attn_res_proj.weight)

        cos, sin = precompute_rope_cos_sin(
            cfg["qk_rope_head_dim"], cfg["context_length"], cfg["rope_base"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        # re-apply zero-init on AttnRes queries after general init
        for mod in self.modules():
            if isinstance(mod, BlockAttnRes):
                nn.init.zeros_(mod.proj.weight)
        nn.init.zeros_(self.output_attn_res_proj.weight)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg["initializer_range"]
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        caches: Optional[List[Optional[dict]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[dict]]]:
        b, t = input_ids.shape
        x = self.tok_emb(input_ids)

        block_size = self.cfg["attn_res_block_size"]
        # banks: [B, T, N, D] — start each block with the current stream
        attn_bank = x.unsqueeze(2)
        mlp_bank = x.unsqueeze(2)
        # collect last-block attention outs for optional output AttnRes
        block_attn_summary = x

        new_caches: List[Optional[dict]] = []
        for i, block in enumerate(self.blocks):
            if i > 0 and i % block_size == 0:
                attn_bank = x.unsqueeze(2)
                mlp_bank = x.unsqueeze(2)

            layer_cache = None if caches is None else caches[i]
            x, attn_bank, mlp_bank, nc = block(
                x,
                self.cos,
                self.sin,
                attn_bank,
                mlp_bank,
                attention_mask=attention_mask,
                cache=layer_cache,
                use_cache=use_cache,
            )
            # mean of attention-bank sources as a cheap block summary
            block_attn_summary = attn_bank.mean(dim=2)
            new_caches.append(nc)

        # output-level residual mix (K3 has output_attn_res_*)
        x = x + self.output_attn_res_proj(self.output_attn_res_norm(block_attn_summary))
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits, (new_caches if use_cache else None)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def describe_architecture(cfg: Dict) -> str:
    lines = [
        f"layers={cfg['n_layers']}  emb={cfg['emb_dim']}  vocab={cfg['vocab_size']}",
        f"KDA layers (1-based): {cfg['kda_layers']}",
        f"MLA layers (1-based): {cfg['full_attn_layers']}",
        f"first_k_dense_replace={cfg['first_k_dense_replace']}",
        f"MoE: experts={cfg['num_experts']} topk={cfg['num_experts_per_token']} "
        f"shared={cfg['num_shared_experts']} latent={cfg['routed_expert_hidden_size']}",
        f"MLA ranks: q_lora={cfg['q_lora_rank']} kv_lora={cfg['kv_lora_rank']} "
        f"nope/rope/v={cfg['qk_nope_head_dim']}/{cfg['qk_rope_head_dim']}/{cfg['v_head_dim']}",
        f"KDA: heads={cfg['linear_num_heads']} head_dim={cfg['linear_head_dim']} "
        f"conv={cfg['linear_conv_kernel_size']}",
        f"AttnRes block size={cfg['attn_res_block_size']}  act={cfg.get('hidden_act', 'situ')}",
    ]
    for i in range(cfg["n_layers"]):
        kind = layer_type(cfg, i)
        ffn = "dense SiTU-GLU" if i < cfg["first_k_dense_replace"] else "LatentMoE"
        lines.append(f"  L{i}: attn={kind:4s}  ffn={ffn}")
    return "\n".join(lines)


def demo_forward(cfg: Optional[Dict] = None, device: str = "cpu") -> None:
    cfg = cfg or KIMI_K3_CONFIG_MICRO
    torch.manual_seed(0)
    model = KimiK3Model(cfg).to(device)
    model.eval()
    print(describe_architecture(cfg))
    n = count_parameters(model)
    print(f"parameters: {n:,} ({n / 1e6:.3f} M)")
    ids = torch.randint(0, min(cfg["vocab_size"], 1000), (2, 16), device=device)
    with torch.no_grad():
        logits, _ = model(ids)
    print(f"logits: {tuple(logits.shape)}  finite={bool(torch.isfinite(logits).all())}")


if __name__ == "__main__":
    demo_forward()
