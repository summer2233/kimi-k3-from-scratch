"""Load Hugging Face tiny Kimi K3 checkpoints (optional).

- ``load_hf_reference_model``: official model via ``trust_remote_code``.
- ``map_and_load_into_from_scratch``: best-effort key map into ``KimiK3Model``.

HF keys use the prefix ``language_model.model.layers.X.*``.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from kimi_k3 import KIMI_K3_CONFIG_0_18B, KimiK3Model


def load_hf_reference_model(
    repo_id: str = "inference-optimization/Kimi-K3-0.18B",
    device: str = "cpu",
    dtype: Optional[torch.dtype] = None,
):
    """Load the HF tiny model with ``trust_remote_code`` (official path)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=None,
    )
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    return model, tokenizer


def _strip_prefix(name: str) -> str:
    for p in (
        "language_model.model.",
        "model.language_model.model.",
        "model.",
    ):
        if name.startswith(p):
            return name[len(p) :]
    return name


def map_hf_key_to_scratch(key: str) -> Optional[str]:
    """Map a subset of HF keys → our ``KimiK3Model`` state_dict keys."""
    k = _strip_prefix(key)
    if k.startswith("language_model."):
        k = k[len("language_model.") :]
    if k.startswith("model."):
        k = k[len("model.") :]

    if k == "embed_tokens.weight":
        return "tok_emb.weight"
    if k == "norm.weight":
        return "final_norm.weight"
    if k.endswith("lm_head.weight") or k == "lm_head.weight":
        return "out_head.weight"
    if k == "output_attn_res_norm.weight":
        return "output_attn_res_norm.weight"
    if k == "output_attn_res_proj.weight":
        return "output_attn_res_proj.weight"

    if not k.startswith("layers."):
        return None

    parts = k.split(".", 2)
    if len(parts) < 3:
        return None
    _, idx, rest = parts
    prefix = f"blocks.{idx}."

    simple = {
        "input_layernorm.weight": "input_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "self_attention_res_norm.weight": "self_attention_res.norm.weight",
        "self_attention_res_proj.weight": "self_attention_res.proj.weight",
        "mlp_res_norm.weight": "mlp_res.norm.weight",
        "mlp_res_proj.weight": "mlp_res.proj.weight",
        "mlp.gate_proj.weight": "mlp.gate_proj.weight",
        "mlp.up_proj.weight": "mlp.up_proj.weight",
        "mlp.down_proj.weight": "mlp.down_proj.weight",
        "self_attn.o_proj.weight": "self_attn.o_proj.weight",
        "self_attn.g_proj.weight": "self_attn.g_proj.weight",
        "self_attn.q_proj.weight": "self_attn.q_proj.weight",
        "self_attn.k_proj.weight": "self_attn.k_proj.weight",
        "self_attn.v_proj.weight": "self_attn.v_proj.weight",
        "self_attn.q_conv1d.weight": "self_attn.q_conv1d.conv.weight",
        "self_attn.k_conv1d.weight": "self_attn.k_conv1d.conv.weight",
        "self_attn.v_conv1d.weight": "self_attn.v_conv1d.conv.weight",
        "self_attn.f_a_proj.weight": "self_attn.f_a_proj.weight",
        "self_attn.f_b_proj.weight": "self_attn.f_b_proj.weight",
        "self_attn.b_proj.weight": "self_attn.b_proj.weight",
        "self_attn.o_norm.weight": "self_attn.o_norm.weight",
        "self_attn.A_log": "self_attn.A_log",
        "self_attn.dt_bias": "self_attn.dt_bias",
        "self_attn.q_a_proj.weight": "self_attn.q_a_proj.weight",
        "self_attn.q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
        "self_attn.q_b_proj.weight": "self_attn.q_b_proj.weight",
        "self_attn.kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
        "self_attn.kv_b_proj.weight": "self_attn.kv_b_proj.weight",
        "block_sparse_moe.gate.weight": "mlp.gate.weight",
        "block_sparse_moe.gate.e_score_correction_bias": "mlp.e_score_correction_bias",
        "block_sparse_moe.routed_expert_up_proj.weight": "mlp.routed_expert_up_proj.weight",
        "block_sparse_moe.routed_expert_down_proj.weight": "mlp.routed_expert_down_proj.weight",
        "block_sparse_moe.routed_expert_norm.weight": "mlp.routed_expert_norm.weight",
        "block_sparse_moe.shared_experts.gate_proj.weight": "mlp.shared_gate.weight",
        "block_sparse_moe.shared_experts.up_proj.weight": "mlp.shared_up.weight",
        "block_sparse_moe.shared_experts.down_proj.weight": "mlp.shared_down.weight",
    }
    if rest in simple:
        return prefix + simple[rest]

    if rest.startswith("block_sparse_moe.experts."):
        segs = rest.split(".")
        if len(segs) >= 5 and segs[3] in ("w1", "w2", "w3"):
            return f"{prefix}mlp.experts.{segs[2]}.{segs[3]}.weight"
    return None


@torch.no_grad()
def map_and_load_into_from_scratch(
    model: KimiK3Model,
    state_dict: Dict[str, torch.Tensor],
    strict_shapes: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """Copy matching tensors from an HF state dict into our model."""
    ours = model.state_dict()
    loaded = skipped = shape_mismatch = 0
    for hk, tensor in state_dict.items():
        sk = map_hf_key_to_scratch(hk)
        if sk is None or sk not in ours:
            skipped += 1
            continue
        if ours[sk].shape != tensor.shape:
            shape_mismatch += 1
            if verbose:
                print(
                    f"shape mismatch {hk} -> {sk}: "
                    f"{tuple(tensor.shape)} vs {tuple(ours[sk].shape)}"
                )
            if strict_shapes:
                continue
        ours[sk].copy_(tensor.to(dtype=ours[sk].dtype))
        loaded += 1
    model.load_state_dict(ours)
    stats = {"loaded": loaded, "skipped": skipped, "shape_mismatch": shape_mismatch}
    if verbose:
        print(stats)
    return stats


def build_0_18b(device: str = "cpu") -> KimiK3Model:
    """Instantiate the 0.18B-shaped from-scratch model (random init)."""
    return KimiK3Model(KIMI_K3_CONFIG_0_18B).to(device)
