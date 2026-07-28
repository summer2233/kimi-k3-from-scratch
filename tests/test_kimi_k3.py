"""Structure and forward tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate import generate_text_basic
from kimi_k3 import (
    KIMI_K3_CONFIG_0_18B,
    KIMI_K3_CONFIG_MICRO,
    KIMI_K3_FULL_REFERENCE,
    BlockAttnRes,
    GatedMLA,
    KimiDeltaAttention,
    KimiK3Model,
    LatentMoE,
    count_parameters,
    layer_type,
)
from kimi_k3_ops import recurrent_kda, soft_cap, situ


def test_layer_schedule_matches_k3_tiny():
    cfg = KIMI_K3_CONFIG_0_18B
    kinds = [layer_type(cfg, i) for i in range(cfg["n_layers"])]
    assert kinds == ["kda", "kda", "kda", "full"]
    assert cfg["first_k_dense_replace"] == 1
    assert cfg["hidden_act"] == "situ"
    assert cfg["mla_use_output_gate"] is True
    assert cfg["mla_use_nope"] is True


def test_0_18b_dims_match_hf_card():
    c = KIMI_K3_CONFIG_0_18B
    assert c["n_layers"] == 4
    assert c["emb_dim"] == 512
    assert c["n_heads"] == 4
    assert c["intermediate_size"] == 1024
    assert c["moe_intermediate_size"] == 256
    assert c["routed_expert_hidden_size"] == 256
    assert c["num_experts"] == 8
    assert c["num_experts_per_token"] == 2
    assert c["num_shared_experts"] == 1
    assert c["kv_lora_rank"] == 64
    assert c["q_lora_rank"] == 128
    assert c["qk_nope_head_dim"] == 64
    assert c["qk_rope_head_dim"] == 32
    assert c["v_head_dim"] == 64
    assert c["linear_head_dim"] == 64


def test_full_reference_is_k3_not_other_arch():
    r = KIMI_K3_FULL_REFERENCE
    assert r["n_layers"] == 93
    assert r["n_kda_layers"] == 69
    assert r["n_mla_layers"] == 24
    assert r["num_experts"] == 896
    assert r["num_experts_per_token"] == 16
    assert r["activation"] == "situ"


def test_situ_is_not_silu():
    x = torch.tensor([[-2.0, 0.0, 2.0]])
    y = situ(x, beta=4.0, linear_beta=25.0)
    silu = x * torch.sigmoid(x)
    # SiTU uses tanh soft-cap * sigmoid, differs from SiLU
    assert not torch.allclose(y, silu, atol=1e-3)
    assert torch.isfinite(y).all()
    assert soft_cap(x, 25.0).abs().max() <= 25.0 + 1e-5


def test_kda_recurrent_shapes():
    b, t, h, dk, dv = 2, 5, 2, 8, 8
    q = torch.randn(b, t, h, dk)
    k = torch.randn(b, t, h, dk)
    v = torch.randn(b, t, h, dv)
    alpha = torch.sigmoid(torch.randn(b, t, h, dk))
    beta = torch.sigmoid(torch.randn(b, t, h))
    o, state = recurrent_kda(q, k, v, alpha, beta, output_final_state=True)
    assert o.shape == (b, t, h, dv)
    assert state.shape == (b, h, dk, dv)


def test_modules_are_k3_types():
    cfg = KIMI_K3_CONFIG_MICRO
    model = KimiK3Model(cfg)
    assert isinstance(model.blocks[0].self_attn, KimiDeltaAttention)
    assert isinstance(model.blocks[3].self_attn, GatedMLA)
    assert model.blocks[0].is_moe is False
    assert isinstance(model.blocks[1].mlp, LatentMoE)
    assert isinstance(model.blocks[0].self_attention_res, BlockAttnRes)
    # LatentMoE norm is on latent dim, not model dim
    assert model.blocks[1].mlp.routed_expert_norm.weight.shape[0] == cfg[
        "routed_expert_hidden_size"
    ]


def test_micro_forward():
    cfg = KIMI_K3_CONFIG_MICRO
    torch.manual_seed(0)
    model = KimiK3Model(cfg).eval()
    ids = torch.randint(0, cfg["vocab_size"], (2, 12))
    with torch.no_grad():
        logits, caches = model(ids, use_cache=False)
    assert logits.shape == (2, 12, cfg["vocab_size"])
    assert torch.isfinite(logits).all()
    assert caches is None


def test_generate_greedy_grows():
    cfg = KIMI_K3_CONFIG_MICRO
    torch.manual_seed(1)
    model = KimiK3Model(cfg)
    prompt = torch.randint(0, cfg["vocab_size"], (1, 4))
    out = generate_text_basic(model, prompt, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 7)


def test_0_18b_instantiates():
    cfg = KIMI_K3_CONFIG_0_18B
    model = KimiK3Model(cfg)
    n = count_parameters(model)
    assert n > 50_000_000
    # ~0.18B scale (embedding-dominated)
    assert 100_000_000 < n < 300_000_000
    ids = torch.randint(0, 1000, (1, 4))
    with torch.no_grad():
        logits, _ = model(ids)
    assert logits.shape == (1, 4, cfg["vocab_size"])


if __name__ == "__main__":
    for fn in [
        test_layer_schedule_matches_k3_tiny,
        test_0_18b_dims_match_hf_card,
        test_full_reference_is_k3_not_other_arch,
        test_situ_is_not_silu,
        test_kda_recurrent_shapes,
        test_modules_are_k3_types,
        test_micro_forward,
        test_generate_greedy_grows,
    ]:
        fn()
        print(f"OK  {fn.__name__}")
    print("running 0.18B smoke…")
    test_0_18b_instantiates()
    print("all tests passed")
