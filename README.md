# Kimi K3 From Scratch

Minimal PyTorch implementation of the **Kimi K3** language-model architecture, for reading and experimentation.

Kimi K3 (Moonshot AI) is a large hybrid MoE model (~2.8T total / ~104B active). This repository reimplements its **text tower structure** at a tiny scale so the distinctive pieces are easy to inspect:

| Building block | Role in Kimi K3 |
| --- | --- |
| **KDA** (Kimi Delta Attention) | Linear / delta-rule attention with **channel-wise** forget gates; ~¾ of layers |
| **Gated MLA** | Full multi-head latent attention with output gate; ~¼ of layers |
| **Stable LatentMoE** | Sparse experts in a **latent** dimension; shared experts + sigmoid top‑k routing |
| **SiTU-GLU** | Activation: soft-capped tanh × sigmoid (not SiLU / SwiGLU) |
| **Block AttnRes** | Softmax mix over residual history (not fixed residual add) |

Default hyperparameters follow the public miniature  
[`inference-optimization/Kimi-K3-0.18B`](https://huggingface.co/inference-optimization/Kimi-K3-0.18B)  
(same layer *types* as full [`moonshotai/Kimi-K3`](https://huggingface.co/moonshotai/Kimi-K3), reduced width / depth / expert count).

**Scope:** structure-level, educational pure PyTorch. Not a drop-in for production kernels (FLA KDA, fused MLA, MXFP4 serving). For real inference use the official HF checkpoint (`trust_remote_code=True`) or vLLM / SGLang.

---

## Architecture

### Full model vs this repo’s default (0.18B-shaped)

| Setting | Full Kimi-K3 | Tiny (this config) |
| --- | ---: | ---: |
| Layers | 93 | **4** |
| Hidden size | 7168 | **512** |
| Attention heads | 96 | **4** |
| Attention mix | 69 KDA + 24 Gated MLA | **3 KDA + 1 MLA** |
| Dense layers | 1 | **1** |
| Dense intermediate | 33792 | **1024** |
| MoE intermediate | 3072 | **256** |
| Latent MoE dim | 3584 | **256** |
| Experts / top‑k / shared | 896 / 16 / 2 | **8 / 2 / 1** |
| `kv_lora` / `q_lora` | 512 / 1536 | **64 / 128** |
| qk_nope / rope / v | 128 / 64 / 128 | **64 / 32 / 64** |
| KDA head dim | 128 | **64** |
| Activation | SiTU-GLU | SiTU-GLU |
| AttnRes block size | ~12 | **4** |
| Vocab / context | ~160K / 1M | 163840 / 4096 |

### Layer stack (0.18B)

Same **3∶1 KDA∶MLA** ratio as the full model (69∶24).

| Layer | Attention | FFN |
| --- | --- | --- |
| 0 | KDA | Dense SiTU-GLU |
| 1 | KDA | LatentMoE |
| 2 | KDA | LatentMoE |
| 3 | Gated MLA | LatentMoE |

### Design highlights

**1. Hybrid attention (KDA + Gated MLA)**  
Most layers use KDA: fixed-size recurrent state, short convolutions on Q/K/V, L2-normalized keys, channel-wise decay \(\alpha\), and a delta-rule write. Every fourth layer is Gated MLA for global, lossless retrieval (low-rank Q/KV, NOPE + RoPE split, sigmoid output gate).

**2. Stable LatentMoE**  
Routed experts do not run in full hidden width. Tokens are projected to a latent size, optionally normalized there, processed by top‑k experts, then projected back. Shared experts stay on full width. Router uses sigmoid scores and a `noaux_tc`-style correction bias.

**3. SiTU-GLU**  
Gate branch: \(\mathrm{soft\_cap}(x)\cdot\sigma(\beta x)\). Up branch is soft-capped. Intended to stabilize activations at large MoE scale versus classic SwiGLU.

**4. Block Attention Residuals**  
Instead of only \(x \leftarrow x + f(x)\), each sublayer mixes a **history bank** of residual sources with a learned (zero-init) query and softmax weights, within blocks of size `attn_res_block_size`.

---

## Quick start

```bash
pip install -r requirements.txt

# interactive walkthrough
jupyter notebook kimi_k3.ipynb

# CLI smoke (micro config)
python kimi_k3.py

# tests
python tests/test_kimi_k3.py
```

```python
import torch
from kimi_k3 import KIMI_K3_CONFIG_MICRO, KimiK3Model, describe_architecture
from generate import generate_text_basic

print(describe_architecture(KIMI_K3_CONFIG_MICRO))
model = KimiK3Model(KIMI_K3_CONFIG_MICRO).eval()
ids = torch.randint(0, 256, (1, 8))
logits, _ = model(ids)
out = generate_text_basic(model, ids, max_new_tokens=4, temperature=0.0)
```

Configs:

- `KIMI_K3_CONFIG_0_18B` — HF tiny dimensions (~0.18B params with large vocab)
- `KIMI_K3_CONFIG_MICRO` — same topology, tiny dims for CPU demos
- `KIMI_K3_FULL_REFERENCE` — full-model numbers for documentation only (do not instantiate)

---

## Repository layout

| Path | Description |
| --- | --- |
| [`kimi_k3.ipynb`](kimi_k3.ipynb) | Main walkthrough: specs, components, forward, generate |
| [`kimi_k3.py`](kimi_k3.py) | Configs, KDA, Gated MLA, LatentMoE, AttnRes, `KimiK3Model` |
| [`kimi_k3_ops.py`](kimi_k3_ops.py) | SiTU, norms, short conv, KDA recurrence, RoPE |
| [`generate.py`](generate.py) | Greedy / sampling decode |
| [`load_weights.py`](load_weights.py) | Optional HF tiny load + best-effort key map |
| [`tests/`](tests/) | Structure and forward checks |
| [`docs/inference-cost-report.md`](docs/inference-cost-report.md) | Kimi K3 vs DeepSeek V4.1 Flash: FLOPs, state, and inference economics |

---

## Optional: official tiny weights

```python
from load_weights import load_hf_reference_model

model, tokenizer = load_hf_reference_model(
    "inference-optimization/Kimi-K3-0.18B",
    device="cuda",
)
```

Requires network download (~0.69 GB) and `transformers` with `trust_remote_code=True`. Mapping those tensors into this code graph (`map_and_load_into_from_scratch`) is best-effort and not guaranteed to match official numerics.

---

## References

- [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- [inference-optimization/Kimi-K3-0.18B](https://huggingface.co/inference-optimization/Kimi-K3-0.18B)
- [Kimi K3 tech blog](https://www.kimi.com/blog/kimi-k3)
- KDA: [Kimi Linear (arXiv:2510.26692)](https://arxiv.org/abs/2510.26692)
- [flash-linear-attention KDA layer](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py)
- Attention Residuals (Kimi Team technical report)

---

## License

Code in this repository is for educational use. Model weights on Hugging Face are under their respective licenses (e.g. Kimi K3 License). Kimi K3 and related names are trademarks of their owners.
