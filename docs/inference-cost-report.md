# Inference Economics of Kimi K3 and DeepSeek V4.1 Flash

> Status: architecture-level analysis based on public model specifications and this repository's implementation  
> Updated: 2026-09-15

## Executive summary

Total parameter count and activated parameter count do not fully explain the inference cost of a long-context model. A more useful first-order model is:

$$
F_{\mathrm{decode}}(N) = A + BN + C(N)
$$

where:

- $N$ is the existing context length before the next token is generated;
- $A$ is the approximately context-independent cost of weight matrices, MoE layers, projections, and fixed-state updates;
- $B$ is the marginal compute required to retain and query one additional history token;
- $C(N)$ contains fixed, bounded, or piecewise costs such as Top-K attention, sliding-window attention, hierarchical indexing, and speculative decoding.

Using raw FLOPs with 1 MAC counted as 2 FLOPs, the dominant terms can be approximated as:

$$
F_{\mathrm{K3}}(N) \approx 208\ \mathrm{GFLOPs} + 5.0135 \times 10^6 N
$$

$$
F_{\mathrm{V4.1}}(N) \approx 35.8\ \mathrm{GFLOPs} + 20{,}480N
$$

These equations are analytical estimates, not performance guarantees. They expose the architectural trade-off:

- Kimi K3 replaces 69 of its 93 attention layers with fixed-state Kimi Delta Attention (KDA), but retains 24 global Gated MLA layers whose history-query cost grows with $N$.
- DeepSeek V4.1 Flash combines a causal encoder-decoder architecture, cross-layer compressed KV sharing, hierarchical sparse indexing, and bounded replay to reduce both the number of full-history scans and the amount of persistent state.

Under these assumptions, K3 has approximately 5.8–6.5 times the fixed decode compute and about 245 times the long-context slope of V4.1. At a 1M-token context, the raw operation-count ratio approaches 95 times. Real throughput and energy ratios will be smaller and workload-dependent because HBM bandwidth, MoE communication, batching, quantization, prefix caching, kernel efficiency, and speculative decoding also matter.

This repository is a small, readable PyTorch reconstruction of the Kimi K3 text architecture. It is suitable for studying KDA, Gated MLA, LatentMoE, SiTU-GLU, and AttnRes, but it does not reproduce the cache layout or optimized kernels of a production K3 serving stack. The full-model estimates in this report therefore cannot be measured directly with the current generation script.

## 1. Cost model

### 1.1 Prefill and decode must be separated

For a conventional decoder-only Transformer with a KV cache:

- During **prefill**, all prompt tokens form queries and attend to earlier prompt tokens. Full attention therefore contains an $O(N^2)$ term.
- During **decode**, past keys and values are cached. A new token queries the existing history, so full-attention work for that step is normally $O(N)$.

Consequently, describing a Transformer simply as “quadratic in $N$” is accurate for the main full-attention prefill term, but not for cached single-token decode.

For a conventional Transformer, a useful coarse expression is:

$$
F_{\mathrm{decode}}(N)
\approx
2P_{\mathrm{active}}
+
4LHdN
+
F_{\mathrm{other}}
$$

The exact coefficient depends on the attention representation, head dimensions, grouping, and whether latent projections are absorbed into adjacent operations. The important distinction is between the context-independent parameter term and the history-dependent attention term.

Prefill should be modeled separately:

$$
F_{\mathrm{prefill}}(N)
\approx
2P_{\mathrm{active}}N
+
F_{\mathrm{attention}}(N)
$$

For dense full attention, $F_{\mathrm{attention}}(N)$ contains an $O(N^2)$ component. Sparse, recurrent, compressed, or hybrid architectures change its coefficient, its coverage across layers, or its asymptotic behavior.

### 1.2 FLOPs are not power consumption

FLOPs are necessary but insufficient for estimating inference performance. A simplified roofline-style lower bound is:

$$
t_{\mathrm{token}}
\gtrsim
\max\left(
\frac{F_{\mathrm{token}}}{\text{effective compute throughput}},
\frac{M_{\mathrm{token}}}{\text{effective memory bandwidth}}
\right)
+ t_{\mathrm{communication}}
+ t_{\mathrm{launch}}
$$

Energy per token must be measured over time:

$$
E_{\mathrm{token}} = \int_0^{t_{\mathrm{token}}} P(t)\,dt
$$

$$
\mathrm{tokens/J} = \frac{1}{E_{\mathrm{token}}}
$$

FLOPs therefore cannot be converted directly into watts. Power and energy require a specified accelerator, precision, parallelism strategy, batch size, input/output length distribution, and serving objective.

## 2. Repository context and scope

This repository focuses on a readable reconstruction of the Kimi K3 text tower:

- `kimi_k3.py` defines the full-scale reference values and the executable 0.18B-shaped and micro configurations.
- `KimiDeltaAttention` demonstrates channel-wise forgetting and a delta-rule recurrent state.
- `GatedMLA` demonstrates low-rank Q/KV projections, NOPE/RoPE decomposition, and output gating.
- `LatentMoE` demonstrates latent-space routed experts, shared experts, and sigmoid Top-K routing.
- `BlockAttnRes` demonstrates learned mixing over residual history.
- `kimi_k3_ops.py` uses a clear token-by-token recurrence instead of production FLA or fused kernels.

The public full-scale configuration matches the reference values used by this repository: 93 layers, 69 KDA layers, 24 Gated MLA layers, hidden size 7168, 96 attention heads, approximately 2.8T total parameters, approximately 104B activated parameters, and a 1,048,576-token context window. The public configuration also specifies `kv_lora_rank=512`, `qk_rope_head_dim=64`, and `v_head_dim=128`. See the [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3) and [public configuration](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json).

The current implementation is not a production inference benchmark:

1. `generate_text_basic` does not use a KV cache. It reruns the complete truncated `ids_cond` sequence for every generated token.
2. The educational `GatedMLA` caches expanded per-head K/V tensors rather than a production absorbed-MLA latent cache.
3. The KDA cache returns the recurrent state but does not preserve the short-convolution state between decode steps.
4. Incremental MLA does not yet offset RoPE positions by the existing cache length.
5. The implementation uses general PyTorch operations and does not include FLA KDA, fused MLA, MXFP4 serving, expert parallelism, production prefix caching, or a production scheduler.

This report should therefore be used as architectural documentation and as a specification for a future cost estimator or benchmark. Wall-clock timings from the current scripts must not be extrapolated to the official Kimi K3 deployment.

## 3. A six-dimensional inference ledger

A practical comparison should track six dimensions:

| Dimension | Recommended representation | Question answered |
| --- | --- | --- |
| Decode compute | $F_{\mathrm{decode}}(N)=A+BN+C(N)$ | How much work is required per output token, and how does history amplify it? |
| Prefill compute | $F_{\mathrm{prefill}}(N)$ and FLOPs/input-token | How do prompt ingestion and time to first token scale? |
| Session state | KV/state bytes per context token plus fixed state | How much HBM or SSD does each long-running session consume? |
| Weight and data movement | Resident bytes and bytes/output-token | Is execution compute-bound or memory-bandwidth-bound? |
| Hardware outcome | tokens/s/GPU, tokens/J, TTFT, TPOT, P95 | What happens on a specified serving system? |
| Quality-adjusted cost | cost/task, joules/task, success rate, steps and retries | Does a cheap token complete the task economically? |

For agent workloads, the final metric should be task-level cost:

$$
\text{Task Cost}
=
\text{input cost}
+
\text{reasoning/output cost}
+
\text{tool-step cost}
+
\text{retry and failure cost}
$$

A model can compensate for a smaller active path by using more reasoning tokens or more agent steps. The lowest cost per token is not necessarily the lowest cost per successful task.

## 4. Kimi K3: high fixed compute and a visible context slope

### 4.1 Context-independent term

Kimi K3 reports approximately 104B activated parameters. Using the common matrix-multiplication estimate of 2 FLOPs per active parameter:

$$
A_{\mathrm{K3}}
\approx
2 \times 104 \times 10^9
=
208\ \mathrm{GFLOPs/token}
$$

This is a scale estimate rather than a complete operator census. It may omit or combine normalization, activation, routing, recurrent updates, the output head, and implementation-specific work. KDA recurrent-state operations grow with layer width but not with context length, so they belong in $A$ rather than $BN$.

### 4.2 History-dependent MLA term

The 69 KDA layers use fixed-size recurrent states. The 24 Gated MLA layers are the layers that continue to query a history whose length grows with $N$.

For an idealized absorbed-MLA decode path, the relevant public dimensions are:

- 96 query heads;
- latent KV dimension 512;
- RoPE key dimension 64;
- 24 MLA layers.

The score and value-aggregation work per history token per MLA layer is approximately:

$$
2 \times 96 \times \left[(512+64)+512\right]
=
208{,}896\ \mathrm{FLOPs}
$$

Across 24 layers:

$$
B_{\mathrm{K3}}
=
24 \times 208{,}896
=
5{,}013{,}504\ \mathrm{FLOPs/history\ token}
$$

This gives:

$$
\boxed{
F_{\mathrm{K3}}(N)
\approx
208\ \mathrm{GFLOPs}
+
5.0135 \times 10^6 N
}
$$

The history-query term exceeds the fixed model term at approximately:

$$
N^*
=
\frac{208 \times 10^9}{5{,}013{,}504}
\approx
41{,}500
$$

Under this model, MLA history access becomes the dominant source of raw decode FLOPs at roughly a 40K-token context.

### 4.3 K3 session state

A production latent cache stores approximately the following elements per token per MLA layer:

$$
512 + 64 = 576
$$

Across 24 MLA layers:

$$
576 \times 24 = 13{,}824\ \mathrm{elements/token}
$$

At one byte per FP8 element, this is approximately 13.8 KB/token, or 13.5 KiB/token. At the full 1,048,576-token context length, it is approximately 14.50 GB in decimal units, in addition to fixed KDA recurrent states and implementation metadata.

KDA eliminates context-proportional KV growth from 69 layers, but the remaining 24 global MLA layers still make both compute and session state sensitive to long contexts.

## 5. DeepSeek V4.1 Flash: flattening history compute and state

The public V4.1 Flash architecture specifies:

- a 552B-parameter backbone plus 196B parameters of Engram memory;
- 40 Transformer layers split into a 20-layer causal encoder and a 20-layer decoder;
- approximately 8B active parameters per prefill token and 16B per decode token;
- global KV source layers 2, 8, 14, and 20;
- index source layers 2, 8, 14, 20, 24, 28, 32, and 36;
- a 32-head, 128-dimensional indexer with Top-K 512;
- a later-stage candidate pool of 2,048 blocks times 8 positions, or 16,384 positions;
- a sliding window of 128 tokens;
- an FP4 global KV footprint of 890 bytes/token.

These values are documented in the [DeepSeek V4.1 Flash README](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md) and [public configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json).

### 5.1 Context-independent term

The weight-GEMM lower-bound estimate from 16B active parameters is:

$$
2 \times 16 \times 10^9
=
32\ \mathrm{GFLOPs/token}
$$

Adding an approximate allowance for fixed Top-K attention, the 128-token sliding window, and bounded-candidate reindexing gives a working value of approximately 35.8 GFLOPs/token. This is an analytical constant, not an official measured FLOP count.

### 5.2 Context-dependent indexer term

A first-order estimate treats the first three full-history index sources as operating over compressed encoder memory of length approximately $N/2$, while layer 20 operates over approximately $N$ positions:

$$
2 \times 32 \times 128 \times
\left(
\frac{N}{2}
+
\frac{N}{2}
+
\frac{N}{2}
+
N
\right)
=
20{,}480N
$$

Later attention layers primarily read Top-K 512 positions, while later reindexing is restricted to a fixed 16,384-position candidate pool. Their principal cost therefore belongs in $C(N)$ rather than repeatedly adding full-history terms to $B$.

The resulting working estimate is:

$$
\boxed{
F_{\mathrm{V4.1}}(N)
\approx
35.8\ \mathrm{GFLOPs}
+
20{,}480N
}
$$

This is not a complete operator-level FLOP census. It omits or simplifies projections, routing, Engram lookup, DSpark acceptance behavior, kernel details, and compressed-length boundaries. Its strongest use is comparing the order of magnitude of the context slope, not predicting absolute latency.

### 5.3 Architectural trade-offs

| Design choice | Resource saved | Capability or engineering trade-off |
| --- | --- | --- |
| 20+20 causal encoder-decoder | Full-depth processing of long prompts | History tokens no longer retain independent representations from every decoder layer |
| Cross-layer compressed KV sharing | HBM capacity, KV traffic, and persistent state | Less freedom for every layer to maintain independent historical memory |
| Hierarchical sparse indexing | Repeated full-history scans | An item missed by the first candidate selection is harder for later layers to recover |
| FP4 KV and experts | Storage, bandwidth, and matrix cost | Reduced numerical precision |
| Sliding-window bounded replay | Persistent local KV storage | State is reconstructed from a bounded recent window rather than persisted exactly for arbitrary history |
| 196B Engram memory | Replaces some neural computation with sparse lookup | Larger static storage and a more complex serving system |
| Agent/RL training and adjustable reasoning | Recovers target-task performance | Higher training investment and potentially more output tokens or agent steps |

The official documentation reports a persistent KV footprint approximately one eighth of the previous generation and a global KV footprint of 890 bytes/token. A full 1M-token context therefore uses approximately 0.93 GB of global KV. The model itself remains large: the vLLM deployment recipe reports an approximately 511 GB checkpoint and a default minimum VRAM budget of 614 GB. See the [vLLM deployment recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml).

Low API cost therefore does not imply a small model or a simple single-machine deployment.

### 5.4 What happens to the quadratic term

It is not accurate to say that V4.1 eliminates every $N^2$ term.

- During prefill, the causal encoder-decoder split reduces full-depth computation, while sparse and compressed attention reduce the coefficient and layer coverage of history interactions. Building full indices can still require work that grows across query-history pairs.
- During decode, shared KV, Top-K selection, and hierarchical candidate pools reduce the coefficient $B$ in the conventional $BN$ term. Deeper retrieval work is bounded independently of the full context length.

The architecture primarily removes repeated, full-resolution, per-layer access to the complete history.

## 6. Numerical comparison

Using the two working equations above:

| Current prefix | Kimi K3 | DeepSeek V4.1 Flash | K3 / V4.1 |
| ---: | ---: | ---: | ---: |
| 4K | 228.5 GFLOPs/token | 35.9 GFLOPs/token | 6.4x |
| 8K | 249.1 GFLOPs/token | 36.0 GFLOPs/token | 6.9x |
| 32K | 372.3 GFLOPs/token | 36.5 GFLOPs/token | 10.2x |
| 128K | 865.1 GFLOPs/token | 38.5 GFLOPs/token | 22.5x |
| 512K | 2.84 TFLOPs/token | 46.5 GFLOPs/token | 61.0x |
| 1M (1,048,576) | 5.46 TFLOPs/token | 57.3 GFLOPs/token | 95.4x |

The ratio between the long-context slopes is:

$$
\frac{5{,}013{,}504}{20{,}480}
\approx
244.8
$$

The state-size comparison is:

| Metric | Kimi K3 | DeepSeek V4.1 Flash |
| --- | ---: | ---: |
| Total/backbone parameters | 2.8T | 552B backbone + 196B Engram |
| Active parameters during decode | 104B | 16B |
| Active parameters during prefill | approximately 104B | 8B |
| Main long-context state | 24 layers of latent MLA cache plus fixed KDA state | Cross-layer shared FP4 global KV plus 128-token local state/replay |
| Bytes per context token | approximately 13.8 KB, assuming FP8 latent cache | 890 bytes of global KV |
| Full 1M context | approximately 14.50 GB plus fixed state | approximately 0.93 GB of global KV plus fixed/local overhead |

These values explain algorithmic and state complexity. They do not imply that V4.1 is always 95 times faster or uses 95 times less energy. Real workloads include shorter average contexts, prefix-cache hits, batching, communication, memory traffic, and different utilization levels.

## 7. Why capability does not fall in proportion to online compute

V4.1 does not obtain a free reduction in inference cost. It moves cost from repeated online computation into static capacity, training, and task-specific optimization:

$$
\text{less general compute and state redundancy}
\rightarrow
\text{larger sparse capacity and Engram memory}
\rightarrow
\text{agent data, RL, distillation, and reasoning-time compensation}
$$

The published base-model results show that the change is not lossless. V4.1 Flash trails V4 Pro on several general-knowledge, mathematics, and long-context evaluations, while performing strongly on code benchmarks. Post-training recovers or exceeds performance on several coding and agent benchmarks, but the published instruct results use the maximum `reasoning_effort=100` setting. Headline performance therefore includes additional inference-time compute and cannot be explained by architectural FLOPs alone.

Three boundaries are especially important for independent evaluation:

1. **Sparse-retrieval recall:** whether the first candidate-selection stage misses critical remote evidence.
2. **Bounded-replay error:** whether reconstructed local state introduces errors that accumulate over a long agent trajectory.
3. **General and genuinely long-range reasoning:** whether agent- and coding-focused post-training hides losses in the base model.

## 8. Implications for this repository

### 8.1 Mapping the analysis to the code

| Report concept | Repository location | What it currently demonstrates |
| --- | --- | --- |
| 69 KDA / 24 MLA topology | `KIMI_K3_FULL_REFERENCE` and `layer_type` | Full-scale reference topology and the tiny 3:1 topology |
| Fixed KDA state | `kimi_k3_ops.recurrent_kda` | A state shaped as $[B,H,D_k,D_v]$ that does not grow with context length |
| Linear MLA history cost | QK and AV operations in `GatedMLA.forward` | Each new cached query reads all historical K/V positions |
| LatentMoE active path | `LatentMoE.forward` | Top-K routed experts plus shared-expert computation |
| Production-cost gap | `generate_text_basic` | Current generation recomputes the context and cannot measure the analytical decode curve |

### 8.2 Recommended next steps

1. Add a static `cost_model.py` that derives parameter terms, KDA state size, MLA KV bytes, and $F(N)$ from a configuration. Cover the micro, 0.18B-shaped, and full-reference configurations.
2. Add estimator tests for $24 \times 576 = 13{,}824$ cache elements/token, KDA state shapes, and the tiny model's 3:1 layer mixture.
3. Implement correct incremental decode by caching KDA short-convolution state, offsetting RoPE positions by cache length, and adding `generate_with_cache`.
4. Expose separate estimates for the educational expanded-K/V cache and a production absorbed-MLA latent cache.
5. Define a reproducible benchmark protocol with device, dtype, batch size, prompt distribution, TTFT, TPOT, throughput, peak memory, and joules/token.

These additions would extend the repository from an architecture reconstruction into an architecture-cost laboratory without confusing readable reference code with production serving behavior.

## 9. Conclusion

The most reusable result is not a single FLOP number, but a way to inspect a model:

$$
\boxed{
\text{Inspect }A\text{ for per-step model weight, and }B\text{ for the marginal cost of history.}
}
$$

- Kimi K3 combines very large sparse capacity with KDA replacing most full-attention layers. This removes context-proportional KV growth from 69 of 93 layers, while 24 global MLA layers retain a substantial decode slope and cache footprint.
- DeepSeek V4.1 Flash combines a smaller active path with a causal encoder-decoder split, shared compressed KV, hierarchical sparse retrieval, and approximate state reconstruction. Its objective is to make the marginal cost of a 1M-context agent closer to that of a short-context model.
- V4.1 gives up per-layer independent historical representations, unrestricted repeated retrieval, higher cache precision, and fully persisted local state. It exchanges those properties for cheaper prefill, a flatter decode curve, and higher concurrency, then compensates with static capacity, training, and reasoning-time compute.
- Model selection must continue beyond raw FLOPs to bytes/token, tokens/J, latency, and quality-adjusted cost per successful task. Price, FLOPs, throughput, and power are related but distinct quantities.

For this project, the current code is best suited to explaining why K3 contains 69 fixed-state layers and 24 global-retrieval layers. It does not yet demonstrate the production efficiency of the official K3 model. A static cost estimator followed by correct incremental caching and controlled hardware benchmarks would make the analysis directly testable.

## References

- [Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Kimi K3 public configuration](https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json)
- [Kimi K3 technical blog](https://www.kimi.com/en/blog/kimi-k3)
- [DeepSeek V4.1 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/README.md)
- [DeepSeek V4.1 Flash public configuration](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json)
- [vLLM DeepSeek V4.1 Flash deployment recipe](https://github.com/vllm-project/recipes/blob/main/models/deepseek-ai/DeepSeek-V4.1-Flash.yaml)

## Estimation conventions

All equations in this report are architecture-level estimates rather than vendor guarantees or independent hardware measurements. The constants 208 GFLOPs, 35.8 GFLOPs, $5.0135 \times 10^6N$, and $20{,}480N$ use a raw-operation approximation. Other sources may use precision-weighted FLOPs, count an FMA as one operation, count it as two operations, or include only selected kernels. Cross-model comparisons must use a consistent convention and disclose the model version, implementation revision, precision, hardware, parallelism, batch size, context length, and output-length distribution.
