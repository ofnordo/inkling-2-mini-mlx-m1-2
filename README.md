# Inkling-2-Mini-MLX-W4W8 & Custom Mac M1/M2 Metal Kernels

> **Experimental / Vibe-Coded Disclaimer:** This repository and companion quantized checkpoint are **vibe-coded experimental test kernels and quants** built to explore how far Apple M1/M2 silicon can push Thinking Machines' [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). **No guarantees, no warranty, and no production SLA are offered.** Everything here is shared as-is to help the local MLX / Apple Silicon community benchmark, reproduce, and build upon.

- **Code, Kernels, CLI & Benchmarks (GitHub):** [ofnordo/inkling-2-mini-mlx-m1-2](https://github.com/ofnordo/inkling-2-mini-mlx-m1-2)
- **Quantized Model Weights + Bundled DSpark MTP (`5.524 bpw`, `18 GB`) (Hugging Face):** [ofnordo/Inkling-2-Mini-MLX-W4W8](https://huggingface.co/ofnordo/Inkling-2-Mini-MLX-W4W8)
- **Base Model:** [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`25.79B` total / `3.0B` active params, `24` physical layers executing `48` looped layer visits per token, `1M` max context)
- **Base `mlx-lm` PR:** [ml-explore/mlx-lm#2086](https://github.com/ml-explore/mlx-lm/pull/2086) (`jundot/mlx-lm` `inkling2-mini` @ `e2b5bbd`)

---

## Why a Purpose-Built Mac M1/M2 Kernel & Quant?

Apple **M1 and M2 (Apple GPU Family 7 / 8 — G13 / G14)** have two major hardware differences compared to NVIDIA Blackwell/Hopper GPUs (and newer M3/M4/M5 Macs):

1. **No Hardware NVFP4 / MXFP8 Tensor Decode:**
   Inkling 2 Mini is trained and deployed natively in **NVFP4** (layers `0..19` experts) and **MXFP8** (last four readout layers `20..23` experts). On M1/M2 Metal, unpacking NVFP4 (`group_size=16`, FP8 scales + `global_scale`) and MXFP8 in software costs **~30–40% decode throughput** compared to MLX's native **`affine` (`group_size=64`)** `gather_qmm` kernels.
2. **Emulated BF16 vs Native FP16 (`half`) on M1/M2:**
   M1/M2 GPUs execute `float16` natively at full throughput while `bfloat16` incurs emulation overhead on compute-heavy prefill and GEMM ops. Recasting BF16 leaves (scales, norms, embeddings, linear weights) to **FP16** at load time—paired with custom FP16-aware Metal kernels—unlocks **+33% to +50% faster prefill (`501–603 tok/s` up to 32k)** and **+11% to +19% faster decode** over base `mlx-lm` PR #2086.
3. **Training-Matched Mixed Precision (`W4W8-Native-A8`, `5.524 bpw`):**
   Uniform 4-bit quantization (`W4A16`) crushes the last 4 MXFP8 readout layers (`20..23`) and BF16 attention/`lm_head` down to 4-bit, causing long-horizon `<|content_thinking|>` loops (`0/12` on our 64k DAG workflow test). Even `oQ4e`-style quants (`4.91 bpw`) compress layers `20..23` down to 4-bit (`10/12`). Matching the exact training precision boundary in native MLX `affine` format solves both speed and accuracy:

| Tensor Group | Layers / Modules | Native Checkpoint | `Inkling-2-Mini-MLX-W4W8` |
| :--- | :--- | :--- | :--- |
| **MoE Experts (`0..19`)** | `layers.0..19.mlp.experts.{gate_up_proj,down_proj}` (`40` tensors, `20.38B` params — prelude + 3× looped core + postlude `18..19`) | **NVFP4** (`gs=16`) | **4-bit Affine (`gs=64`)** |
| **MoE Experts (`20..23`)** | `layers.20..23.mlp.experts.{gate_up_proj,down_proj}` (`8` tensors, `4.08B` params — final 4 readout layers) | **MXFP8** (`gs=32`) | **8-bit Affine (`gs=64`)** |
| **Attention, Embed, Head** | `q/k/v/r/o/gate/sink_proj`, `embed_tokens`, `lm_head` | **BF16** | **8-bit Affine (`gs=64`)** |
| **Norms, ShortConv, Gates** | `sconv`, `controllers`, RMSNorms | **BF16** | **BF16** *(recast to FP16 at load on M1/M2)* |
| **AttnRes & Router Queries** | `attn_res.*_queries`, `middle_cycle_cross_pass_*_queries`, `mlp.gate.bias` | **FP32** | **FP32** *(untouched)* |
| **DSpark MTP Drafter** | `mtp.safetensors` (`3` draft layers + Markov RNN head, taps `[5, 23, 41, 48]`) | **BF16** | **Bundled** *(8-bit draft layers + 4-bit folded head at runtime)* |

---

## Improvements Over Base `mlx-lm` PR #2086 (`e2b5bbd`)

All patches live in [`mlx_lm_patch/mlx_lm/models/inkling2/`](https://github.com/ofnordo/inkling-2-mini-mlx-m1-2/tree/main/mlx_lm_patch/mlx_lm/models/inkling2) (with exact unified diff in [`inkling2_pr2086_diff.patch`](https://github.com/ofnordo/inkling-2-mini-mlx-m1-2/blob/main/mlx_lm_patch/inkling2_pr2086_diff.patch)):

1. **5 Custom Fused Metal Kernels (`kernels.py`, `fp16` + `bf16`)**:
   - **`short_conv`**: Fused causal depthwise short-convolution + residual + state update for any `L >= 1` (**1.21×–1.29×** faster than `nn.Conv1d` across `B=1..16`).
   - **`moe_short_conv`**: Fuses the weighted `top-k = 8` MoE expert reduction directly into the post-MLP causal depthwise short-convolution + residual + state update.
   - **`dynamic_conv`**: Fused dynamic short-convolution for the 8 full-attention K/V projections.
   - **`qkv_post`**: Fused Q/K/Rel L2 clipping + unweighted V RMSNorm + `tau` scaling + transpose in a single Metal dispatch.
   - **`relative_attention_splitk`**: Two-pass Split-K, GQA-grouped (`G = 3` query heads per KV head read), full-`float32` online-softmax Metal kernel with circular ring-buffer indexing (`rot_idx = cache._idx`). Accelerates both `L = 2..8` MTP verify (**1.85×–3.02×**) and `B > 1, L = 1` batched decode (**1.50×–2.16×**, replacing upstream's unfused `_attention_reference` fallback).
2. **M1/M2 Native FP16 Activation Path (`--fp16`, auto-enabled on M1/M2)**:
   - Recasts BF16 leaves to `float16` at load while keeping AttnRes/router queries in `float32`, delivering **+33% to +50% faster prefill** and **+11% to +19% faster decode** on M2 Max.
3. **Linear Projection Fusion (`model.fuse_projections()`)**:
   - Concatenates `q_proj + r_proj + gate_proj + sink_proj` into `_qrgs_proj` and `MessageGate` projections into `_gates_proj`, plus fused QKV / KV-ctx / gate-up inside the DSpark drafter.
4. **`SpeculativeRotatingKVCache` (6× Lower MTP Memory at Long Context)**:
   - Adds 16 tokens of tail slack (`max_size = 1024 + 16`) with exact tail `trim()` on all 40 sliding-window visits so MTP rollback never loses sliding-window history—cutting 512k speculative KV cache from **`>100 GB`** down to **`16.9 GB`**.
5. **DSpark MTP Speculative Decoding + `AdaptiveMTPController` (`mtp.py`)**:
   - 8-bit affine draft transformer layers + 4-bit affine folded Markov head (`[201024, 256]`) + exact `depth=1` linearity fusion.
   - Automatic EMA-based adaptive controller backs off to `d=0` on high-entropy prose (avoiding `d=2` slowdowns) while staying speculative on code, JSON, and tool calls (`43–48 tok/s`).
6. **Offline KV + ShortConv + DSpark Prompt Caching (`inkling_cache.py`)**:
   - Save/load full prefilled contexts (lossless 16-bit or 2×-smaller 8-bit `.safetensors`), watchdog-safe adaptive prefill chunking (`512 -> 256 -> 128 -> 64`) for 512k+ tokens, incremental layering, and **`< 1 ms` in-memory cache branching**.

---

## Benchmark Results (Apple M2 Max, 64 GB Unified Memory)

### 1. Head-to-Head up to `32k` Context vs Base `mlx-lm` PR #2086 & `oMLX 0.8.3`
*Prompt: Chesterton's The Man Who Was Thursday (`512` to `32,768` tokens), `256` generated tokens.*

#### Prefill Throughput (`tok/s` & Time to First Token)
| Context | oMLX `0.8.3` (BF16, NVFP4) | oMLX `0.8.3` (M1/M2 FP16, NVFP4) | Base `mlx-lm` PR #2086 (`e2b5bbd`, W4W8 BF16) | **Our Custom (`W4W8` + FP16 kernels)** | Gain vs PR #2086 | Gain vs oMLX M1/M2 FP16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`512` tok** | `256.6` (`1.9s`) | `301.3` (`1.7s`) | `365.3` (`1.4s`) | **`549.0 tok/s`** (`0.9s`) | **+50.3% (1.50×)** | **+82.2% (1.82×)** |
| **`4,096` tok** | `411.7` (`9.9s`) | `498.1` (`8.2s`) | `416.6` (`9.8s`) | **`602.8 tok/s`** (`6.8s`) | **+44.7% (1.45×)** | **+21.0% (1.21×)** |
| **`16,384` tok** | `403.6` (`40.6s`) | `503.5` (`32.5s`) | `407.1` (`40.2s`) | **`543.0 tok/s`** (`30.2s`) | **+33.4% (1.33×)** | **+7.8% (1.08×)** |
| **`32,768` tok** | `384.6` (`85.2s`) | `465.5` (`70.4s`) | `373.6` (`87.7s`) | **`500.9 tok/s`** (`65.4s`) | **+34.1% (1.34×)** | **+7.6% (1.08×)** |

#### Decode Throughput (`tok/s`) & Peak Memory
| Context | oMLX `0.8.3` (BF16, NVFP4) | oMLX `0.8.3` (M1/M2 FP16, NVFP4) | Base PR #2086 (`e2b5bbd`, W4W8 `d=0`) | **Our Custom (`W4W8` + FP16 `d=0`)** | Peak Mem (`PR #2086` $\rightarrow$ `Custom`) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| **`512` tok** | `21.92` | `24.03` | `25.87` | **`30.86 tok/s`** *(+19.3%)* | `19.65 GB` $\rightarrow$ **`19.49 GB`** |
| **`4,096` tok** | `23.92` | `20.99` | `26.68` | **`29.87 tok/s`** *(+12.0%)* | `21.60 GB` $\rightarrow$ **`20.25 GB`** |
| **`16,384` tok** | `23.04` | `20.54` | `25.16` | **`28.15 tok/s`** *(+11.9%)* | `21.96 GB` $\rightarrow$ **`20.70 GB`** |
| **`32,768` tok** | `21.25` | `18.97` | `23.55` | **`26.12 tok/s`** *(+10.9%)* | `22.49 GB` $\rightarrow$ **`21.23 GB`** |

---

### 2. Adaptive MTP Depth Controller (`Static` vs `Adaptive`, FP16 Kernels)

| Workload | `Static d=0` | `Static d=2` | **`Adaptive d<=2`** *(default)* | `Adaptive d<=3` | Result |
| :--- | ---: | ---: | ---: | ---: | :--- |
| **Coding (`44` tok ctx)** | `30.10 tok/s` | `43.39 tok/s` (`75.0%` acc) | **`43.17 tok/s`** (`76.9%` acc) | `42.27 tok/s` (`68.9%` acc) | **+43.4% over `d=0`** |
| **Prose (`4k` ctx)** | `29.61 tok/s` | `27.89 tok/s` (`36.9%` acc) | **`30.64 tok/s`** (`53.6%` acc) | **`30.88 tok/s`** (`48.5%` acc) | **Beats both `d=2` (+9.9%) and `d=0` (+3.5%)** |
| **Prose (`32k` ctx)** | `25.95 tok/s` | `22.37 tok/s` (`31.3%` acc) | **`25.53 tok/s`** (`199/230` `d0`) | **`25.66 tok/s`** (`172/213` `d0`) | **+14.1% over `d=2`**, eliminates slowdown |

---

### 3. `512k` (`512,444` Tokens) Needle-in-a-Haystack Recall, Scaling & Memory
*Non-repeating literary corpus: full text of G.K. Chesterton's The Man Who Was Thursday (`141.8k` tok) + opening `~370k` tok of Dumas's The Count of Monte Cristo, with 5 hidden archive needles planted from 5% to 95% depth (`benchmarks/bench_512k_niah.py`).*

- **Recall Score:** **5/5 verbatim exact recall** (plus accurate plot/opening synthesis of both novels at tokens `0` and `142k`!)
  - `CODE-ALPHA` (`5.0%`, token `25,607`): `'VELVET-COMPASS-7419'` — **PASS (verbatim)**
  - `CODE-BETA` (`25.0%`, token `128,026`): `4,832 silver florins, Captain Maren Vance` — **PASS (verbatim)**
  - `CODE-GAMMA` (`50.0%`, token `256,034`): `Thursday at 16:45 sharp` — **PASS (verbatim)**
  - `CODE-DELTA` (`75.0%`, token `384,026`): `'Inspector Orlov-9'` — **PASS (verbatim)**
  - `CODE-EPSILON` (`95.0%`, token `486,400`): `19 crates of saffron, October 14th` — **PASS (verbatim)**
- **512k Decode Speed:** **`8.56 tok/s`** (`MTP d=2`, `51.4%` acc) / **`8.35 tok/s`** (`d=0`)
- **512k Memory Footprint:** **`35.75 GB` active / `37.99 GB` peak** (`13.8 GB` under the `51.8 GB` safe Metal working set on a 64 GB Mac!)

| Context Reached | Adaptive Chunk | Recent Prefill | Avg Prefill | Elapsed Time | Active RAM | Peak RAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`32,768`** | `512` | `470.7 tok/s` | `470.7 tok/s` | `1.2 min` | `20.01 GB` | `20.99 GB` |
| **`65,536`** | `512` | `344.6 tok/s` | `397.9 tok/s` | `2.7 min` | `21.09 GB` | `22.11 GB` |
| **`131,072`** | `256` | `209.4 tok/s` | `296.5 tok/s` | `7.4 min` | `23.18 GB` | `24.22 GB` |
| **`262,144`** | `128` | `112.1 tok/s` | `192.5 tok/s` | `22.7 min` | `27.45 GB` | `28.74 GB` |
| **`393,216`** | `128` | `81.4 tok/s` | `139.6 tok/s` | `46.9 min` | `31.75 GB` | `33.61 GB` |
| **`512,444`** | `64` | `64.3 tok/s` | `94.1 tok/s` | `90.8 min` | **`35.75 GB`** | **`37.99 GB`** |

*(Why KV fits so easily: 40/48 visits are capped at `1,024`-token sliding windows (`160 MB` total); only 8 full-attention visits grow with context at **`32 KB/token`** = `3.2 GB` per `100k` tokens!)*

---

### 4. Quantization Intelligence Comparison (`64k` 5-Subsystem DAG Workflow Engine, `MTP d=2`)
*One-shot 12-test complex programming spec (`reasoning_effort="medium"`, `max_tokens=65536`, `benchmarks/eval_quants_workflow_64k.py`).*

| Quant Recipe | Bits/Weight | Peak Mem | Total Gen Tok (`Think` / `Code`) | Decode `tok/s` (`d=2`) | Wall Time | Test Score |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **`W4W8-Native-A8` (this export)** | **5.52** | **26.37 GB** | **16,654** (`12,865` / `3,789`) | **32.02** | **526.3s** *(8.8m)* | **12/12** |
| `W4W8-Native-BF16` (`BF16 attn/head`) | 5.97 | 27.84 GB | 12,279 (`8,543` / `3,736`) | 23.12 | 536.9s *(8.9m)* | **12/12** |
| `W6A16` (uniform 6-bit affine) | 6.64 | 29.88 GB | 19,161 (`15,294` / `3,867`) | 31.01 | 624.1s *(10.4m)* | **12/12** |
| `Original` (NVFP4/MXFP8/BF16) | 5.93 | 27.67 GB | 16,969 (`13,253` / `3,716`) | 21.77 | 785.6s *(13.1m)* | **11/12** |
| `W4W8-Mixed` (`4b` all experts `0..23`, `8b` attn) | 4.91 | 24.44 GB | 15,743 (`11,988` / `3,755`) | 27.94 | 569.3s *(9.5m)* | **10/12** |
| `W4A16` (uniform 4-bit affine) | 4.67 | 23.62 GB | 65,536 (`65,536` / `0`) | 34.55 | 1902.6s *(31.7m)* | **0/12** *(looped)* |
| `W3A16` (uniform 3-bit affine) | 3.68 | 20.53 GB | 65,536 (`65,536` / `0`) | 20.40 | 3218.2s *(53.6m)* | **0/12** *(looped)* |

---

### 5. Fictional-World Agentic Tool-Calling Benchmarks (`12/12` PASS, `0/116` Malformed Calls)
*All facts/code/logs live strictly in the sandbox so low-knowledge reasoning & tool discipline are tested cleanly (`benchmarks/agent_bench.py`, `MTP d=2`, `effort=low`).*

| Suite / Task | Category | Result | Turns / Calls (`err`/`mal`) | Final Context | Decode `tok/s` | Wall Time |
| :--- | :--- | :---: | ---: | ---: | ---: | ---: |
| **Tier-1** (9 tasks: `support_refund`, `ops_deploy`, 3× `wiki_*`, 2× `data_*`, `repo_bugfix`, `spec_implement`) | API policy, SRE 503 recovery, multi-hop retrieval, Python analytics, SWE bugfix & 14-test hidden spec parser | **9/9** | `58` / `79` (`2`/`0`) | up to `8k` | **`38.7–48.3`** | `5.0 min` |
| **Tier-2:** `refactor_multifile` | 9-file v1$\rightarrow$v2 multi-currency refactor (**18/18** hidden tests) | **PASS** | `13` / `16` (`0`/`0`) | `23,610` tok | **`38.9 tok/s`** (`79%` acc) | `6.3 min` |
| **Tier-2:** `needle_debug_longctx` | Tool-assisted log triage over `240 KB` noisy logs (**4/4** hidden replay tests) | **PASS** | `11` / `13` (`0`/`0`) | `9,521` tok | **`40.1 tok/s`** (`75%` acc) | `1.4 min` |
| **Tier-2:** `needle_pure_longctx` | **Pure `24.8k`-token log haystack in prompt, no `search` tool** (**4/4** hidden tests) | **PASS** | `7` / `8` (`0`/`0`) | **`28,642` tok** | **`36.7 tok/s`** (`77%` acc) | `1.7 min` |

---

### 6. Continuous Batching Scaling (`B = 1..16`, `batch_generate_inkling`)

| Batch Size (`B`) | Base PR #2086 Decode | **Custom Kernels + Split-K Batched Decode** | Speedup | Peak Memory |
| ---: | ---: | ---: | ---: | ---: |
| `B = 1` | `26.02 tok/s` | **`28.83 tok/s`** (`28.83`/seq) | **+10.8%** | `18.37 GB` |
| `B = 2` | `40.12 tok/s` | **`44.72 tok/s`** (`22.36`/seq) | **+11.5%** | `18.50 GB` |
| `B = 4` | `59.22 tok/s` | **`71.48 tok/s`** (`17.87`/seq) | **+20.7%** | `18.77 GB` |
| `B = 8` | `80.41 tok/s` | **`89.61 tok/s`** (`11.20`/seq) | **+11.4%** | `19.36 GB` |
| `B = 16` | `106.06 tok/s` | **`115.75 tok/s`** (`7.23`/seq) | **+9.1%** | `19.76 GB` |

---

## Quick Start

### 1. Install the Patched `mlx-lm` (PR #2086 + Custom M1/M2 Kernels)
```bash
git clone https://github.com/ofnordo/inkling-2-mini-mlx-m1-2.git
cd inkling-2-mini-mlx-m1-2
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./install_patch.sh
```

### 2. Run Inference (Auto-Downloads `ofnordo/Inkling-2-Mini-MLX-W4W8`, Adaptive MTP + FP16 Enabled on M1/M2)
```bash
# Single-turn prompt
python3 inkling_mlx.py -p "Implement a thread-safe LRU cache with TTL in Python." -e minimal

# Interactive chat REPL (/effort, /depth, /think, /clear, /stats, /quit)
python3 inkling_mlx.py --chat

# Hide <|content_thinking|> and stream only the final answer
python3 inkling_mlx.py -p "Explain wait-free vs lock-free queues." -e low --hide-thinking

# Disable adaptive MTP or FP16 for A/B testing
python3 inkling_mlx.py -p "Write a binary search tree." --no-adaptive-mtp --no-fp16
```

### 3. Build & Reuse Offline Prompt Caches (Save Up to 90 Minutes of Prefill!)
```bash
# Build a reusable lossless 16-bit cache (or pass --cache-bits 8 for ~2x smaller disk size)
python3 inkling_mlx.py \
  --build-cache caches/my_repo.safetensors \
  --context-file ./my_project/ \
  --cache-bits 16

# Query the cached prefix instantly (~0.25s-3.0s reload) at any reasoning effort
python3 inkling_mlx.py \
  --cache caches/my_repo.safetensors \
  -p "Audit the concurrency model in the cached repo." \
  -e low

# Interactive chat with <1 ms in-memory cache branching (/reset-cache, /save-cache <file>)
python3 inkling_mlx.py --cache caches/my_repo.safetensors --chat
```

| Cached Context | Disk (`16-bit`) | Disk (`8-bit`) | Reload on M2 Max SSD |
| ---: | ---: | ---: | ---: |
| `32k` tok | `1.18 GB` | `0.68 GB` | `~0.25s` |
| `128k` tok | `4.26 GB` | `2.31 GB` | `~0.8s` |
| `256k` tok | `8.36 GB` | `4.48 GB` | `~1.5s` |
| `512k` tok | `16.56 GB` | `8.82 GB` | `~3.0s` |

---

## Included Benchmarks & Converter Scripts

- `convert/convert_w4w8_native.py` — Reproduce `Inkling-2-Mini-MLX-W4W8` from `thinkingmachines/Inkling-2-Mini` (`INKLING_SRC=... INKLING_DST=... python3 convert/convert_w4w8_native.py`).
- `benchmarks/bench_vs_mainline_32k.py` — 512 $\rightarrow$ 32k prefill/decode comparison vs PR #2086.
- `benchmarks/bench_512k_niah.py` — 512k Needle-in-a-Haystack + decode speed test.
- `benchmarks/bench_omlx_32k.py` — 512 $\rightarrow$ 32k comparison against `oMLX`.
- `benchmarks/bench_mtp.py` & `benchmarks/bench_workloads.py` — MTP depth `0..7` sweep and 1k-token coding/writing workloads.
- `benchmarks/eval_quants_coding.py` & `benchmarks/eval_quants_workflow_64k.py` — 12-test MVCC Store and 64k 5-Subsystem DAG Workflow coding evaluations across quants.
- `benchmarks/agent_bench.py` (`--suite {tier1,tier2,all}`) — 12-task persistent-cache multi-turn agentic benchmark suite (`agent_tasks.py` + `agent_tasks_tier2.py`).

## License

Apache 2.0, following `thinkingmachines/Inkling-2-Mini` and `ml-explore/mlx-lm`.
