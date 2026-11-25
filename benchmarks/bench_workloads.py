"""Large-token real-world coding & writing benchmarks for Inkling-2-Mini W4A16 (depth=0 vs depth=2)."""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import gc
import time
from pathlib import Path
from typing import Any

import mlx.core as mx

from mlx_lm.models.inkling2 import layers as inkling2_layers
from mlx_lm.utils import load, maybe_set_recommended_wired_limit

from bench_mtp import Inkling2DSparkDrafter, format_prompt, run_generation


MODEL_PATH = Path(os.environ.get("INKLING_W4A16", "Inkling-2-Mini-MLX-W4A16"))
KERNELS_SRC = Path("mlx_lm_patch/mlx_lm/models/inkling2/kernels.py")
README_SRC = Path("README.md")


def _clip_tokens(tokenizer: Any, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens])


def build_workloads(tokenizer: Any) -> list[tuple[str, list[int]]]:
    kernels_snippet = _clip_tokens(tokenizer, KERNELS_SRC.read_text(), 1950)
    readme_snippet = _clip_tokens(tokenizer, README_SRC.read_text(), 3950)

    specs = [
        (
            "Coding: Async Scheduler (short prompt)",
            "Design and implement a production-ready Python asyncio job scheduler with: "
            "1) per-tenant token-bucket rate limiting, 2) bounded priority queues with backpressure, "
            "3) exponential backoff with full jitter and dead-letter handling, and 4) graceful cancellation on SIGTERM. "
            "Provide complete runnable code and concise pytest tests covering concurrency edge cases.",
        ),
        (
            "Coding: Metal Kernel Review (~2k ctx)",
            "Review the following MLX Metal kernel file (`inkling2/kernels.py`). Identify the key performance bottlenecks "
            "during small-batch speculative verification (L = 2..8, large sequence length S), then write a clean "
            "split-K relative attention Metal kernel design and explain its threadgroup memory layout and numerical precision.\n\n"
            f"```python\n{kernels_snippet}\n```",
        ),
        (
            "Writing: Technical Deep-Dive (short prompt)",
            "Write a thorough engineering article comparing Apple Silicon unified memory architecture against discrete PCIe GPUs "
            "for running sparse Mixture-of-Experts and looped transformer models locally. Cover memory bandwidth rooflines, "
            "weight quantization trade-offs (W4A16 vs FP4/FP8), KV cache pressure, and when speculative decoding helps vs hurts.",
        ),
        (
            "Writing: Architecture Brief (~4k ctx)",
            "Using the following model documentation and specifications, write a comprehensive deployment & engineering brief "
            "covering architecture highlights, quantization choices, hardware memory sizing across 32GB/64GB/128GB systems, "
            "and practical latency vs throughput trade-offs.\n\n"
            f"{readme_snippet}",
        ),
    ]
    return [(name, format_prompt(tokenizer, prompt)) for name, prompt in specs]


def main() -> None:
    maybe_set_recommended_wired_limit()
    inkling2_layers._USE_SHORT_CONV_KERNEL = True

    print(f"[INFO] Loading optimized W4A16 model from {MODEL_PATH}...")
    model, tokenizer = load(str(MODEL_PATH))
    model.fuse_projections()

    drafter = Inkling2DSparkDrafter(MODEL_PATH, model, draft_bits=8, opt_drafter=True)
    workloads = build_workloads(tokenizer)

    # Warmup small and medium prefill + decode/verify Metal kernels.
    print("[INFO] Warming up Metal kernels (d=0 and d=2)...")
    warm_prompt = workloads[0][1]
    _ = run_generation(model, tokenizer, warm_prompt, 16, mtp_depth=0, stop_on_eos=False)
    _ = run_generation(
        model, tokenizer, warm_prompt, 16, mtp_depth=2, drafter=drafter, stop_on_eos=False
    )

    max_tokens = 1024
    rows = []

    for name, prompt_tokens in workloads:
        print(f"\n[RUN] {name} (prompt={len(prompt_tokens)} tok, max_gen={max_tokens} tok)...")
        mx.clear_cache()
        t0 = time.perf_counter()
        r0 = run_generation(
            model, tokenizer, prompt_tokens, max_tokens, mtp_depth=0, stop_on_eos=True
        )
        r0["wall_s"] = time.perf_counter() - t0
        r0["ttft_s"] = len(prompt_tokens) / r0["prompt_tps"]
        print(
            f"  d=0: gen_tokens={r0['gen_tokens']} | prompt={r0['prompt_tps']:.1f} tok/s (TTFT={r0['ttft_s']:.2f}s) | "
            f"gen={r0['gen_tps']:.2f} tok/s | wall={r0['wall_s']:.2f}s | peak={r0['peak_gb']:.2f} GB"
        )

        mx.clear_cache()
        t0 = time.perf_counter()
        r2 = run_generation(
            model,
            tokenizer,
            prompt_tokens,
            max_tokens,
            mtp_depth=2,
            drafter=drafter,
            stop_on_eos=True,
        )
        r2["wall_s"] = time.perf_counter() - t0
        r2["ttft_s"] = len(prompt_tokens) / r2["prompt_tps"]

        if r2["tokens"] == r0["tokens"]:
            verify_tag = "[EXACT MATCH]"
        else:
            diff_idx = next(
                (
                    i
                    for i, (a, b) in enumerate(zip(r0["tokens"], r2["tokens"]))
                    if a != b
                ),
                min(len(r0["tokens"]), len(r2["tokens"])),
            )
            assert diff_idx < len(r0["tokens"]) and diff_idx < len(r2["tokens"]), (
                f"[{name}] Length mismatch without divergence: {len(r0['tokens'])} vs {len(r2['tokens'])}"
            )
            a, b = r0["tokens"][diff_idx], r2["tokens"][diff_idx]
            v_cache = model.make_cache()
            v_logits = model(mx.array([prompt_tokens], dtype=mx.int32), cache=v_cache)[0, -1]
            for tok in r0["tokens"][:diff_idx]:
                v_logits = model(mx.array([[tok]], dtype=mx.int32), cache=v_cache)[0, -1]
            mx.eval(v_logits)
            del v_cache
            gap = abs(float(v_logits[a]) - float(v_logits[b]))
            tok_a, tok_b = tokenizer.decode([a]), tokenizer.decode([b])
            if gap >= 0.25:
                raise AssertionError(
                    f"[{name}] Non-tie mismatch at token {diff_idx}: d=0={a} ({tok_a!r}) vs d=2={b} ({tok_b!r}), gap={gap:.3f}"
                )
            verify_tag = f"[MATCH {diff_idx}/{r0['gen_tokens']}, near-tie gap={gap:.3f} ({tok_a!r} vs {tok_b!r})]"

        print(
            f"  d=2: gen_tokens={r2['gen_tokens']} | prompt={r2['prompt_tps']:.1f} tok/s (TTFT={r2['ttft_s']:.2f}s) | "
            f"gen={r2['gen_tps']:.2f} tok/s ({r2['gen_tps']/r0['gen_tps']:.2f}x) | "
            f"acc={r2['mean_accepted']:.2f}/2 ({r2['accept_rate']*100:.1f}%) | "
            f"wall={r2['wall_s']:.2f}s | peak={r2['peak_gb']:.2f} GB | {verify_tag}"
        )
        rows.append((name, len(prompt_tokens), r0, r2, verify_tag))
        gc.collect()

    print("\n### Large-Token Real-World Benchmark Summary (Apple M2 Max, max_tokens=1024, d=0 vs d=2)\n")
    print(
        "| Workload | Prompt Tok | Gen Tok (d=0 / d=2) | Prompt tok/s (d=0 / d=2) | Decode tok/s d=0 | Decode tok/s d=2 | Speedup | Mean Acc / Step (Accept %) | Wall Time (d=0 -> d=2) | Peak Mem (d=0 / d=2) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, plen, r0, r2, _tag in rows:
        speedup = r2["gen_tps"] / r0["gen_tps"]
        print(
            f"| {name} | {plen} | {r0['gen_tokens']} / {r2['gen_tokens']} | "
            f"{r0['prompt_tps']:.1f} / {r2['prompt_tps']:.1f} | "
            f"{r0['gen_tps']:.2f} | {r2['gen_tps']:.2f} | {speedup:.2f}x | "
            f"{r2['mean_accepted']:.2f}/2 ({r2['accept_rate']*100:.1f}%) | "
            f"{r0['wall_s']:.1f}s -> {r2['wall_s']:.1f}s | "
            f"{r0['peak_gb']:.2f} / {r2['peak_gb']:.2f} GB |"
        )


if __name__ == "__main__":
    main()
