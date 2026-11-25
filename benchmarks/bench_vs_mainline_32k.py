#!/usr/bin/env python3
"""Head-to-head benchmark up to 32k context:
Latest PR #2086 (e2b5bbd) vs Custom Kernels + Fusion + FP16 + DSpark MTP (d=0 & d=2).
"""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

CONTEXTS = [512, 4096, 16384, 32768]
GEN_TOKENS = 256
MODEL_PATH = Path(os.environ.get("INKLING_MODEL", "ofnordo/Inkling-2-Mini-MLX-W4W8"))
BOOK_PATH = Path("niah_512k/thursday.txt")


def build_prompt(tokenizer, target_tokens: int) -> list[int]:
    book = BOOK_PATH.read_text(errors="ignore")
    toks = tokenizer.encode(book, add_special_tokens=False)
    prefix = tokenizer.encode(
        "<|message_system|><|content_text|>Thinking effort level: 0.1<|end_message|>"
        "<|message_user|><|content_text|>Read the following passage carefully and then write a "
        "concise literary analysis of its tone, setting, and characters:\n\n",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode("\n\nProvide your analysis now.<|end_message|>", add_special_tokens=False)
    body_len = max(16, target_tokens - len(prefix) - len(suffix))
    return prefix + toks[:body_len] + suffix


def bench_stock_pr2086(ctx_lengths: list[int]) -> list[dict]:
    from mlx_lm.utils import load, maybe_set_recommended_wired_limit

    maybe_set_recommended_wired_limit()
    model, tok = load(str(MODEL_PATH))

    # Warmup
    w_cache = model.make_cache()
    _ = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32), cache=w_cache)
    mx.eval([c.state for row in w_cache for c in row])
    del w_cache
    mx.clear_cache()

    rows = []
    for ctx in ctx_lengths:
        prompt = build_prompt(tok, ctx)
        mx.reset_peak_memory()
        cache = model.make_cache()
        arr = mx.array([prompt], dtype=mx.int32)
        n = len(prompt)
        step = 2048

        t0 = time.perf_counter()
        pos = 0
        while n - pos > 1:
            c = min(step, n - pos - 1)
            _ = model(arr[:, pos : pos + c], cache=cache)
            mx.eval([x.state for row in cache for x in row])
            pos += c
        lg = model(arr[:, pos:], cache=cache)
        cur = mx.argmax(lg[:, -1, :], axis=-1)
        mx.eval(cur)
        prefill_s = time.perf_counter() - t0
        cur_id = int(cur.item())

        t1 = time.perf_counter()
        for _ in range(GEN_TOKENS - 1):
            lg = model(mx.array([[cur_id]], dtype=mx.int32), cache=cache)
            cur = mx.argmax(lg[:, -1, :], axis=-1)
            mx.eval(cur)
            cur_id = int(cur.item())
        dec_s = time.perf_counter() - t1
        peak_gb = mx.get_peak_memory() / 1e9

        row = {
            "mode": "Mainline PR#2086 (e2b5bbd, BF16 d=0)",
            "ctx": n,
            "prefill_tps": n / prefill_s,
            "prefill_s": prefill_s,
            "decode_tps": (GEN_TOKENS - 1) / dec_s,
            "decode_s": dec_s,
            "accept_rate": 0.0,
            "peak_gb": peak_gb,
        }
        rows.append(row)
        print(
            f"[STOCK e2b5bbd] ctx={n:>6,} | prefill={row['prefill_tps']:6.1f} tok/s ({prefill_s:5.1f}s) | "
            f"decode={row['decode_tps']:5.2f} tok/s | peak={peak_gb:.2f} GB",
            flush=True,
        )
        del cache
        gc.collect()
        mx.clear_cache()
    return rows


def bench_custom(ctx_lengths: list[int], fp16: bool, depth: int) -> list[dict]:
    from inkling_mlx import load_inkling, generate_inkling

    label = f"Custom ({'FP16' if fp16 else 'BF16'}, d={depth})"
    model, tok, drafter = load_inkling(
        MODEL_PATH,
        mtp_depth=depth,
        enable_fast_kernels=True,
        fuse_projections=True,
        fp16_activations=fp16,
    )
    # Warmup
    _ = generate_inkling(
        model,
        tok,
        build_prompt(tok, 64),
        max_tokens=16,
        mtp_depth=depth,
        drafter=drafter,
        stream=False,
        stop_on_eos=False,
    )

    rows = []
    for ctx in ctx_lengths:
        prompt = build_prompt(tok, ctx)
        res = generate_inkling(
            model,
            tok,
            prompt,
            max_tokens=GEN_TOKENS,
            mtp_depth=depth,
            drafter=drafter,
            stream=False,
            stop_on_eos=False,
            prefill_step_size=2048,
        )
        row = {
            "mode": label,
            "ctx": len(prompt),
            "prefill_tps": res.prompt_tps,
            "prefill_s": res.prompt_s,
            "decode_tps": res.gen_tps,
            "decode_s": res.gen_s,
            "accept_rate": res.accept_rate,
            "peak_gb": res.peak_memory_gb,
        }
        rows.append(row)
        acc_s = f" acc={res.accept_rate*100:.0f}%" if depth > 0 else ""
        print(
            f"[{label:18s}] ctx={len(prompt):>6,} | prefill={row['prefill_tps']:6.1f} tok/s ({row['prefill_s']:5.1f}s) | "
            f"decode={row['decode_tps']:5.2f} tok/s{acc_s} | peak={row['peak_gb']:.2f} GB",
            flush=True,
        )
        gc.collect()
        mx.clear_cache()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stock", "custom_bf16_d0", "custom_fp16_d0", "custom_fp16_d2"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if args.mode == "stock":
        rows = bench_stock_pr2086(CONTEXTS)
    elif args.mode == "custom_bf16_d0":
        rows = bench_custom(CONTEXTS, fp16=False, depth=0)
    elif args.mode == "custom_fp16_d0":
        rows = bench_custom(CONTEXTS, fp16=True, depth=0)
    else:
        rows = bench_custom(CONTEXTS, fp16=True, depth=2)

    args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
