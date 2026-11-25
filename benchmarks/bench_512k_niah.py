#!/usr/bin/env python3
"""512k-Context Needle-in-a-Haystack & Decode Speed Benchmark for Inkling-2-Mini (M2 Max 64 GB).

Corpus: Full text of G.K. Chesterton's *The Man Who Was Thursday* (~142k tokens)
        + opening ~370k tokens of Alexandre Dumas's *The Count of Monte Cristo*
        => ~512,000 non-repeating tokens of prose with 5 buried needles at 5%, 25%, 50%, 75%, 95%.

Uses pre-allocated BF16 full-attention KVCache buffers to avoid O(N^2) concatenation
copies and memory spikes during 512k prefill.
"""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import gc
import time
from pathlib import Path

import mlx.core as mx

from mlx_lm.models.cache import KVCache
from inkling_mlx import (
    DEFAULT_MODEL_PATH,
    TOK_END_SAMPLING,
    format_chat_prompt,
    load_inkling,
)

CORPUS_DIR = Path("niah_512k")
OUT_PATH = CORPUS_DIR / "niah_512k_result.txt"
TARGET_HAYSTACK_TOKENS = 512_000
PREFILL_STEP = 512

NEEDLES = [
    (
        0.05,
        "CODE-ALPHA",
        "VELVET-COMPASS-7419",
        "\n\n[ARCHIVE NOTE CODE-ALPHA: The secret harbor passkey for the S.S. Albatross is 'VELVET-COMPASS-7419'.]\n\n",
    ),
    (
        0.25,
        "CODE-BETA",
        "4,832",
        "\n\n[ARCHIVE NOTE CODE-BETA: The hidden vault in Lyon holds exactly 4,832 silver florins registered to Captain Maren Vance.]\n\n",
    ),
    (
        0.50,
        "CODE-GAMMA",
        "16:45",
        "\n\n[ARCHIVE NOTE CODE-GAMMA: The clockmaker's workshop on Rue de l'Horloge closes every Thursday at 16:45 sharp.]\n\n",
    ),
    (
        0.75,
        "CODE-DELTA",
        "Inspector Orlov-9",
        "\n\n[ARCHIVE NOTE CODE-DELTA: The coded telegram to Marseille must be signed with the alias 'Inspector Orlov-9'.]\n\n",
    ),
    (
        0.95,
        "CODE-EPSILON",
        "19 crates",
        "\n\n[ARCHIVE NOTE CODE-EPSILON: The botanical ledger lists 19 crates of saffron shipped from Genoa on October 14th.]\n\n",
    ),
]


def build_512k_prompt(tokenizer) -> list[int]:
    t1 = (CORPUS_DIR / "thursday.txt").read_text(errors="ignore")
    t2 = (CORPUS_DIR / "monte_cristo.txt").read_text(errors="ignore")

    toks1 = tokenizer.encode(t1, add_special_tokens=False)
    toks2 = tokenizer.encode(t2, add_special_tokens=False)
    combined = (toks1 + toks2)[:TARGET_HAYSTACK_TOKENS]

    # Insert needles from back to front so earlier indices don't shift
    needle_positions = []
    for frac, code, _, text in reversed(NEEDLES):
        ntoks = tokenizer.encode(text, add_special_tokens=False)
        pos = int(len(combined) * frac)
        combined[pos:pos] = ntoks
        needle_positions.append((code, pos, frac))
    needle_positions.reverse()
    print(f"[CORPUS] Haystack tokens (with 5 needles): {len(combined):,}")
    for code, pos, frac in needle_positions:
        print(f"  - {code:12s} @ token {pos:>7,} ({frac*100:4.1f}%)")

    prefix_txt = (
        "Below is a long literary archive containing two classic novels with five bracketed "
        "ARCHIVE NOTE entries (CODE-ALPHA, CODE-BETA, CODE-GAMMA, CODE-DELTA, CODE-EPSILON) "
        "hidden at different points in the text. Read the entire archive carefully.\n\n"
        "=== BEGIN ARCHIVE ===\n"
    )
    suffix_txt = (
        "\n=== END ARCHIVE ===\n\n"
        "Answer ALL of the following accurately and concisely:\n"
        "1. [CODE-ALPHA] What is the secret harbor passkey for the S.S. Albatross?\n"
        "2. [CODE-BETA] How many silver florins are in the hidden vault in Lyon, and who is it registered to?\n"
        "3. [CODE-GAMMA] What day and exact time does the clockmaker's workshop on Rue de l'Horloge close?\n"
        "4. [CODE-DELTA] What alias must be used to sign the coded telegram to Marseille?\n"
        "5. [CODE-EPSILON] How many crates of saffron were shipped from Genoa, and on what date?\n"
        "6. Finally, write a 100-word summary contrasting the opening atmosphere of the first novel with the second novel in the archive."
    )

    # Build chat-templated wrapper tokens around the pre-tokenized haystack
    marker = "<<<HAYSTACK_PLACEHOLDER>>>"
    shell_tokens = format_chat_prompt(
        tokenizer,
        [{"role": "user", "content": prefix_txt + marker + suffix_txt}],
        reasoning_effort="minimal",
    )
    marker_toks = tokenizer.encode(marker, add_special_tokens=False)
    # Find marker_toks subsequence in shell_tokens
    idx = -1
    for i in range(len(shell_tokens) - len(marker_toks) + 1):
        if shell_tokens[i : i + len(marker_toks)] == marker_toks:
            idx = i
            break
    assert idx >= 0, "Could not locate placeholder marker in formatted prompt"
    full_prompt = shell_tokens[:idx] + combined + shell_tokens[idx + len(marker_toks) :]
    print(f"[PROMPT] Total formatted prompt tokens: {len(full_prompt):,}")
    return full_prompt


def preallocate_full_kv(cache: list, max_len: int, n_kv_heads: int = 8, head_dim: int = 128) -> int:
    """Pre-allocate full-attention KVCache tensors so 512k prefill runs in-place without concat copies."""
    n_full = 0
    for entry in cache:
        kv = entry[0]
        if type(kv) is KVCache:
            kv.keys = mx.zeros((1, n_kv_heads, max_len, head_dim), dtype=mx.bfloat16)
            kv.values = mx.zeros((1, n_kv_heads, max_len, head_dim), dtype=mx.bfloat16)
            kv.offset = 0
            mx.eval(kv.keys, kv.values)
            n_full += 1
    return n_full


def ingest_drafter_taps(drafter, taps: list[mx.array]) -> None:
    ctx = mx.concatenate(taps, axis=-1)
    n_total = int(ctx.shape[1])
    if n_total == 0:
        return
    skip = max(0, n_total - drafter.ring_slots)
    if skip:
        ctx = ctx[:, skip:]
    n = int(ctx.shape[1])
    base = drafter.fed + skip
    drafter.fed = base + n
    slots = mx.arange(base, base + n, dtype=mx.int32) % drafter.ring_slots
    drafter.pos[slots] = mx.arange(base, base + n, dtype=mx.int32)
    for li, layer in enumerate(drafter.model.layers):
        attn = layer.self_attn
        ck, cv = attn.project_context_kv(ctx)
        ck = attn.k_norm_ctx(ck.reshape(1, n, attn.n_kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
        cv = cv.reshape(1, n, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        ck = drafter.model.rope(ck, offset=base)
        drafter.ring_keys[li][:, :, slots, :] = ck
        drafter.ring_values[li][:, :, slots, :] = cv
    mx.eval(*drafter.ring_keys, *drafter.ring_values, drafter.pos)


def main() -> None:
    print("[LOAD] Loading Inkling-2-Mini-MLX-W4W8-Native (MTP depth=2)...")
    t0 = time.perf_counter()
    model, tokenizer, drafter = load_inkling(DEFAULT_MODEL_PATH, mtp_depth=2)
    print(f"[LOAD] Ready in {time.perf_counter() - t0:.1f}s | active={mx.get_active_memory()/1e9:.2f} GB")

    prompt_tokens = build_512k_prompt(tokenizer)
    max_len = len(prompt_tokens) + 1024

    mx.reset_peak_memory()
    KVCache.step = 8192
    cache = model.make_speculative_cache()
    print(
        f"[CACHE] Dynamic KVCache(step=8192) + SpeculativeRotatingKVCache(1040) "
        f"| active={mx.get_active_memory()/1e9:.2f} GB"
    )

    drafter.reset()
    capture = {lid: None for lid in drafter.target_layer_ids}
    prompt_arr = mx.array([prompt_tokens], dtype=mx.int32)
    n_prompt = len(prompt_tokens)

    # 1. Chunked 512k prefill (only capture drafter taps for the final 2048 tokens since ring_slots=1023!)
    print(f"[PREFILL] Starting {n_prompt:,}-token prefill (chunk={PREFILL_STEP})...")
    t_pre0 = time.perf_counter()
    t_window = t_pre0
    pos = 0
    report_every = 32768
    next_report = report_every

    while n_prompt - pos > 1:
        # Adaptive chunk size keeps every Metal command buffer < ~1.6s to avoid macOS GPU watchdog
        if pos < 98_304:
            step_sz = 512
        elif pos < 229_376:
            step_sz = 256
        elif pos < 393_216:
            step_sz = 128
        else:
            step_sz = 64
        chunk_len = min(step_sz, n_prompt - pos - 1)
        need_taps = (n_prompt - pos) <= (drafter.ring_slots + step_sz + 64)
        cur_cap = capture if need_taps else None
        _ = model(prompt_arr[:, pos : pos + chunk_len], cache=cache, capture=cur_cap)
        mx.eval([c.state for row in cache for c in row])
        if need_taps:
            taps = [capture[lid] for lid in drafter.target_layer_ids]
            mx.eval(*taps)
            ingest_drafter_taps(drafter, taps)
        else:
            drafter.fed += chunk_len
        pos += chunk_len
        if pos % 16384 == 0:
            mx.clear_cache()

        if pos >= next_report or pos == n_prompt - 1:
            now = time.perf_counter()
            dt_win = max(now - t_window, 1e-6)
            dt_tot = max(now - t_pre0, 1e-6)
            win_toks = min(report_every, pos)
            print(
                f"  [PREFILL] {pos:>7,}/{n_prompt:,} ({pos*100/n_prompt:5.1f}%) | "
                f"recent={win_toks/dt_win:6.1f} tok/s | avg={pos/dt_tot:6.1f} tok/s | "
                f"elapsed={dt_tot:6.1f}s | active={mx.get_active_memory()/1e9:.2f} GB | "
                f"peak={mx.get_peak_memory()/1e9:.2f} GB",
                flush=True,
            )
            t_window = now
            next_report += report_every

    # Final prompt token
    logits = model(prompt_arr[:, pos:], cache=cache, capture=capture)
    first_tok_arr = mx.argmax(logits[:, -1, :], axis=-1)
    pending_taps = [capture[lid] for lid in drafter.target_layer_ids]
    mx.eval(first_tok_arr, *pending_taps)
    prefill_s = time.perf_counter() - t_pre0
    print(
        f"[PREFILL DONE] {n_prompt:,} tokens in {prefill_s:.1f}s ({n_prompt/prefill_s:.1f} tok/s) | "
        f"active={mx.get_active_memory()/1e9:.2f} GB | peak={mx.get_peak_memory()/1e9:.2f} GB",
        flush=True,
    )

    # 2. Decode answer at MTP depth = 2 (max 512 tokens)
    print("\n[DECODE d=2] Generating NIAH answers + summary at ~512k context...", flush=True)
    max_gen = 512
    depth = 2
    cur_tok = int(first_tok_arr.item())
    gen_tokens = [cur_tok]
    drafted = 0
    accepted_total = 0
    steps = 0
    t_dec0 = time.perf_counter()

    while len(gen_tokens) < max_gen:
        rem = max_gen - len(gen_tokens)
        cur_depth = min(depth, rem)
        draft_toks = drafter.draft(pending_taps, cur_tok, cur_depth)
        dl = draft_toks.tolist()

        verify_in = mx.concatenate([mx.array([cur_tok], dtype=mx.int32), draft_toks])[None, :]
        for e in cache:
            e[1].record_verify_windows = True
        v_logits = model(verify_in, cache=cache, capture=capture)
        for e in cache:
            e[1].record_verify_windows = False
        greedy = mx.argmax(v_logits[0], axis=-1)
        mx.eval(greedy, *[capture[lid] for lid in drafter.target_layer_ids])
        gl = greedy.tolist()

        steps += 1
        drafted += cur_depth
        acc = 0
        hit_eos = False
        step_toks = []
        for i in range(cur_depth):
            pred = int(gl[i])
            if pred == TOK_END_SAMPLING:
                hit_eos = True
                break
            step_toks.append(pred)
            if pred == int(dl[i]):
                acc += 1
            else:
                break
        if not hit_eos and acc == cur_depth and (len(gen_tokens) + len(step_toks)) < max_gen:
            bonus = int(gl[cur_depth])
            if bonus == TOK_END_SAMPLING:
                hit_eos = True
            else:
                step_toks.append(bonus)

        accepted_total += acc
        gen_tokens.extend(step_toks)
        if hit_eos or len(gen_tokens) >= max_gen:
            break
        if acc < cur_depth:
            model.rollback_speculative_cache(cache, acc, cur_depth)
        keep = 1 + acc
        pending_taps = [capture[lid][:, :keep, :] for lid in drafter.target_layer_ids]
        cur_tok = step_toks[-1]

    dec_s = time.perf_counter() - t_dec0
    dec_tps = max(len(gen_tokens) - 1, 1) / max(dec_s, 1e-6)
    acc_rate = accepted_total / max(drafted, 1)
    mean_acc = accepted_total / max(steps, 1)
    peak_gb = mx.get_peak_memory() / 1e9
    active_gb = mx.get_active_memory() / 1e9

    output_text = tokenizer.decode(gen_tokens)
    OUT_PATH.write_text(output_text)

    print("\n================ 512k NIAH MODEL OUTPUT ================")
    print(output_text)
    print("========================================================")

    # Check recall of all 5 needles
    hits = 0
    print("\n[RECALL CHECK]")
    for frac, code, expected, _ in NEEDLES:
        found = expected.lower() in output_text.lower()
        hits += int(found)
        print(f"  - {code:12s} ({frac*100:4.1f}% depth, expected {expected!r}): {'PASS' if found else 'FAIL'}")

    # 3. Measure d=0 decode speed at 512k context for 32 tokens on the same KV cache
    mx.clear_cache()
    t_d0 = time.perf_counter()
    d0_tok = cur_tok
    n_d0 = 32
    for _ in range(n_d0):
        lg0 = model(mx.array([[d0_tok]], dtype=mx.int32), cache=cache)
        nxt0 = mx.argmax(lg0[:, -1, :], axis=-1)
        mx.eval(nxt0)
        d0_tok = int(nxt0.item())
    d0_s = time.perf_counter() - t_d0
    d0_tps = n_d0 / max(d0_s, 1e-6)

    print(
        f"\n[SUMMARY 512k] Recall: {hits}/{len(NEEDLES)} | "
        f"Prefill: {n_prompt:,} tok in {prefill_s:.1f}s ({n_prompt/prefill_s:.1f} tok/s) | "
        f"Decode (d=2): {len(gen_tokens)} tok in {dec_s:.1f}s ({dec_tps:.2f} tok/s, "
        f"acc={mean_acc:.2f}/2 = {acc_rate*100:.1f}%) | "
        f"Decode (d=0): {d0_tps:.2f} tok/s | "
        f"Active: {active_gb:.2f} GB | Peak: {peak_gb:.2f} GB"
    )
    gc.collect()
    mx.clear_cache()


if __name__ == "__main__":
    main()
