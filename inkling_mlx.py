#!/usr/bin/env python3
"""Inkling-MLX: High-performance Apple Silicon wrapper around mlx-lm for Inkling-2-Mini.

Features:
- W4W8-Native-A8 mixed-precision checkpoint support (4-bit affine L0..19 experts,
  8-bit affine L20..23 experts + attention/embed/lm_head, FP32 AttnRes/router queries).
- Custom Metal kernels enabled by default:
  * Fused K=4 causal depthwise short convolution + residual + state update (`short_conv`).
  * Two-pass GQA-grouped Split-K relative attention for MTP verification (`relative_attention_splitk`).
- Linear projection fusion (`model.fuse_projections()`) across Q/R/Gate/Sink and MessageGate.
- DSpark Multi-Token Prediction (MTP) speculative decoding (`mtp.safetensors`, default depth=2)
  with 8-bit affine draft layers, fused draft QKV/KV-ctx/gate-up projections, and 4-bit folded Markov head.
- Real-time streaming CLI, interactive multi-turn chat REPL (`--chat`), and Python API.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_lm.generate import BatchResponse, batch_generate
from mlx_lm.models.inkling2 import (
    AdaptiveMTPController,
    Inkling2DSparkDrafter,
    layers as inkling2_layers,
)
from mlx_lm.utils import (
    compute_bits_per_weight,
    load,
    maybe_set_recommended_wired_limit,
)

from inkling_cache import (
    CacheSnapshot,
    collect_context_text,
    load_inkling_cache,
    prefill_into_cache,
    restore_cache_tail,
    save_inkling_cache,
    snapshot_cache_tail,
)

WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "ofnordo/Inkling-2-Mini-MLX-W4W8"
DEFAULT_MODEL_PATH = (
    WORKSPACE_DIR / "Inkling-2-Mini-MLX-W4W8"
    if (WORKSPACE_DIR / "Inkling-2-Mini-MLX-W4W8").exists()
    else Path(os.environ.get("INKLING_MODEL", DEFAULT_MODEL_ID))
)
DEFAULT_RAW_MODEL_PATH = WORKSPACE_DIR / "Inkling-2-Mini"

# Special token IDs in Inkling-2-Mini tokenizer
TOK_MESSAGE_USER = 200000
TOK_MESSAGE_MODEL = 200001
TOK_MESSAGE_SYSTEM = 200002
TOK_MESSAGE_TOOL = 200003
TOK_CONTENT_TEXT = 200004
TOK_CONTENT_IMAGE = 200005
TOK_END_SAMPLING = 200006
TOK_CONTENT_THINKING = 200008
TOK_END_MESSAGE = 200010

CONTROL_TOKEN_IDS = {
    TOK_MESSAGE_USER,
    TOK_MESSAGE_MODEL,
    TOK_MESSAGE_SYSTEM,
    TOK_MESSAGE_TOOL,
    TOK_CONTENT_TEXT,
    TOK_CONTENT_IMAGE,
    TOK_END_SAMPLING,
    TOK_CONTENT_THINKING,
    TOK_END_MESSAGE,
}


@dataclass
class GenerationChunk:
    """Single streamed token event emitted during generation."""

    token_id: int
    text: str
    phase: str  # "thinking" | "response" | "control"
    tokens_generated: int
    think_tokens: int
    response_tokens: int
    step_accepted: int
    mtp_depth: int
    elapsed_s: float
    step_depth: int = -1

    @property
    def gen_tps(self) -> float:
        return self.tokens_generated / max(self.elapsed_s, 1e-9)


@dataclass
class GenerationResult:
    """Aggregated result and telemetry from an Inkling-2-Mini generation run."""

    text: str
    thinking: str
    raw_text: str
    tokens: list[int] = field(default_factory=list)
    prompt_tokens: int = 0
    gen_tokens: int = 0
    think_tokens: int = 0
    response_tokens: int = 0
    prompt_s: float = 0.0
    prompt_tps: float = 0.0
    gen_s: float = 0.0
    gen_tps: float = 0.0
    total_wall_s: float = 0.0
    mtp_depth: int = 2
    target_steps: int = 0
    d0_steps: int = 0
    draft_proposed: int = 0
    draft_accepted: int = 0
    accept_rate: float = 0.0
    mean_accepted: float = 0.0
    tokens_per_step: float = 1.0
    peak_memory_gb: float = 0.0


def configure_kernels(
    *,
    short_conv_kernel: bool = True,
    splitk_attn_kernel: bool = True,
    wired_limit: bool = True,
) -> None:
    """Enable or disable custom Metal kernels and unified memory wired limit."""
    if wired_limit:
        maybe_set_recommended_wired_limit()
    inkling2_layers._USE_SHORT_CONV_KERNEL = bool(short_conv_kernel)
    inkling2_layers._USE_SPLITK_ATTN_KERNEL = bool(splitk_attn_kernel)


def _default_fp16_activations() -> bool:
    """Enable FP16 activations by default on M1/M2 where BF16 is emulated."""
    try:
        name = str(mx.device_info().get("device_name", ""))
    except Exception:
        return False
    return "M1" in name or "M2" in name


def recast_bf16_to_fp16(model: nn.Module, *, chunk_size: int = 128) -> int:
    """Recast bfloat16 parameter leaves (scales, biases, norms, embeddings) to float16.

    Quantized U32 payloads and FP32 AttnRes/router parameters are left untouched.
    """
    leaves = tree_flatten(model.parameters())
    bf16_indices = [
        i for i, (_, v) in enumerate(leaves) if isinstance(v, mx.array) and v.dtype == mx.bfloat16
    ]
    if not bf16_indices:
        return 0
    updated: list[tuple[str, mx.array]] = list(leaves)
    for start in range(0, len(bf16_indices), chunk_size):
        batch_idx = bf16_indices[start : start + chunk_size]
        casted = []
        for idx in batch_idx:
            k, v = updated[idx]
            cv = v.astype(mx.float16)
            updated[idx] = (k, cv)
            casted.append(cv)
        mx.eval(*casted)
    model.update(tree_unflatten(updated))
    mx.clear_cache()
    return len(bf16_indices)


def load_inkling(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    *,
    mtp_depth: int = 2,
    draft_bits: int = 8,
    enable_fast_kernels: bool = True,
    fuse_projections: bool = True,
    opt_drafter: bool = True,
    fp16_activations: bool | None = None,
) -> tuple[nn.Module, Any, Inkling2DSparkDrafter | None]:
    """Load an Inkling-2-Mini MLX model, apply optional FP16 recast + projection fusion, and bind DSpark MTP."""
    configure_kernels(
        short_conv_kernel=enable_fast_kernels,
        splitk_attn_kernel=enable_fast_kernels,
        wired_limit=True,
    )
    raw_str = str(model_path)
    resolved_path = Path(raw_str)
    if not resolved_path.exists():
        from huggingface_hub import snapshot_download
        resolved_path = Path(snapshot_download(repo_id=raw_str))

    use_fp16 = _default_fp16_activations() if fp16_activations is None else bool(fp16_activations)
    model, tokenizer = load(str(resolved_path))
    if use_fp16:
        recast_bf16_to_fp16(model)
    if fuse_projections and hasattr(model, "fuse_projections"):
        model.fuse_projections()

    drafter: Inkling2DSparkDrafter | None = None
    mtp_file = resolved_path / "mtp.safetensors"
    if mtp_depth > 0:
        if not mtp_file.exists() and (DEFAULT_RAW_MODEL_PATH / "mtp.safetensors").exists():
            drafter_root = DEFAULT_RAW_MODEL_PATH
        else:
            drafter_root = resolved_path
        drafter = Inkling2DSparkDrafter(
            drafter_root,
            model,
            draft_bits=draft_bits,
            opt_drafter=opt_drafter,
        )
    return model, tokenizer, drafter


def _set_verify_recording(cache: list[Any], enabled: bool) -> None:
    for entry in cache:
        entry[1].record_verify_windows = enabled


def format_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    reasoning_effort: str | float = "medium",
) -> list[int]:
    """Format chat messages with Inkling-2-Mini's Jinja chat template and reasoning effort."""
    effort_arg: str | float = reasoning_effort
    if isinstance(reasoning_effort, str):
        try:
            effort_arg = float(reasoning_effort)
        except ValueError:
            effort_arg = reasoning_effort
    return list(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            reasoning_effort=effort_arg,
        )
    )


def stream_generate_inkling(
    model: nn.Module,
    tokenizer: Any,
    prompt_tokens: list[int],
    *,
    max_tokens: int = 8192,
    mtp_depth: int = 2,
    drafter: Inkling2DSparkDrafter | None = None,
    cache: list[Any] | None = None,
    adaptive_mtp: bool = True,
    stop_on_eos: bool = True,
    prefill_step_size: int = 2048,
) -> Iterator[GenerationChunk]:
    """Stream tokens using greedy decoding at mtp_depth=0 or DSpark MTP speculative verification (1..7)."""
    if mtp_depth > 0 and drafter is None:
        raise ValueError("drafter must be provided when mtp_depth > 0")

    eos_ids = set(getattr(tokenizer, "eos_token_ids", ()) or ()) | {
        TOK_END_SAMPLING,
    }
    resuming_cache = cache is not None
    if cache is None:
        cache = (
            model.make_speculative_cache()
            if mtp_depth > 0
            else model.make_cache()
        )
    capture = (
        {lid: None for lid in drafter.target_layer_ids}
        if (mtp_depth > 0 and drafter is not None)
        else None
    )
    if drafter is not None and not resuming_cache:
        drafter.reset()

    prev_fed = int(drafter.fed) if (mtp_depth > 0 and drafter is not None) else 0
    prompt_arr = mx.array([prompt_tokens], dtype=mx.int32)
    n_prompt = len(prompt_tokens)
    drafter_keep_start = (
        max(0, (n_prompt - 1) - drafter.ring_slots)
        if (mtp_depth > 0 and drafter is not None)
        else n_prompt
    )
    pos = 0
    while n_prompt - pos > 1:
        chunk_len = min(prefill_step_size, n_prompt - pos - 1)
        chunk_end = pos + chunk_len
        need_drafter_taps = (
            mtp_depth > 0
            and drafter is not None
            and capture is not None
            and chunk_end > drafter_keep_start
        )
        _ = model(
            prompt_arr[:, pos:chunk_end],
            cache=cache,
            capture=capture if need_drafter_taps else None,
        )
        if need_drafter_taps and drafter is not None and capture is not None:
            taps = [capture[lid] for lid in drafter.target_layer_ids]
            ctx = mx.concatenate(taps, axis=-1)
            skip = max(0, drafter_keep_start - pos)
            if skip:
                ctx = ctx[:, skip:]
            n_use = int(ctx.shape[1])
            base = prev_fed + pos + skip
            drafter.fed = base + n_use
            write_slots = (
                mx.arange(base, base + n_use, dtype=mx.int32) % drafter.ring_slots
            )
            drafter.pos[write_slots] = mx.arange(base, base + n_use, dtype=mx.int32)
            for l_idx, layer in enumerate(drafter.model.layers):
                attn = layer.self_attn
                ck, cv = attn.project_context_kv(ctx)
                ck = attn.k_norm_ctx(
                    ck.reshape(1, n_use, attn.n_kv_heads, attn.head_dim)
                ).transpose(0, 2, 1, 3)
                cv = cv.reshape(1, n_use, attn.n_kv_heads, attn.head_dim).transpose(
                    0, 2, 1, 3
                )
                ck = drafter.model.rope(ck, offset=base)
                drafter.ring_keys[l_idx][:, :, write_slots, :] = ck
                drafter.ring_values[l_idx][:, :, write_slots, :] = cv
            mx.eval(
                [c.state for row in cache for c in row],
                *drafter.ring_keys,
                *drafter.ring_values,
                drafter.pos,
            )
        else:
            mx.eval([c.state for row in cache for c in row])
            if mtp_depth > 0 and drafter is not None:
                drafter.fed = prev_fed + chunk_end
        pos = chunk_end

    logits = model(prompt_arr[:, pos:], cache=cache, capture=capture)
    first_tok_arr = mx.argmax(logits[:, -1, :], axis=-1)
    if capture is not None and drafter is not None:
        mx.eval(
            first_tok_arr,
            *[capture[lid] for lid in drafter.target_layer_ids],
        )
    else:
        mx.eval(first_tok_arr)

    detok = tokenizer.detokenizer
    detok.reset()

    phase = "response"
    tokens_generated = 0
    think_tokens = 0
    response_tokens = 0
    t_gen0 = time.perf_counter()

    def _make_chunk(tid: int, step_acc: int, step_d: int = -1) -> GenerationChunk:
        nonlocal phase, tokens_generated, think_tokens, response_tokens
        tokens_generated += 1
        elapsed = time.perf_counter() - t_gen0
        if tid == TOK_CONTENT_THINKING:
            phase = "thinking"
            return GenerationChunk(
                token_id=tid,
                text="",
                phase="control",
                tokens_generated=tokens_generated,
                think_tokens=think_tokens,
                response_tokens=response_tokens,
                step_accepted=step_acc,
                mtp_depth=mtp_depth,
                elapsed_s=elapsed,
                step_depth=step_d,
            )
        if tid == TOK_CONTENT_TEXT:
            phase = "response"
            detok.reset()
            return GenerationChunk(
                token_id=tid,
                text="",
                phase="control",
                tokens_generated=tokens_generated,
                think_tokens=think_tokens,
                response_tokens=response_tokens,
                step_accepted=step_acc,
                mtp_depth=mtp_depth,
                elapsed_s=elapsed,
                step_depth=step_d,
            )
        if tid in CONTROL_TOKEN_IDS:
            return GenerationChunk(
                token_id=tid,
                text="",
                phase="control",
                tokens_generated=tokens_generated,
                think_tokens=think_tokens,
                response_tokens=response_tokens,
                step_accepted=step_acc,
                mtp_depth=mtp_depth,
                elapsed_s=elapsed,
                step_depth=step_d,
            )

        detok.add_token(tid)
        seg = detok.last_segment
        if phase == "thinking":
            think_tokens += 1
        else:
            response_tokens += 1
        return GenerationChunk(
            token_id=tid,
            text=seg,
            phase=phase,
            tokens_generated=tokens_generated,
            think_tokens=think_tokens,
            response_tokens=response_tokens,
            step_accepted=step_acc,
            mtp_depth=mtp_depth,
            elapsed_s=elapsed,
            step_depth=step_d,
        )

    cur_tok = int(first_tok_arr.item())
    if stop_on_eos and cur_tok in eos_ids:
        return
    yield _make_chunk(cur_tok, 0, -1)

    if mtp_depth == 0:
        while tokens_generated < max_tokens:
            logits = model(mx.array([[cur_tok]], dtype=mx.int32), cache=cache)
            next_tok_arr = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(next_tok_arr)
            cur_tok = int(next_tok_arr.item())
            if stop_on_eos and cur_tok in eos_ids:
                break
            yield _make_chunk(cur_tok, 0, 0)
        return

    assert drafter is not None and capture is not None
    pending_tap_chunks: list[list[mx.array]] = [
        [capture[lid] for lid in drafter.target_layer_ids]
    ]
    controller = (
        AdaptiveMTPController(max_depth=mtp_depth) if adaptive_mtp else None
    )

    while tokens_generated < max_tokens:
        rem = max_tokens - tokens_generated
        cur_depth = (
            controller.next_depth(rem)
            if controller is not None
            else min(mtp_depth, rem)
        )

        if cur_depth == 0:
            logits = model(
                mx.array([[cur_tok]], dtype=mx.int32), cache=cache, capture=capture
            )
            next_tok_arr = mx.argmax(logits[:, -1, :], axis=-1)
            step_taps = [capture[lid] for lid in drafter.target_layer_ids]
            mx.eval(next_tok_arr, *step_taps)
            pending_tap_chunks.append(step_taps)
            cur_tok = int(next_tok_arr.item())
            if stop_on_eos and cur_tok in eos_ids:
                break
            yield _make_chunk(cur_tok, 0, 0)
            continue

        if len(pending_tap_chunks) == 1:
            pending_taps = pending_tap_chunks[0]
        else:
            pending_taps = [
                mx.concatenate([chunk[i] for chunk in pending_tap_chunks], axis=1)
                for i in range(len(drafter.target_layer_ids))
            ]
        draft_toks = drafter.draft(pending_taps, cur_tok, cur_depth)
        draft_list = draft_toks.tolist()

        verify_in = mx.concatenate(
            [mx.array([cur_tok], dtype=mx.int32), draft_toks], axis=0
        )[None, :]
        _set_verify_recording(cache, True)
        v_logits = model(verify_in, cache=cache, capture=capture)
        _set_verify_recording(cache, False)
        greedy = mx.argmax(v_logits[0], axis=-1)
        mx.eval(greedy, *[capture[lid] for lid in drafter.target_layer_ids])
        greedy_list = greedy.tolist()

        accepted = 0
        hit_eos = False
        new_step_tokens: list[int] = []
        for i in range(cur_depth):
            pred = int(greedy_list[i])
            if stop_on_eos and pred in eos_ids:
                hit_eos = True
                break
            new_step_tokens.append(pred)
            if pred == int(draft_list[i]):
                accepted += 1
            else:
                break

        if not hit_eos and accepted == cur_depth and (tokens_generated + len(new_step_tokens)) < max_tokens:
            bonus = int(greedy_list[cur_depth])
            if stop_on_eos and bonus in eos_ids:
                hit_eos = True
            else:
                new_step_tokens.append(bonus)

        if controller is not None:
            controller.observe(cur_depth, accepted)

        for sub_idx, tok_id in enumerate(new_step_tokens):
            yield _make_chunk(
                tok_id,
                accepted if sub_idx == 0 else -1,
                cur_depth if sub_idx == 0 else -1,
            )

        if hit_eos or tokens_generated >= max_tokens:
            break

        if accepted < cur_depth:
            model.rollback_speculative_cache(cache, accepted, cur_depth)

        keep = 1 + accepted
        pending_tap_chunks = [
            [capture[lid][:, :keep, :] for lid in drafter.target_layer_ids]
        ]
        cur_tok = new_step_tokens[-1]


def generate_inkling(
    model: nn.Module,
    tokenizer: Any,
    prompt: str | list[dict[str, str]] | list[int],
    *,
    max_tokens: int = 8192,
    mtp_depth: int = 2,
    drafter: Inkling2DSparkDrafter | None = None,
    cache: list[Any] | None = None,
    adaptive_mtp: bool = True,
    reasoning_effort: str | float = "medium",
    system_prompt: str | None = None,
    stream: bool = True,
    show_thinking: bool = True,
    stop_on_eos: bool = True,
    prefill_step_size: int = 2048,
) -> GenerationResult:
    """High-level generation helper supporting streaming display, offline KV caches, and telemetry."""
    if isinstance(prompt, str):
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        prompt_tokens = format_chat_prompt(
            tokenizer, messages, reasoning_effort=reasoning_effort
        )
    elif isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        prompt_tokens = format_chat_prompt(
            tokenizer, prompt, reasoning_effort=reasoning_effort
        )
    else:
        prompt_tokens = list(prompt)  # type: ignore[arg-type]

    mx.reset_peak_memory()
    t_wall0 = time.perf_counter()

    gen_iter = stream_generate_inkling(
        model,
        tokenizer,
        prompt_tokens,
        max_tokens=max_tokens,
        mtp_depth=mtp_depth,
        drafter=drafter,
        cache=cache,
        adaptive_mtp=adaptive_mtp,
        stop_on_eos=stop_on_eos,
        prefill_step_size=prefill_step_size,
    )

    tokens: list[int] = []
    thinking_parts: list[str] = []
    response_parts: list[str] = []
    prompt_s = 0.0
    t_first_tok = 0.0
    last_phase: str | None = None
    target_steps = 0
    d0_steps = 0
    draft_proposed = 0
    draft_accepted = 0
    think_tokens = 0
    response_tokens = 0

    for idx, chunk in enumerate(gen_iter):
        if idx == 0:
            t_first_tok = time.perf_counter()
            prompt_s = max(t_first_tok - t_wall0, 1e-6)
        elif chunk.step_depth >= 0:
            target_steps += 1
            if chunk.step_depth == 0:
                d0_steps += 1
            else:
                draft_proposed += chunk.step_depth
                draft_accepted += chunk.step_accepted

        tokens.append(chunk.token_id)
        think_tokens = chunk.think_tokens
        response_tokens = chunk.response_tokens

        if chunk.phase == "thinking":
            thinking_parts.append(chunk.text)
            if stream:
                if show_thinking:
                    if last_phase != "thinking":
                        sys.stdout.write("\033[2m[Thinking]\n")
                        last_phase = "thinking"
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
                elif think_tokens % 16 == 0:
                    sys.stderr.write(
                        f"\r\033[2m[Thinking... {think_tokens} tokens ({chunk.gen_tps:.1f} tok/s)]\033[0m"
                    )
                    sys.stderr.flush()
        elif chunk.phase == "response":
            response_parts.append(chunk.text)
            if stream:
                if last_phase == "thinking":
                    if show_thinking:
                        sys.stdout.write("\033[0m\n\n[Response]\n")
                    else:
                        sys.stderr.write("\r\033[K")
                        sys.stderr.flush()
                elif last_phase is None and not show_thinking and think_tokens > 0:
                    sys.stderr.write("\r\033[K")
                    sys.stderr.flush()
                last_phase = "response"
                sys.stdout.write(chunk.text)
                sys.stdout.flush()

    if stream:
        if last_phase == "thinking" and show_thinking:
            sys.stdout.write("\033[0m")
        sys.stdout.write("\n")
        sys.stdout.flush()

    t_end = time.perf_counter()
    total_wall_s = t_end - t_wall0
    if t_first_tok == 0.0:
        prompt_s = total_wall_s
        gen_s = 0.0
    else:
        gen_s = max(t_end - t_first_tok, 1e-6)

    gen_tokens = len(tokens)
    verify_steps = max(target_steps - d0_steps, 0)
    accept_rate = draft_accepted / draft_proposed if draft_proposed > 0 else 0.0
    mean_accepted = (
        draft_accepted / verify_steps if (mtp_depth > 0 and verify_steps > 0) else 0.0
    )
    tokens_per_step = (
        max(gen_tokens - 1, 0) / target_steps if target_steps > 0 else 1.0
    )

    peak_gb = mx.get_peak_memory() / 1e9
    raw_text = tokenizer.decode(tokens)
    thinking_str = "".join(thinking_parts).strip()
    response_str = "".join(response_parts).strip()

    return GenerationResult(
        text=response_str if response_str else raw_text,
        thinking=thinking_str,
        raw_text=raw_text,
        tokens=tokens,
        prompt_tokens=len(prompt_tokens),
        gen_tokens=gen_tokens,
        think_tokens=think_tokens,
        response_tokens=response_tokens,
        prompt_s=prompt_s,
        prompt_tps=len(prompt_tokens) / max(prompt_s, 1e-6),
        gen_s=gen_s,
        gen_tps=max(gen_tokens - 1, 1) / max(gen_s, 1e-6) if gen_tokens > 0 else 0.0,
        total_wall_s=total_wall_s,
        mtp_depth=mtp_depth,
        target_steps=target_steps,
        d0_steps=d0_steps,
        draft_proposed=draft_proposed,
        draft_accepted=draft_accepted,
        accept_rate=accept_rate,
        mean_accepted=mean_accepted,
        tokens_per_step=tokens_per_step,
        peak_memory_gb=peak_gb,
    )


def batch_generate_inkling(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str] | list[list[dict[str, str]]] | list[list[int]],
    *,
    max_tokens: int | list[int] = 2048,
    reasoning_effort: str | float = "medium",
    system_prompt: str | None = None,
    completion_batch_size: int = 32,
    prefill_batch_size: int = 8,
    prefill_step_size: int = 2048,
    verbose: bool = False,
) -> BatchResponse:
    """Run continuous batched generation across multiple prompts using our fused kernels + Split-K batched decode."""
    token_prompts: list[list[int]] = []
    for p in prompts:
        if isinstance(p, str):
            msgs: list[dict[str, str]] = []
            if system_prompt:
                msgs.append({"role": "system", "content": system_prompt})
            msgs.append({"role": "user", "content": p})
            token_prompts.append(
                format_chat_prompt(tokenizer, msgs, reasoning_effort=reasoning_effort)
            )
        elif isinstance(p, list) and p and isinstance(p[0], dict):
            token_prompts.append(
                format_chat_prompt(tokenizer, p, reasoning_effort=reasoning_effort)
            )
        else:
            token_prompts.append(list(p))  # type: ignore[arg-type]

    return batch_generate(
        model,
        tokenizer,
        token_prompts,
        max_tokens=max_tokens,
        verbose=verbose,
        completion_batch_size=completion_batch_size,
        prefill_batch_size=prefill_batch_size,
        prefill_step_size=prefill_step_size,
    )


def print_stats(res: GenerationResult) -> None:
    """Print concise generation telemetry to stderr."""
    mtp_str = f"MTP depth={res.mtp_depth}"
    if res.mtp_depth > 0 and res.target_steps > 0:
        d0_note = f", d0_backoff={res.d0_steps}/{res.target_steps}" if res.d0_steps > 0 else ""
        mtp_str += (
            f" ({res.accept_rate*100:.1f}% acc, "
            f"{res.tokens_per_step:.2f} tok/step{d0_note})"
        )
    sys.stderr.write(
        f"\033[2m──────────────────────────────────────────────────────────────────────\n"
        f"Prompt:     {res.prompt_tokens:,} tok in {res.prompt_s:.2f}s ({res.prompt_tps:.1f} tok/s)\n"
        f"Generation: {res.gen_tokens:,} tok (think={res.think_tokens:,}, answer={res.response_tokens:,}) "
        f"in {res.gen_s:.2f}s ({res.gen_tps:.2f} tok/s, wall={res.total_wall_s:.2f}s)\n"
        f"Config:     {mtp_str} | Peak memory={res.peak_memory_gb:.2f} GB\n"
        f"──────────────────────────────────────────────────────────────────────\033[0m\n"
    )
    sys.stderr.flush()


def run_interactive_chat(
    model: nn.Module,
    tokenizer: Any,
    drafter: Inkling2DSparkDrafter | None,
    *,
    mtp_depth: int,
    reasoning_effort: str,
    max_tokens: int,
    show_thinking: bool,
    show_stats: bool,
    system_prompt: str | None,
    prefill_step_size: int = 2048,
    loaded_cache: list[Any] | None = None,
    cache_bits: int = 16,
) -> None:
    """Interactive multi-turn chat loop with slash commands and instant <50ms cache branching."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    base_snap: CacheSnapshot | None = None
    if loaded_cache is not None:
        base_snap = snapshot_cache_tail(loaded_cache, drafter)

    extra_cmds = ", /reset-cache, /save-cache <file>" if loaded_cache is not None else ""
    print(
        f"Inkling-2-Mini Interactive Chat (MTP depth={mtp_depth}, effort={reasoning_effort}, "
        f"max_tokens={max_tokens}"
        + (f", cached_prefix={base_snap.prefix_offset:,} tok" if base_snap else "")
        + f")\nCommands: /clear, /effort <minimal|low|medium|high>, /depth <0..7>, /think, /stats{extra_cmds}, /quit\n"
    )

    while True:
        try:
            user_in = input("\n\033[1;36mUser>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_in:
            continue
        if user_in in ("/quit", "/exit"):
            break
        if user_in in ("/clear", "/reset-cache", "/fork"):
            messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
            if loaded_cache is not None and base_snap is not None:
                t_r = time.perf_counter()
                restore_cache_tail(loaded_cache, base_snap, drafter)
                print(
                    f"[Rewound to cached prefix ({base_snap.prefix_offset:,} tok) in "
                    f"{(time.perf_counter()-t_r)*1000:.1f} ms]"
                )
            else:
                if drafter is not None:
                    drafter.reset()
                gc.collect()
                mx.clear_cache()
                print("[Conversation cleared]")
            continue
        if user_in.startswith("/save-cache "):
            if loaded_cache is None:
                print("[No active cache loaded; use --build-cache or --cache]")
                continue
            out_p = Path(user_in.split(maxsplit=1)[1].strip())
            save_inkling_cache(out_p, loaded_cache, drafter, quant_bits=cache_bits)
            print(f"[Saved cache ({loaded_cache[0][0].offset:,} tok) -> {out_p}]")
            continue
        if user_in == "/think":
            show_thinking = not show_thinking
            print(f"[Show thinking: {show_thinking}]")
            continue
        if user_in == "/stats":
            show_stats = not show_stats
            print(f"[Show telemetry stats: {show_stats}]")
            continue
        if user_in.startswith("/effort "):
            reasoning_effort = user_in.split(maxsplit=1)[1].strip()
            print(f"[Reasoning effort set to: {reasoning_effort}]")
            continue
        if user_in.startswith("/depth "):
            new_d = int(user_in.split(maxsplit=1)[1].strip())
            if not (0 <= new_d <= 7):
                print("[Error: depth must be between 0 and 7]")
                continue
            mtp_depth = new_d
            print(f"[MTP depth set to: {mtp_depth}]")
            continue

        messages.append({"role": "user", "content": user_in})
        print("\033[1;32mInkling>\033[0m ", end="", flush=True)
        if loaded_cache is not None and base_snap is not None:
            # Always branch cleanly from the cached system prefix so multi-turn history stays consistent
            restore_cache_tail(loaded_cache, base_snap, drafter)
            res = generate_inkling(
                model,
                tokenizer,
                messages,
                max_tokens=max_tokens,
                mtp_depth=mtp_depth if drafter is not None else 0,
                drafter=drafter,
                cache=loaded_cache,
                reasoning_effort=reasoning_effort,
                stream=True,
                show_thinking=show_thinking,
                prefill_step_size=prefill_step_size,
            )
        else:
            res = generate_inkling(
                model,
                tokenizer,
                messages,
                max_tokens=max_tokens,
                mtp_depth=mtp_depth if drafter is not None else 0,
                drafter=drafter,
                reasoning_effort=reasoning_effort,
                stream=True,
                show_thinking=show_thinking,
                prefill_step_size=prefill_step_size,
            )
        messages.append({"role": "assistant", "content": res.text})
        if show_stats:
            print_stats(res)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inkling-MLX: Apple Silicon CLI & wrapper for Inkling-2-Mini (W4W8-Native-A8 + DSpark MTP)"
    )
    parser.add_argument(
        "-m",
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Path to MLX checkpoint (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=None,
        help="Single-turn prompt to generate from",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        default=None,
        help="Read prompt from a file",
    )
    parser.add_argument(
        "-i",
        "--chat",
        action="store_true",
        help="Start interactive multi-turn chat REPL",
    )
    parser.add_argument(
        "-d",
        "--mtp-depth",
        type=int,
        default=2,
        help="DSpark MTP speculative decoding max depth 0..7 (default: 2)",
    )
    parser.add_argument(
        "--no-adaptive-mtp",
        action="store_true",
        help="Disable adaptive MTP depth backoff/ramp-up (force static --mtp-depth)",
    )
    parser.add_argument(
        "-e",
        "--effort",
        type=str,
        default="medium",
        help="Reasoning effort: minimal (0.1), low (0.4), medium (0.7), high (0.9), or float (default: medium)",
    )
    parser.add_argument(
        "-n",
        "--max-tokens",
        type=int,
        default=8192,
        help="Maximum tokens to generate (default: 8192, supports up to 65536)",
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        default=None,
        help="Optional system prompt",
    )
    parser.add_argument(
        "--hide-thinking",
        action="store_true",
        help="Hide <|content_thinking|> stream and display only the final response",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Suppress telemetry summary footer",
    )
    parser.add_argument(
        "--no-fast-kernels",
        action="store_true",
        help="Disable custom short_conv and Split-K attention Metal kernels",
    )
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="Disable Q/R/Gate/Sink and MessageGate linear projection fusion",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action="store_true",
        default=None,
        help="Recast BF16 leaves to FP16 at load (default: auto-enabled on M1/M2)",
    )
    parser.add_argument(
        "--no-fp16",
        dest="fp16",
        action="store_false",
        help="Keep BF16 activations even on M1/M2",
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=2048,
        help="Prefill chunk size in tokens (default: 2048)",
    )
    parser.add_argument(
        "--build-cache",
        type=Path,
        default=None,
        help="Prefill context from --context-file (and/or --prompt) and save an offline .safetensors cache",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        action="append",
        default=[],
        help="File or directory to include in --build-cache (may be repeated)",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Load a pre-computed offline .safetensors cache before querying, chatting, or layering",
    )
    parser.add_argument(
        "--cache-bits",
        type=int,
        choices=[8, 16],
        default=16,
        help="On-disk precision for full-attention KV cache: 16 (lossless, default) or 8 (~2x smaller)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Run the W4W8-Native-A8 checkpoint exporter (convert_w4w8_native.py) and exit",
    )
    args = parser.parse_args()

    if args.export:
        from convert.convert_w4w8_native import main as export_main

        export_main()
        return

    if not (0 <= args.mtp_depth <= 7):
        parser.error("--mtp-depth must be between 0 and 7")

    prompt_text: str | None = args.prompt
    if prompt_text is None and args.file is not None:
        prompt_text = args.file.read_text()
    elif prompt_text is None and args.build_cache is None and not args.chat and not sys.stdin.isatty():
        prompt_text = sys.stdin.read().strip()

    if prompt_text is None and args.build_cache is None and not args.chat:
        args.chat = True

    use_fp16 = _default_fp16_activations() if args.fp16 is None else args.fp16
    t_load0 = time.perf_counter()
    sys.stderr.write(
        f"\033[2m[Loading {args.model.name} (MTP depth={args.mtp_depth}, "
        f"fp16={use_fp16}, prefill_step={args.prefill_step_size}, "
        f"fast_kernels={not args.no_fast_kernels}, fuse={not args.no_fuse})...]\033[0m\n"
    )
    model, tokenizer, drafter = load_inkling(
        args.model,
        mtp_depth=args.mtp_depth,
        enable_fast_kernels=not args.no_fast_kernels,
        fuse_projections=not args.no_fuse,
        fp16_activations=use_fp16,
    )
    bpw = compute_bits_per_weight(model)
    load_s = time.perf_counter() - t_load0
    sys.stderr.write(
        f"\033[2m[Ready in {load_s:.1f}s | {bpw:.2f} bits/weight]\033[0m\n"
    )

    loaded_cache: list[Any] | None = None
    if args.cache is not None:
        t_c0 = time.perf_counter()
        loaded_cache, c_meta = load_inkling_cache(args.cache, model, drafter)
        sys.stderr.write(
            f"\033[2m[Loaded cache {args.cache.name} in {time.perf_counter()-t_c0:.2f}s | "
            f"{c_meta['num_tokens']:,} tokens, {c_meta['quant_bits']}-bit, "
            f"drafter={c_meta['has_drafter']}]\033[0m\n"
        )

    if args.build_cache is not None:
        if not args.context_file and not prompt_text:
            parser.error("--build-cache requires at least one --context-file or --prompt")
        ctx_body = collect_context_text(args.context_file, inline_text=prompt_text)
        prefix_header = (
            "Additional reference context:\n\n"
            if loaded_cache is not None
            else "You have the following reference context available to answer user questions:\n\n"
        )
        # Format as a system message so Thinking effort level can be chosen freely at query time!
        sys_block = f"<|message_system|><|content_text|>{prefix_header}{ctx_body}<|end_message|>"
        ctx_tokens = tokenizer.encode(sys_block, add_special_tokens=False)
        cache = loaded_cache if loaded_cache is not None else model.make_speculative_cache()
        if loaded_cache is None and drafter is not None:
            drafter.reset()
        sys.stderr.write(
            f"\033[2m[Prefilling {len(ctx_tokens):,} context tokens into cache...]\033[0m\n"
        )
        p_stats = prefill_into_cache(model, ctx_tokens, cache, drafter, verbose=True)
        t_s0 = time.perf_counter()
        out_path = save_inkling_cache(
            args.build_cache,
            cache,
            drafter,
            quant_bits=args.cache_bits,
            metadata={
                "model": args.model.name,
                "sources": [str(p) for p in args.context_file],
            },
        )
        size_gb = out_path.stat().st_size / (1024**3)
        sys.stderr.write(
            f"\033[2m[Saved offline cache -> {out_path} ({size_gb:.2f} GB, {args.cache_bits}-bit) "
            f"in {time.perf_counter()-t_s0:.2f}s | total ctx={p_stats['total_offset']:,} tok | "
            f"prefill={p_stats['prefill_tps']:.1f} tok/s]\033[0m\n"
        )
        return

    if args.chat:
        run_interactive_chat(
            model,
            tokenizer,
            drafter,
            mtp_depth=args.mtp_depth,
            reasoning_effort=args.effort,
            max_tokens=args.max_tokens,
            show_thinking=not args.hide_thinking,
            show_stats=not args.no_stats,
            system_prompt=args.system,
            prefill_step_size=args.prefill_step_size,
            loaded_cache=loaded_cache,
            cache_bits=args.cache_bits,
        )
        return

    assert prompt_text is not None
    res = generate_inkling(
        model,
        tokenizer,
        prompt_text,
        max_tokens=args.max_tokens,
        mtp_depth=args.mtp_depth,
        drafter=drafter,
        cache=loaded_cache,
        adaptive_mtp=not args.no_adaptive_mtp,
        reasoning_effort=args.effort,
        system_prompt=args.system,
        stream=True,
        show_thinking=not args.hide_thinking,
        prefill_step_size=args.prefill_step_size,
    )
    if not args.no_stats:
        print_stats(res)


if __name__ == "__main__":
    main()
