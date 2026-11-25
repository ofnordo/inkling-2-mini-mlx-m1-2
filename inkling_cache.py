"""Offline KV + ShortConv + DSpark MTP Drafter prompt caching for Inkling-2-Mini.

Supports:
- Full lossless BF16 or 2x-smaller 8-bit quantized on-disk `.safetensors` caches.
- Watchdog-safe adaptive chunked prefill (512 -> 256 -> 128 -> 64) for up to 512k+ tokens.
- Incremental cache layering (load base cache -> append new context -> save child cache).
- Instant (<50 ms) in-memory branching (`CacheSnapshot`) by trimming only the 8 full-attention
  KVCaches and restoring the ~175 MB sliding-window + conv + drafter tail.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache
from mlx_lm.models.inkling2 import Inkling2DSparkDrafter
from mlx_lm.models.inkling2.model import SpeculativeRotatingKVCache

CACHE_FORMAT_VERSION = "inkling2-cache-v1"


@dataclass
class CacheSnapshot:
    """Lightweight in-memory snapshot of the mutable tail (~175 MB) for instant branching."""

    prefix_offset: int
    sliding_states: dict[int, tuple[mx.array | None, mx.array | None, int, int]]
    conv_states: list[list[mx.array | None]]
    drafter_state: dict[str, Any] | None


def _ingest_drafter_taps(drafter: Inkling2DSparkDrafter, taps: list[mx.array]) -> None:
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
        ck = attn.k_norm_ctx(
            ck.reshape(1, n, attn.n_kv_heads, attn.head_dim)
        ).transpose(0, 2, 1, 3)
        cv = cv.reshape(1, n, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
        ck = drafter.model.rope(ck, offset=base)
        drafter.ring_keys[li][:, :, slots, :] = ck
        drafter.ring_values[li][:, :, slots, :] = cv
    drafter.f_rnn = drafter.model.markov_head.context_feature(ctx[:, -1:])
    mx.eval(*drafter.ring_keys, *drafter.ring_values, drafter.pos, drafter.f_rnn)


def _adaptive_chunk_size(pos: int) -> int:
    """Keep every Metal command buffer safely below macOS's ~2.5s GPU watchdog."""
    if pos < 98_304:
        return 512
    if pos < 229_376:
        return 256
    if pos < 393_216:
        return 128
    return 64


def prefill_into_cache(
    model: nn.Module,
    tokens: list[int],
    cache: list[CacheList],
    drafter: Inkling2DSparkDrafter | None = None,
    *,
    verbose: bool = True,
    report_every: int = 16384,
) -> dict[str, Any]:
    """Prefill `tokens` into an existing (or fresh) Inkling cache using watchdog-safe adaptive chunks."""
    if not tokens:
        return {"prefill_tokens": 0, "prefill_s": 0.0, "prefill_tps": 0.0}

    KVCache.step = 8192
    capture = (
        {lid: None for lid in drafter.target_layer_ids} if drafter is not None else None
    )
    arr = mx.array([tokens], dtype=mx.int32)
    n_tok = len(tokens)
    base_offset = int(cache[0][0].offset)
    pos = 0
    t0 = time.perf_counter()
    t_win = t0
    next_report = report_every
    next_clear = 16384

    while pos < n_tok:
        abs_pos = base_offset + pos
        step_sz = _adaptive_chunk_size(abs_pos)
        chunk_len = min(step_sz, n_tok - pos)
        need_taps = drafter is not None and (n_tok - pos) <= (
            drafter.ring_slots + step_sz + 64
        )
        cur_cap = capture if need_taps else None
        _ = model(arr[:, pos : pos + chunk_len], cache=cache, capture=cur_cap)
        mx.eval([c.state for row in cache for c in row])

        if drafter is not None:
            if need_taps and cur_cap is not None:
                taps = [cur_cap[lid] for lid in drafter.target_layer_ids]
                mx.eval(*taps)
                _ingest_drafter_taps(drafter, taps)
            else:
                drafter.fed += chunk_len

        pos += chunk_len
        if pos >= next_clear:
            mx.clear_cache()
            next_clear += 16384

        if verbose and (pos >= next_report or pos == n_tok):
            now = time.perf_counter()
            dt_tot = max(now - t0, 1e-6)
            dt_win = max(now - t_win, 1e-6)
            win_toks = min(report_every, pos)
            sys.stderr.write(
                f"  [prefill] {pos:>7,}/{n_tok:,} (total ctx={base_offset+pos:>7,}) | "
                f"recent={win_toks/dt_win:6.1f} tok/s | avg={pos/dt_tot:6.1f} tok/s | "
                f"active={mx.get_active_memory()/1e9:.2f} GB\n"
            )
            sys.stderr.flush()
            t_win = now
            next_report += report_every

    dt = max(time.perf_counter() - t0, 1e-6)
    return {
        "prefill_tokens": n_tok,
        "total_offset": base_offset + n_tok,
        "prefill_s": dt,
        "prefill_tps": n_tok / dt,
    }


def save_inkling_cache(
    path: Path | str,
    cache: list[CacheList],
    drafter: Inkling2DSparkDrafter | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    quant_bits: int = 16,
) -> Path:
    """Save target KVCache/SpeculativeRotatingKVCache + ShortConv + DSpark drafter state to `.safetensors`."""
    if quant_bits not in (16, 8):
        raise ValueError("quant_bits must be 16 (lossless BF16) or 8 (8-bit affine full-attn KV)")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, mx.array] = {}
    layer_meta: list[dict[str, Any]] = []

    for i, entry in enumerate(cache):
        kv, conv = entry[0], entry[1]
        is_rotating = isinstance(kv, RotatingKVCache)
        info: dict[str, Any] = {
            "kind": "rotating" if is_rotating else "full",
            "offset": int(kv.offset),
        }
        if is_rotating:
            info["max_size"] = int(kv.max_size)
            info["keep"] = int(kv.keep)
            info["idx"] = int(kv._idx)
            if kv.keys is not None:
                k, v = kv.keys_and_values()
                tensors[f"v{i}.k"] = k
                tensors[f"v{i}.v"] = v
                info["has_kv"] = True
            else:
                info["has_kv"] = False
        else:
            if kv.keys is not None and kv.offset > 0:
                k, v = kv.keys_and_values()
                if quant_bits == 8:
                    kq, ks, kb = mx.quantize(k, group_size=64, bits=8, mode="affine")
                    vq, vs, vb = mx.quantize(v, group_size=64, bits=8, mode="affine")
                    tensors[f"v{i}.kq"] = kq
                    tensors[f"v{i}.ks"] = ks
                    tensors[f"v{i}.kb"] = kb
                    tensors[f"v{i}.vq"] = vq
                    tensors[f"v{i}.vs"] = vs
                    tensors[f"v{i}.vb"] = vb
                else:
                    tensors[f"v{i}.k"] = k
                    tensors[f"v{i}.v"] = v
                info["has_kv"] = True
            else:
                info["has_kv"] = False

        conv_present = []
        for ci in range(len(conv.cache)):
            c_arr = conv.cache[ci]
            if c_arr is not None:
                tensors[f"v{i}.conv{ci}"] = c_arr
                conv_present.append(ci)
        info["conv_present"] = conv_present
        layer_meta.append(info)

    drafter_meta: dict[str, Any] | None = None
    if drafter is not None:
        for li in range(len(drafter.ring_keys)):
            tensors[f"d.rk{li}"] = drafter.ring_keys[li]
            tensors[f"d.rv{li}"] = drafter.ring_values[li]
        tensors["d.pos"] = drafter.pos
        if drafter.f_rnn is not None:
            tensors["d.f_rnn"] = drafter.f_rnn
        drafter_meta = {
            "num_layers": len(drafter.ring_keys),
            "fed": int(drafter.fed),
            "has_f_rnn": drafter.f_rnn is not None,
        }

    header = {
        "format": CACHE_FORMAT_VERSION,
        "quant_bits": str(quant_bits),
        "num_tokens": str(int(cache[0][0].offset) if cache else 0),
        "num_visits": str(len(cache)),
        "layers_json": json.dumps(layer_meta, separators=(",", ":")),
        "drafter_json": json.dumps(drafter_meta, separators=(",", ":")),
        "user_meta_json": json.dumps(metadata or {}, separators=(",", ":")),
    }
    mx.eval(tensors)
    mx.save_safetensors(str(out_path), tensors, metadata=header)
    return out_path


def load_inkling_cache(
    path: Path | str,
    model: nn.Module,
    drafter: Inkling2DSparkDrafter | None = None,
) -> tuple[list[CacheList], dict[str, Any]]:
    """Load an offline Inkling `.safetensors` cache into a fresh speculative cache + drafter state."""
    in_path = Path(path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Cache file not found: {in_path}")

    tensors, header = mx.load(str(in_path), return_metadata=True)
    if header.get("format") != CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported cache format: {header.get('format')!r}")

    quant_bits = int(header["quant_bits"])
    layer_meta = json.loads(header["layers_json"])
    drafter_meta = json.loads(header["drafter_json"])
    user_meta = json.loads(header["user_meta_json"])

    cache = model.make_speculative_cache()
    if len(cache) != len(layer_meta):
        raise ValueError(
            f"Cache has {len(layer_meta)} layer visits, but model expects {len(cache)}"
        )

    target_dtype = model.model.language_model.embed_norm.weight.dtype
    for i, (entry, info) in enumerate(zip(cache, layer_meta)):
        kv, conv = entry[0], entry[1]
        if info["kind"] == "rotating":
            if not isinstance(kv, SpeculativeRotatingKVCache):
                kv = SpeculativeRotatingKVCache(
                    max_size=info["max_size"], keep=info["keep"]
                )
                entry[0] = kv
            kv.offset = int(info["offset"])
            kv._idx = int(info["idx"])
            if info["has_kv"]:
                kv.keys = tensors[f"v{i}.k"].astype(target_dtype)
                kv.values = tensors[f"v{i}.v"].astype(target_dtype)
        else:
            kv.offset = int(info["offset"])
            if info["has_kv"]:
                if quant_bits == 8:
                    kv.keys = mx.dequantize(
                        tensors[f"v{i}.kq"],
                        tensors[f"v{i}.ks"],
                        tensors[f"v{i}.kb"],
                        group_size=64,
                        bits=8,
                        mode="affine",
                    ).astype(target_dtype)
                    kv.values = mx.dequantize(
                        tensors[f"v{i}.vq"],
                        tensors[f"v{i}.vs"],
                        tensors[f"v{i}.vb"],
                        group_size=64,
                        bits=8,
                        mode="affine",
                    ).astype(target_dtype)
                else:
                    kv.keys = tensors[f"v{i}.k"].astype(target_dtype)
                    kv.values = tensors[f"v{i}.v"].astype(target_dtype)

        for ci in info["conv_present"]:
            conv[ci] = tensors[f"v{i}.conv{ci}"]

    if drafter is not None:
        drafter.reset()
        if drafter_meta is not None:
            for li in range(drafter_meta["num_layers"]):
                drafter.ring_keys[li] = tensors[f"d.rk{li}"].astype(target_dtype)
                drafter.ring_values[li] = tensors[f"d.rv{li}"].astype(target_dtype)
            drafter.pos = tensors["d.pos"]
            drafter.fed = int(drafter_meta["fed"])
            drafter.f_rnn = (
                tensors["d.f_rnn"].astype(target_dtype)
                if drafter_meta["has_f_rnn"]
                else None
            )
            mx.eval(*drafter.ring_keys, *drafter.ring_values, drafter.pos)
            if drafter.f_rnn is not None:
                mx.eval(drafter.f_rnn)

    mx.eval([c.state for row in cache for c in row])
    del tensors
    gc.collect()
    mx.clear_cache()

    info_out = {
        "num_tokens": int(header["num_tokens"]),
        "quant_bits": quant_bits,
        "has_drafter": drafter_meta is not None,
        **user_meta,
    }
    return cache, info_out


def snapshot_cache_tail(
    cache: list[CacheList],
    drafter: Inkling2DSparkDrafter | None = None,
) -> CacheSnapshot:
    """Capture a lightweight (~175 MB) in-memory snapshot for instant <50ms branching."""
    prefix_offset = int(cache[0][0].offset)
    sliding_states: dict[int, tuple[mx.array | None, mx.array | None, int, int]] = {}
    conv_states: list[list[mx.array | None]] = []

    for i, entry in enumerate(cache):
        kv, conv = entry[0], entry[1]
        if isinstance(kv, RotatingKVCache):
            k_copy = mx.array(kv.keys) if kv.keys is not None else None
            v_copy = mx.array(kv.values) if kv.values is not None else None
            sliding_states[i] = (k_copy, v_copy, int(kv.offset), int(kv._idx))
        conv_states.append(
            [mx.array(c) if c is not None else None for c in conv.cache]
        )

    drafter_state: dict[str, Any] | None = None
    if drafter is not None:
        drafter_state = {
            "ring_keys": [mx.array(k) for k in drafter.ring_keys],
            "ring_values": [mx.array(v) for v in drafter.ring_values],
            "pos": mx.array(drafter.pos),
            "fed": int(drafter.fed),
            "f_rnn": mx.array(drafter.f_rnn) if drafter.f_rnn is not None else None,
        }

    return CacheSnapshot(
        prefix_offset=prefix_offset,
        sliding_states=sliding_states,
        conv_states=conv_states,
        drafter_state=drafter_state,
    )


def restore_cache_tail(
    cache: list[CacheList],
    snap: CacheSnapshot,
    drafter: Inkling2DSparkDrafter | None = None,
) -> None:
    """Instantly rewind `cache` + `drafter` back to `snap` without touching the 8 full-attn KV buffers."""
    for i, entry in enumerate(cache):
        kv, conv = entry[0], entry[1]
        if i in snap.sliding_states:
            k_copy, v_copy, off, idx = snap.sliding_states[i]
            kv.keys = mx.array(k_copy) if k_copy is not None else None
            kv.values = mx.array(v_copy) if v_copy is not None else None
            kv.offset = off
            kv._idx = idx
        else:
            extra = int(kv.offset) - snap.prefix_offset
            if extra > 0:
                kv.trim(extra)
        for ci, c_arr in enumerate(snap.conv_states[i]):
            conv[ci] = mx.array(c_arr) if c_arr is not None else None
        if hasattr(conv, "_verify_windows"):
            conv._verify_windows = None

    if drafter is not None and snap.drafter_state is not None:
        ds = snap.drafter_state
        drafter.ring_keys = [mx.array(k) for k in ds["ring_keys"]]
        drafter.ring_values = [mx.array(v) for v in ds["ring_values"]]
        drafter.pos = mx.array(ds["pos"])
        drafter.fed = int(ds["fed"])
        drafter.f_rnn = mx.array(ds["f_rnn"]) if ds["f_rnn"] is not None else None


def collect_context_text(paths: list[Path], inline_text: str | None = None) -> str:
    """Read one or more files/directories into a formatted context block."""
    parts: list[str] = []
    for p in paths:
        if p.is_file():
            parts.append(f"=== FILE: {p.name} ===\n{p.read_text(errors='ignore')}")
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if (
                    f.is_file()
                    and not f.name.startswith(".")
                    and "__pycache__" not in f.parts
                    and ".git" not in f.parts
                ):
                    try:
                        txt = f.read_text()
                    except UnicodeDecodeError:
                        continue
                    parts.append(f"=== FILE: {f.relative_to(p)} ===\n{txt}")
        else:
            raise FileNotFoundError(f"Context path not found: {p}")
    if inline_text:
        parts.append(inline_text)
    return "\n\n".join(parts)
