"""DSpark MTP speculative decoding implementation and depth benchmark for Inkling-2-Mini in MLX."""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import argparse
import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.inkling2 import layers as inkling2_layers
from mlx_lm.models.inkling2.layers import fuse_linears
from mlx_lm.utils import load, maybe_set_recommended_wired_limit


DRAFT_FILENAME = "mtp.safetensors"

DEFAULT_PROMPT = (
    "Explain the difference between lock-free and wait-free concurrent data structures, "
    "then implement a clean Python lock-free ring buffer (SPSC) and explain its memory ordering guarantees."
)


@dataclass
class Inkling2DSparkConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    block_size: int
    markov_rank: int
    target_layer_ids: list[int]
    vocab_size: int
    unpadded_vocab_size: int
    final_logit_softcapping: float

    @classmethod
    def from_target_config(cls, config: dict) -> Inkling2DSparkConfig:
        text = config["text_config"]
        mtp = config["mtp_config"]
        if mtp.get("q_bias") or mtp.get("o_bias"):
            raise ValueError("Inkling 2 DSpark does not support projection biases.")
        if mtp.get("markov_head_type") != "rnn_hidden":
            raise ValueError(f"Unsupported markov_head_type: {mtp.get('markov_head_type')}")
        return cls(
            hidden_size=int(mtp["hidden_size"]),
            num_hidden_layers=int(mtp["num_hidden_layers"]),
            num_attention_heads=int(mtp["num_attention_heads"]),
            num_key_value_heads=int(mtp["num_key_value_heads"]),
            head_dim=int(mtp["head_dim"]),
            intermediate_size=int(mtp["intermediate_size"]),
            rms_norm_eps=float(mtp.get("rms_norm_eps", 1e-6)),
            rope_theta=float(mtp.get("rope_theta", 1e6)),
            sliding_window=int(mtp["sliding_window_size"]),
            block_size=int(mtp["block_size"]),
            markov_rank=int(mtp["markov_rank"]),
            target_layer_ids=[int(i) for i in mtp["target_layer_ids"]],
            vocab_size=int(text["vocab_size"]),
            unpadded_vocab_size=int(text.get("unpadded_vocab_size") or text["vocab_size"]),
            final_logit_softcapping=float(text.get("final_logit_softcapping") or 0.0),
        )


def mix_rmsnorm(context: mx.array, alpha: mx.array, weight: mx.array, eps: float) -> mx.array:
    """Softmax-weighted mixture over captured target taps followed by RMSNorm."""
    b, n, width = context.shape
    taps = alpha.shape[0]
    x = context.reshape(b, n, taps, width // taps).astype(mx.float32)
    mixed = mx.sum(x * alpha.astype(mx.float32)[:, None], axis=2)
    return mx.fast.rms_norm(mixed, weight.astype(mx.float32), eps).astype(context.dtype)


class DraftAttention(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        h = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim**-0.5
        dq = self.n_heads * self.head_dim
        dkv = self.n_kv_heads * self.head_dim
        self.dq = dq
        self.dkv = dkv
        self.q_proj = nn.Linear(h, dq, bias=False)
        self.k_proj = nn.Linear(h, dkv, bias=False)
        self.v_proj = nn.Linear(h, dkv, bias=False)
        self.o_proj = nn.Linear(dq, h, bias=False)
        self.k_proj_ctx = nn.Linear(h, dkv, bias=False)
        self.v_proj_ctx = nn.Linear(h, dkv, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_ctx = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self._eps = config.rms_norm_eps
        self._alpha: mx.array | None = None
        self._mix_weight: mx.array | None = None

    def project_context_kv(self, context: mx.array) -> tuple[mx.array, mx.array]:
        mixed = mix_rmsnorm(context, self._alpha, self._mix_weight, self._eps)
        if getattr(self, "_kv_ctx_proj", None) is not None:
            return mx.split(self._kv_ctx_proj(mixed), 2, axis=-1)
        return self.k_proj_ctx(mixed), self.v_proj_ctx(mixed)

    def project_block_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        if getattr(self, "_qkv_proj", None) is not None:
            return mx.split(self._qkv_proj(x), (self.dq, self.dq + self.dkv), axis=-1)
        return self.q_proj(x), self.k_proj(x), self.v_proj(x)


class DraftMLP(nn.Module):
    def __init__(self, hidden: int, inter: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, inter, bias=False)
        self.up_proj = nn.Linear(hidden, inter, bias=False)
        self.down_proj = nn.Linear(inter, hidden, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if getattr(self, "_gate_up_proj", None) is not None:
            g, u = mx.split(self._gate_up_proj(x), 2, axis=-1)
            return self.down_proj(nn.silu(g) * u)
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DraftLayer(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        h = config.hidden_size
        self.self_attn = DraftAttention(config)
        self.mlp = DraftMLP(h, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self.dflare_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)


class Inkling2MarkovHead(nn.Module):
    """Recurrent Markov transition head with folded LM head projection."""

    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        r, h = config.markov_rank, config.hidden_size
        self.rank = r
        self.markov_w1 = nn.Embedding(config.vocab_size, r)
        self.rnn_proj = nn.Linear(2 * r + 2 * h, 3 * r, bias=True)
        self.hidden_out = nn.Linear(r, h, bias=True)
        self.dflare_norm = nn.RMSNorm(h, eps=config.rms_norm_eps)
        self._eps = config.rms_norm_eps
        self._softcap = config.final_logit_softcapping
        self._vocab_bias = None
        if config.unpadded_vocab_size < config.vocab_size:
            self._vocab_bias = mx.where(
                mx.arange(config.vocab_size) < config.unpadded_vocab_size,
                0.0,
                -mx.inf,
            ).astype(mx.float32)
        self._fold_weight: mx.array | None = None
        self._fold_qweight: mx.array | None = None
        self._fold_scales: mx.array | None = None
        self._fold_biases: mx.array | None = None
        self._fold_bias: mx.array | None = None
        self._alpha: mx.array | None = None

    def fold(self, lm_head: nn.Module, quantize_fold: bool = True) -> None:
        if isinstance(lm_head, nn.QuantizedLinear):
            weight = mx.dequantize(
                lm_head.weight,
                lm_head.scales,
                lm_head.get("biases"),
                lm_head.group_size,
                lm_head.bits,
                mode=getattr(lm_head, "mode", "affine"),
            )
        else:
            weight = lm_head.weight
        proj = self.hidden_out.weight.astype(mx.float32)
        bias = self.hidden_out.bias.astype(mx.float32)
        rows, biases = [], []
        for start in range(0, weight.shape[0], 16384):
            head = weight[start : start + 16384].astype(mx.float32)
            rows.append((head @ proj).astype(mx.bfloat16))
            biases.append((head @ bias).astype(mx.bfloat16))
            mx.eval(rows[-1], biases[-1])
        fold_w = mx.concatenate(rows, axis=0)
        self._fold_bias = mx.concatenate(biases, axis=0)
        if quantize_fold:
            self._fold_qweight, self._fold_scales, self._fold_biases = mx.quantize(
                fold_w, group_size=64, bits=4, mode="affine"
            )
            self._fold_weight = None
            mx.eval(self._fold_qweight, self._fold_scales, self._fold_biases, self._fold_bias)
        else:
            self._fold_weight = fold_w
            self._fold_qweight = self._fold_scales = self._fold_biases = None
            mx.eval(self._fold_weight, self._fold_bias)

    def context_feature(self, last_tap: mx.array) -> mx.array:
        return mix_rmsnorm(last_tap, self._alpha, self.dflare_norm.weight, self._eps)[0]

    def _finalize_logits(self, logits: mx.array) -> mx.array:
        if self._softcap:
            logits = mx.tanh(logits / self._softcap) * self._softcap
        if self._vocab_bias is not None:
            logits = logits + self._vocab_bias
        return logits

    def draft_depth1(
        self,
        lm_head_fn: Any,
        prev_token: mx.array,
        hidden_row: mx.array,
        f_rnn: mx.array,
    ) -> mx.array:
        """Exact linearity fusion for depth=1: lm_head(h + hidden_out(latent)) in 1 LM-head pass."""
        dtype = hidden_row.dtype
        inputs = mx.concatenate(
            [
                mx.zeros((1, self.rank), dtype=dtype),
                self.markov_w1(prev_token.reshape(-1)).astype(dtype),
                hidden_row.astype(dtype),
                f_rnn.astype(dtype),
            ],
            axis=-1,
        )
        proj = self.rnn_proj(inputs).astype(mx.float32)
        latent = mx.tanh(proj[:, 2 * self.rank :]).astype(dtype)
        h_eff = (hidden_row + self.hidden_out(latent)).astype(dtype)[:, None, :]
        logits = lm_head_fn(h_eff)[:, 0].astype(mx.float32)
        return mx.argmax(self._finalize_logits(logits), axis=-1)

    def step_logits(
        self,
        base_logits: mx.array,
        prev_token: mx.array,
        state: mx.array,
        hidden_row: mx.array,
        f_rnn: mx.array,
    ) -> tuple[mx.array, mx.array]:
        dtype = state.dtype
        inputs = mx.concatenate(
            [
                state,
                self.markov_w1(prev_token.reshape(-1)).astype(dtype),
                hidden_row.astype(dtype),
                f_rnn.astype(dtype),
            ],
            axis=-1,
        )
        proj = self.rnn_proj(inputs).astype(mx.float32)
        r = self.rank
        gate = mx.sigmoid(proj[:, :r]).astype(dtype)
        new_state = (gate * state).astype(dtype) + (
            (1.0 - gate) * mx.tanh(proj[:, r : 2 * r]).astype(dtype)
        ).astype(dtype)
        latent = mx.tanh(proj[:, 2 * r :]).astype(dtype)
        if self._fold_qweight is not None:
            bias = (
                mx.quantized_matmul(
                    latent,
                    self._fold_qweight,
                    self._fold_scales,
                    self._fold_biases,
                    transpose=True,
                    group_size=64,
                    bits=4,
                    mode="affine",
                )
                + self._fold_bias
            )
        else:
            bias = latent @ self._fold_weight.T + self._fold_bias
        logits = base_logits.astype(mx.float32) + bias.astype(mx.float32)
        return self._finalize_logits(logits), new_state


class Inkling2DSparkModel(nn.Module):
    def __init__(self, config: Inkling2DSparkConfig):
        super().__init__()
        self.config = config
        self.layers = [DraftLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mask_embed = mx.zeros((config.hidden_size,))
        self.fuse_weight = mx.zeros(
            (config.num_hidden_layers + 1, len(config.target_layer_ids))
        )
        self.markov_head = Inkling2MarkovHead(config)
        self.rope = nn.RoPE(config.head_dim, traditional=False, base=config.rope_theta)
        self._embed_fn = None
        self._lm_head_fn = None

    def bind(self, target: nn.Module, quantize_fold: bool = True) -> Inkling2DSparkModel:
        self._embed_fn = target.model.language_model.embed
        self._lm_head_fn = target.lm_head
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(
            mx.bfloat16
        )
        for index, layer in enumerate(self.layers):
            layer.self_attn._alpha = alpha[index]
            layer.self_attn._mix_weight = layer.dflare_norm.weight
        self.markov_head._alpha = alpha[-1]
        self.markov_head.fold(target.lm_head, quantize_fold=quantize_fold)
        lazy = [layer.self_attn._alpha for layer in self.layers]
        lazy.append(self.markov_head._alpha)
        if self.markov_head._vocab_bias is not None:
            lazy.append(self.markov_head._vocab_bias)
        mx.eval(lazy)
        return self

    def fuse_projections(self) -> None:
        for layer in self.layers:
            attn = layer.self_attn
            if getattr(attn, "_qkv_proj", None) is None:
                fused = fuse_linears([attn.q_proj, attn.k_proj, attn.v_proj])
                if fused is not None:
                    attn._qkv_proj = fused
                    del attn.q_proj, attn.k_proj, attn.v_proj
            if getattr(attn, "_kv_ctx_proj", None) is None:
                fused = fuse_linears([attn.k_proj_ctx, attn.v_proj_ctx])
                if fused is not None:
                    attn._kv_ctx_proj = fused
                    del attn.k_proj_ctx, attn.v_proj_ctx
            mlp = layer.mlp
            if getattr(mlp, "_gate_up_proj", None) is None:
                fused = fuse_linears([mlp.gate_proj, mlp.up_proj])
                if fused is not None:
                    mlp._gate_up_proj = fused
                    del mlp.gate_proj, mlp.up_proj


class Inkling2DSparkDrafter:
    """Single-sequence DSpark MTP drafter with static-size context ring buffers."""

    def __init__(
        self,
        model_path: Path,
        target_model: nn.Module,
        *,
        draft_bits: int = 8,
        opt_drafter: bool = True,
    ):
        config = Inkling2DSparkConfig.from_target_config(
            json.loads((model_path / "config.json").read_text())
        )
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.block_size = config.block_size
        self.max_depth = config.block_size - 1
        self.ring_slots = config.sliding_window - 1
        self.opt_drafter = opt_drafter

        draft_model = Inkling2DSparkModel(config)
        weights = mx.load(str(model_path / DRAFT_FILENAME))
        draft_model.load_weights(list(weights.items()), strict=True)
        nn.quantize(
            draft_model,
            group_size=64,
            bits=draft_bits,
            class_predicate=lambda p, m: isinstance(m, nn.Linear)
            and p.startswith("layers."),
        )
        if opt_drafter:
            draft_model.fuse_projections()
        mx.eval(draft_model.parameters())
        draft_model.bind(target_model, quantize_fold=opt_drafter)
        self.model = draft_model

        self.reset()

    def reset(self) -> None:
        cfg = self.config
        shape = (1, cfg.num_key_value_heads, self.ring_slots, cfg.head_dim)
        self.ring_keys = [
            mx.zeros(shape, dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)
        ]
        self.ring_values = [
            mx.zeros(shape, dtype=mx.bfloat16) for _ in range(cfg.num_hidden_layers)
        ]
        self.pos = mx.full((self.ring_slots,), -1, dtype=mx.int32)
        self.fed = 0
        self.f_rnn: mx.array | None = None
        mx.eval(*self.ring_keys, *self.ring_values, self.pos)

    def draft(
        self,
        new_taps: list[mx.array],
        anchor: int,
        depth: int,
    ) -> mx.array:
        """Ingest committed target taps ``[(1, n, H), ...]`` and draft ``depth`` tokens."""
        assert 1 <= depth <= self.max_depth
        model = self.model
        cfg = self.config
        ctx = mx.concatenate(new_taps, axis=-1)
        n_total = int(ctx.shape[1])
        skip = max(0, n_total - self.ring_slots)
        if skip:
            ctx = ctx[:, skip:]
        n = int(ctx.shape[1])
        base = self.fed + skip
        total = base + n
        self.fed = total

        if n > 0:
            self.f_rnn = model.markov_head.context_feature(ctx[:, -1:])
            write_slots = (mx.arange(base, base + n, dtype=mx.int32) % self.ring_slots)
            self.pos[write_slots] = mx.arange(base, base + n, dtype=mx.int32)

        ring_valid = (
            (self.pos >= 0)
            & (self.pos >= total - self.ring_slots)
            & (self.pos < total)
        )
        mask = mx.concatenate(
            [ring_valid, mx.ones((self.block_size,), dtype=mx.bool_)], axis=0
        )[None, None, None, :]

        anchor_arr = mx.array([[anchor]], dtype=mx.int32)
        anchor_emb = model._embed_fn(anchor_arr)
        masks_emb = mx.broadcast_to(
            model.mask_embed.astype(anchor_emb.dtype),
            (1, self.block_size - 1, cfg.hidden_size),
        )
        h = mx.concatenate([anchor_emb, masks_emb], axis=1)

        for layer_idx, layer in enumerate(model.layers):
            attn = layer.self_attn
            residual = h
            x = layer.input_layernorm(h)

            q, prop_k, prop_v = attn.project_block_qkv(x)

            if n > 0:
                ctx_k, ctx_v = attn.project_context_kv(ctx)
                ctx_k = attn.k_norm_ctx(
                    ctx_k.reshape(1, n, attn.n_kv_heads, attn.head_dim)
                ).transpose(0, 2, 1, 3)
                ctx_v = ctx_v.reshape(1, n, attn.n_kv_heads, attn.head_dim).transpose(
                    0, 2, 1, 3
                )
                ctx_k = model.rope(ctx_k, offset=base)
                self.ring_keys[layer_idx][:, :, write_slots, :] = ctx_k
                self.ring_values[layer_idx][:, :, write_slots, :] = ctx_v

            q = attn.q_norm(
                q.reshape(1, self.block_size, attn.n_heads, attn.head_dim)
            ).transpose(0, 2, 1, 3)
            prop_k = attn.k_norm(
                prop_k.reshape(1, self.block_size, attn.n_kv_heads, attn.head_dim)
            ).transpose(0, 2, 1, 3)
            prop_v = prop_v.reshape(
                1, self.block_size, attn.n_kv_heads, attn.head_dim
            ).transpose(0, 2, 1, 3)

            q = model.rope(q, offset=total)
            prop_k = model.rope(prop_k, offset=total)

            keys = mx.concatenate([self.ring_keys[layer_idx], prop_k], axis=2)
            values = mx.concatenate([self.ring_values[layer_idx], prop_v], axis=2)
            attended = mx.fast.scaled_dot_product_attention(
                q, keys, values, scale=attn.scale, mask=mask
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(1, self.block_size, -1)
            h = residual + attn.o_proj(attended)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        # Only the first `depth` mask positions need LM-head + Markov RNN steps.
        draft_hidden = model.norm(h)[:, 1 : 1 + depth]
        prev = mx.array([anchor], dtype=mx.int32)
        if depth == 1 and self.opt_drafter:
            draft_tokens = model.markov_head.draft_depth1(
                model._lm_head_fn, prev, draft_hidden[:, 0], self.f_rnn
            ).astype(mx.int32)
        else:
            base_logits = model._lm_head_fn(draft_hidden)
            state = mx.zeros((1, cfg.markov_rank), dtype=draft_hidden.dtype)
            proposals = []
            for step in range(depth):
                step_logits, state = model.markov_head.step_logits(
                    base_logits[:, step],
                    prev,
                    state,
                    draft_hidden[:, step],
                    self.f_rnn,
                )
                prev = mx.argmax(step_logits, axis=-1)
                proposals.append(prev)
            draft_tokens = mx.concatenate(proposals).astype(mx.int32)
        mx.eval(draft_tokens, *self.ring_keys, *self.ring_values, self.pos, self.f_rnn)
        return draft_tokens


def _set_verify_recording(cache: list[Any], enabled: bool) -> None:
    for entry in cache:
        entry[1].record_verify_windows = enabled


def run_generation(
    model: nn.Module,
    tokenizer: Any,
    prompt_tokens: list[int],
    max_tokens: int,
    mtp_depth: int,
    drafter: Inkling2DSparkDrafter | None = None,
    stop_on_eos: bool = True,
) -> dict[str, Any]:
    """Run greedy generation at mtp_depth=0 (autoregressive) or 1..7 (DSpark MTP)."""
    eos_ids = set(getattr(tokenizer, "eos_token_ids", ()) or ())
    if not eos_ids and getattr(tokenizer, "eos_token_id", None) is not None:
        eos_ids = {int(tokenizer.eos_token_id)}

    if mtp_depth > 0:
        assert drafter is not None
        drafter.reset()
        cache = model.make_speculative_cache()
        target_ids = drafter.target_layer_ids
    else:
        cache = model.make_cache()
        target_ids = []

    mx.clear_cache()
    mx.reset_peak_memory()

    prompt_arr = mx.array([prompt_tokens], dtype=mx.int32)

    # --- Prefill ---
    t_prefill_start = time.perf_counter()
    if mtp_depth > 0:
        capture = {idx: None for idx in target_ids}
        logits = model(prompt_arr, cache=cache, capture=capture)
        anchor_arr = mx.argmax(logits[0, -1], axis=-1)
        pending_taps = [capture[idx] for idx in target_ids]
        mx.eval(anchor_arr, *pending_taps)
    else:
        logits = model(prompt_arr, cache=cache)
        anchor_arr = mx.argmax(logits[0, -1], axis=-1)
        mx.eval(anchor_arr)
    t_prefill_end = time.perf_counter()

    anchor = int(anchor_arr.item())
    generated: list[int] = [anchor]
    if stop_on_eos and anchor in eos_ids:
        return {
            "tokens": generated,
            "prompt_tokens": len(prompt_tokens),
            "gen_tokens": len(generated),
            "prompt_tps": len(prompt_tokens) / max(1e-9, t_prefill_end - t_prefill_start),
            "gen_tps": 0.0,
            "mean_accepted": 0.0,
            "accept_rate": 0.0,
            "steps": 0,
            "peak_gb": mx.get_peak_memory() / 1e9,
        }

    # --- Decode ---
    t_gen_start = time.perf_counter()
    total_drafted = 0
    total_accepted = 0
    steps = 0

    if mtp_depth == 0:
        while len(generated) < max_tokens:
            logits = model(mx.array([[anchor]], dtype=mx.int32), cache=cache)
            next_tok = int(mx.argmax(logits[0, -1], axis=-1).item())
            generated.append(next_tok)
            anchor = next_tok
            steps += 1
            if stop_on_eos and anchor in eos_ids:
                break
    else:
        while len(generated) < max_tokens:
            remaining = max_tokens - len(generated)
            d = min(mtp_depth, remaining)
            draft_arr = drafter.draft(pending_taps, anchor, d)
            draft_list = draft_arr.tolist()
            total_drafted += d

            verify_input = mx.concatenate(
                [mx.array([anchor], dtype=mx.int32), draft_arr]
            )[None, :]
            capture = {idx: None for idx in target_ids}
            _set_verify_recording(cache, True)
            logits = model(verify_input, cache=cache, capture=capture)
            _set_verify_recording(cache, False)

            target_preds = mx.argmax(logits[0], axis=-1)
            taps_out = [capture[idx] for idx in target_ids]
            mx.eval(target_preds, *taps_out)
            preds_list = target_preds.tolist()

            accepted = 0
            for i in range(d):
                if preds_list[i] == draft_list[i]:
                    accepted += 1
                else:
                    break
            total_accepted += accepted
            steps += 1

            model.rollback_speculative_cache(cache, accepted, d)

            emitted = draft_list[:accepted] + [preds_list[accepted]]
            for tok in emitted:
                generated.append(tok)
                if (stop_on_eos and tok in eos_ids) or len(generated) >= max_tokens:
                    break
            if (stop_on_eos and generated[-1] in eos_ids) or len(generated) >= max_tokens:
                break

            anchor = preds_list[accepted]
            pending_taps = [t[:, : accepted + 1] for t in taps_out]

    t_gen_end = time.perf_counter()
    gen_time = max(1e-9, t_gen_end - t_gen_start)
    # Prompt timing covers the first token; decode timing covers remaining generated tokens.
    decode_tokens = max(0, len(generated) - 1)
    return {
        "tokens": generated,
        "prompt_tokens": len(prompt_tokens),
        "gen_tokens": len(generated),
        "prompt_tps": len(prompt_tokens) / max(1e-9, t_prefill_end - t_prefill_start),
        "gen_tps": decode_tokens / gen_time,
        "mean_accepted": total_accepted / max(1, steps),
        "accept_rate": (total_accepted / total_drafted) if total_drafted else 0.0,
        "steps": steps,
        "peak_gb": mx.get_peak_memory() / 1e9,
    }


def format_prompt(tokenizer: Any, user_prompt: str) -> list[int]:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_prompt}]
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
    return tokenizer.encode(user_prompt)


def verify_equivalence(
    model: nn.Module,
    tokenizer: Any,
    drafter: Inkling2DSparkDrafter,
    prompt_tokens: list[int],
    depths: list[int],
    verify_tokens: int = 64,
) -> None:
    print(f"\n[VERIFY] Checking exact greedy token equivalence for {verify_tokens} tokens across depths [0, {', '.join(map(str, depths))}]...")
    baseline = run_generation(
        model, tokenizer, prompt_tokens, verify_tokens, mtp_depth=0, stop_on_eos=False
    )
    base_toks = baseline["tokens"]
    for d in depths:
        res = run_generation(
            model,
            tokenizer,
            prompt_tokens,
            verify_tokens,
            mtp_depth=d,
            drafter=drafter,
            stop_on_eos=False,
        )
        if res["tokens"] != base_toks:
            for idx, (a, b) in enumerate(zip(base_toks, res["tokens"])):
                if a != b:
                    raise AssertionError(
                        f"Mismatch at depth={d} token index {idx}: baseline={a} ({tokenizer.decode([a])!r}) vs mtp={b} ({tokenizer.decode([b])!r})"
                    )
            raise AssertionError(f"Length mismatch at depth={d}: {len(base_toks)} vs {len(res['tokens'])}")
        print(
            f"  [PASS] depth={d}: exact 100% match ({len(base_toks)} tokens, "
            f"mean_accepted={res['mean_accepted']:.2f}/{d}, accept_rate={res['accept_rate']*100:.1f}%)"
        )


def benchmark_depths(
    model: nn.Module,
    tokenizer: Any,
    drafter: Inkling2DSparkDrafter,
    prompt_tokens: list[int],
    depths: list[int],
    max_tokens: int,
    runs: int,
    label: str,
) -> list[dict[str, Any]]:
    results = []
    for d in depths:
        # 1 warmup run to compile Metal graphs/kernels for this depth shape.
        _ = run_generation(
            model,
            tokenizer,
            prompt_tokens,
            min(32, max_tokens),
            mtp_depth=d,
            drafter=drafter if d > 0 else None,
            stop_on_eos=False,
        )
        run_stats = []
        for _ in range(runs):
            stat = run_generation(
                model,
                tokenizer,
                prompt_tokens,
                max_tokens,
                mtp_depth=d,
                drafter=drafter if d > 0 else None,
                stop_on_eos=False,
            )
            run_stats.append(stat)
        avg = {
            "label": label,
            "depth": d,
            "prompt_tps": sum(s["prompt_tps"] for s in run_stats) / runs,
            "gen_tps": sum(s["gen_tps"] for s in run_stats) / runs,
            "mean_accepted": sum(s["mean_accepted"] for s in run_stats) / runs,
            "accept_rate": sum(s["accept_rate"] for s in run_stats) / runs,
            "peak_gb": max(s["peak_gb"] for s in run_stats),
            "gen_tokens": run_stats[0]["gen_tokens"],
        }
        results.append(avg)
        print(
            f"  [{label} | depth={d}] prompt={avg['prompt_tps']:.1f} tok/s | "
            f"gen={avg['gen_tps']:.2f} tok/s | mean_acc={avg['mean_accepted']:.2f} | "
            f"acc_rate={avg['accept_rate']*100:.1f}% | peak={avg['peak_gb']:.2f} GB"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Inkling-2-Mini W4A16 across MTP depths")
    parser.add_argument(
        "--w4a16-model",
        type=Path,
        default=Path(os.environ.get("INKLING_W4A16", "Inkling-2-Mini-MLX-W4A16")),
    )
    parser.add_argument(
        "--orig-model",
        type=Path,
        default=Path(os.environ.get("INKLING_RAW", "Inkling-2-Mini")),
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--skip-orig", action="store_true")
    parser.add_argument("--depths", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--fuse", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--short-conv-kernel", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--draft-bits", type=int, choices=(4, 8), default=8)
    parser.add_argument(
        "--opt-drafter", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args()

    maybe_set_recommended_wired_limit()
    inkling2_layers._USE_SHORT_CONV_KERNEL = args.short_conv_kernel

    print(f"[INFO] Loading W4A16 model from {args.w4a16_model}...")
    model, tokenizer = load(str(args.w4a16_model))
    prompt_tokens = format_prompt(tokenizer, DEFAULT_PROMPT)
    print(
        f"[INFO] Prompt length: {len(prompt_tokens)} tokens, generating {args.max_tokens} tokens/run "
        f"({args.runs} runs + 1 warmup) | fuse={args.fuse} short_conv={args.short_conv_kernel} "
        f"draft_bits={args.draft_bits} opt_drafter={args.opt_drafter}"
    )

    if args.ablation:
        print("\n[ABLATION] Running same-session optimization ablation on W4A16...")
        inkling2_layers._USE_SHORT_CONV_KERNEL = False
        unopt_drafter = Inkling2DSparkDrafter(
            args.w4a16_model, model, draft_bits=8, opt_drafter=False
        )
        ablation_rows = []
        ablation_rows.extend(
            benchmark_depths(
                model,
                tokenizer,
                unopt_drafter,
                prompt_tokens,
                [0, 1, 2],
                args.max_tokens,
                args.runs,
                "1. Baseline W4A16",
            )
        )
        del unopt_drafter
        gc.collect()
        mx.clear_cache()

        inkling2_layers._USE_SHORT_CONV_KERNEL = True
        ablation_rows.extend(
            benchmark_depths(
                model,
                tokenizer,
                None,  # type: ignore[arg-type]
                prompt_tokens,
                [0],
                args.max_tokens,
                args.runs,
                "2. +short_conv kernel",
            )
        )

        model.fuse_projections()
        ablation_rows.extend(
            benchmark_depths(
                model,
                tokenizer,
                None,  # type: ignore[arg-type]
                prompt_tokens,
                [0],
                args.max_tokens,
                args.runs,
                "3. +fuse_projections",
            )
        )

        opt_drafter_8 = Inkling2DSparkDrafter(
            args.w4a16_model, model, draft_bits=8, opt_drafter=True
        )
        ablation_rows.extend(
            benchmark_depths(
                model,
                tokenizer,
                opt_drafter_8,
                prompt_tokens,
                [1, 2],
                args.max_tokens,
                args.runs,
                "4. +opt_drafter (8-bit)",
            )
        )
        del opt_drafter_8
        gc.collect()
        mx.clear_cache()

        opt_drafter_4 = Inkling2DSparkDrafter(
            args.w4a16_model, model, draft_bits=4, opt_drafter=True
        )
        verify_equivalence(
            model,
            tokenizer,
            opt_drafter_4,
            prompt_tokens,
            [d for d in args.depths if d > 0],
            verify_tokens=64,
        )
        ablation_rows.extend(
            benchmark_depths(
                model,
                tokenizer,
                opt_drafter_4,
                prompt_tokens,
                args.depths,
                args.max_tokens,
                args.runs,
                "5. All Opts (4-bit drafter)",
            )
        )
        base_d0 = ablation_rows[0]["gen_tps"]
        print("\n### Same-Session Ablation & MTP Sweep (Apple M2 Max, 256 generated tokens)\n")
        print("| Config | MTP Depth | Prompt tok/s | Gen tok/s | Speedup vs Baseline d=0 | Mean Accepted / Step | Accept Rate | Peak Memory |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in ablation_rows:
            speedup = f"{r['gen_tps'] / base_d0:.2f}x"
            acc_str = f"{r['mean_accepted']:.2f}/{r['depth']}" if r["depth"] > 0 else "-"
            rate_str = f"{r['accept_rate']*100:.1f}%" if r["depth"] > 0 else "-"
            print(
                f"| {r['label']} | {r['depth']} | {r['prompt_tps']:.1f} | {r['gen_tps']:.2f} | "
                f"{speedup} | {acc_str} | {rate_str} | {r['peak_gb']:.2f} GB |"
            )
        return

    if args.fuse:
        model.fuse_projections()

    drafter = Inkling2DSparkDrafter(
        args.w4a16_model,
        model,
        draft_bits=args.draft_bits,
        opt_drafter=args.opt_drafter,
    )

    mtp_depths = [d for d in args.depths if d > 0]
    if mtp_depths:
        verify_equivalence(model, tokenizer, drafter, prompt_tokens, mtp_depths, verify_tokens=64)

    print("\n[BENCH] Running W4A16 MTP depth sweep...")
    all_rows = benchmark_depths(
        model,
        tokenizer,
        drafter,
        prompt_tokens,
        args.depths,
        args.max_tokens,
        args.runs,
        "W4A16 (4-bit affine)",
    )

    if not args.skip_orig:
        del drafter, model
        gc.collect()
        mx.clear_cache()
        print(f"\n[INFO] Loading original checkpoint from {args.orig_model} for d=0 baseline comparison...")
        orig_model, orig_tok = load(str(args.orig_model))
        orig_prompt = format_prompt(orig_tok, DEFAULT_PROMPT)
        orig_rows = benchmark_depths(
            orig_model,
            orig_tok,
            None,  # type: ignore[arg-type]
            orig_prompt,
            [0],
            args.max_tokens,
            args.runs,
            "Original (NVFP4/MXFP8)",
        )
        all_rows = orig_rows + all_rows

    w4_d0 = next((r["gen_tps"] for r in all_rows if r["label"].startswith("W4A16") and r["depth"] == 0), None)

    print("\n### Benchmark Summary (Apple M2 Max, 256 generated tokens)\n")
    print("| Model / Quant | MTP Depth | Prompt tok/s | Gen tok/s | Speedup vs W4A16 d=0 | Mean Accepted / Step | Accept Rate | Peak Memory |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in all_rows:
        speedup = f"{r['gen_tps'] / w4_d0:.2f}x" if w4_d0 else "-"
        acc_str = f"{r['mean_accepted']:.2f}/{r['depth']}" if r["depth"] > 0 else "-"
        rate_str = f"{r['accept_rate']*100:.1f}%" if r["depth"] > 0 else "-"
        print(
            f"| {r['label']} | {r['depth']} | {r['prompt_tps']:.1f} | {r['gen_tps']:.2f} | "
            f"{speedup} | {acc_str} | {rate_str} | {r['peak_gb']:.2f} GB |"
        )


if __name__ == "__main__":
    main()
