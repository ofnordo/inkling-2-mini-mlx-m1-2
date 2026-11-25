# Copyright © 2026 Apple Inc.

"""DSpark Multi-Token Prediction (MTP) speculative drafter for Inkling 2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.inkling2.layers import fuse_linears

DRAFT_FILENAME = "mtp.safetensors"


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
            raise ValueError(
                f"Unsupported markov_head_type: {mtp.get('markov_head_type')}"
            )
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
            unpadded_vocab_size=int(
                text.get("unpadded_vocab_size") or text["vocab_size"]
            ),
            final_logit_softcapping=float(text.get("final_logit_softcapping") or 0.0),
        )


def mix_rmsnorm(
    context: mx.array, alpha: mx.array, weight: mx.array, eps: float
) -> mx.array:
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
            return mx.split(
                self._qkv_proj(x), (self.dq, self.dq + self.dkv), axis=-1
            )
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
        target_dtype = (
            lm_head.scales.dtype
            if isinstance(lm_head, nn.QuantizedLinear)
            else lm_head.weight.dtype
        )
        if target_dtype not in (mx.bfloat16, mx.float16):
            target_dtype = mx.bfloat16
        rows, biases = [], []
        for start in range(0, weight.shape[0], 16384):
            head = weight[start : start + 16384].astype(mx.float32)
            rows.append((head @ proj).astype(target_dtype))
            biases.append((head @ bias).astype(target_dtype))
            mx.eval(rows[-1], biases[-1])
        fold_w = mx.concatenate(rows, axis=0)
        self._fold_bias = mx.concatenate(biases, axis=0)
        if quantize_fold:
            self._fold_qweight, self._fold_scales, self._fold_biases = mx.quantize(
                fold_w, group_size=64, bits=4, mode="affine"
            )
            self._fold_weight = None
            mx.eval(
                self._fold_qweight,
                self._fold_scales,
                self._fold_biases,
                self._fold_bias,
            )
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

    def bind(
        self, target: nn.Module, quantize_fold: bool = True
    ) -> Inkling2DSparkModel:
        self._embed_fn = target.model.language_model.embed
        self._lm_head_fn = target.lm_head
        target_dtype = target.model.language_model.embed_norm.weight.dtype
        if target_dtype not in (mx.bfloat16, mx.float16):
            target_dtype = mx.bfloat16
        alpha = mx.softmax(self.fuse_weight.astype(mx.float32), axis=-1).astype(
            target_dtype
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
        model_path: Path | str,
        target_model: nn.Module,
        *,
        draft_bits: int = 8,
        opt_drafter: bool = True,
    ):
        model_path = Path(model_path)
        config = Inkling2DSparkConfig.from_target_config(
            json.loads((model_path / "config.json").read_text())
        )
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.block_size = config.block_size
        self.max_depth = config.block_size - 1
        self.ring_slots = config.sliding_window - 1
        self.opt_drafter = opt_drafter

        target_dtype = target_model.model.language_model.embed_norm.weight.dtype
        if target_dtype not in (mx.bfloat16, mx.float16):
            target_dtype = mx.bfloat16
        self.dtype = target_dtype

        draft_model = Inkling2DSparkModel(config)
        weights = mx.load(str(model_path / DRAFT_FILENAME))
        if target_dtype == mx.float16:
            weights = {
                k: (v.astype(mx.float16) if v.dtype == mx.bfloat16 else v)
                for k, v in weights.items()
            }
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
            mx.zeros(shape, dtype=self.dtype) for _ in range(cfg.num_hidden_layers)
        ]
        self.ring_values = [
            mx.zeros(shape, dtype=self.dtype) for _ in range(cfg.num_hidden_layers)
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
            write_slots = (
                mx.arange(base, base + n, dtype=mx.int32) % self.ring_slots
            )
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


class AdaptiveMTPController:
    """Adaptive MTP depth controller that backs off to d=0 on high-entropy spans
    and ramps up to max_depth on predictable code / JSON / structured spans.
    """

    def __init__(
        self,
        max_depth: int = 2,
        *,
        ema_alpha: float = 0.25,
        backoff_threshold: float = 0.42,
        initial_cooldown: int = 12,
        max_cooldown: int = 32,
    ) -> None:
        self.max_depth = max(0, int(max_depth))
        self.ema_alpha = float(ema_alpha)
        self.backoff_threshold = float(backoff_threshold)
        self.initial_cooldown = int(initial_cooldown)
        self.max_cooldown = int(max_cooldown)
        self.reset()

    def reset(self) -> None:
        self.cur_depth = self.max_depth
        self.ema_accept = 0.65
        self.consec_zero = 0
        self.consec_full = 0
        self.cooldown_left = 0
        self.cur_cooldown = self.initial_cooldown
        self.probing = False

    def next_depth(self, remaining_tokens: int) -> int:
        if self.max_depth <= 0 or remaining_tokens <= 0:
            return 0
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            return 0
        if self.probing:
            return min(2, self.max_depth, remaining_tokens)
        return min(self.cur_depth, remaining_tokens)

    def observe(self, depth_used: int, accepted: int) -> None:
        if depth_used <= 0:
            return
        rate = accepted / depth_used

        if self.probing:
            self.probing = False
            if accepted >= 1:
                self.cur_depth = min(2, self.max_depth)
                self.ema_accept = max(self.ema_accept, 0.55)
                self.consec_zero = 0
                self.consec_full = 1 if accepted == depth_used else 0
                self.cur_cooldown = self.initial_cooldown
            else:
                self.cooldown_left = self.cur_cooldown
                self.cur_cooldown = min(self.max_cooldown, self.cur_cooldown * 2)
                self.probing = True
            return

        self.ema_accept = (1.0 - self.ema_alpha) * self.ema_accept + self.ema_alpha * rate
        if accepted == 0:
            self.consec_zero += 1
            self.consec_full = 0
        else:
            self.consec_zero = 0
            if accepted == depth_used:
                self.consec_full += 1
            else:
                self.consec_full = 0

        if self.consec_zero >= 2 or self.ema_accept < self.backoff_threshold:
            if self.cur_depth > 2:
                self.cur_depth = max(2, self.cur_depth - 1)
                self.consec_zero = 0
            else:
                self.cooldown_left = self.cur_cooldown
                self.cur_cooldown = min(self.max_cooldown, self.cur_cooldown * 2)
                self.probing = True
                self.consec_zero = 0
        elif self.consec_full >= 2 and self.cur_depth < self.max_depth:
            self.cur_depth += 1
            self.consec_full = 0

