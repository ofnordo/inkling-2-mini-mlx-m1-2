# Copyright © 2026 Apple Inc.

import math
from functools import partial
from typing import Any

import mlx.core as mx
from mlx import nn
from mlx.nn.layers.distributed import sum_gradients

from mlx_lm.kernels.expert_routing import expert_routing
from mlx_lm.layers.moe import SwitchLinear, _gather_sort, _scatter_unsort
from mlx_lm.models.cache import BatchRotatingKVCache, RotatingKVCache
from mlx_lm.models.inkling2.config import TextArgs
from mlx_lm.models.inkling2.kernels import (
    depth_attention,
    dynamic_conv,
    moe_short_conv,
    qkv_post,
    relative_attention,
    relative_attention_splitk,
    short_conv,
)

__all__ = [
    "Attention",
    "BlockAttentionResidual",
    "DecoderLayer",
    "DynamicShortConvolution",
    "Experts",
    "FullAttention",
    "FullDecoderLayer",
    "L2Clip",
    "MessageGate",
    "MessageInput",
    "RelativeLogits",
    "ScaledL2Clip",
    "ShortConvolution",
    "SlidingAttention",
    "SlidingDecoderLayer",
    "SoftcappedSwiGLU",
    "SparseMoeBlock",
    "TopKRouter",
    "fuse_linears",
    "unweighted_rms_norm",
]


def fuse_linears(modules: list[nn.Module]) -> nn.Module | None:
    """Concatenate bias-free Linear or QuantizedLinear modules along output rows."""
    if not modules:
        return None
    first = modules[0]
    if all(isinstance(m, nn.QuantizedLinear) and "bias" not in m for m in modules):
        gs, bits, mode = first.group_size, first.bits, first.mode
        if not all(
            m.group_size == gs and m.bits == bits and m.mode == mode
            and m.weight.shape[1] == first.weight.shape[1]
            for m in modules
        ):
            return None
        in_dims = first.scales.shape[1] * gs
        out_dims = sum(m.weight.shape[0] for m in modules)
        ql = nn.QuantizedLinear(
            in_dims, out_dims, bias=False, group_size=gs, bits=bits, mode=mode
        )
        ql.weight = mx.concatenate([m.weight for m in modules], axis=0)
        ql.scales = mx.concatenate([m.scales for m in modules], axis=0)
        if first.get("biases") is not None and all(m.get("biases") is not None for m in modules):
            ql.biases = mx.concatenate([m.biases for m in modules], axis=0)
        else:
            ql.biases = None
        mx.eval(ql.parameters())
        return ql
    if all(isinstance(m, nn.Linear) and "bias" not in m for m in modules):
        in_dims = first.weight.shape[1]
        if not all(m.weight.shape[1] == in_dims for m in modules):
            return None
        out_dims = sum(m.weight.shape[0] for m in modules)
        lin = nn.Linear(in_dims, out_dims, bias=False)
        lin.weight = mx.concatenate([m.weight for m in modules], axis=0)
        mx.eval(lin.parameters())
        return lin
    return None


_ROUTER_SCORE_EPS = 1e-20


def _use_kernels() -> bool:
    return mx.default_device() == mx.gpu and mx.metal.is_available()


_USE_SHORT_CONV_KERNEL = True
_USE_SPLITK_ATTN_KERNEL = True


def unweighted_rms_norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return mx.fast.rms_norm(x, None, eps)


@partial(mx.compile, shapeless=True)
def _l2_clip(hidden_states: mx.array, scale: mx.array, threshold: float) -> mx.array:
    states = hidden_states.astype(mx.float32) * scale
    norm = mx.sqrt((states * states).sum(axis=-1, keepdims=True))
    scaled = states * mx.minimum(threshold / (norm + 1e-6), 1.0)
    return scaled.astype(hidden_states.dtype)


@partial(mx.compile, shapeless=True)
def _softcapped_swiglu(
    gate: mx.array, up: mx.array, gate_softcap: float, up_softcap: float
) -> mx.array:
    input_dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    gate_capped = mx.tanh(gate / gate_softcap) * gate_softcap
    up_capped = mx.tanh(up / up_softcap) * up_softcap
    return (gate_capped * mx.sigmoid(gate) * up_capped).astype(input_dtype)


class L2Clip(nn.Module):
    def __init__(self, threshold: float):
        super().__init__()
        self.threshold = threshold

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, 1.0, self.threshold)


class ScaledL2Clip(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, threshold: float):
        super().__init__()
        self.threshold = threshold
        self.scale = mx.ones((num_heads, head_dim))

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return _l2_clip(hidden_states, self.scale, self.threshold)


class RelativeLogits(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.proj = mx.zeros((d_rel, rel_extent))

    def __call__(self, relative_states: mx.array) -> mx.array:
        return relative_states @ self.proj.astype(relative_states.dtype)


def _conv_input(
    hidden_states: mx.array,
    conv_state: mx.array | None,
    conv_mask: mx.array | None,
    kernel_size: int,
) -> mx.array:
    if conv_mask is not None:
        hidden_states = mx.where(conv_mask[..., None], hidden_states, 0)
    if conv_state is None:
        shape = (hidden_states.shape[0], kernel_size - 1, hidden_states.shape[-1])
        conv_state = mx.zeros(shape, hidden_states.dtype)
    return mx.concatenate([conv_state, hidden_states], axis=1)


def _conv_state(window: mx.array, lengths: mx.array | None, n: int) -> mx.array:
    if lengths is None:
        pos = mx.arange(window.shape[1] - n, window.shape[1])[None, :, None]
    else:
        ends = mx.clip(lengths, 0, window.shape[1] - n)
        pos = (ends[:, None] + mx.arange(n))[..., None]
    # A gather copies, so the state does not keep the input in memory.
    return mx.take_along_axis(window, pos, axis=1)


class ShortConvolution(nn.Module):
    def __init__(self, dims: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(dims, dims, kernel_size, groups=dims, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
        *,
        return_window: bool = True,
        residual: mx.array | None = None,
        out_dtype: mx.Dtype | None = None,
    ) -> tuple[mx.array, mx.array]:
        K = self.kernel_size
        if (
            _USE_SHORT_CONV_KERNEL
            and _use_kernels()
            and conv_mask is None
            and hidden_states.dtype in (mx.bfloat16, mx.float16)
            and self.conv1d.weight.dtype == hidden_states.dtype
        ):
            out, next_state = short_conv(
                hidden_states,
                conv_state,
                self.conv1d.weight,
                1.0 / math.sqrt(K + 1),
                residual=residual,
                out_dtype=out_dtype,
            )
            if not return_window:
                return out, next_state
            return out, _conv_input(hidden_states, conv_state, None, K)
        res_in = hidden_states.astype(mx.float32)
        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        # As the reference, the conv runs in the weight dtype.
        out = self.conv1d(window.astype(self.conv1d.weight.dtype))
        out = ((out.astype(mx.float32) + res_in) / math.sqrt(K + 1)).astype(
            hidden_states.dtype
        )
        if residual is not None:
            out = residual.astype(mx.float32) + out.astype(mx.float32)
        return out.astype(out_dtype or hidden_states.dtype), window


class DynamicShortConvolution(nn.Module):
    def __init__(self, dims: int, input_dims: int, num_heads: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.static_weight = mx.zeros((kernel_size, dims))
        self.weight_proj = nn.Linear(input_dims, num_heads * kernel_size, bias=False)

    def __call__(
        self,
        hidden_states: mx.array,
        dynamic_input: mx.array,
        conv_state: mx.array | None,
        conv_mask: mx.array | None,
        *,
        return_window: bool = True,
    ) -> tuple[mx.array, mx.array]:
        B, L, C = hidden_states.shape
        K, H = self.kernel_size, self.num_heads
        dyn_raw = self.weight_proj(dynamic_input)
        if (
            _USE_SHORT_CONV_KERNEL
            and _use_kernels()
            and conv_mask is None
            and hidden_states.dtype in (mx.bfloat16, mx.float16)
            and self.static_weight.dtype == hidden_states.dtype
        ):
            out, next_state = dynamic_conv(
                hidden_states,
                dyn_raw,
                self.static_weight,
                conv_state,
                H,
                1.0 / math.sqrt(2 * K + 1),
            )
            if not return_window:
                return out, next_state
            return out, _conv_input(hidden_states, conv_state, None, K)
        dynamic_weights = dyn_raw.astype(mx.float32).reshape(B, L, H, 1, K)
        static_weights = self.static_weight.astype(mx.float32).T.reshape(H, -1, K)
        weights = static_weights + dynamic_weights

        window = _conv_input(hidden_states, conv_state, conv_mask, K)
        windows = mx.stack(
            [window[:, K - 1 - lag : K - 1 - lag + L] for lag in range(K)], axis=-1
        )
        windows = windows.reshape(B, L, H, -1, K)
        out = (windows * weights).sum(axis=-1).reshape(B, L, C)
        out = (out + hidden_states.astype(mx.float32)) / math.sqrt(2 * K + 1)
        return out.astype(hidden_states.dtype), window


def _positions(offset, length: int) -> mx.array:
    pos = mx.arange(length)
    if isinstance(offset, mx.array):
        return offset[:, None] + pos
    return pos + offset


def _key_positions(cache: Any, num_keys: int, num_queries: int, end) -> mx.array:
    k = mx.arange(num_keys)
    if num_queries == 1 and isinstance(cache, (RotatingKVCache, BatchRotatingKVCache)):
        # A decode step writes in place, so the keys can be rotated.
        k = (k - cache._idx) % num_keys
    if isinstance(end, mx.array):
        return (end - num_keys)[:, None] + k
    return k + (end - num_keys)


def _attention_reference(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    position_bias: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    query_positions: mx.array,
    key_positions: mx.array,
) -> mx.array:
    B, Hq, L, D = query.shape
    Hkv, S = key.shape[1], key.shape[2]
    R = position_bias.shape[-1]
    distance = query_positions[..., :, None] - key_positions[..., None, :]
    visible = (distance >= 0) & (key_positions[..., None, :] >= 0)
    if window:
        visible = visible & (distance < window)
    if distance.ndim == 3:
        distance, visible = distance[:, None], visible[:, None]
    bias = mx.take_along_axis(
        position_bias.transpose(0, 2, 1, 3),
        mx.broadcast_to(mx.clip(distance, 0, R - 1), (B, Hq, L, S)),
        axis=-1,
    )
    bias = mx.where(distance < R, bias.astype(mx.float32), 0.0)
    attention_mask = mx.where(visible, bias, -mx.inf).reshape(B, Hkv, -1, L, S)

    query = query.astype(mx.float32).reshape(B, Hkv, -1, L, D)
    key = key.astype(mx.float32)[:, :, None]
    attn_weights = (query @ key.swapaxes(-1, -2)) * scale + attention_mask
    sinks = sinks.astype(mx.float32).transpose(0, 2, 1).reshape(B, Hkv, -1, L, 1)
    combined_logits = mx.concatenate([attn_weights, sinks], axis=-1)
    attn_weights = mx.softmax(combined_logits, axis=-1)[..., :S]
    attn_output = attn_weights.astype(value.dtype) @ value[:, :, None]
    return attn_output.reshape(B, Hq, L, D).transpose(0, 2, 1, 3)


class Attention(nn.Module):
    def __init__(self, args: TextArgs, *, sliding: bool):
        super().__init__()
        if sliding:
            self.num_heads = args.swa_num_attention_heads
            self.num_key_value_heads = args.swa_num_key_value_heads
            self.head_dim = args.swa_head_dim
            self.sliding_window = args.sliding_window_size
            rel_extent = args.sliding_window_size
        else:
            self.num_heads = args.num_attention_heads
            self.num_key_value_heads = args.num_key_value_heads
            self.head_dim = args.head_dim
            self.sliding_window = 0
            rel_extent = args.rel_extent
        self.scaling = self.head_dim**-0.5

        dim = args.hidden_size
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(dim, q_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.r_proj = nn.Linear(dim, self.num_heads * args.d_rel, bias=False)
        self.o_proj = nn.Linear(q_dim, dim, bias=False)
        self.gate_proj = nn.Linear(dim, q_dim, bias=False)
        self.sink_proj = nn.Linear(dim, self.num_heads, bias=False)
        self.rel_logits_proj = RelativeLogits(args.d_rel, rel_extent)
        self.q_l2_clip = ScaledL2Clip(
            self.num_heads, self.head_dim, args.qk_l2_clip_threshold
        )
        self.k_l2_clip = L2Clip(args.qk_l2_clip_threshold)
        self.rel_l2_clip = ScaledL2Clip(
            self.num_heads, args.d_rel, args.rel_l2_clip_threshold
        )
        self.eps = args.rms_norm_eps

    def _project_qrgs(
        self, hidden_states: mx.array
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if getattr(self, "_qrgs_proj", None) is not None:
            return mx.split(self._qrgs_proj(hidden_states), self._qrgs_splits, axis=-1)
        return (
            self.q_proj(hidden_states),
            self.r_proj(hidden_states),
            self.gate_proj(hidden_states),
            self.sink_proj(hidden_states),
        )

    def _attend(
        self,
        hidden_states: mx.array,
        query_states: mx.array,
        key_states: mx.array,
        value_states: mx.array,
        relative_states: mx.array,
        cache: Any | None,
        tau: mx.array | None = None,
        gate: mx.array | None = None,
        sinks: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = hidden_states.shape
        if gate is None:
            gate = self.gate_proj(hidden_states)
        if sinks is None:
            sinks = self.sink_proj(hidden_states)

        if (
            _USE_SHORT_CONV_KERNEL
            and _use_kernels()
            and self.head_dim == 128
            and query_states.dtype in (mx.bfloat16, mx.float16)
            and query_states.dtype
            == key_states.dtype
            == value_states.dtype
            == relative_states.dtype
        ):
            query_states, key_states, value_states, relative_states = qkv_post(
                query_states,
                key_states,
                value_states,
                relative_states,
                self.q_l2_clip.scale,
                self.rel_l2_clip.scale,
                self.q_l2_clip.threshold,
                self.rel_l2_clip.threshold,
                self.eps,
                tau,
                self.num_heads,
                self.num_key_value_heads,
            )
            position_bias = self.rel_logits_proj(relative_states)
        else:
            query_states = query_states.reshape(B, L, self.num_heads, -1)
            key_states = key_states.reshape(B, L, self.num_key_value_heads, -1)
            value_states = value_states.reshape(B, L, self.num_key_value_heads, -1)
            relative_states = relative_states.reshape(B, L, self.num_heads, -1)
            query_states = self.q_l2_clip(query_states)
            key_states = self.k_l2_clip(key_states)
            value_states = unweighted_rms_norm(
                value_states.astype(mx.float32), self.eps
            ).astype(value_states.dtype)
            relative_states = self.rel_l2_clip(relative_states)
            if tau is not None:
                query_states = (query_states.astype(mx.float32) * tau).astype(
                    query_states.dtype
                )
                relative_states = (relative_states.astype(mx.float32) * tau).astype(
                    relative_states.dtype
                )
            position_bias = self.rel_logits_proj(relative_states)

            query_states = query_states.transpose(0, 2, 1, 3)
            key_states = key_states.transpose(0, 2, 1, 3)
            value_states = value_states.transpose(0, 2, 1, 3)
        if cache is not None:
            key_states, value_states = cache.update_and_fetch(key_states, value_states)
        attn_output = self._attention(
            query_states, key_states, value_states, position_bias, sinks, cache
        )

        attn_output = attn_output.reshape(B, L, -1).astype(mx.float32)
        attn_output = attn_output * mx.sigmoid(gate.astype(mx.float32))
        return self.o_proj(attn_output.astype(gate.dtype))

    def _attention(self, query, key, value, position_bias, sinks, cache) -> mx.array:
        B, _, L, _ = query.shape
        S = key.shape[2]
        end = cache.offset if cache is not None else L
        half_or_bf16 = (
            query.dtype == key.dtype == value.dtype
            and query.dtype in (mx.bfloat16, mx.float16)
        )
        if _use_kernels() and half_or_bf16 and (L > 1 or B > 1):
            if (
                _USE_SPLITK_ATTN_KERNEL
                and L <= 8
                and self.head_dim == 128
                and self.num_heads % self.num_key_value_heads == 0
            ):
                key_start = mx.broadcast_to(S - end, (B,))
                rot_idx = (
                    cache._idx
                    if (
                        L == 1
                        and isinstance(cache, (RotatingKVCache, BatchRotatingKVCache))
                    )
                    else S
                )
                return relative_attention_splitk(
                    query,
                    key,
                    value,
                    position_bias,
                    sinks,
                    scale=self.scaling,
                    window=self.sliding_window,
                    key_start=key_start,
                    rot_idx=rot_idx,
                )
            if L > 1:
                key_start = mx.broadcast_to(S - end, (B,))
                return relative_attention(
                    query,
                    key,
                    value,
                    position_bias,
                    sinks,
                    scale=self.scaling,
                    window=self.sliding_window,
                    key_start=key_start,
                )

        query_positions = _positions(end - L, L)
        key_positions = _key_positions(cache, S, L, end)
        if L == 1 and B == 1:
            # The fused attention takes the sinks of one row.
            distance = query_positions.reshape(1) - key_positions.reshape(S)
            R = position_bias.shape[-1]
            bias = mx.take(position_bias[0, 0], mx.clip(distance, 0, R - 1), axis=-1)
            bias = mx.where(distance < R, bias, mx.zeros_like(bias))
            visible = (distance >= 0) & (key_positions.reshape(S) >= 0)
            if self.sliding_window:
                visible = visible & (distance < self.sliding_window)
            mask = mx.where(visible, bias, -mx.inf).astype(query.dtype)
            attn_output = mx.fast.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=self.scaling,
                mask=mask[None, :, None, :],
                sinks=sinks[0, 0].astype(query.dtype),
            )
            return attn_output.transpose(0, 2, 1, 3)

        return _attention_reference(
            query,
            key,
            value,
            position_bias,
            sinks,
            scale=self.scaling,
            window=self.sliding_window,
            query_positions=query_positions,
            key_positions=key_positions,
        )


class SlidingAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=True)
        dims = self.num_key_value_heads * self.head_dim
        self.k_sconv = ShortConvolution(dims, args.sconv_kernel_size)
        self.v_sconv = ShortConvolution(dims, args.sconv_kernel_size)

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
        return_window: bool = True,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        key_states, conv_states[0] = self.k_sconv(
            self.k_proj(key_hidden_states),
            conv_states[0],
            conv_mask,
            return_window=return_window,
        )
        value_states, conv_states[1] = self.v_sconv(
            self.v_proj(value_hidden_states),
            conv_states[1],
            conv_mask,
            return_window=return_window,
        )
        query_states, relative_states, gate, sinks = self._project_qrgs(hidden_states)
        return self._attend(
            hidden_states,
            query_states,
            key_states,
            value_states,
            relative_states,
            kv_cache,
            gate=gate,
            sinks=sinks,
        )


class FullAttention(Attention):
    def __init__(self, args: TextArgs):
        super().__init__(args, sliding=False)
        dims = self.num_key_value_heads * self.head_dim
        conv_args = (args.hidden_size, self.num_key_value_heads, args.sconv_kernel_size)
        self.k_sconv = DynamicShortConvolution(dims, *conv_args)
        self.v_sconv = DynamicShortConvolution(dims, *conv_args)
        self.log_scaling_n_floor = args.log_scaling_n_floor
        self.log_scaling_alpha = args.log_scaling_alpha

    def __call__(
        self,
        hidden_states: mx.array,
        key_hidden_states: mx.array,
        value_hidden_states: mx.array,
        *,
        cache: Any | None,
        conv_states: list,
        conv_mask: mx.array | None,
        return_window: bool = True,
    ) -> mx.array:
        kv_cache = cache[0] if cache is not None else None
        key_states, conv_states[0] = self.k_sconv(
            self.k_proj(key_hidden_states),
            key_hidden_states,
            conv_states[0],
            conv_mask,
            return_window=return_window,
        )
        value_states, conv_states[1] = self.v_sconv(
            self.v_proj(value_hidden_states),
            value_hidden_states,
            conv_states[1],
            conv_mask,
            return_window=return_window,
        )

        offset = kv_cache.offset if kv_cache is not None else 0
        position_ids = _positions(offset, hidden_states.shape[1]).astype(mx.float32)
        tau = 1.0 + self.log_scaling_alpha * mx.log(
            mx.maximum((position_ids + 1) / self.log_scaling_n_floor, 1.0)
        )
        query_states, relative_states, gate, sinks = self._project_qrgs(hidden_states)
        return self._attend(
            hidden_states,
            query_states,
            key_states,
            value_states,
            relative_states,
            kv_cache,
            tau=tau[..., None, None],
            gate=gate,
            sinks=sinks,
        )


class SoftcappedSwiGLU(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate_softcap = args.mlp_gate_softcap
        self.up_softcap = args.mlp_up_softcap

    def __call__(self, gate: mx.array, up: mx.array) -> mx.array:
        return _softcapped_swiglu(gate, up, self.gate_softcap, self.up_softcap)


class Experts(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        dims, hidden = args.hidden_size, args.intermediate_size
        num_experts = args.n_routed_experts
        # The rows alternate between gate and up, as in the checkpoint.
        self.gate_up_proj = SwitchLinear(dims, 2 * hidden, num_experts, bias=False)
        self.down_proj = SwitchLinear(hidden, dims, num_experts, bias=False)
        self.act_fn = SoftcappedSwiGLU(args)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        x = mx.expand_dims(x, (-2, -3))
        indices = mx.stop_gradient(indices)
        do_sort = indices.size >= 64
        idx, inv_order = indices, None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)
        x = self.act_fn(gate_up[..., 0::2], gate_up[..., 1::2])
        x = self.down_proj(x, idx, sorted_indices=do_sort)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


class TopKRouter(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.top_k = args.num_experts_per_tok
        self.routed_scaling_factor = args.route_scale
        num_experts, dims = args.n_routed_experts, args.hidden_size
        self.weight = mx.zeros((num_experts, dims), dtype=mx.float32)
        self.e_score_correction_bias = mx.zeros((num_experts,), dtype=mx.float32)

    def __call__(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        logits = hidden_states.astype(mx.float32) @ self.weight.T
        if _use_kernels():
            return expert_routing(
                logits,
                self.e_score_correction_bias,
                top_k=self.top_k,
                multiplier=1.0,
                score_eps=_ROUTER_SCORE_EPS,
                route_scale=self.routed_scaling_factor,
            )
        scores = mx.sigmoid(logits)
        biased = scores + self.e_score_correction_bias
        k = self.top_k
        indices = mx.argpartition(-biased, kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + _ROUTER_SCORE_EPS)
        return indices, weights * self.routed_scaling_factor


class SparseMoeBlock(nn.Module):
    def __init__(self, args: TextArgs):
        super().__init__()
        self.gate = TopKRouter(args)
        self.experts = Experts(args)
        self.sharding_group = None

    def unreduced(self, hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        selected_experts, routing_weights = self.gate(hidden_states)
        expert_states = self.experts(hidden_states, selected_experts)
        return expert_states, routing_weights

    def __call__(self, hidden_states: mx.array) -> mx.array:
        group = self.sharding_group
        if group is not None:
            hidden_states = sum_gradients(group)(hidden_states)
        selected_experts, routing_weights = self.gate(hidden_states)
        expert_states = self.experts(hidden_states, selected_experts)
        expert_states = expert_states.astype(mx.float32) * routing_weights[..., None]
        final_hidden_states = expert_states.sum(axis=-2)
        if group is not None:
            final_hidden_states = mx.distributed.all_sum(
                final_hidden_states, group=group
            )
        return final_hidden_states.astype(hidden_states.dtype)


class BlockAttentionResidual(nn.Module):
    def __init__(self, dims: int, num_queries: int = 1):
        super().__init__()
        shape = (dims,) if num_queries == 1 else (num_queries, dims)
        self.query = mx.zeros(shape, dtype=mx.float32)

    def __call__(
        self, block_states: mx.array, partial_block_states: mx.array
    ) -> mx.array:
        dims = partial_block_states.shape[-1]
        return _depth_attention(
            block_states, partial_block_states, self.query.reshape(-1, dims)
        )


def _depth_attention(
    block_states: mx.array, partial_block_states: mx.array, query: mx.array
) -> mx.array:
    query = query.astype(mx.float32)
    scale = partial_block_states.shape[-1] ** -0.5
    if _use_kernels():
        return depth_attention(block_states, partial_block_states, query, scale=scale)
    stacked_states = mx.concatenate([block_states, partial_block_states[None]])
    stacked_states = mx.moveaxis(stacked_states, 0, -2)
    weights = mx.softmax((stacked_states @ query.T) * scale, axis=-2, precise=True)
    return weights.swapaxes(-1, -2) @ stacked_states


class MessageGate(nn.Module):
    def __init__(self, dims: int, rank: int):
        super().__init__()
        self.down_proj = nn.Linear(2 * dims, rank, bias=False)
        self.forget_gate_proj = nn.Linear(rank, dims, bias=False)
        self.message_gate_proj = nn.Linear(rank, dims, bias=False)

    def __call__(self, hidden_states: mx.array, message: mx.array) -> mx.array:
        states = mx.concatenate(
            [unweighted_rms_norm(hidden_states), unweighted_rms_norm(message)],
            axis=-1,
        )
        states = self.down_proj(states.astype(self.down_proj.weight.dtype))
        states = states.astype(mx.float32)
        if getattr(self, "_gates_proj", None) is not None:
            gates = mx.tanh(self._gates_proj(states))
            forget_gate, message_gate = mx.split(gates, 2, axis=-1)
        else:
            forget_gate = mx.tanh(self.forget_gate_proj(states))
            message_gate = mx.tanh(self.message_gate_proj(states))
        return (1 - forget_gate) * hidden_states + message_gate * message


class MessageInput(nn.Module):
    def __init__(self, dims: int, num_looped_layers: int):
        super().__init__()
        # The query, key, value and mlp queries of each looped layer.
        self.query = mx.zeros((num_looped_layers, 4, dims), mx.float32)

    def __call__(self, bank: tuple, loop_layer_idx: int) -> mx.array:
        return _depth_attention(*bank, self.query[loop_layer_idx])


class DecoderLayer(nn.Module):
    def __init__(self, args: TextArgs, layer_idx: int, *, num_queries: int):
        super().__init__()
        dims = args.hidden_size
        self.is_block_boundary = layer_idx % args.n_layers_per_attn_res_block == 0
        self.attention_input = (
            BlockAttentionResidual(dims, num_queries) if layer_idx > 0 else None
        )
        self.mlp_input = BlockAttentionResidual(dims)
        self.mlp = SparseMoeBlock(args)
        self.post_attention_layernorm = nn.RMSNorm(dims, eps=args.rms_norm_eps)
        kernel_size = args.sconv_kernel_size
        self.post_attention_convolution = ShortConvolution(dims, kernel_size)
        self.post_mlp_convolution = ShortConvolution(dims, kernel_size)
        if layer_idx in args.loop_core:
            rank = args.loop_config["gate_rank"]
            self.q_message_gate = MessageGate(dims, rank)
            self.k_message_gate = MessageGate(dims, rank)
            self.v_message_gate = MessageGate(dims, rank)
            self.mlp_message_gate = MessageGate(dims, rank)

    def _mix_messages(self, states: tuple, messages: mx.array) -> tuple:
        q_states, k_states, v_states = states
        return (
            self.q_message_gate(q_states, messages[..., 0, :]),
            self.k_message_gate(k_states, messages[..., 1, :]),
            self.v_message_gate(v_states, messages[..., 2, :]),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        block_states: mx.array | None,
        *,
        cache: Any | None = None,
        conv_mask: mx.array | None = None,
        messages: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        dtype = hidden_states.dtype
        conv = cache[1] if cache is not None else None
        conv_states = list(conv.cache) if conv is not None else [None] * 4
        return_window = (
            conv is None
            or conv.lengths is not None
            or (
                hidden_states.shape[1] > 1
                and getattr(conv, "record_verify_windows", False)
            )
        )
        block_residual = hidden_states
        normed_residual = unweighted_rms_norm(block_residual.astype(mx.float32))

        attention_inputs = None
        if self.attention_input is not None:
            attention_inputs = self.attention_input(block_states, normed_residual)
        hidden_states = self._self_attention(
            hidden_states,
            attention_inputs,
            cache,
            conv_states,
            conv_mask,
            messages,
            return_window=return_window,
        )
        if self.is_block_boundary:
            hidden_states, conv_states[2] = self.post_attention_convolution(
                hidden_states,
                conv_states[2],
                conv_mask,
                return_window=return_window,
                out_dtype=mx.float32,
            )
            residual = hidden_states
            block = normed_residual[None]
            if block_states is None:
                block_states = block
            else:
                block_states = mx.concatenate([block_states, block])
        else:
            hidden_states, conv_states[2] = self.post_attention_convolution(
                hidden_states,
                conv_states[2],
                conv_mask,
                return_window=return_window,
                residual=block_residual,
                out_dtype=mx.float32,
            )
            residual = hidden_states

        normed_states = unweighted_rms_norm(hidden_states)
        hidden_states = self.mlp_input(block_states, normed_states)[..., 0, :]
        if messages is not None:
            hidden_states = self.mlp_message_gate(hidden_states, messages[..., 3, :])
        mlp_in = self.post_attention_layernorm(hidden_states).astype(dtype)
        fused_moe_conv = (
            _USE_SHORT_CONV_KERNEL
            and _use_kernels()
            and conv_mask is None
            and not return_window
            and self.mlp.sharding_group is None
            and dtype in (mx.bfloat16, mx.float16)
            and self.post_mlp_convolution.conv1d.weight.dtype == dtype
        )
        if fused_moe_conv:
            expert_states, routing_weights = self.mlp.unreduced(mlp_in)
            hidden_states, conv_states[3] = moe_short_conv(
                expert_states,
                routing_weights,
                conv_states[3],
                self.post_mlp_convolution.conv1d.weight,
                residual,
                1.0 / math.sqrt(self.post_mlp_convolution.kernel_size + 1),
                dtype=dtype,
            )
        else:
            hidden_states = self.mlp(mlp_in)
            hidden_states, conv_states[3] = self.post_mlp_convolution(
                hidden_states,
                conv_states[3],
                conv_mask,
                return_window=return_window,
                residual=residual,
                out_dtype=dtype,
            )

        if conv is not None:
            n = self.post_mlp_convolution.kernel_size - 1
            if block_residual.shape[1] > 1 and getattr(
                conv, "record_verify_windows", False
            ):
                conv._verify_windows = list(conv_states)
            for i, window in enumerate(conv_states):
                conv[i] = (
                    window
                    if window.shape[1] == n
                    else _conv_state(window, conv.lengths, n)
                )
            conv.advance(block_residual.shape[1])
        return hidden_states, block_states


class SlidingDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=1)
        self.self_attn = SlidingAttention(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def _self_attention(
        self,
        hidden_states,
        attention_inputs,
        cache,
        conv_states,
        conv_mask,
        messages,
        return_window: bool = True,
    ):
        dtype = hidden_states.dtype
        if attention_inputs is not None:
            hidden_states = attention_inputs[..., 0, :]
        if messages is None:
            states = (self.input_layernorm(hidden_states).astype(dtype),) * 3
        else:
            states = self._mix_messages((hidden_states,) * 3, messages)
            states = [self.input_layernorm(s).astype(dtype) for s in states]
        return self.self_attn(
            *states,
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
            return_window=return_window,
        )


class FullDecoderLayer(DecoderLayer):
    def __init__(self, args: TextArgs, layer_idx: int):
        super().__init__(args, layer_idx, num_queries=3)
        self.self_attn = FullAttention(args)
        eps = args.rms_norm_eps
        self.q_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.k_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)
        self.v_input_layernorm = nn.RMSNorm(args.hidden_size, eps=eps)

    def _self_attention(
        self,
        hidden_states,
        attention_inputs,
        cache,
        conv_states,
        conv_mask,
        messages,
        return_window: bool = True,
    ):
        dtype = hidden_states.dtype
        if attention_inputs is None:
            query_states = key_states = value_states = hidden_states
        else:
            query_states, key_states, value_states = (
                attention_inputs[..., i, :] for i in range(3)
            )
        if messages is not None:
            query_states, key_states, value_states = self._mix_messages(
                (query_states, key_states, value_states), messages
            )
        return self.self_attn(
            self.q_input_layernorm(query_states).astype(dtype),
            self.k_input_layernorm(key_states).astype(dtype),
            self.v_input_layernorm(value_states).astype(dtype),
            cache=cache,
            conv_states=conv_states,
            conv_mask=conv_mask,
        )
