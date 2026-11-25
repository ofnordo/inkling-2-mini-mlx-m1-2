# Copyright © 2026 Apple Inc.

import mlx.core as mx

from mlx_lm.kernels.common import row_threads, scalar

__all__ = [
    "depth_attention",
    "dynamic_conv",
    "moe_short_conv",
    "qkv_post",
    "relative_attention",
    "relative_attention_splitk",
    "short_conv",
]


# Attention of learned queries over the depth sources of each token, in one
# kernel.
_DEPTH_ATTENTION_SOURCE = """
    constexpr int N = K + 1;
    uint row = threadgroup_position_in_grid.x;
    uint rows = threadgroups_per_grid.x;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;

    float acc[N * Q];
    for (int i = 0; i < N * Q; i++) {
        acc[i] = 0.0f;
    }
    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float qv = queries[j * D + d];
            for (int i = 0; i < N; i++) {
                acc[i * Q + j] += x[i] * qv;
            }
        }
    }

    threadgroup float buf[N * Q][TG / 32];
    for (int i = 0; i < N * Q; i++) {
        float v = simd_sum(acc[i]);
        if (lane == 0) {
            buf[i][sg] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float w[N * Q];
    for (int i = 0; i < N * Q; i++) {
        float t = 0.0f;
        for (int s = 0; s < TG / 32; s++) {
            t += buf[i][s];
        }
        w[i] = t * scale;
    }
    for (int j = 0; j < Q; j++) {
        float m = -INFINITY;
        for (int i = 0; i < N; i++) {
            m = max(m, w[i * Q + j]);
        }
        float z = 0.0f;
        for (int i = 0; i < N; i++) {
            w[i * Q + j] = metal::exp(w[i * Q + j] - m);
            z += w[i * Q + j];
        }
        for (int i = 0; i < N; i++) {
            w[i * Q + j] /= z;
        }
    }

    for (int d = tid; d < D; d += TG) {
        float x[N];
        for (int i = 0; i < K; i++) {
            x[i] = blocks[(size_t(i) * rows + row) * D + d];
        }
        x[K] = partial[size_t(row) * D + d];
        for (int j = 0; j < Q; j++) {
            float o = 0.0f;
            for (int i = 0; i < N; i++) {
                o += w[i * Q + j] * x[i];
            }
            out[(size_t(row) * Q + j) * D + d] = o;
        }
    }
"""

_depth_attention_kernel = mx.fast.metal_kernel(
    name="depth_attention",
    input_names=["blocks", "partial", "queries", "scale"],
    output_names=["out"],
    source=_DEPTH_ATTENTION_SOURCE,
)


def depth_attention(
    blocks: mx.array, partial: mx.array, queries: mx.array, *, scale: float
) -> mx.array:
    """blocks: (K, ..., D) and partial: (..., D), float32, also the values.
    queries: float32 (Q, D). Returns (..., Q, D)."""
    D = partial.shape[-1]
    Q = queries.shape[0]
    TG = row_threads(D)
    return _depth_attention_kernel(
        inputs=[blocks, partial, queries, scalar(scale)],
        template=[("K", blocks.shape[0]), ("D", D), ("Q", Q), ("TG", TG)],
        grid=(partial.size // D * TG, 1, 1),
        threadgroup=(TG, 1, 1),
        output_shapes=[partial.shape[:-1] + (Q, D)],
        output_dtypes=[mx.float32],
    )[0]


# Causal flash attention with a relative position bias and a sink logit per
# query.
# Queries per threadgroup, 8 for each of its 4 simdgroups.
_BQ = 32

_TG = 128

# The online softmax of each row starts from its sink logit, so the zero-value
# sink needs no key.
_RELATIVE_ATTENTION_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    constexpr int BK = 16;
    constexpr int LDK = BK + 8;
    constexpr int LDV = D + 8;
    const int L = sizes[0];
    const int S = sizes[1];
    const uint tid = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int batch = threadgroup_position_in_grid.z;
    const int head = threadgroup_position_in_grid.y;
    const int kv_head = head / (HQ / HKV);
    const int q0 = threadgroup_position_in_grid.x * BQ;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    // The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
    const short fm = ((lane / 4) & 4) + (lane / 2) % 4;
    const short fn = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
    const int row = q0 + int(sg) * 8 + fm;
    const bool row_ok = row < L;
    const int diag = S - L + row;

    // q and k are clipped and v has unit RMS, so half holds them.
    threadgroup half Ks[D * LDK];
    threadgroup half Vs[BK * LDV];

    simdgroup_half8x8 q_frag[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        half2 qv = half2(0.0h);
        if (row_ok) {
            const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
                + row * q_strides[2] + db * 8 + fn;
            if constexpr (IS_BF16) {
                qv = half2(float(qp[0]), float(qp[1]));
            } else {
                qv = half2(qp[0], qp[1]);
            }
        }
        q_frag[db].thread_elements()[0] = qv.x;
        q_frag[db].thread_elements()[1] = qv.y;
    }
    simdgroup_float8x8 acc[D / 8];
    for (int db = 0; db < D / 8; ++db) {
        acc[db] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float row_max = row_ok ? float(sinks[(batch * L + row) * HQ + head]) * LOG2E : 0.0f;
    float row_sum = 1.0f;
    const int bias_row = batch * L + (row_ok ? row : 0);
    const device T* bias = rel + (size_t(bias_row) * HQ + head) * R;

    const int first_diag = S - L + q0;
    const int last_diag = S - L + min(q0 + BQ, L) - 1;
    int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

    for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int c = tid; c < BK * D / 8; c += TG) {
            const int r = c / (D / 8);
            const int col = (c % (D / 8)) * 8;
            const int key = k0 + r;
            vec<T, 4> kv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            vec<T, 4> vv[2] = {vec<T, 4>(0), vec<T, 4>(0)};
            if (key < S) {
                auto kp = (const device vec<T, 4>*)(k_head + key * k_strides[2] + col);
                auto vp = (const device vec<T, 4>*)(v_head + key * v_strides[2] + col);
                kv[0] = kp[0];
                kv[1] = kp[1];
                vv[0] = vp[0];
                vv[1] = vp[1];
            }
            for (int j = 0; j < 8; ++j) {
                if constexpr (IS_BF16) {
                    Ks[(col + j) * LDK + r] = half(float(kv[j / 4][j % 4]));
                    Vs[r * LDV + col + j] = half(float(vv[j / 4][j % 4]));
                } else {
                    Ks[(col + j) * LDK + r] = kv[j / 4][j % 4];
                    Vs[r * LDV + col + j] = vv[j / 4][j % 4];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (q0 + int(sg) * 8 >= L) {
            continue;
        }

        simdgroup_float8x8 scores[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (int db = 0; db < D / 8; ++db) {
                simdgroup_half8x8 k_frag;
                simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
                simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
            }
        }

        float tile_max = -INFINITY;
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const int key = k0 + kb * 8 + fn + e;
                const int d = diag - key;
                const bool visible = row_ok && d >= 0 && key >= start
                    && (WINDOW == 0 || d < WINDOW);
                float s = -INFINITY;
                if (visible) {
                    s = scores[kb].thread_elements()[e] * qscale;
                    if (d < R) {
                        s += float(bias[d]) * LOG2E;
                    }
                }
                scores[kb].thread_elements()[e] = s;
                tile_max = max(tile_max, s);
            }
        }
        // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 8));
        const float new_max = max(row_max, tile_max);
        const float factor = fast::exp2(row_max - new_max);
        float tile_sum = 0.0f;
        simdgroup_half8x8 probs[BK / 8];
        for (int kb = 0; kb < BK / 8; ++kb) {
            for (int e = 0; e < 2; ++e) {
                const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
                probs[kb].thread_elements()[e] = half(p);
                tile_sum += p;
            }
        }
        tile_sum += simd_shuffle_xor(tile_sum, 1);
        tile_sum += simd_shuffle_xor(tile_sum, 8);
        row_sum = row_sum * factor + tile_sum;
        row_max = new_max;
        for (int db = 0; db < D / 8; ++db) {
            acc[db].thread_elements()[0] *= factor;
            acc[db].thread_elements()[1] *= factor;
            for (int kb = 0; kb < BK / 8; ++kb) {
                simdgroup_half8x8 v_frag;
                simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
                simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
            }
        }
    }

    if (row_ok) {
        device T* op = out + (size_t(batch * L + row) * HQ + head) * D;
        for (int db = 0; db < D / 8; ++db) {
            op[db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
            op[db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
        }
    }
"""

_relative_attention_kernel = mx.fast.metal_kernel(
    name="relative_attention",
    input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
    output_names=["out"],
    source=_RELATIVE_ATTENTION_SOURCE,
    ensure_row_contiguous=False,
)


def relative_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
) -> mx.array:
    """q: (B, Hq, L, D), k and v: (B, Hkv, S, D), all bfloat16 or float16, with the queries
    at the last L keys. rel: (B, L, Hq, R), the bias at distances below R.
    sinks: (B, L, Hq). key_start: (B,), the first key of each row. Returns
    (B, L, Hq, D)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    return _relative_attention_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            mx.array([L, S], dtype=mx.int32),
            scalar(scale),
        ],
        template=[
            ("T", q.dtype),
            ("IS_BF16", int(q.dtype == mx.bfloat16)),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("R", rel.shape[-1]),
            ("WINDOW", window),
            ("BQ", _BQ),
            ("TG", _TG),
        ],
        grid=((L + _BQ - 1) // _BQ * _TG, Hq, B),
        threadgroup=(_TG, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]


_SHORT_CONV_HEADER = """
template <typename T, typename S, typename X>
inline float inkling2_padded(
    const device S* srow, int64_t ss,
    const device X* xrow, int64_t xs,
    int P, int idx
) {
    if (idx < P) {
        return float(T(srow[idx * ss]));
    }
    return float(T(xrow[(idx - P) * xs]));
}

template <typename T, typename S, typename X>
inline float inkling2_padded_raw(
    const device S* srow, int64_t ss,
    const device X* xrow, int64_t xs,
    int P, int idx
) {
    if (idx < P) {
        return float(srow[idx * ss]);
    }
    return float(xrow[(idx - P) * xs]);
}

template <typename T, typename S, typename X>
inline float inkling2_moe_padded(
    const device S* srow, int64_t ss,
    const device X* zcol, const device float* wt,
    int C, int KE, int P, int idx
) {
    if (idx < P) {
        return float(srow[idx * ss]);
    }
    const int t = idx - P;
    const device X* zt = zcol + (size_t)t * KE * C;
    const device float* w = wt + (size_t)t * KE;
    float sum = 0.0f;
    for (int e = 0; e < KE; ++e) {
        sum += float(zt[(size_t)e * C]) * w[e];
    }
    return float(T(sum));
}
"""

_SHORT_CONV_SOURCE = """
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int t = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = x_shape[1];
    const int C = x_shape[2];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const device X* xrow = x + b * x_strides[0] + c * x_strides[2];
    const int64_t ss = state_strides[1];
    const int64_t xs = x_strides[1];

    const device W* wrow = weight + c * weight_strides[0];
    const int64_t ws = weight_strides[1];
    float acc = 0.0f;
    for (int j = 0; j < K; ++j) {
        acc += float(wrow[j * ws]) * inkling2_padded<T>(srow, ss, xrow, xs, P, t + j);
    }
    const float own = inkling2_padded_raw<T>(srow, ss, xrow, xs, P, t + P);
    const float conv = float(W(acc));
    float val = float(T((conv + own) * inv_scale));
    if (HAS_RESIDUAL) {
        val = float(residual[b * residual_strides[0] + t * residual_strides[1]
                             + c * residual_strides[2]]) + val;
    }
    out[((size_t)b * L + t) * C + c] = O(val);

    if (t == L - 1) {
        device float* st_out = next_state + ((size_t)b * P) * C + c;
        for (int i = 0; i < P; ++i) {
            st_out[i * C] = inkling2_padded_raw<T>(srow, ss, xrow, xs, P, L + i);
        }
    }
"""

_short_conv_kernel = mx.fast.metal_kernel(
    name="inkling2_short_conv_v2",
    input_names=["x", "state", "weight", "residual", "inv_scale"],
    output_names=["out", "next_state"],
    header=_SHORT_CONV_HEADER,
    source=_SHORT_CONV_SOURCE,
    ensure_row_contiguous=False,
)


def short_conv(
    x: mx.array,
    state: mx.array | None,
    weight: mx.array,
    inv_scale: float,
    *,
    residual: mx.array | None = None,
    out_dtype: mx.Dtype | None = None,
) -> tuple[mx.array, mx.array]:
    """Fused causal depthwise short convolution + residual + state update for any L >= 1."""
    B, L, C = x.shape
    w = weight.reshape(C, -1)
    K = w.shape[1]
    P = K - 1
    if state is None:
        state = mx.zeros((B, P, C), dtype=mx.float32)
    has_residual = residual is not None
    res_in = residual if has_residual else x
    o_dtype = out_dtype if out_dtype is not None else x.dtype
    return _short_conv_kernel(
        inputs=[x, state, w, res_in, scalar(inv_scale)],
        template=[
            ("T", x.dtype),
            ("X", x.dtype),
            ("W", w.dtype),
            ("S", state.dtype),
            ("O", o_dtype),
            ("K", K),
            ("HAS_RESIDUAL", has_residual),
        ],
        grid=(C, L, B),
        threadgroup=(min(256, C), 1, 1),
        output_shapes=[(B, L, C), (B, P, C)],
        output_dtypes=[o_dtype, mx.float32],
    )


_MOE_SHORT_CONV_SOURCE = """
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int t = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = z_shape[1];
    const int C = z_shape[3];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const int64_t ss = state_strides[1];
    const device X* zcol = z + (size_t)b * L * KE * C + c;
    const device float* wt = moe_w + (size_t)b * L * KE;

    const device W* wrow = weight + c * weight_strides[0];
    const int64_t ws = weight_strides[1];
    float acc = 0.0f;
    float own = 0.0f;
    for (int j = 0; j < K; ++j) {
        const float xj = inkling2_moe_padded<T>(srow, ss, zcol, wt, C, KE, P, t + j);
        acc += float(wrow[j * ws]) * float(T(xj));
        if (j == P) {
            own = xj;
        }
    }
    const float conv = float(W(acc));
    float val = float(T((conv + own) * inv_scale));
    val = float(residual[b * residual_strides[0] + t * residual_strides[1]
                         + c * residual_strides[2]]) + val;
    out[((size_t)b * L + t) * C + c] = O(val);

    if (t == L - 1) {
        device float* st_out = next_state + ((size_t)b * P) * C + c;
        for (int i = 0; i < P; ++i) {
            st_out[i * C] = inkling2_moe_padded<T>(srow, ss, zcol, wt, C, KE, P, L + i);
        }
    }
"""

_moe_short_conv_kernel = mx.fast.metal_kernel(
    name="inkling2_moe_short_conv",
    input_names=["z", "moe_w", "state", "weight", "residual", "inv_scale"],
    output_names=["out", "next_state"],
    header=_SHORT_CONV_HEADER,
    source=_MOE_SHORT_CONV_SOURCE,
    ensure_row_contiguous=False,
)


def moe_short_conv(
    z: mx.array,
    moe_weights: mx.array,
    state: mx.array | None,
    weight: mx.array,
    residual: mx.array,
    inv_scale: float,
    *,
    dtype: mx.Dtype,
) -> tuple[mx.array, mx.array]:
    """Fused MoE weighted expert reduction + post-MLP short conv + block residual add."""
    B, L, KE, C = z.shape
    w = weight.reshape(C, -1)
    K = w.shape[1]
    P = K - 1
    if state is None:
        state = mx.zeros((B, P, C), dtype=mx.float32)
    return _moe_short_conv_kernel(
        inputs=[
            mx.contiguous(z),
            mx.contiguous(moe_weights.astype(mx.float32)),
            state,
            w,
            residual,
            scalar(inv_scale),
        ],
        template=[
            ("T", dtype),
            ("X", z.dtype),
            ("W", w.dtype),
            ("S", state.dtype),
            ("O", dtype),
            ("K", K),
            ("KE", KE),
        ],
        grid=(C, L, B),
        threadgroup=(min(256, C), 1, 1),
        output_shapes=[(B, L, C), (B, P, C)],
        output_dtypes=[dtype, mx.float32],
    )


_DYNAMIC_CONV_SOURCE = """
    constexpr int P = K - 1;
    const int c = thread_position_in_grid.x;
    const int t = thread_position_in_grid.y;
    const int b = thread_position_in_grid.z;
    const int L = x_shape[1];
    const int C = x_shape[2];
    const device S* srow = state + b * state_strides[0] + c * state_strides[2];
    const device T* xrow = x + b * x_strides[0] + c * x_strides[2];
    const int64_t ss = state_strides[1];
    const int64_t xs = x_strides[1];

    const int head = c / (C / H);
    const device DYN* drow = dyn + b * dyn_strides[0] + t * dyn_strides[1]
        + head * K * dyn_strides[2];
    const int64_t ds = dyn_strides[2];

    float acc = 0.0f;
    for (int lag = 0; lag < K; ++lag) {
        const float sw = float(static_w[lag * static_w_strides[0] + c * static_w_strides[1]]);
        const float tap = sw + float(drow[lag * ds]);
        acc += inkling2_padded_raw<T>(srow, ss, xrow, xs, P, t + P - lag) * tap;
    }
    const float own = inkling2_padded_raw<T>(srow, ss, xrow, xs, P, t + P);
    out[((size_t)b * L + t) * C + c] = T((acc + own) * inv_scale);

    if (t == L - 1) {
        device float* st_out = next_state + ((size_t)b * P) * C + c;
        for (int i = 0; i < P; ++i) {
            st_out[i * C] = inkling2_padded_raw<T>(srow, ss, xrow, xs, P, L + i);
        }
    }
"""

_dynamic_conv_kernel = mx.fast.metal_kernel(
    name="inkling2_dynamic_conv",
    input_names=["x", "dyn", "static_w", "state", "inv_scale"],
    output_names=["out", "next_state"],
    header=_SHORT_CONV_HEADER,
    source=_DYNAMIC_CONV_SOURCE,
    ensure_row_contiguous=False,
)


def dynamic_conv(
    x: mx.array,
    dyn: mx.array,
    static_w: mx.array,
    state: mx.array | None,
    num_heads: int,
    inv_scale: float,
) -> tuple[mx.array, mx.array]:
    """Fused dynamic causal short convolution + residual + state update for full attention."""
    B, L, C = x.shape
    K = static_w.shape[0]
    P = K - 1
    if state is None:
        state = mx.zeros((B, P, C), dtype=mx.float32)
    return _dynamic_conv_kernel(
        inputs=[x, dyn, static_w, state, scalar(inv_scale)],
        template=[
            ("T", x.dtype),
            ("DYN", dyn.dtype),
            ("S", state.dtype),
            ("K", K),
            ("H", num_heads),
        ],
        grid=(C, L, B),
        threadgroup=(min(256, C), 1, 1),
        output_shapes=[(B, L, C), (B, P, C)],
        output_dtypes=[x.dtype, mx.float32],
    )


# Fused Q/K/V/R post-projection epilogue:
#   q -> ScaledL2Clip (+ optional tau) -> head-major (B, Hq, L, D)
#   k -> L2Clip                        -> head-major (B, Hkv, L, D)
#   v -> unweighted_rms_norm           -> head-major (B, Hkv, L, D)
#   r -> ScaledL2Clip (+ optional tau) -> token-major (B, L, Hq, R)
# One 32-thread simdgroup per head/task per token (assumes D == 128).
_QKV_POST_SOURCE = """
    const uint lane = thread_index_in_simdgroup;
    const uint task = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const int L = q_shape[1];
    const int b = row / L;
    const int t = row % L;

    if (task < HQ + HKV) {
        const bool is_q = task < HQ;
        const int head = is_q ? int(task) : int(task) - HQ;
        const int col = head * D + int(lane) * 4;
        float xf[4];
        if (is_q) {
            const device T* src = q + b * q_strides[0] + t * q_strides[1];
            const int64_t cs = q_strides[2];
            for (int i = 0; i < 4; ++i) {
                xf[i] = float(src[(col + i) * cs]) * float(q_scale[col + i]);
            }
        } else {
            const device T* src = k + b * k_strides[0] + t * k_strides[1];
            const int64_t cs = k_strides[2];
            for (int i = 0; i < 4; ++i) {
                xf[i] = float(src[(col + i) * cs]);
            }
        }
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            acc += xf[i] * xf[i];
        }
        const float norm = metal::precise::sqrt(simd_sum(acc));
        const float f = metal::min(params[0] / (norm + 1e-6f), 1.0f);
        const int heads = is_q ? HQ : HKV;
        device T* dst = (is_q ? qo : ko)
            + (((size_t)b * heads + head) * L + t) * D + lane * 4;
        const float tau_v = (HAS_TAU && is_q)
            ? float(tau[b * tau_strides[0] + t * tau_strides[1]])
            : 1.0f;
        for (int i = 0; i < 4; ++i) {
            T o = T(xf[i] * f);
            if (HAS_TAU && is_q) {
                o = T(float(o) * tau_v);
            }
            dst[i] = o;
        }
    } else if (task < HQ + 2 * HKV) {
        const int head = int(task) - HQ - HKV;
        const int col = head * D + int(lane) * 4;
        const device T* src = v + b * v_strides[0] + t * v_strides[1];
        const int64_t cs = v_strides[2];
        float xv[4];
        float acc = 0.0f;
        for (int i = 0; i < 4; ++i) {
            xv[i] = float(src[(col + i) * cs]);
            acc = metal::fma(xv[i], xv[i], acc);
        }
        const float mean_sq = simd_sum(acc) / float(D);
        const float inv = metal::precise::rsqrt(mean_sq + params[2]);
        device T* dst = vo + (((size_t)b * HKV + head) * L + t) * D + lane * 4;
        for (int i = 0; i < 4; ++i) {
            dst[i] = T(xv[i] * inv);
        }
    } else {
        const int head = (int(task) - HQ - 2 * HKV) * 32 + int(lane);
        if (head >= HQ) {
            return;
        }
        const device T* src = r + b * r_strides[0] + t * r_strides[1];
        const int64_t cs = r_strides[2];
        float xf[R];
        float acc = 0.0f;
        for (int j = 0; j < R; ++j) {
            xf[j] = float(src[(head * R + j) * cs]) * float(r_scale[head * R + j]);
            acc += xf[j] * xf[j];
        }
        const float norm = metal::precise::sqrt(acc);
        const float f = metal::min(params[1] / (norm + 1e-6f), 1.0f);
        device T* dst = ro + ((size_t)row * HQ + head) * R;
        const float tau_v = HAS_TAU
            ? float(tau[b * tau_strides[0] + t * tau_strides[1]])
            : 1.0f;
        for (int j = 0; j < R; ++j) {
            T o = T(xf[j] * f);
            if (HAS_TAU) {
                o = T(float(o) * tau_v);
            }
            dst[j] = o;
        }
    }
"""

_qkv_post_kernel = mx.fast.metal_kernel(
    name="inkling2_qkv_post",
    input_names=["q", "k", "v", "r", "q_scale", "r_scale", "tau", "params"],
    output_names=["qo", "ko", "vo", "ro"],
    source=_QKV_POST_SOURCE,
    ensure_row_contiguous=False,
)


def qkv_post(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    r: mx.array,
    q_scale: mx.array,
    r_scale: mx.array,
    qk_clip: float,
    r_clip: float,
    eps: float,
    tau: mx.array | None,
    num_heads: int,
    num_kv_heads: int,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    """Fused Q/K/R L2-clip + V unit RMSNorm + optional tau + head-major transpose."""
    B, L, q_dim = q.shape
    D = q_dim // num_heads
    R = r.shape[-1] // num_heads
    assert D == 128
    has_tau = tau is not None
    tau_in = (
        mx.broadcast_to(tau.reshape(-1, L), (B, L)).astype(mx.float32)
        if has_tau
        else mx.zeros((1, 1), dtype=mx.float32)
    )
    tasks = num_heads + 2 * num_kv_heads + (num_heads + 31) // 32
    params = mx.array([qk_clip, r_clip, eps], dtype=mx.float32)
    return _qkv_post_kernel(
        inputs=[
            q,
            k,
            v,
            r,
            mx.contiguous(q_scale.reshape(-1).astype(mx.float32)),
            mx.contiguous(r_scale.reshape(-1).astype(mx.float32)),
            tau_in,
            params,
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", num_heads),
            ("HKV", num_kv_heads),
            ("R", R),
            ("HAS_TAU", has_tau),
        ],
        grid=(tasks * 32, B * L, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[
            (B, num_heads, L, D),
            (B, num_kv_heads, L, D),
            (B, num_kv_heads, L, D),
            (B, L, num_heads, R),
        ],
        output_dtypes=[q.dtype, k.dtype, v.dtype, r.dtype],
    )


_RELATIVE_SPLITK_SOURCE = """
    constexpr float LOG2E = 1.4426950408889634f;
    const int L = sizes[0];
    const int S = sizes[1];
    const int rot_idx = sizes[2];
    const int split = threadgroup_position_in_grid.x;
    const int row_kv = threadgroup_position_in_grid.y;
    const int row = row_kv / HKV;
    const int kv_head = row_kv % HKV;
    const int batch = threadgroup_position_in_grid.z;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int start = key_start[batch];
    const float qscale = scale * LOG2E;

    const int diag = S - L + row;
    int first_key = WINDOW > 0 ? max(0, diag - WINDOW + 1) : 0;
    first_key = max(first_key, start);
    const int total_keys = max(0, diag - first_key + 1);
    const int chunk = (total_keys + NS - 1) / NS;
    const int k_start = first_key + split * chunk;
    const int k_end = min(diag + 1, k_start + chunk);

    float q_reg[G][4];
    float m[G];
    float l[G];
    float acc[G][4];
    for (int g = 0; g < G; ++g) {
        const int head = kv_head * G + g;
        const device T* qp = q + batch * q_strides[0] + head * q_strides[1]
            + row * q_strides[2] + lane * 4;
        q_reg[g][0] = float(qp[0]);
        q_reg[g][1] = float(qp[1]);
        q_reg[g][2] = float(qp[2]);
        q_reg[g][3] = float(qp[3]);
        if (split == 0 && sg == 0) {
            m[g] = float(sinks[(batch * L + row) * HQ + head]) * LOG2E;
            l[g] = 1.0f;
        } else {
            m[g] = -INFINITY;
            l[g] = 0.0f;
        }
        acc[g][0] = 0.0f;
        acc[g][1] = 0.0f;
        acc[g][2] = 0.0f;
        acc[g][3] = 0.0f;
    }

    const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
    const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];
    const size_t rel_row_base = size_t(batch * L + row) * HQ;

    for (int key = k_start + int(sg); key < k_end; key += 4) {
        int phys_key = key + rot_idx;
        if (phys_key >= S) {
            phys_key -= S;
        }
        const device T* kp = k_head + phys_key * k_strides[2] + lane * 4;
        const device T* vp = v_head + phys_key * v_strides[2] + lane * 4;
        const float k0 = float(kp[0]);
        const float k1 = float(kp[1]);
        const float k2 = float(kp[2]);
        const float k3 = float(kp[3]);
        const float v0 = float(vp[0]);
        const float v1 = float(vp[1]);
        const float v2 = float(vp[2]);
        const float v3 = float(vp[3]);
        const int dist = diag - key;

        for (int g = 0; g < G; ++g) {
            const int head = kv_head * G + g;
            const float dot = simd_sum(
                q_reg[g][0] * k0 + q_reg[g][1] * k1
                + q_reg[g][2] * k2 + q_reg[g][3] * k3
            );
            float s = dot * qscale;
            if (dist < R) {
                s += float(rel[(rel_row_base + head) * R + dist]) * LOG2E;
            }
            const float new_m = max(m[g], s);
            const float factor = (m[g] > -INFINITY) ? fast::exp2(m[g] - new_m) : 0.0f;
            const float p = fast::exp2(s - new_m);
            l[g] = l[g] * factor + p;
            acc[g][0] = acc[g][0] * factor + p * v0;
            acc[g][1] = acc[g][1] * factor + p * v1;
            acc[g][2] = acc[g][2] * factor + p * v2;
            acc[g][3] = acc[g][3] * factor + p * v3;
            m[g] = new_m;
        }
    }

    threadgroup float tg_m[4][G];
    threadgroup float tg_l[4][G];
    threadgroup float tg_acc[4][G][D];
    for (int g = 0; g < G; ++g) {
        if (lane == 0) {
            tg_m[sg][g] = m[g];
            tg_l[sg][g] = l[g];
        }
        for (int i = 0; i < 4; ++i) {
            tg_acc[sg][g][lane * 4 + i] = acc[g][i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        for (int g = 0; g < G; ++g) {
            const float m_all = max(
                max(tg_m[0][g], tg_m[1][g]),
                max(tg_m[2][g], tg_m[3][g])
            );
            float l_all = 0.0f;
            float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
            for (int s = 0; s < 4; ++s) {
                const float ms = tg_m[s][g];
                const float ls = tg_l[s][g];
                if (ls > 0.0f) {
                    const float w = fast::exp2(ms - m_all);
                    l_all += ls * w;
                    o0 += tg_acc[s][g][lane * 4 + 0] * w;
                    o1 += tg_acc[s][g][lane * 4 + 1] * w;
                    o2 += tg_acc[s][g][lane * 4 + 2] * w;
                    o3 += tg_acc[s][g][lane * 4 + 3] * w;
                }
            }
            const int head = kv_head * G + g;
            const size_t base_ml = (size_t(batch * L + row) * HQ + head) * NS + split;
            if (lane == 0) {
                partial_m[base_ml] = m_all;
                partial_l[base_ml] = l_all;
            }
            device float* po = partial_out + base_ml * D + lane * 4;
            po[0] = o0;
            po[1] = o1;
            po[2] = o2;
            po[3] = o3;
        }
    }
"""

_RELATIVE_SPLITK_REDUCE_SOURCE = """
    const int L = sizes[0];
    const int row_h = threadgroup_position_in_grid.y;
    const int row = row_h / HQ;
    const int head = row_h % HQ;
    const int batch = threadgroup_position_in_grid.z;
    const uint lane = thread_index_in_simdgroup;
    const size_t base_ml = (size_t(batch * L + row) * HQ + head) * NS;

    float m_all = -INFINITY;
    for (int s = 0; s < NS; ++s) {
        m_all = max(m_all, partial_m[base_ml + s]);
    }
    float l_all = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    for (int s = 0; s < NS; ++s) {
        const float ls = partial_l[base_ml + s];
        if (ls > 0.0f) {
            const float w = fast::exp2(partial_m[base_ml + s] - m_all);
            l_all += ls * w;
            const device float* po = partial_out + (base_ml + s) * D + lane * 4;
            o0 += po[0] * w;
            o1 += po[1] * w;
            o2 += po[2] * w;
            o3 += po[3] * w;
        }
    }
    device T* op = out + (size_t(batch * L + row) * HQ + head) * D + lane * 4;
    const float inv_l = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
    op[0] = T(o0 * inv_l);
    op[1] = T(o1 * inv_l);
    op[2] = T(o2 * inv_l);
    op[3] = T(o3 * inv_l);
"""

_relative_splitk_kernel = mx.fast.metal_kernel(
    name="relative_attention_splitk",
    input_names=["q", "k", "v", "rel", "sinks", "key_start", "sizes", "scale"],
    output_names=["partial_out", "partial_m", "partial_l"],
    source=_RELATIVE_SPLITK_SOURCE,
    ensure_row_contiguous=False,
)

_relative_splitk_reduce_kernel = mx.fast.metal_kernel(
    name="relative_attention_splitk_reduce",
    input_names=["partial_out", "partial_m", "partial_l", "sizes"],
    output_names=["out"],
    source=_RELATIVE_SPLITK_REDUCE_SOURCE,
)


def relative_attention_splitk(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    rel: mx.array,
    sinks: mx.array,
    *,
    scale: float,
    window: int,
    key_start: mx.array,
    rot_idx: int | None = None,
) -> mx.array:
    """Split-K GQA causal relative attention with sinks for small L (D == 128)."""
    B, Hq, L, D = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    assert D == 128 and Hq % Hkv == 0
    G = Hq // Hkv
    eff = min(S, window) if window > 0 else S
    if eff <= 128:
        ns = 1
    elif eff <= 1024:
        ns = 4
    elif eff <= 2560:
        ns = 8
    else:
        ns = 16

    r_idx = S if rot_idx is None else int(rot_idx)
    sizes = mx.array([L, S, r_idx], dtype=mx.int32)
    partial_out, partial_m, partial_l = _relative_splitk_kernel(
        inputs=[
            q,
            k,
            v,
            mx.contiguous(rel),
            mx.contiguous(sinks.astype(mx.float32)),
            mx.contiguous(key_start.astype(mx.int32)),
            sizes,
            scalar(scale),
        ],
        template=[
            ("T", q.dtype),
            ("D", D),
            ("HQ", Hq),
            ("HKV", Hkv),
            ("G", G),
            ("R", rel.shape[-1]),
            ("WINDOW", window),
            ("NS", ns),
        ],
        grid=(ns * 128, Hkv * L, B),
        threadgroup=(128, 1, 1),
        output_shapes=[(B, L, Hq, ns, D), (B, L, Hq, ns), (B, L, Hq, ns)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    return _relative_splitk_reduce_kernel(
        inputs=[partial_out, partial_m, partial_l, sizes],
        template=[("T", q.dtype), ("D", D), ("HQ", Hq), ("NS", ns)],
        grid=(32, Hq * L, B),
        threadgroup=(32, 1, 1),
        output_shapes=[(B, L, Hq, D)],
        output_dtypes=[q.dtype],
    )[0]
