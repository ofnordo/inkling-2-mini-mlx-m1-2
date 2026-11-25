"""Convert ThinkingMachines Inkling-2-Mini to MLX W4W8-Native-A8:
- Layers 0..19 MoE experts (trained in NVFP4) -> 4-bit affine (group_size=64)
- Layers 20..23 MoE experts (trained in MXFP8) -> 8-bit affine (group_size=64)
- Attention projections, embeddings, and lm_head (trained in BF16) -> 8-bit affine (group_size=64)
- Norms, short-conv, controllers, and AttnRes/router queries -> preserved in BF16/FP32
"""

from __future__ import annotations

import gc
import os
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path, tree_unflatten

from mlx_lm.layers.moe import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.utils import compute_bits_per_weight, load, save

from convert_w4a16 import dequantize_switch_module


SRC_PATH = Path(os.environ.get("INKLING_SRC", "Inkling-2-Mini"))
DST_PATH = Path(os.environ.get("INKLING_DST", "Inkling-2-Mini-MLX-W4W8"))
GROUP_SIZE = 64
DENSE_BITS = 8
MODE = "affine"


def requantize_switch_layers_native(model, group_size: int, mode: str) -> dict[str, dict]:
    """Re-quantize each QuantizedSwitchLinear preserving its native bit-width (4-bit for L0..19, 8-bit for L20..23)."""
    switch_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, QuantizedSwitchLinear)
    ]
    per_layer_quant: dict[str, dict] = {}
    print(
        f"[INFO] Re-quantizing {len(switch_modules)} QuantizedSwitchLinear tensors "
        f"(preserving 4-bit for L0..19 NVFP4 and 8-bit for L20..23 MXFP8) to {mode} (gs={group_size})..."
    )
    for idx, (name, module) in enumerate(switch_modules, 1):
        target_bits = int(module.bits)
        bias = "bias" in module
        weight = dequantize_switch_module(module)
        mx.eval(weight)
        num_experts, out_dims, in_dims = weight.shape

        dense = SwitchLinear(in_dims, out_dims, num_experts, bias=bias)
        dense.weight = weight
        if bias:
            dense.bias = module.bias.astype(mx.bfloat16)
        ql = dense.to_quantized(group_size=group_size, bits=target_bits, mode=mode)
        mx.eval(ql.parameters())
        model.update_modules(tree_unflatten([(name, ql)]))
        per_layer_quant[name] = {
            "group_size": group_size,
            "bits": target_bits,
            "mode": mode,
        }

        del weight, dense, ql
        if idx % 8 == 0 or idx == len(switch_modules):
            gc.collect()
            mx.clear_cache()
            print(
                f"  [{idx}/{len(switch_modules)}] {name} -> {target_bits}-bit {mode} "
                f"(active={mx.get_active_memory()/1e9:.2f} GB)"
            )
    return per_layer_quant


def main() -> None:
    t0 = time.perf_counter()
    print(f"[INFO] Loading {SRC_PATH} (lazy=True)...")
    model, tokenizer, config = load(str(SRC_PATH), lazy=True, return_config=True)

    # 1. Convert NVFP4 (4-bit) -> 4-bit affine and MXFP8 (8-bit) -> 8-bit affine one-by-one.
    requantize_switch_layers_native(model, GROUP_SIZE, MODE)

    # 2. Cast remaining floating parameters to bfloat16, respecting model.cast_predicate (keeps FP32 queries/router biases).
    cast_predicate = getattr(model, "cast_predicate", lambda _: True)

    def set_dtype(k: str, v: mx.array) -> mx.array:
        if k.endswith("global_scale"):
            return v
        if cast_predicate(k) and mx.issubdtype(v.dtype, mx.floating) and v.dtype != mx.bfloat16:
            return v.astype(mx.bfloat16)
        return v

    model.update(tree_map_with_path(set_dtype, model.parameters()))
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()

    # 3. Quantize remaining eligible dense layers (attention, embed_tokens, lm_head) to 8-bit affine.
    config.pop("quantization", None)
    config.pop("quantization_config", None)

    quant_pred = getattr(model, "quant_predicate", lambda p, m: True)

    def dense_predicate(path: str, module: nn.Module) -> bool:
        if isinstance(module, QuantizedSwitchLinear):
            return False
        if not hasattr(module, "to_quantized"):
            return False
        if module.weight.shape[-1] % GROUP_SIZE != 0:
            return False
        return bool(quant_pred(path, module))

    print(f"[INFO] Quantizing eligible dense layers (attention/embed/lm_head) to {DENSE_BITS}-bit {MODE} (gs={GROUP_SIZE})...")
    nn.quantize(
        model,
        group_size=GROUP_SIZE,
        bits=DENSE_BITS,
        mode=MODE,
        class_predicate=dense_predicate,
    )
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()

    bpw = compute_bits_per_weight(model)
    print(f"[INFO] Effective bits per weight: {bpw:.3f} bpw")

    # Record top-level quantization dict; mlx_lm.utils.load_model infers per-tensor (4-bit vs 8-bit) from weight/scales shapes.
    quant_cfg = {"group_size": GROUP_SIZE, "bits": 4, "mode": MODE}
    config["quantization"] = quant_cfg
    config["quantization_config"] = quant_cfg

    # 4. Save converted model + tokenizer + config.
    if DST_PATH.exists():
        shutil.rmtree(DST_PATH)
    print(f"[INFO] Saving MLX W4W8-Native checkpoint to {DST_PATH}...")
    save(DST_PATH, SRC_PATH, model, tokenizer, config, donate_model=True)

    # 5. Ensure mtp_config, mtp.safetensors, processor/chat_template/tiktoken assets are preserved,
    #    and remove hf_quant_config.json if copied.
    hf_qcfg = DST_PATH / "hf_quant_config.json"
    if hf_qcfg.exists():
        hf_qcfg.unlink()

    src_cfg = json.loads((SRC_PATH / "config.json").read_text())
    dst_cfg_path = DST_PATH / "config.json"
    dst_cfg = json.loads(dst_cfg_path.read_text())
    if "mtp_config" in src_cfg and "mtp_config" not in dst_cfg:
        dst_cfg["mtp_config"] = src_cfg["mtp_config"]
        dst_cfg = dict(sorted(dst_cfg.items()))
        dst_cfg_path.write_text(json.dumps(dst_cfg, indent=4))

    for filename in (
        "mtp.safetensors",
        "processor_config.json",
        "chat_template.jinja",
        "special_tokens_map.json",
    ):
        src_file = SRC_PATH / filename
        dst_file = DST_PATH / filename
        if src_file.is_file() and not dst_file.exists():
            shutil.copy2(src_file, dst_file)

    tiktoken_src = SRC_PATH / "tiktoken"
    tiktoken_dst = DST_PATH / "tiktoken"
    if tiktoken_src.is_dir() and not tiktoken_dst.exists():
        shutil.copytree(tiktoken_src, tiktoken_dst)

    elapsed = time.perf_counter() - t0
    print(f"[INFO] Conversion complete in {elapsed:.1f}s -> {DST_PATH}")


if __name__ == "__main__":
    main()
