"""Convert ThinkingMachines Inkling-2-Mini (NVFP4/MXFP8/BF16) to MLX W4A16 (4-bit affine, group_size=64)."""

from __future__ import annotations

import gc
import os
import json
import shutil
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map_with_path, tree_unflatten

from mlx_lm.layers.moe import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.utils import load, quantize_model, save


SRC_PATH = Path(os.environ.get("INKLING_SRC", "Inkling-2-Mini"))
DST_PATH = Path(os.environ.get("INKLING_DST", "Inkling-2-Mini-MLX-W4A16"))
GROUP_SIZE = 64
BITS = 4
MODE = "affine"


def dequantize_switch_module(module: QuantizedSwitchLinear) -> mx.array:
    """Dequantize one QuantizedSwitchLinear (NVFP4 with global_scale or MXFP8/affine) to bfloat16."""
    global_scale = module.get("global_scale")
    if global_scale is not None:
        weight = mx.stack(
            [
                mx.dequantize(
                    module.weight[e],
                    module.scales[e],
                    group_size=module.group_size,
                    bits=module.bits,
                    mode=module.mode,
                    global_scale=global_scale[e],
                )
                for e in range(global_scale.shape[0])
            ]
        )
    else:
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.get("biases"),
            module.group_size,
            module.bits,
            module.mode,
        )
    return weight.astype(mx.bfloat16)


def requantize_switch_layers(model, group_size: int, bits: int, mode: str) -> None:
    """Dequantize and immediately re-quantize QuantizedSwitchLinear layers one at a time to bound RAM."""
    switch_modules = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, QuantizedSwitchLinear)
    ]
    print(f"[INFO] Re-quantizing {len(switch_modules)} QuantizedSwitchLinear expert tensors to {bits}-bit {mode} (gs={group_size})...")
    for idx, (name, module) in enumerate(switch_modules, 1):
        bias = "bias" in module
        weight = dequantize_switch_module(module)
        mx.eval(weight)
        _, out_dims, in_dims = weight.shape
        num_experts = weight.shape[0]

        dense = SwitchLinear(in_dims, out_dims, num_experts, bias=bias)
        dense.weight = weight
        if bias:
            dense.bias = module.bias.astype(mx.bfloat16)
        ql = dense.to_quantized(group_size=group_size, bits=bits, mode=mode)
        mx.eval(ql.parameters())
        model.update_modules(tree_unflatten([(name, ql)]))

        del weight, dense, ql
        if idx % 8 == 0 or idx == len(switch_modules):
            gc.collect()
            mx.clear_cache()
            print(f"  [{idx}/{len(switch_modules)}] {name} done (active={mx.get_active_memory()/1e9:.2f} GB)")


def main() -> None:
    t0 = time.perf_counter()
    print(f"[INFO] Loading {SRC_PATH} (lazy=True)...")
    model, tokenizer, config = load(str(SRC_PATH), lazy=True, return_config=True)

    # 1. Convert NVFP4 / MXFP8 QuantizedSwitchLinear layers one-by-one to W4A16 affine.
    requantize_switch_layers(model, GROUP_SIZE, BITS, MODE)

    # 2. Cast remaining floating parameters to bfloat16, respecting model.cast_predicate.
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

    # 3. Strip source NVFP4/modelopt quantization metadata and quantize remaining linear/embed layers.
    config.pop("quantization", None)
    config.pop("quantization_config", None)

    print(f"[INFO] Quantizing remaining eligible layers to {BITS}-bit {MODE} (gs={GROUP_SIZE})...")
    model, config = quantize_model(
        model,
        config,
        group_size=GROUP_SIZE,
        bits=BITS,
        mode=MODE,
    )
    mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()

    # 4. Save converted model + tokenizer + config.
    if DST_PATH.exists():
        shutil.rmtree(DST_PATH)
    print(f"[INFO] Saving MLX W4A16 checkpoint to {DST_PATH}...")
    save(DST_PATH, SRC_PATH, model, tokenizer, config, donate_model=True)

    # 5. Ensure mtp_config, mtp.safetensors, processor/chat_template/tiktoken assets are preserved.
    src_cfg = json.loads((SRC_PATH / "config.json").read_text())
    dst_cfg_path = DST_PATH / "config.json"
    dst_cfg = json.loads(dst_cfg_path.read_text())
    if "mtp_config" in src_cfg and "mtp_config" not in dst_cfg:
        dst_cfg["mtp_config"] = src_cfg["mtp_config"]
        dst_cfg = dict(sorted(dst_cfg.items()))
        dst_cfg_path.write_text(json.dumps(dst_cfg, indent=4))

    for filename in ("mtp.safetensors", "processor_config.json", "chat_template.jinja"):
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
