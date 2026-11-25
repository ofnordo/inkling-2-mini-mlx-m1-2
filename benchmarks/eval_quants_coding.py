"""Evaluate Inkling-2-Mini coding intelligence and throughput across quantization recipes at MTP depth=2."""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import argparse
import gc
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_map_with_path, tree_unflatten

from mlx_lm.layers.moe import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.models.inkling2 import layers as inkling2_layers
from mlx_lm.utils import compute_bits_per_weight, load, maybe_set_recommended_wired_limit

from bench_mtp import Inkling2DSparkDrafter, run_generation
from convert_w4a16 import dequantize_switch_module


SRC_PATH = Path(os.environ.get("INKLING_RAW", "Inkling-2-Mini"))
W4A16_PATH = Path(os.environ.get("INKLING_W4A16", "Inkling-2-Mini-MLX-W4A16"))
OUT_DIR = Path("outputs/quant_eval_outputs")

CODING_PROMPT = """Write a single, self-contained Python 3 module implementing an in-memory Multi-Version Concurrency Control (MVCC) key-value store with Snapshot Isolation.
Output ONLY one ```python ... ``` code block (standard library only). Keep your thinking concise and go straight to the code.

### Specification

1. `class ConflictError(Exception)`:
   - Raised when `commit(tx_id)` detects a write-write conflict under First-Committer-Wins Snapshot Isolation.

2. `class MVCCStore`:
   - `__init__(self)`:
     - Maintains a monotonic integer version clock `self.current_ts: int = 0`.
     - Stores committed version histories per key (each committed version has `(commit_ts, value_or_tombstone)`).

   - `begin(self) -> int`:
     - Starts a new active transaction with a unique integer `tx_id` (`1, 2, 3, ...`) and snapshot timestamp `start_ts = self.current_ts`.
     - Returns `tx_id`.

   - **Active Transaction Validation**:
     - Calling `get`, `put`, `delete`, `scan`, `savepoint`, `rollback_to`, `commit`, or `abort` with an unknown `tx_id`, or a `tx_id` that has already committed or aborted, MUST raise `ValueError`.

   - `get(self, tx_id: int, key: str) -> Any | None`:
     - If `key` was modified (put or deleted) by `tx_id` in its currently staged write set, return the staged value (or `None` if staged as deleted).
     - Otherwise, return the value from the latest committed version of `key` with `commit_ts <= tx.start_ts` (or `None` if no such version exists or if that version was a deletion tombstone).

   - `get_at(self, key: str, ts: int) -> Any | None`:
     - Point-in-time historical read (outside any transaction): return the value from the latest committed version of `key` with `commit_ts <= ts` (or `None` if absent or deleted at `ts`).

   - `put(self, tx_id: int, key: str, value: Any) -> None`:
     - Stages `key -> value` in `tx_id`'s write set (`value` is never `None`).

   - `delete(self, tx_id: int, key: str) -> None`:
     - Stages a deletion (tombstone) for `key` in `tx_id`'s write set.

   - `savepoint(self, tx_id: int, name: str) -> None`:
     - Records a named savepoint capturing `tx_id`'s current staged write/delete set. Creating a savepoint with an existing `name` overwrites it at the current position.

   - `rollback_to(self, tx_id: int, name: str) -> None`:
     - Restores `tx_id`'s staged write/delete set to the exact state it had when `savepoint(tx_id, name)` was called, and removes any savepoints created *after* `name` (keeping `name` itself so it can be rolled back to again).
     - If `name` does not exist in `tx_id`, raise `KeyError(name)`.
     - Note: Keys whose staged modifications were undone by `rollback_to` (so they are no longer in `tx_id`'s staged write set) must NOT be treated as written by `tx_id` during conflict detection!

   - `scan(self, tx_id: int, start_key: str | None = None, end_key: str | None = None) -> list[tuple[str, Any]]`:
     - Returns a list of `(key, value)` pairs sorted lexicographically by `key` for all visible keys where:
       - `start_key is None or key >= start_key`, AND
       - `end_key is None or key <= end_key`.
     - Must merge `tx_id`'s staged puts/deletes over the committed snapshot as of `tx.start_ts`, omitting any key whose effective state is deleted/absent.

   - `commit(self, tx_id: int) -> int`:
     - **First-Committer-Wins Conflict Detection**:
       - If `tx_id` has a non-empty staged write/delete set: check whether ANY key `k` in `tx_id`'s current staged write/delete set has a committed version with `commit_ts > tx.start_ts`.
       - If a conflict exists on any key `k`: automatically abort `tx_id` (mark it aborted and discard its staged writes) and raise `ConflictError(k)`.
     - If no conflict:
       - If `tx_id` has a non-empty staged write/delete set: increment `self.current_ts += 1`, append `(self.current_ts, val_or_tombstone)` to the committed history of each key in `tx_id`'s staged set, mark `tx_id` as committed, and return `self.current_ts`.
       - If `tx_id` is read-only (empty staged write/delete set): mark `tx_id` as committed and return `self.current_ts` (without incrementing `self.current_ts`).

   - `abort(self, tx_id: int) -> None`:
     - Marks `tx_id` as aborted and discards its staged write/delete set and savepoints.

   - `compact(self) -> int`:
     - Let `watermark = min((tx.start_ts for tx in active_transactions), default=self.current_ts)`.
     - For every key's committed version list, keep:
       - all versions with `commit_ts > watermark`, AND
       - among versions with `commit_ts <= watermark`, keep ONLY the single version with the largest `commit_ts` (pruning all older versions with `commit_ts <= watermark`).
     - Return the total number of pruned version entries across all keys.
"""


def build_model_variant(variant: str) -> tuple[nn.Module, Any, float]:
    """Load or construct the requested quantization variant in memory."""
    if variant == "W4A16":
        model, tok = load(str(W4A16_PATH))
        bpw = compute_bits_per_weight(model)
        return model, tok, bpw

    if variant == "Original (NVFP4/MXFP8/BF16)":
        model, tok = load(str(SRC_PATH))
        bpw = compute_bits_per_weight(model)
        return model, tok, bpw

    model, tok, _ = load(str(SRC_PATH), lazy=True, return_config=True)
    preserve_native_expert_bits = False
    if variant == "W3A16 (3-bit affine)":
        expert_bits, dense_bits = 3, 3
    elif variant == "W4W8-Mixed (4b experts, 8b attn/head)":
        expert_bits, dense_bits = 4, 8
    elif variant == "W4W8-Native-A8 (4b L0-19, 8b L20-23 + 8b attn/head)":
        expert_bits, dense_bits = None, 8
        preserve_native_expert_bits = True
    elif variant == "W4W8-Native-BF16 (4b L0-19, 8b L20-23 + BF16 attn/head)":
        expert_bits, dense_bits = None, None
        preserve_native_expert_bits = True
    elif variant == "W6A16 (6-bit affine)":
        expert_bits, dense_bits = 6, 6
    else:
        raise ValueError(f"Unknown variant: {variant}")

    switch_modules = [
        (name, mod)
        for name, mod in model.named_modules()
        if isinstance(mod, QuantizedSwitchLinear)
    ]
    for idx, (name, mod) in enumerate(switch_modules, 1):
        bias = "bias" in mod
        target_bits = int(mod.bits) if preserve_native_expert_bits else int(expert_bits)
        w = dequantize_switch_module(mod)
        mx.eval(w)
        num_experts, out_dims, in_dims = w.shape
        dense = SwitchLinear(in_dims, out_dims, num_experts, bias=bias)
        dense.weight = w
        if bias:
            dense.bias = mod.bias.astype(mx.bfloat16)
        ql = dense.to_quantized(group_size=64, bits=target_bits, mode="affine")
        mx.eval(ql.parameters())
        model.update_modules(tree_unflatten([(name, ql)]))
        del w, dense, ql
        if idx % 12 == 0 or idx == len(switch_modules):
            gc.collect()
            mx.clear_cache()

    cast_pred = getattr(model, "cast_predicate", lambda _: True)

    def set_dtype(k: str, v: mx.array) -> mx.array:
        if k.endswith("global_scale"):
            return v
        if cast_pred(k) and mx.issubdtype(v.dtype, mx.floating) and v.dtype != mx.bfloat16:
            return v.astype(mx.bfloat16)
        return v

    model.update(tree_map_with_path(set_dtype, model.parameters()))
    mx.eval(model.parameters())

    if dense_bits is not None:
        quant_pred = getattr(model, "quant_predicate", lambda p, m: True)

        def dense_predicate(path: str, module: nn.Module) -> bool:
            if isinstance(module, QuantizedSwitchLinear):
                return False
            if not hasattr(module, "to_quantized"):
                return False
            if module.weight.shape[-1] % 64 != 0:
                return False
            return bool(quant_pred(path, module))

        nn.quantize(
            model,
            group_size=64,
            bits=dense_bits,
            mode="affine",
            class_predicate=dense_predicate,
        )
        mx.eval(model.parameters())
    gc.collect()
    mx.clear_cache()
    bpw = compute_bits_per_weight(model)
    return model, tok, bpw


def extract_code_and_stats(tokenizer: Any, token_ids: list[int]) -> tuple[str, int, int, str]:
    """Split thinking vs response tokens and extract the Python code block."""
    full_text = tokenizer.decode(token_ids)
    think_tokens = 0
    if "<|content_text|>" in full_text:
        think_part, text_part = full_text.split("<|content_text|>", 1)
        think_tokens = len(tokenizer.encode(think_part, add_special_tokens=False))
    else:
        text_part = full_text
        if "<|content_thinking|>" in full_text:
            think_tokens = len(token_ids)
    code_tokens = max(0, len(token_ids) - think_tokens)

    matches = re.findall(r"```(?:python)?\s*\n(.*?)```", text_part, flags=re.DOTALL)
    if matches:
        code = max(matches, key=len).strip()
    else:
        m = re.search(r"```(?:python)?\s*\n(.*)$", text_part, flags=re.DOTALL)
        code = m.group(1).strip() if m else text_part.strip()
    return code, think_tokens, code_tokens, full_text


def run_test_suite(code: str) -> tuple[int, int, list[tuple[str, bool, str]]]:
    """Execute 12 deterministic unit tests against the generated MVCCStore implementation."""
    ns: dict[str, Any] = {}
    results: list[tuple[str, bool, str]] = []

    try:
        exec(compile(code, "<generated_mvcc_store>", "exec"), ns)
    except Exception as exc:
        return 0, 12, [("00_compile_and_exec", False, f"{type(exc).__name__}: {exc}")]

    MVCCStore = ns.get("MVCCStore")
    ConflictError = ns.get("ConflictError")
    if MVCCStore is None or ConflictError is None:
        return 0, 12, [("00_exports", False, "Missing MVCCStore or ConflictError")]

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            results.append((name, True, "OK"))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    # 1. Basic read-your-own-writes and monotonic commit_ts
    def test_01_basic_crud_and_ts():
        s = MVCCStore()
        t1 = s.begin()
        assert s.get(t1, "x") is None
        s.put(t1, "x", 10)
        assert s.get(t1, "x") == 10
        ts1 = s.commit(t1)
        assert ts1 == 1 and s.current_ts == 1
        t2 = s.begin()
        assert s.get(t2, "x") == 10
        s.put(t2, "x", 20)
        ts2 = s.commit(t2)
        assert ts2 == 2 and s.current_ts == 2

    # 2. Snapshot isolation visibility (concurrent tx does not see later commit)
    def test_02_snapshot_isolation():
        s = MVCCStore()
        t0 = s.begin()
        s.put(t0, "k", "v0")
        s.commit(t0)

        t_reader = s.begin()
        t_writer = s.begin()
        s.put(t_writer, "k", "v1")
        s.put(t_writer, "new_k", "nv")
        # Uncommitted writes invisible to t_reader
        assert s.get(t_reader, "k") == "v0"
        assert s.get(t_reader, "new_k") is None
        s.commit(t_writer)
        # Committed after t_reader.start_ts -> STILL invisible to t_reader!
        assert s.get(t_reader, "k") == "v0"
        assert s.get(t_reader, "new_k") is None
        # Fresh tx sees both
        t_fresh = s.begin()
        assert s.get(t_fresh, "k") == "v1"
        assert s.get(t_fresh, "new_k") == "nv"

    # 3. Tombstone deletions & historical visibility
    def test_03_delete_tombstone_and_get_at():
        s = MVCCStore()
        t1 = s.begin()
        s.put(t1, "a", 100)
        ts1 = s.commit(t1)

        t_snap = s.begin()
        t2 = s.begin()
        s.delete(t2, "a")
        assert s.get(t2, "a") is None  # read your own delete
        assert s.get(t_snap, "a") == 100
        ts2 = s.commit(t2)

        assert s.get(t_snap, "a") == 100  # old snapshot still sees 100
        t3 = s.begin()
        assert s.get(t3, "a") is None
        assert s.get_at("a", 0) is None
        assert s.get_at("a", ts1) == 100
        assert s.get_at("a", ts2) is None

    # 4. First-Committer-Wins write-write conflict detection & auto-abort
    def test_04_write_write_conflict():
        s = MVCCStore()
        t1 = s.begin()
        t2 = s.begin()
        s.put(t1, "shared", "from_t1")
        s.put(t2, "shared", "from_t2")
        s.put(t2, "side_effect", "should_discard")
        s.commit(t1)
        try:
            s.commit(t2)
            raise AssertionError("Expected ConflictError on t2 commit")
        except ConflictError:
            pass
        # t2 must be auto-aborted (further ops raise ValueError, side_effect discarded)
        try:
            s.get(t2, "shared")
            raise AssertionError("Expected ValueError on auto-aborted tx")
        except ValueError:
            pass
        t3 = s.begin()
        assert s.get(t3, "shared") == "from_t1"
        assert s.get(t3, "side_effect") is None

    # 5. Write skew (disjoint write sets) allowed under Snapshot Isolation
    def test_05_write_skew_allowed():
        s = MVCCStore()
        t0 = s.begin()
        s.put(t0, "a", 1)
        s.put(t0, "b", 1)
        s.commit(t0)

        t1 = s.begin()
        t2 = s.begin()
        _ = s.get(t1, "a")
        _ = s.get(t2, "b")
        s.put(t1, "b", 2)
        s.put(t2, "a", 2)
        s.commit(t1)
        s.commit(t2)  # Must succeed since write sets {'b'} and {'a'} are disjoint!
        t3 = s.begin()
        assert s.get(t3, "a") == 2 and s.get(t3, "b") == 2

    # 6. Aborted transaction does not block concurrent writers
    def test_06_aborted_tx_no_conflict():
        s = MVCCStore()
        t1 = s.begin()
        t2 = s.begin()
        s.put(t1, "k", "v1")
        s.abort(t1)
        s.put(t2, "k", "v2")
        ts = s.commit(t2)
        assert ts == 1
        assert s.get_at("k", ts) == "v2"

    # 7. Nested savepoints & rollback_to (including pruning later savepoints)
    def test_07_nested_savepoints():
        s = MVCCStore()
        tx = s.begin()
        s.put(tx, "k1", "v1")
        s.savepoint(tx, "sp1")
        s.put(tx, "k1", "v2")
        s.put(tx, "k2", "v2")
        s.savepoint(tx, "sp2")
        s.delete(tx, "k1")
        assert s.get(tx, "k1") is None

        s.rollback_to(tx, "sp2")
        assert s.get(tx, "k1") == "v2" and s.get(tx, "k2") == "v2"

        s.rollback_to(tx, "sp1")
        assert s.get(tx, "k1") == "v1" and s.get(tx, "k2") is None
        # sp2 was created after sp1, so rolling back to sp2 now must raise KeyError!
        try:
            s.rollback_to(tx, "sp2")
            raise AssertionError("Expected KeyError for pruned savepoint sp2")
        except KeyError:
            pass
        # sp1 itself must remain reusable
        s.put(tx, "k1", "v3")
        s.rollback_to(tx, "sp1")
        assert s.get(tx, "k1") == "v1"

    # 8. Rollback of a staged key removes it from conflict set
    def test_08_savepoint_rollback_avoids_conflict():
        s = MVCCStore()
        t1 = s.begin()
        t2 = s.begin()
        s.put(t1, "safe", "ok")
        s.savepoint(t1, "sp")
        s.put(t1, "contested", "t1_temp")
        s.rollback_to(t1, "sp")  # 'contested' is no longer touched by t1!

        s.put(t2, "contested", "t2_win")
        s.commit(t2)
        # t1 should commit cleanly without ConflictError on 'contested'
        s.commit(t1)
        assert s.get_at("safe", s.current_ts) == "ok"
        assert s.get_at("contested", s.current_ts) == "t2_win"

    # 9. Lexicographical range scan merging staged puts & deletes over snapshot
    def test_09_range_scan_merged():
        s = MVCCStore()
        t0 = s.begin()
        for k, v in [("d", 4), ("b", 2), ("a", 1), ("c", 3), ("e", 5)]:
            s.put(t0, k, v)
        s.commit(t0)

        t1 = s.begin()
        t_other = s.begin()
        s.put(t_other, "b2", 99)
        s.commit(t_other)  # invisible to t1

        s.put(t1, "b", 20)       # overwrite
        s.delete(t1, "c")        # stage delete
        s.put(t1, "c2", 35)      # stage insert
        assert s.scan(t1, "b", "d") == [("b", 20), ("c2", 35), ("d", 4)]
        assert s.scan(t1, None, "b") == [("a", 1), ("b", 20)]
        assert s.scan(t1, "d", None) == [("d", 4), ("e", 5)]

    # 10. Read-only transaction does not advance current_ts
    def test_10_readonly_tx_no_ts_bump():
        s = MVCCStore()
        t1 = s.begin()
        s.put(t1, "k", 1)
        ts1 = s.commit(t1)
        assert ts1 == 1 and s.current_ts == 1

        t_ro = s.begin()
        _ = s.get(t_ro, "k")
        ts_ro = s.commit(t_ro)
        assert ts_ro == 1 and s.current_ts == 1

    # 11. Invalid transaction lifecycle raises ValueError
    def test_11_invalid_tx_raises_value_error():
        s = MVCCStore()
        try:
            s.get(999, "k")
            raise AssertionError("Expected ValueError for unknown tx_id")
        except ValueError:
            pass
        tx = s.begin()
        s.commit(tx)
        for fn in [
            lambda: s.get(tx, "k"),
            lambda: s.put(tx, "k", 1),
            lambda: s.delete(tx, "k"),
            lambda: s.scan(tx),
            lambda: s.commit(tx),
            lambda: s.abort(tx),
        ]:
            try:
                fn()
                raise AssertionError("Expected ValueError on committed tx")
            except ValueError:
                pass

    # 12. Watermark garbage collection (compact) preserves active snapshots
    def test_12_compact_watermark_gc():
        s = MVCCStore()
        for i in range(1, 5):
            tx = s.begin()
            s.put(tx, "k", f"v{i}")
            s.commit(tx)
        # Versions of 'k' are at ts=1, 2, 3, 4.
        # Open active transaction at start_ts=2? Let's test with an active tx opened at ts=2:
        s2 = MVCCStore()
        for i in range(1, 3):
            tx = s2.begin()
            s2.put(tx, "k", f"v{i}")
            s2.commit(tx)
        tx_pin = s2.begin()  # start_ts = 2 (pins ts=2, allows pruning ts=1!)
        for i in range(3, 5):
            tx = s2.begin()
            s2.put(tx, "k", f"v{i}")
            s2.commit(tx)
        # Versions of 'k': ts=1, 2, 3, 4. Watermark = 2 -> ts <= 2 has [1, 2], prune ts=1 (1 entry), keep [2, 3, 4].
        pruned1 = s2.compact()
        assert pruned1 == 1, f"Expected 1 version pruned at watermark=2, got {pruned1}"
        assert s2.get(tx_pin, "k") == "v2"
        s2.abort(tx_pin)
        # Now no active tx -> watermark = 4 -> versions [2, 3, 4] <= 4 -> prune ts=2, 3 (2 entries), keep [4].
        pruned2 = s2.compact()
        assert pruned2 == 2, f"Expected 2 versions pruned at watermark=4, got {pruned2}"
        tx_final = s2.begin()
        assert s2.get(tx_final, "k") == "v4"

    for name, fn in [
        ("01_basic_crud_and_ts", test_01_basic_crud_and_ts),
        ("02_snapshot_isolation", test_02_snapshot_isolation),
        ("03_delete_tombstone_get_at", test_03_delete_tombstone_and_get_at),
        ("04_write_write_conflict", test_04_write_write_conflict),
        ("05_write_skew_allowed", test_05_write_skew_allowed),
        ("06_aborted_tx_no_conflict", test_06_aborted_tx_no_conflict),
        ("07_nested_savepoints", test_07_nested_savepoints),
        ("08_rollback_avoids_conflict", test_08_savepoint_rollback_avoids_conflict),
        ("09_range_scan_merged", test_09_range_scan_merged),
        ("10_readonly_tx_no_ts_bump", test_10_readonly_tx_no_ts_bump),
        ("11_invalid_tx_value_error", test_11_invalid_tx_raises_value_error),
        ("12_compact_watermark_gc", test_12_compact_watermark_gc),
    ]:
        check(name, fn)

    passed = sum(1 for _, ok, _ in results if ok)
    return passed, len(results), results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=5120)
    args = parser.parse_args()

    maybe_set_recommended_wired_limit()
    inkling2_layers._USE_SHORT_CONV_KERNEL = True
    inkling2_layers._USE_SPLITK_ATTN_KERNEL = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        ("Original (NVFP4/MXFP8/BF16)", "orig_nvfp4_mxfp8"),
        ("W6A16 (6-bit affine)", "w6a16"),
        ("W4W8-Mixed (4b experts, 8b attn/head)", "w4w8_mixed"),
        ("W4A16", "w4a16"),
        ("W3A16 (3-bit affine)", "w3a16"),
    ]
    if args.only:
        variants = [v for v in variants if args.only.lower() in v[0].lower() or args.only.lower() in v[1].lower()]
    if args.skip:
        variants = [v for v in variants if args.skip.lower() not in v[0].lower() and args.skip.lower() not in v[1].lower()]

    summary_rows = []

    for label, slug in variants:
        print(f"\n======================================================================")
        print(f"[QUANT] Building & loading: {label}")
        t_load = time.perf_counter()
        model, tokenizer, bpw = build_model_variant(label)
        model.fuse_projections()
        drafter = Inkling2DSparkDrafter(SRC_PATH, model, draft_bits=8, opt_drafter=True)
        load_s = time.perf_counter() - t_load

        prompt_tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": CODING_PROMPT}],
            add_generation_prompt=True,
            tokenize=True,
            reasoning_effort="minimal",
        )
        print(f"[INFO] {label} ready in {load_s:.1f}s (bpw={bpw:.3f}, prompt={len(prompt_tokens)} tokens). Warming up...")

        _ = run_generation(
            model,
            tokenizer,
            prompt_tokens[:64],
            16,
            mtp_depth=2,
            drafter=drafter,
            stop_on_eos=False,
        )

        print(f"[GEN] Generating solution at MTP depth=2 (max_tokens={args.max_tokens})...")
        t0 = time.perf_counter()
        res = run_generation(
            model,
            tokenizer,
            prompt_tokens,
            args.max_tokens,
            mtp_depth=2,
            drafter=drafter,
            stop_on_eos=True,
        )
        wall_s = time.perf_counter() - t0

        code, think_toks, code_toks, full_text = extract_code_and_stats(
            tokenizer, res["tokens"]
        )
        (OUT_DIR / f"{slug}_raw.txt").write_text(full_text)
        (OUT_DIR / f"{slug}.py").write_text(code)

        passed, total, details = run_test_suite(code)
        print(
            f"[RESULT] {label} | bpw={bpw:.2f} | gen={res['gen_tokens']} tok "
            f"(think={think_toks}, code={code_toks}) | speed={res['gen_tps']:.2f} tok/s | "
            f"acc={res['mean_accepted']:.2f}/2 ({res['accept_rate']*100:.1f}%) | "
            f"wall={wall_s:.1f}s | peak={res['peak_gb']:.2f} GB | SCORE: {passed}/{total}"
        )
        for tname, ok, msg in details:
            status_str = "PASS" if ok else f"FAIL ({msg})"
            print(f"    - {tname}: {status_str}")

        summary_rows.append(
            {
                "label": label,
                "slug": slug,
                "bpw": bpw,
                "gen_tokens": res["gen_tokens"],
                "think_tokens": think_toks,
                "code_tokens": code_toks,
                "prompt_tps": res["prompt_tps"],
                "gen_tps": res["gen_tps"],
                "mean_accepted": res["mean_accepted"],
                "accept_rate": res["accept_rate"],
                "wall_s": wall_s,
                "peak_gb": res["peak_gb"],
                "passed": passed,
                "total": total,
                "details": details,
            }
        )

        del drafter, model, tokenizer
        gc.collect()
        mx.clear_cache()

    print("\n### One-Shot Programming Test Summary across Quants (MTP Depth = 2)\n")
    print(
        "| Quant Recipe | Bits/Weight | Peak Mem | Gen Tok (Think / Code) | Decode tok/s (`d=2`) | Mean Acc (Accept %) | Wall Time | Test Score |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in summary_rows:
        print(
            f"| {r['label']} | {r['bpw']:.2f} | {r['peak_gb']:.2f} GB | "
            f"{r['gen_tokens']} ({r['think_tokens']} / {r['code_tokens']}) | "
            f"{r['gen_tps']:.2f} | {r['mean_accepted']:.2f}/2 ({r['accept_rate']*100:.1f}%) | "
            f"{r['wall_s']:.1f}s | **{r['passed']}/{r['total']}** |"
        )


if __name__ == "__main__":
    main()
