"""Run the tough 12-test Deterministic Virtual-Clock DAG Workflow Engine benchmark with full thinking and 64k max tokens at MTP depth=2."""

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

from mlx_lm.models.inkling2 import layers as inkling2_layers
from mlx_lm.utils import maybe_set_recommended_wired_limit

from bench_mtp import Inkling2DSparkDrafter, run_generation
from eval_quants_coding import SRC_PATH, build_model_variant, extract_code_and_stats


OUT_DIR = Path("outputs/quant_eval_outputs_workflow_64k")

WORKFLOW_PROMPT = """Write a single, self-contained Python 3 module implementing a deterministic virtual-clock DAG workflow engine.
Output ONLY one ```python ... ``` code block containing the implementation (no external third-party imports; use only the Python standard library).

### Required Classes & Exceptions

1. `class CycleError(Exception)`:
   - Raised when the workflow graph contains a directed cycle.
   - Must set `self.cycle: list[str]` to a list of task IDs forming a cycle `[u, ..., u]` where `cycle[0] == cycle[-1]` and each consecutive pair `(a, b)` means task `b` depends on task `a` (or `a` depends on `b` — either direction is accepted as long as all nodes in `self.cycle[:-1]` belong to the cycle and `self.cycle[0] == self.cycle[-1]`).

2. `class WorkflowEngine`:
   - `__init__(self, max_concurrent: int = 2, resources: dict[str, int] | None = None)`:
     - `max_concurrent`: maximum number of tasks allowed to be `"RUNNING"` simultaneously in any tick.
     - `resources`: total pool of named integer resources (e.g. `{"gpu": 2, "mem": 8}`). Default `{}`.
     - `self.clock: int = 0`: virtual clock in integer ticks.
     - `self.status: dict[str, str]`: maps `task_id` to one of `"PENDING"`, `"READY"`, `"RUNNING"`, `"WAITING_RETRY"`, `"COMPLETED"`, `"FAILED"`, `"SKIPPED"`, `"CANCELLED"`.
     - `self.results: dict[str, Any]`: maps completed `task_id` to its return value.
     - `self.errors: dict[str, Exception]`: maps failed `task_id` to the last exception raised.
     - `self.attempts: dict[str, int]`: maps `task_id` to number of attempts started or failed so far (initial `0`).

   - `add_task(self, task_id: str, fn: Callable[[dict[str, Any]], Any], *, deps: list[str] | None = None, priority: int = 0, retries: int = 0, backoff_base: int = 1, timeout_ticks: int | None = None, resources: dict[str, int] | None = None, condition: Callable[[dict[str, Any]], bool] | None = None) -> None`:
     - Registers a task with initial status `"PENDING"`.

   - `validate(self) -> list[str]`:
     - If any task lists a dependency `d` that has not been added via `add_task`, raise `KeyError(d)`.
     - If the dependency graph has any cycle (including a self-loop), raise `CycleError` with `.cycle` populated.
     - Return a valid topological ordering (`list[str]`) where every dependency appears before its dependents and ties among currently zero-indegree nodes are broken by `(-priority, task_id)`.

   - **State Resolution & Scheduling Rules (executed on every `step()` call)**:
     First, call `validate()` if not yet validated.
     Each tick `self.clock` proceeds in three phases:

     **Phase A — Retry Wakeups & Dependency / Condition Resolution (cascade to fixed point)**:
     - Any task in `"WAITING_RETRY"` whose `wake_tick <= self.clock` transitions to `"READY"`.
     - Repeatedly inspect all `"PENDING"` tasks until no more transitions occur:
       - A `"PENDING"` task `T` is eligible to resolve only when ALL of its direct dependencies `deps` are in terminal states (`{"COMPLETED", "SKIPPED", "FAILED", "CANCELLED"}`). (If `deps` is empty, `T` resolves immediately at tick 0).
       - Among terminal dependencies:
         1. If ANY direct dependency is `"FAILED"` or `"CANCELLED"`, `T` immediately becomes `"CANCELLED"`.
         2. Else if ANY direct dependency is `"SKIPPED"`, `T` immediately becomes `"SKIPPED"`.
         3. Else (all direct dependencies are `"COMPLETED"`): build `ctx = {d: self.results[d] for d in T.deps}` (ONLY direct dependencies!). If `T.condition is not None` and `not T.condition(ctx)`, `T` immediately becomes `"SKIPPED"`. Otherwise `T` becomes `"READY"`.

     **Phase B — Greedy Non-Blocking Admission of `"READY"` Tasks**:
     - Consider all currently `"READY"` tasks sorted by `(-priority, task_id)` (highest `priority` first; ties broken by lexicographically smallest `task_id`).
     - Greedily admit a `"READY"` task `T` into `"RUNNING"` if:
       - current number of `"RUNNING"` tasks `< self.max_concurrent`, AND
       - for every `(res_name, req_amount)` in `(T.resources or {}).items()`, `req_amount <= available_resources[res_name]`.
     - **Important**: Do NOT head-of-line block! If the highest-priority `"READY"` task does not fit available resources, skip it and keep checking lower-priority `"READY"` tasks that might fit.
     - When a task is admitted to `"RUNNING"`, increment `self.attempts[task_id] += 1`, allocate its resources, and initialize its current-attempt elapsed running ticks to `0`.

     **Phase C — Execute 1 Tick for All `"RUNNING"` Tasks & Advance Clock**:
     - Execute all currently `"RUNNING"` tasks in `(-priority, task_id)` order:
       - Increment the task's current-attempt running ticks `run_ticks += 1`.
       - Build `ctx = {d: self.results[d] for d in T.deps}` (only direct dependencies).
       - On the first tick of this attempt (`run_ticks == 1`), call `val = T.fn(ctx)`.
         - If `inspect.isgenerator(val)` is `False`: the task finishes immediately on this tick with return value `val`!
         - If `inspect.isgenerator(val)` is `True`: advance it once on this first tick via `next(val)`!
       - On subsequent ticks of this attempt (`run_ticks > 1`), advance the active generator once via `next(gen)`.
       - If `next(gen)` raises `StopIteration as stop`: the task finishes on this tick with return value `stop.value`!
       - If the generator yielded without finishing AND `T.timeout_ticks is not None` and `run_ticks >= T.timeout_ticks`: raise `TimeoutError("Task timed out")` for this attempt!
       - **On Attempt Success**:
         - Release the task's concurrency slot and allocated resources.
         - Set `self.status[task_id] = "COMPLETED"` and `self.results[task_id] = result`.
       - **On Attempt Exception (including `TimeoutError`)**:
         - Release the task's concurrency slot, allocated resources, and discard any active generator.
         - Store `self.errors[task_id] = exc`.
         - Let `fail_count = self.attempts[task_id]`.
         - If `fail_count <= T.retries`:
           - Compute `delay = int(T.backoff_base * (2 ** (fail_count - 1)))`.
           - Set `wake_tick = (self.clock + 1) + delay`.
           - If `delay <= 0`, set `self.status[task_id] = "READY"`; else set `self.status[task_id] = "WAITING_RETRY"`.
         - Else:
           - Set `self.status[task_id] = "FAILED"`.
     - Increment `self.clock += 1`.
     - After advancing the clock, run **Phase A** once more so any newly unblocked downstream `"CANCELLED"` / `"SKIPPED"` / `"READY"` states and retry wakeups at the new `self.clock` are immediately visible!

   - `step(self, ticks: int = 1) -> int`:
     - Run up to `ticks` ticks (stopping early if all tasks are in terminal states `{"COMPLETED", "FAILED", "SKIPPED", "CANCELLED"}`).
     - Return `self.clock`.

   - `run_until_complete(self, max_ticks: int = 10000) -> dict[str, Any]`:
     - Run ticks until all registered tasks are in terminal states (or raise `RuntimeError` if `max_ticks` ticks elapse without finishing).
     - Return `dict(self.results)`.

   - `snapshot(self) -> dict[str, Any]` and `restore(self, state: dict[str, Any]) -> None`:
     - `snapshot()` returns a JSON-serializable `dict` (must survive `json.loads(json.dumps(snap))`) containing `clock`, `status`, `results`, and `attempts`.
     - In `restore(state)`, restore `self.clock`, `self.results`, and `self.attempts`; any task whose saved status was `"RUNNING"` or `"WAITING_RETRY"` must be placed in `"READY"` (with its allocated resources cleared and its saved `attempts` count preserved so its remaining retry budget is intact), while `"COMPLETED"`, `"FAILED"`, `"SKIPPED"`, `"CANCELLED"`, `"PENDING"`, and `"READY"` keep their saved status.
"""


def run_workflow_test_suite(code: str) -> tuple[int, int, list[tuple[str, bool, str]]]:
    """Execute 12 deterministic unit tests against the generated WorkflowEngine code."""
    ns: dict[str, Any] = {}
    results: list[tuple[str, bool, str]] = []

    try:
        exec(compile(code, "<generated_workflow_engine>", "exec"), ns)
    except Exception as exc:
        return 0, 12, [("00_compile_and_exec", False, f"{type(exc).__name__}: {exc}")]

    WorkflowEngine = ns.get("WorkflowEngine")
    CycleError = ns.get("CycleError")
    if WorkflowEngine is None or CycleError is None:
        return 0, 12, [("00_exports", False, "Missing WorkflowEngine or CycleError")]

    def check(name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
            results.append((name, True, "OK"))
        except Exception as exc:
            results.append((name, False, f"{type(exc).__name__}: {exc}"))

    def test_01_toposort():
        eng = WorkflowEngine()
        eng.add_task("c", lambda ctx: 1, priority=0)
        eng.add_task("b", lambda ctx: 2, priority=0)
        eng.add_task("a", lambda ctx: 3, priority=5)
        eng.add_task("d", lambda ctx: 4, deps=["b", "c"], priority=10)
        order = eng.validate()
        assert order == ["a", "b", "c", "d"], f"Expected ['a', 'b', 'c', 'd'], got {order}"

    def test_02_cycle_and_missing_dep():
        eng_missing = WorkflowEngine()
        eng_missing.add_task("a", lambda ctx: 1, deps=["nonexistent"])
        try:
            eng_missing.validate()
            raise AssertionError("Expected KeyError for missing dep")
        except KeyError:
            pass

        eng_cyc = WorkflowEngine()
        eng_cyc.add_task("n1", lambda ctx: 1, deps=["n3"])
        eng_cyc.add_task("n2", lambda ctx: 2, deps=["n1"])
        eng_cyc.add_task("n3", lambda ctx: 3, deps=["n2"])
        eng_cyc.add_task("root", lambda ctx: 0)
        try:
            eng_cyc.validate()
            raise AssertionError("Expected CycleError")
        except CycleError as ce:
            cyc = getattr(ce, "cycle", None)
            assert isinstance(cyc, list) and len(cyc) >= 4, f"Invalid cycle attr: {cyc}"
            assert cyc[0] == cyc[-1], f"Cycle must start and end at same node: {cyc}"
            assert set(cyc[:-1]) == {"n1", "n2", "n3"}, f"Cycle nodes mismatch: {cyc}"

    def test_03_diamond_direct_ctx_only():
        eng = WorkflowEngine(max_concurrent=4)
        seen_by_d: dict[str, Any] = {}
        eng.add_task("a", lambda ctx: 10)
        eng.add_task("b", lambda ctx: ctx["a"] + 5, deps=["a"])
        eng.add_task("c", lambda ctx: ctx["a"] * 2, deps=["a"])

        def fn_d(ctx):
            seen_by_d.update(ctx)
            return ctx["b"] + ctx["c"]

        eng.add_task("d", fn_d, deps=["b", "c"])
        res = eng.run_until_complete()
        assert res["d"] == 35, f"Expected 35, got {res.get('d')}"
        assert set(seen_by_d.keys()) == {"b", "c"}, f"ctx must contain ONLY direct deps, got {seen_by_d}"
        assert eng.clock == 3, f"Expected 3 ticks for 3-level diamond with max_concurrent=4, got {eng.clock}"

    def test_04_concurrency_and_priority_order():
        eng = WorkflowEngine(max_concurrent=2)
        start_ticks: dict[str, int] = {}
        for tid, prio in [("t3", 1), ("t1", 1), ("t2", 5), ("t4", 0)]:
            eng.add_task(tid, lambda ctx, t=tid: start_ticks.setdefault(t, eng.clock) or t, priority=prio)
        eng.run_until_complete()
        assert start_ticks == {"t2": 0, "t1": 0, "t3": 1, "t4": 1}, f"Got start_ticks={start_ticks}"

    def test_05_resource_non_blocking():
        eng = WorkflowEngine(max_concurrent=3, resources={"gpu": 2})
        start_ticks: dict[str, int] = {}

        def gen_a(ctx):
            start_ticks["a"] = eng.clock
            yield 1
            return "a"

        eng.add_task("a", gen_a, priority=10, resources={"gpu": 1})
        eng.add_task("b_heavy", lambda ctx: start_ticks.setdefault("b_heavy", eng.clock), priority=5, resources={"gpu": 2})
        eng.add_task("c_light", lambda ctx: start_ticks.setdefault("c_light", eng.clock), priority=1, resources={"gpu": 1})
        eng.run_until_complete()
        assert start_ticks["a"] == 0, f"Expected a at 0: {start_ticks}"
        assert start_ticks["c_light"] == 0, f"c_light should not be blocked by b_heavy: {start_ticks}"
        assert start_ticks["b_heavy"] == 2, f"Expected b_heavy at tick 2 after a releases gpu: {start_ticks}"

    def test_06_generator_ticks():
        eng = WorkflowEngine(max_concurrent=2)

        def slow_double(ctx):
            yield "step1"
            yield "step2"
            return 42

        eng.add_task("g", slow_double)
        eng.step(1)
        assert eng.status["g"] == "RUNNING" and eng.clock == 1
        eng.step(1)
        assert eng.status["g"] == "RUNNING" and eng.clock == 2
        eng.step(1)
        assert eng.status["g"] == "COMPLETED" and eng.results["g"] == 42 and eng.clock == 3

    def test_07_timeout_and_retry_recovery():
        eng = WorkflowEngine(max_concurrent=1)
        calls = 0

        def flaky_gen(ctx):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield 1
                yield 2
                return "too_slow"
            yield 1
            return "recovered"

        eng.add_task("flaky", flaky_gen, retries=1, backoff_base=1, timeout_ticks=2)
        res = eng.run_until_complete()
        assert res["flaky"] == "recovered", f"Expected 'recovered', got {res}"
        assert isinstance(eng.errors.get("flaky"), TimeoutError), f"Expected TimeoutError recorded, got {eng.errors}"
        assert eng.attempts["flaky"] == 2, f"Expected 2 attempts, got {eng.attempts['flaky']}"
        assert eng.clock == 5, f"Expected clock=5, got {eng.clock}"

    def test_08_exponential_backoff_timing():
        eng = WorkflowEngine(max_concurrent=1)
        call_ticks: list[int] = []

        def fail_twice(ctx):
            call_ticks.append(eng.clock)
            if len(call_ticks) < 3:
                raise ValueError(f"boom {len(call_ticks)}")
            return "ok"

        eng.add_task("t", fail_twice, retries=2, backoff_base=2)
        res = eng.run_until_complete()
        assert res["t"] == "ok"
        assert call_ticks == [0, 3, 8], f"Expected call_ticks=[0, 3, 8], got {call_ticks}"
        assert eng.clock == 9, f"Expected final clock=9, got {eng.clock}"

    def test_09_cascading_cancelled():
        eng = WorkflowEngine(max_concurrent=2)
        eng.add_task("bad", lambda ctx: (_ for _ in ()).throw(RuntimeError("fatal")), retries=0)
        eng.add_task("child", lambda ctx: 1, deps=["bad"])
        eng.add_task("grandchild", lambda ctx: 2, deps=["child"])
        eng.run_until_complete()
        assert eng.status["bad"] == "FAILED"
        assert eng.status["child"] == "CANCELLED"
        assert eng.status["grandchild"] == "CANCELLED"
        assert eng.clock == 1

    def test_10_condition_skipped_and_dominance():
        eng = WorkflowEngine(max_concurrent=4)
        eng.add_task("root", lambda ctx: 5)
        eng.add_task("branch_skip", lambda ctx: 10, deps=["root"], condition=lambda ctx: ctx["root"] > 100)
        eng.add_task("after_skip", lambda ctx: 20, deps=["branch_skip"])
        eng.add_task("fail_node", lambda ctx: (_ for _ in ()).throw(ValueError("x")))
        eng.add_task("join_fail_skip", lambda ctx: 30, deps=["after_skip", "fail_node"])
        eng.run_until_complete()
        assert eng.status["branch_skip"] == "SKIPPED"
        assert eng.status["after_skip"] == "SKIPPED"
        assert eng.status["join_fail_skip"] == "CANCELLED"

    def test_11_snapshot_and_restore():
        eng1 = WorkflowEngine(max_concurrent=1)
        a_runs = 0
        b_runs = 0

        def run_a(ctx):
            nonlocal a_runs
            a_runs += 1
            return 100

        def run_b(ctx):
            nonlocal b_runs
            b_runs += 1
            yield 1
            return ctx["a"] + 23

        eng1.add_task("a", run_a)
        eng1.add_task("b", run_b, deps=["a"])
        eng1.step(2)
        assert eng1.status["a"] == "COMPLETED" and eng1.status["b"] == "RUNNING"

        raw_snap = eng1.snapshot()
        roundtrip_snap = json.loads(json.dumps(raw_snap))

        eng2 = WorkflowEngine(max_concurrent=1)
        eng2.add_task("a", run_a)
        eng2.add_task("b", run_b, deps=["a"])
        eng2.restore(roundtrip_snap)
        assert eng2.status["a"] == "COMPLETED"
        assert eng2.status["b"] == "READY"

        res = eng2.run_until_complete()
        assert res["b"] == 123
        assert a_runs == 1

    def test_12_self_loop_and_early_stop():
        eng_self = WorkflowEngine()
        eng_self.add_task("self_node", lambda ctx: 1, deps=["self_node"])
        try:
            eng_self.validate()
            raise AssertionError("Expected CycleError on self-loop")
        except CycleError as ce:
            assert ce.cycle == ["self_node", "self_node"]

        eng_fast = WorkflowEngine()
        eng_fast.add_task("only", lambda ctx: "done")
        c = eng_fast.step(50)
        assert c == 1 and eng_fast.clock == 1

    for name, fn in [
        ("01_toposort", test_01_toposort),
        ("02_cycle_and_missing_dep", test_02_cycle_and_missing_dep),
        ("03_diamond_direct_ctx_only", test_03_diamond_direct_ctx_only),
        ("04_concurrency_and_priority", test_04_concurrency_and_priority_order),
        ("05_resource_non_blocking", test_05_resource_non_blocking),
        ("06_generator_ticks", test_06_generator_ticks),
        ("07_timeout_and_retry", test_07_timeout_and_retry_recovery),
        ("08_exponential_backoff", test_08_exponential_backoff_timing),
        ("09_cascading_cancelled", test_09_cascading_cancelled),
        ("10_condition_skip_dominance", test_10_condition_skipped_and_dominance),
        ("11_snapshot_and_restore", test_11_snapshot_and_restore),
        ("12_self_loop_and_early_stop", test_12_self_loop_and_early_stop),
    ]:
        check(name, fn)

    passed = sum(1 for _, ok, _ in results if ok)
    return passed, len(results), results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=65536)
    parser.add_argument("--effort", type=str, default="medium")
    args = parser.parse_args()

    maybe_set_recommended_wired_limit()
    inkling2_layers._USE_SHORT_CONV_KERNEL = True
    inkling2_layers._USE_SPLITK_ATTN_KERNEL = True
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    variants = [
        ("Original (NVFP4/MXFP8/BF16)", "orig_nvfp4_mxfp8"),
        ("W6A16 (6-bit affine)", "w6a16"),
        ("W4W8-Native-BF16 (4b L0-19, 8b L20-23 + BF16 attn/head)", "w4w8_native_bf16"),
        ("W4W8-Native-A8 (4b L0-19, 8b L20-23 + 8b attn/head)", "w4w8_native_a8"),
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
            [{"role": "user", "content": WORKFLOW_PROMPT}],
            add_generation_prompt=True,
            tokenize=True,
            reasoning_effort=args.effort,
        )
        print(
            f"[INFO] {label} ready in {load_s:.1f}s (bpw={bpw:.3f}, prompt={len(prompt_tokens)} tokens, "
            f"effort={args.effort}). Warming up..."
        )

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

        passed, total, details = run_workflow_test_suite(code)
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

    print(f"\n### Tough DAG Workflow Engine Summary (effort={args.effort}, max_tokens={args.max_tokens}, MTP Depth = 2)\n")
    print(
        "| Quant Recipe | Bits/Weight | Peak Mem | Total Gen Tok (`Think` / `Code`) | Decode tok/s (`d=2`) | Mean Acc (`Accept %`) | Total Wall Time | Test Score |"
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
