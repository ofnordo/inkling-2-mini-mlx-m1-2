#!/usr/bin/env python3
"""Agentic benchmark harness for Inkling-2-Mini (W4W8-Native + custom kernels + DSpark MTP).

Runs multi-turn native tool calling:
  model:  <|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|> ... <|content_model_end_sampling|>
  tools:  <|message_tool|>NAME<|content_text|>RESULT<|end_message|>

The conversation is kept as a raw token stream and the KV cache / conv state / MTP drafter ring
persist across turns, so each turn only prefills the new tool-result tokens.
"""

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
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import mlx.core as mx

from agent_tasks import Task, ToolError, all_tasks
from agent_tasks_tier2 import tier2_tasks
from inkling_mlx import DEFAULT_MODEL_PATH, load_inkling

TOK_EOS = 200006  # <|content_model_end_sampling|>
OUT_ROOT = Path(__file__).resolve().parent / "agent_bench_outputs"


class Session:
    """Persistent single-sequence generation state (cache + drafter ring) across turns."""

    def __init__(self, model, tokenizer, drafter, depth: int, prefill_step: int = 512):
        self.model = model
        self.tok = tokenizer
        self.drafter = drafter if depth > 0 else None
        self.depth = depth
        self.prefill_step = prefill_step
        self.cache = model.make_speculative_cache() if depth > 0 else model.make_cache()
        self.capture = {lid: None for lid in drafter.target_layer_ids} if self.drafter else None
        if self.drafter:
            self.drafter.reset()
        self.pending_taps: list[mx.array] | None = None
        self.n_ctx = 0
        self.prefill_s = 0.0
        self.prefill_tokens = 0
        self.decode_s = 0.0
        self.decode_tokens = 0
        self.steps = 0
        self.drafted = 0
        self.accepted = 0

    # -- drafter ring ingestion (same math as Inkling2DSparkDrafter.draft's first half) --
    def _ingest(self, taps: list[mx.array]) -> None:
        d = self.drafter
        ctx = mx.concatenate(taps, axis=-1)
        n_total = int(ctx.shape[1])
        if n_total == 0:
            return
        skip = max(0, n_total - d.ring_slots)
        if skip:
            ctx = ctx[:, skip:]
        n = int(ctx.shape[1])
        base = d.fed + skip
        d.fed = base + n
        slots = mx.arange(base, base + n, dtype=mx.int32) % d.ring_slots
        d.pos[slots] = mx.arange(base, base + n, dtype=mx.int32)
        for li, layer in enumerate(d.model.layers):
            attn = layer.self_attn
            ck, cv = attn.project_context_kv(ctx)
            ck = attn.k_norm_ctx(ck.reshape(1, n, attn.n_kv_heads, attn.head_dim)).transpose(0, 2, 1, 3)
            cv = cv.reshape(1, n, attn.n_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            ck = d.model.rope(ck, offset=base)
            d.ring_keys[li][:, :, slots, :] = ck
            d.ring_values[li][:, :, slots, :] = cv
        mx.eval(*d.ring_keys, *d.ring_values, d.pos)

    def _taps(self) -> list[mx.array]:
        return [self.capture[lid] for lid in self.drafter.target_layer_ids]

    def prefill(self, tokens: list[int]) -> int:
        """Feed new tokens; return greedy next token (not yet fed)."""
        t0 = time.perf_counter()
        if self.drafter and self.pending_taps is not None:
            self._ingest(self.pending_taps)
            self.pending_taps = None
        arr = mx.array([tokens], dtype=mx.int32)
        n = len(tokens)
        pos = 0
        while n - pos > 1:
            c = min(self.prefill_step, n - pos - 1)
            self.model(arr[:, pos : pos + c], cache=self.cache, capture=self.capture)
            if self.drafter:
                taps = self._taps()
                mx.eval(*taps)
                self._ingest(taps)
            else:
                mx.eval([s.state for row in self.cache for s in row])
            pos += c
        logits = self.model(arr[:, pos:], cache=self.cache, capture=self.capture)
        nxt = mx.argmax(logits[:, -1, :], axis=-1)
        if self.drafter:
            self.pending_taps = self._taps()
            mx.eval(nxt, *self.pending_taps)
        else:
            mx.eval(nxt)
        self.n_ctx += n
        self.prefill_s += time.perf_counter() - t0
        self.prefill_tokens += n
        return int(nxt.item())

    def generate(self, first_tok: int, max_tokens: int) -> tuple[list[int], bool]:
        """Greedy (d=0) or DSpark-MTP decode. Returns (tokens excluding EOS, hit_eos).

        Invariant on return: cache holds every returned token (EOS is NOT fed)."""
        t0 = time.perf_counter()
        out: list[int] = []
        cur = first_tok
        if cur == TOK_EOS:
            return out, True
        out.append(cur)
        hit_eos = False
        if self.depth == 0:
            while len(out) < max_tokens:
                logits = self.model(mx.array([[cur]], dtype=mx.int32), cache=self.cache)
                nxt = mx.argmax(logits[:, -1, :], axis=-1)
                mx.eval(nxt)
                self.n_ctx += 1
                self.steps += 1
                cur = int(nxt.item())
                if cur == TOK_EOS:
                    hit_eos = True
                    break
                out.append(cur)
            else:
                # last token not fed yet; feed it so cache holds all returned tokens
                self.model(mx.array([[cur]], dtype=mx.int32), cache=self.cache)
                self.n_ctx += 1
        else:
            d = self.drafter
            pending = self.pending_taps
            self.pending_taps = None
            while True:
                if len(out) >= max_tokens:
                    # feed `cur` (not yet in cache) so invariant holds
                    self.model(mx.array([[cur]], dtype=mx.int32), cache=self.cache, capture=self.capture)
                    self.n_ctx += 1
                    if pending is not None:
                        self._ingest(pending)
                    self.pending_taps = self._taps()
                    mx.eval(*self.pending_taps)
                    break
                depth = self.depth
                drafts = d.draft(pending, cur, depth)
                dl = drafts.tolist()
                inp = mx.concatenate([mx.array([cur], dtype=mx.int32), drafts])[None]
                for e in self.cache:
                    e[1].record_verify_windows = True
                v = self.model(inp, cache=self.cache, capture=self.capture)
                for e in self.cache:
                    e[1].record_verify_windows = False
                g = mx.argmax(v[0], axis=-1)
                mx.eval(g, *self._taps())
                gl = g.tolist()
                self.steps += 1
                self.drafted += depth
                acc = 0
                new: list[int] = []
                for i in range(depth):
                    p = int(gl[i])
                    if p == TOK_EOS:
                        hit_eos = True
                        break
                    new.append(p)
                    if p == int(dl[i]):
                        acc += 1
                    else:
                        break
                if not hit_eos and acc == depth:
                    b = int(gl[depth])
                    if b == TOK_EOS:
                        hit_eos = True
                    else:
                        new.append(b)
                # committed fed tokens: cur + first `acc` drafts
                self.accepted += acc
                if acc < depth:
                    self.model.rollback_speculative_cache(self.cache, acc, depth)
                keep = 1 + acc
                self.n_ctx += keep
                pending = [t[:, :keep, :] for t in self._taps()]
                if hit_eos:
                    out.extend(new)
                    self.pending_taps = pending
                    break
                # new[-1] is the correction/bonus token: not yet fed
                out.extend(new)
                cur = new[-1]
        self.decode_s += time.perf_counter() - t0
        self.decode_tokens += len(out)
        return out, hit_eos


MSG_RE = re.compile(r"(?P<name>[^<]*)<\|(?P<ctype>content_[a-z_]+)\|>(?P<body>.*?)(?:<\|end_message\|>|$)", re.S)


def parse_turn(text: str) -> list[dict]:
    msgs = []
    for seg in text.split("<|message_model|>"):
        if not seg.strip():
            continue
        m = MSG_RE.match(seg)
        if not m:
            msgs.append({"ctype": "unparsed", "name": "", "body": seg})
            continue
        msgs.append({"ctype": m.group("ctype"), "name": m.group("name").strip(), "body": m.group("body")})
    return msgs


def run_task(task: Task, model, tok, drafter, *, depth: int, effort: str, turn_tokens: int,
             total_tokens: int, out_dir: Path, verbose: bool) -> dict:
    sandbox = Path(tempfile.mkdtemp(prefix=f"agent_{task.name}_"))
    task.setup(sandbox)
    msgs = [{"role": "system", "content": task.system}, {"role": "user", "content": task.user}]
    prompt = list(tok.apply_chat_template(msgs, tools=task.tools(), add_generation_prompt=True,
                                          tokenize=True, reasoning_effort=effort))
    sess = Session(model, tok, drafter, depth)
    transcript: list[int] = list(prompt)
    stats = {"turns": 0, "tool_calls": 0, "tool_errors": 0, "malformed_calls": 0, "nudges": 0,
             "think_tokens": 0, "gen_tokens": 0, "stop": ""}
    t_start = time.perf_counter()
    feed = prompt
    final_text = ""
    try:
        while True:
            if stats["turns"] >= task.max_turns:
                stats["stop"] = "max_turns"
                break
            if stats["gen_tokens"] >= total_tokens:
                stats["stop"] = "token_budget"
                break
            first = sess.prefill(feed)
            toks, hit_eos = sess.generate(first, min(turn_tokens, total_tokens - stats["gen_tokens"]))
            stats["turns"] += 1
            stats["gen_tokens"] += len(toks)
            transcript.extend(toks + ([TOK_EOS] if hit_eos else []))
            text = tok.decode(toks)
            parsed = parse_turn(text)
            calls, results = [], []
            for p in parsed:
                if p["ctype"] == "content_thinking":
                    stats["think_tokens"] += len(tok.encode(p["body"], add_special_tokens=False))
                elif p["ctype"] == "content_text":
                    final_text = p["body"]
                elif p["ctype"].startswith("content_invoke_tool"):
                    calls.append(p)
            if verbose:
                for p in parsed:
                    body = p["body"].strip().replace("\n", " ")
                    print(f"    [{p['ctype'].replace('content_', '')}{':' + p['name'] if p['name'] else ''}] "
                          f"{body[:220]}{'…' if len(body) > 220 else ''}")
            if not hit_eos:
                stats["stop"] = "turn_token_limit"
                break
            if not calls:
                if task.done:
                    stats["stop"] = "done"
                    break
                if stats["nudges"] >= 1:
                    stats["stop"] = "no_tool_call"
                    break
                stats["nudges"] += 1
                nudge = ("Continue working on the task using the tools. When you are finished, call the "
                         "finish / submit_answer tool.")
                feed_s = f"<|message_user|><|content_text|>{nudge}<|end_message|>"
                feed = [TOK_EOS] + tok.encode(feed_s, add_special_tokens=False)
                transcript.extend(feed[1:])
                continue
            feed_s = ""
            for c in calls:
                stats["tool_calls"] += 1
                name = c["name"]
                try:
                    obj = json.loads(c["body"])
                    name = obj.get("name", name)
                    args = obj.get("args", obj.get("arguments", {})) or {}
                    if not isinstance(args, dict):
                        raise ValueError("args must be an object")
                except Exception as e:  # noqa: BLE001
                    stats["malformed_calls"] += 1
                    res = f"ERROR: could not parse tool call JSON ({e}). Use {{\"name\": ..., \"args\": {{...}}}}."
                else:
                    try:
                        res = task.call(name, args)
                    except ToolError as e:
                        stats["tool_errors"] += 1
                        res = f"ERROR: {e}"
                    except Exception as e:  # noqa: BLE001
                        stats["tool_errors"] += 1
                        res = f"ERROR: {type(e).__name__}: {e}"
                if verbose:
                    print(f"      -> {name}: {res[:200].replace(chr(10), ' ')}{'…' if len(res) > 200 else ''}")
                feed_s += f"<|message_tool|>{name}<|content_text|>{res}<|end_message|>"
            feed = [TOK_EOS] + tok.encode(feed_s, add_special_tokens=False)
            transcript.extend(feed[1:])
            if task.done:
                stats["stop"] = "done"
                break
    except Exception:  # noqa: BLE001
        stats["stop"] = "harness_error: " + traceback.format_exc(limit=3)
    wall = time.perf_counter() - t_start
    try:
        verdict = task.check(final_text)
    except Exception as e:  # noqa: BLE001
        verdict = {"success": False, "score": 0.0, "detail": f"checker error {e}"}
    (out_dir / f"{task.name}.txt").write_text(tok.decode(transcript))
    shutil.rmtree(sandbox, ignore_errors=True)
    res = {
        "task": task.name, "category": task.category, **verdict, **stats,
        "wall_s": wall, "ctx_tokens": sess.n_ctx,
        "prefill_tokens": sess.prefill_tokens, "prefill_s": sess.prefill_s,
        "decode_tokens": sess.decode_tokens, "decode_s": sess.decode_s,
        "decode_tps": sess.decode_tokens / max(sess.decode_s, 1e-9),
        "accept_rate": sess.accepted / max(sess.drafted, 1) if depth else 0.0,
    }
    del sess
    gc.collect()
    mx.clear_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--suite", choices=["tier1", "tier2", "all"], default="tier1")
    ap.add_argument("--only", default=None, help="comma-separated substrings of task names")
    ap.add_argument("--turn-tokens", type=int, default=8192)
    ap.add_argument("--total-tokens", type=int, default=48000)
    ap.add_argument("--tag", default=None)
    ap.add_argument("-q", "--quiet", action="store_true")
    args = ap.parse_args()

    tag = args.tag or f"{args.model.name}_d{args.depth}_{args.effort}_{time.strftime('%m%d-%H%M')}"
    out_dir = OUT_ROOT / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model, tok, drafter = load_inkling(args.model, mtp_depth=args.depth)

    if args.suite == "tier1":
        tasks = all_tasks()
    elif args.suite == "tier2":
        tasks = tier2_tasks()
    else:
        tasks = all_tasks() + tier2_tasks()
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        tasks = [t for t in tasks if any(k in t.name for k in keys)]

    results = []
    for t in tasks:
        print(f"\n=== {t.name} ({t.category}) ===", flush=True)
        r = run_task(t, model, tok, drafter, depth=args.depth, effort=args.effort,
                     turn_tokens=args.turn_tokens, total_tokens=args.total_tokens,
                     out_dir=out_dir, verbose=not args.quiet)
        results.append(r)
        print(f"  RESULT {'PASS' if r['success'] else 'FAIL'} score={r['score']:.2f} turns={r['turns']} "
              f"calls={r['tool_calls']} err={r['tool_errors']} malformed={r['malformed_calls']} "
              f"gen={r['gen_tokens']} (think={r['think_tokens']}) wall={r['wall_s']:.0f}s "
              f"decode={r['decode_tps']:.1f} tok/s acc={r['accept_rate']*100:.0f}% stop={r['stop'][:60]}\n"
              f"  detail: {r['detail'][:400]}", flush=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n### Agentic benchmark: {args.model.name}, MTP d={args.depth}, effort={args.effort}\n")
    print("| Task | Category | Result | Score | Turns | Tool calls (err/malformed) | Gen tok (think) | Wall | Decode tok/s |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['task']} | {r['category']} | {'✅' if r['success'] else '❌'} | {r['score']:.2f} | {r['turns']} | "
              f"{r['tool_calls']} ({r['tool_errors']}/{r['malformed_calls']}) | {r['gen_tokens']} ({r['think_tokens']}) | "
              f"{r['wall_s']:.0f}s | {r['decode_tps']:.1f} |")
    n_pass = sum(r["success"] for r in results)
    print(f"\nPassed {n_pass}/{len(results)} | mean score {sum(r['score'] for r in results)/len(results):.2f} | "
          f"total wall {sum(r['wall_s'] for r in results)/60:.1f} min")


if __name__ == "__main__":
    main()
