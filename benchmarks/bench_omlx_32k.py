#!/usr/bin/env python3
"""Benchmark oMLX 0.8.3 (with its built-in inkling2 + dspark + kernels) up to 32k context
on the exact same Inkling-2-Mini-MLX-W4W8-Native model and prompts as bench_vs_mainline_32k.py.
"""

from __future__ import annotations

import os, sys
from pathlib import Path as _Path
for _d in (_Path(__file__).resolve().parent, _Path(__file__).resolve().parent.parent, _Path(__file__).resolve().parent.parent / "convert"):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import json
import subprocess
import time
import urllib.request
from pathlib import Path

from mlx_lm.utils import load_tokenizer

MODEL_DIR = Path("/tmp/omlx_bench_models")
MODEL_ID = "Inkling-2-Mini"
BOOK_PATH = Path("niah_512k/thursday.txt")
PORT = 8787
BASE_URL = f"http://127.0.0.1:{PORT}"
CONTEXTS = [512, 4096, 16384, 32768]
GEN_TOKENS = 256


def wait_for_server(timeout_s: float = 120.0) -> list[str]:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=2) as resp:
                data = json.loads(resp.read().decode())
                ids = [m["id"] for m in data.get("data", [])]
                if ids:
                    return ids
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("Timed out waiting for omlx serve")


def build_user_prompt(tok, target_tokens: int) -> str:
    book = BOOK_PATH.read_text(errors="ignore")
    toks = tok.encode(book, add_special_tokens=False)
    head = "Read the following passage carefully and then write a concise literary analysis of its tone, setting, and characters:\n\n"
    tail = "\n\nProvide your analysis now."
    overhead = len(tok.encode(head + tail, add_special_tokens=False)) + 28
    body_len = max(16, target_tokens - overhead)
    body = tok.decode(toks[:body_len])
    return head + body + tail


def run_streaming_request(prompt_text: str, max_tokens: int) -> dict:
    payload = json.dumps(
        {
            "model": MODEL_ID,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_start = time.perf_counter()
    t_first = None
    chunks = 0
    usage = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            obj = json.loads(body)
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    if t_first is None:
                        t_first = time.perf_counter()
                    chunks += 1
    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first is not None else (t_end - t_start)
    dec_s = max(t_end - (t_first or t_start), 1e-6)
    p_tok = int(usage.get("prompt_tokens", 0))
    c_tok = int(usage.get("completion_tokens", chunks))
    return {
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "ttft_s": ttft,
        "prefill_tps": p_tok / max(ttft, 1e-6),
        "decode_s": dec_s,
        "decode_tps": max(c_tok - 1, 1) / dec_s,
        "wall_s": t_end - t_start,
        "usage": usage,
    }


def main() -> None:
    tok = load_tokenizer(str(MODEL_DIR / MODEL_ID))
    log_file = open("/tmp/omlx_server.log", "w")
    proc = subprocess.Popen(
        [
            "omlx",
            "serve",
            "--model-dir",
            str(MODEL_DIR),
            "--port",
            str(PORT),
            "--no-cache",
            "--headless",
        ],
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    try:
        print("[oMLX] Starting omlx 0.8.3 server on port 8787...", flush=True)
        models = wait_for_server()
        print(f"[oMLX] Ready! Discovered models: {models}", flush=True)

        # Warmup (also triggers model load)
        print("[oMLX] Running warmup request (loads model)...", flush=True)
        w = run_streaming_request("Say hello in 5 words.", max_tokens=16)
        print(f"[oMLX] Warmup done in {w['wall_s']:.1f}s (usage={w['usage']})", flush=True)

        rows = []
        for ctx in CONTEXTS:
            prompt = build_user_prompt(tok, ctx)
            r = run_streaming_request(prompt, max_tokens=GEN_TOKENS)
            rows.append(r)
            print(
                f"[oMLX 0.8.3] ctx={r['prompt_tokens']:>6,} | prefill={r['prefill_tps']:6.1f} tok/s ({r['ttft_s']:5.1f}s) | "
                f"decode={r['decode_tps']:5.2f} tok/s ({r['completion_tokens']} tok in {r['decode_s']:.1f}s)",
                flush=True,
            )
        Path("/tmp/bench_32k_omlx.json").write_text(json.dumps(rows, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()


if __name__ == "__main__":
    main()
