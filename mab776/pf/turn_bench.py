#!/usr/bin/env python3
"""mab776 overnight: agent-style turns on a Strata server - a fresh prompt, then follow-up user turns of N new tokens
on the same conversation (the KV prefix is reused, so only the new tokens are prefilled).  Prints one JSON line per
turn from the server's own /metrics timings, like strata_bench.py.

  python3 turn_bench.py URL --nonce X [--base 8000] [--turns 300,1500] [--gen 32] [--label L]
"""
import argparse, json, os, sys, time, urllib.request

sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
from ple_bench import corpus  # noqa: E402
from strata_bench import call, last_request  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--nonce", required=True)
    ap.add_argument("--base", type=int, default=8000)
    ap.add_argument("--turns", default="300,1500")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--ratio", type=float, default=3.1)
    ap.add_argument("--offset", type=int, default=7000000)
    ap.add_argument("--label", default="")
    ap.add_argument("--temp", type=float, default=float(__import__("os").environ.get("TURN_TEMP", "0.7")))
    a = ap.parse_args()
    text = corpus()
    at = a.offset
    def chunk(ntok):
        nonlocal at
        n = int(ntok * a.ratio)
        s = text[at:at + n]
        at += n
        return s
    msgs = [{"role": "user", "content": f"[run {a.nonce}]\n" + chunk(a.base)}]
    for i, nt in enumerate([0] + [int(x) for x in a.turns.split(",") if x]):
        if nt:
            msgs.append({"role": "user", "content": chunk(nt)})
        t0 = time.time()
        r = call(a.url, "/v1/chat/completions", {"messages": msgs, "max_tokens": a.gen, "temperature": a.temp,
                                                  "top_p": 0.95, "stream": False})
        wall = time.time() - t0
        msgs.append({"role": "assistant", "content": r["choices"][0]["message"].get("content") or "ok"})
        m = last_request(call(a.url, "/metrics"))
        read = m["prompt_tokens"] - (m.get("reused") or 0)
        print(json.dumps({"label": a.label, "turn": i, "new": nt, "depth": m["prompt_tokens"], "reused": m.get("reused"),
                          "read": read, "prefill_s": round(m["prompt_ms"] / 1000, 2) if m.get("prompt_ms") else None,
                          "prefill_tps": round(read / (m["prompt_ms"] / 1000), 1) if m.get("prompt_ms") else None,
                          "decode_tps": m.get("decode_tok_s"), "gen": m.get("output_tokens"),
                          "wall_s": round(wall, 1)}), flush=True)


if __name__ == "__main__":
    main()
