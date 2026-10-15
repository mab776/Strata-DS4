#!/usr/bin/env python3
"""mab776: conversation-parking bench - several conversations, their turns in a given order; every assistant reply in
the history is a fixed "ok" (the prompts are then the same in every run, whatever the engine generates), greedy.
  python3 park_bench.py URL --plan A1,B1,A2,C1,B2,A3 [--base 20000] [--turn 300] [--label L] [--nonce X]
One JSON line per step: the server's prompt time, reused tokens, read tokens."""
import argparse, json, sys, time
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
from ple_bench import corpus  # noqa: E402
from strata_bench import call, last_request  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("--plan", required=True)
ap.add_argument("--base", type=int, default=20000); ap.add_argument("--turn", type=int, default=300)
ap.add_argument("--ratio", type=float, default=3.1); ap.add_argument("--label", default=""); ap.add_argument("--nonce", default="park")
a = ap.parse_args()
text = corpus()
offs = {c: 2000000 + 4000000 * i for i, c in enumerate("ABCDEFGH")}
def chunk(conv, k, ntok):   # the k-th text block of a conversation (0 = its base)
    start = offs[conv] + (0 if k == 0 else int(a.base * a.ratio) + (k - 1) * int(a.turn * a.ratio))
    return text[start:start + int(ntok * a.ratio)]
for step in a.plan.split(","):
    conv, turn = step[0], int(step[1:])
    msgs = [{"role": "user", "content": f"[{a.nonce} {conv}]\n" + chunk(conv, 0, a.base)}]
    for k in range(1, turn):
        msgs += [{"role": "assistant", "content": "ok"}, {"role": "user", "content": chunk(conv, k, a.turn)}]
    t0 = time.time()
    r = call(a.url, "/v1/chat/completions", {"messages": msgs, "max_tokens": 16, "temperature": 0, "stream": False})
    m = last_request(call(a.url, "/metrics"))
    print(json.dumps({"label": a.label, "step": step, "depth": m["prompt_tokens"], "reused": m.get("reused"),
                      "read": m["prompt_tokens"] - (m.get("reused") or 0),
                      "prompt_s": round(m["prompt_ms"] / 1000, 3) if m.get("prompt_ms") else None,
                      "wall_s": round(time.time() - t0, 2)}), flush=True)
