#!/usr/bin/env python3
"""mab776: a real ~500k check - a code word hidden at 50 % of a ~500k-token text, then an unrelated conversation (the
big one gets parked), then a follow-up on the big conversation (restored from the park store, no 500k re-read).
  python3 long_park.py URL [--tokens 500000] [--label L]"""
import argparse, json, sys, time
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
from ple_bench import corpus  # noqa: E402
from strata_bench import call, last_request  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("--tokens", type=int, default=500000)
ap.add_argument("--ratio", type=float, default=3.1); ap.add_argument("--label", default="")
a = ap.parse_args()
text = corpus()
n = int(a.tokens * a.ratio)
body = text[1000000:1000000 + n]
word = "TANGERINE-4471"
mid = len(body) // 2
mid = body.rfind("\n", 0, mid) + 1
body = body[:mid] + f"\nThe secret code word is {word}. Remember it.\n" + body[mid:]
q = "\n\nWhat is the secret code word written in the text above? Reply with the code word only."
def ask(msgs, max_tokens=64):
    t0 = time.time()
    r = call(a.url, "/v1/chat/completions", {"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "stream": False},)
    m = last_request(call(a.url, "/metrics"))
    c = r["choices"][0]["message"].get("content") or ""
    return c, m, time.time() - t0
big = [{"role": "user", "content": body + q}]
c1, m1, w1 = ask(big, 256)
print(json.dumps({"label": a.label, "step": "needle", "depth": m1["prompt_tokens"], "reused": m1.get("reused"),
                  "prompt_s": round(m1["prompt_ms"] / 1000, 1), "found": word in c1, "answer": c1[-80:], "wall_s": round(w1, 1)}), flush=True)
other = [{"role": "user", "content": text[9000000:9000000 + int(3000 * a.ratio)] + "\n\nSummarize the text above in one sentence."}]
c2, m2, w2 = ask(other, 32)
print(json.dumps({"label": a.label, "step": "other", "depth": m2["prompt_tokens"], "reused": m2.get("reused"),
                  "prompt_s": round(m2["prompt_ms"] / 1000, 1), "wall_s": round(w2, 1)}), flush=True)
follow = big + [{"role": "assistant", "content": c1 or word}, {"role": "user", "content": "Write the code word once more, then nothing else."}]
c3, m3, w3 = ask(follow, 64)
print(json.dumps({"label": a.label, "step": "follow-up", "depth": m3["prompt_tokens"], "reused": m3.get("reused"),
                  "read": m3["prompt_tokens"] - (m3.get("reused") or 0), "prompt_s": round(m3["prompt_ms"] / 1000, 2),
                  "found": word in c3, "answer": c3[-80:], "wall_s": round(w3, 1)}), flush=True)
