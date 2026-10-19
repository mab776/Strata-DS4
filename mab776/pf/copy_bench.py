#!/usr/bin/env python3
"""mab776: copy-heavy decode bench (prompt-lookup drafts): a source file + "output it again with one rename", the
answer saved for an exact diff between engine variants (greedy) and the server's own decode timing printed.
  python3 copy_bench.py URL FILE --old NAME --new NAME --label L [--temp 0] [--max 3000] [--out DIR]"""
import argparse, json, os, sys, time
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
from strata_bench import call, last_request  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("file")
ap.add_argument("--old", required=True); ap.add_argument("--new", required=True)
ap.add_argument("--label", default=""); ap.add_argument("--temp", type=float, default=0.0)
ap.add_argument("--max", type=int, default=3000); ap.add_argument("--out", default="")
ap.add_argument("--nonce", default="", help="a first line that makes the prompt new (a repeated prompt takes the engine's regenerate path: no DSpark)")
a = ap.parse_args()
src = open(a.file).read()
msg = (f"[run {a.nonce}]\n" if a.nonce else "") + (f"Here is a file:\n\n```\n{src}\n```\n\nRename every occurrence of `{a.old}` to `{a.new}` and output the COMPLETE "
       "file again, unchanged otherwise. Output only the code block, no explanation.")
t0 = time.time()
r = call(a.url, "/v1/chat/completions", {"messages": [{"role": "user", "content": msg}], "max_tokens": a.max,
                                         "temperature": a.temp, "top_p": 0.95, "stream": False})
wall = time.time() - t0
m = r["choices"][0]["message"]
text = m.get("content") or ""
if a.out:
    os.makedirs(a.out, exist_ok=True)
    open(os.path.join(a.out, f"{a.label}.txt"), "w").write((m.get("reasoning_content") or "") + "\n=====\n" + text)
lr = last_request(call(a.url, "/metrics"))
print(json.dumps({"label": a.label, "temp": a.temp, "prompt": lr.get("prompt_tokens"), "gen": lr.get("output_tokens"),
                  "decode_tps": lr.get("decode_tok_s"), "wall_s": round(wall, 1), "copied_ok": src.replace(a.old, a.new).strip()[:200] in text,
                  "finish": r["choices"][0].get("finish_reason")}), flush=True)
