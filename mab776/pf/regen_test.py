#!/usr/bin/env python3
"""mab776: the regenerate path - the same greedy prompt three times (the 2nd and 3rd take the snapshot + saved logits,
no prefill).  The output must not change; with STRATA_DSPARK_SNAP=1 the repeats draft with DSpark from the first token.
  regen_test.py URL LABEL"""
import json, os, sys
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
from ple_bench import corpus  # noqa: E402
from strata_bench import call, last_request  # noqa: E402

url, label = sys.argv[1], sys.argv[2]
text = corpus()
msgs = [{"role": "user", "content": text[5000000:5000000 + 6000] + "\n\nWrite a detailed summary of the text above."}]
first = None
prev = None
for i in range(3):
    r = call(url, "/v1/chat/completions", {"messages": msgs, "max_tokens": 384, "temperature": 0, "stream": False})
    m = last_request(call(url, "/metrics"))
    msg = r["choices"][0]["message"]
    out = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    first = out if first is None else first
    print(json.dumps({"label": label, "run": i, "depth": m["prompt_tokens"], "reused": m.get("reused"),
                      "prompt_s": round(m["prompt_ms"] / 1000, 2), "decode_tps": m.get("decode_tok_s"),
                      "gen": m.get("output_tokens"), "drafts": [m.get("drafts_offered"), m.get("drafts_accepted")],
                      "same_as_first": out == first, "same_as_prev": out == prev}), flush=True)
    prev = out
    if os.environ.get("REGEN_OUT"):
        open(os.environ["REGEN_OUT"] + f".{i}.txt", "w").write(out)
