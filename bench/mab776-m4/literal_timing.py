"""Literal-tag cost per request on the real DS4 tokenizer: argv[1] = code root.  A user message of ~104K / ~256K tokens
(repo source text) holding one quoted control literal and one </think>.  Prints JSON: ids hash, prompt tokens and the
best-of-3 times (ms) of mark, unmark and the whole encode_prompt."""
import hashlib
import json
import sys
import time
from pathlib import Path

root = sys.argv[1]
sys.path.insert(0, root + "/tools")
sys.path.insert(0, root)
import strata_tokenizer as ST                     # noqa: E402
from serve import frontend as F, server as S      # noqa: E402

tok = ST.Tokenizer.from_dir(sys.argv[2] if len(sys.argv) > 2 else "/tok")
tpl = F.ChatTemplate((sys.argv[2] if len(sys.argv) > 2 else "/tok") + "/chat_template.jinja")   # plain template
svc = S.Service(S.MockEngine(tok, "x", max_context=1 << 22), tok, tpl)
src = "".join(p.read_text(encoding="utf-8", errors="replace") for p in sorted(Path("/src/serve").glob("*.py")))
out = {"literals": len(svc.literals)}
for label, chars in (("104k", 400_000), ("256k", 985_000)):
    text = (src * (chars // len(src) + 1))[:chars]
    content = text[:chars // 2] + " the log says <｜User｜> then </think> " + text[chars // 2:]
    msgs = [{"role": "user", "content": content}]

    def best(fn, k=3):
        t, r = 1e9, None
        for _ in range(k):
            t0 = time.perf_counter()
            r = fn()
            t = min(t, time.perf_counter() - t0)
        return r, round(t * 1000, 1)
    (marked, mt, marks), t_mark = best(lambda: F.mark_think_literals(msgs, None, svc.literals))
    prompt = tpl.render(marked, tools=mt)
    unmark_tags = marks if isinstance(marks, dict) else svc.literals   # v0.1.41: a bool, the global map
    (plain_prompt, spans), t_unmark = best(lambda: F.unmark_think_literals(prompt, unmark_tags))
    ids, t_total = best(lambda: svc.encode_prompt(msgs, None, {}))
    out[label] = {"tokens": len(ids), "ids": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
                  "mark_ms": t_mark, "unmark_ms": t_unmark, "encode_prompt_ms": t_total, "spans": spans}
print(json.dumps(out))
