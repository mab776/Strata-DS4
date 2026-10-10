import json, sys
from pathlib import Path
sys.path.insert(0, "/src"); sys.path.insert(0, "/src/tools")
import strata_tokenizer as ST
from serve import frontend as F
from serve.deepseek import DeepSeekTemplate
tok = ST.Tokenizer.from_dir("/tok")
tpl = DeepSeekTemplate(Path("/src/serve/fixtures/ds4/chat_template.jinja"), tok)
print("caps", tpl.caps)
M2 = Path("/src/bench/mab776-m2")
ok = 0
for conv in sorted((M2 / "conv").glob("*.json")):
    req = json.loads(conv.read_text(encoding="utf-8"))
    want = json.loads((M2 / "renders" / (conv.stem + ".render.json")).read_text(encoding="utf-8"))["prompt"]
    msgs, tools, kw = F.openai_to_messages(req, tpl)
    got = tpl.render(msgs, tools, **kw)
    same = got == want
    ok += same
    note = ""
    if not same:
        k = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
        note = f"\n   at {k}: got {got[max(0,k-40):k+80]!r}\n   want {want[max(0,k-40):k+80]!r}"
    print(f"{conv.stem:42s} {'EQUAL' if same else 'DIFF'}{note}")
print(ok, "of 14")
