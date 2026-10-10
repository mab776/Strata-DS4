#!/usr/bin/env python3
"""mab776 M2: the SAME prompts stack 73 (llama.cpp ds4-lcpp) was measured on, as token-id files for ds4_generate.

Depth prompts = deepseek-v4/lcpp_depths.py: "[run <id>] " + PARA * k + tail, sent to llama.cpp's raw /completion
(no chat template, no BOS on this GGUF), temperature 0, 512 tokens; tails = the filler continuation and the three
CODE tails.  Needles = deepseek-v4/needle_ds4.py at "250k" (Strata's src-0140 haystack, Random(7) words, CPT 3.431,
depths 10/50/90 %), as ONE user message through the chat template the way stack 73 renders it (--template-prefix /
--template-suffix: the text around the user content, taken from a captured llama.cpp /apply-template render).

    python3 make_prompts.py GGUF OUTDIR [--depths 8192,130048,256000] [--needles 250k]
                            [--template-prefix '<｜begin▁of▁sentence｜><｜User｜>' --template-suffix '<｜Assistant｜><think>']

Writes OUTDIR/<name>.i32 (int32 LE ids) + OUTDIR/prompts.json (name, tokens, kind, target, word, ...).
Needs the `regex` module (tools/strata_tokenizer.py).
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST  # noqa: E402

PARA = ("The quick brown fox jumps over the lazy dog while 12 engineers debug a "
        "distributed cache at 3am, logging metric 4567 and tracing span 89ab. ")
CODE_TAILS = [
    "Write a complete, well-commented Python module implementing an LRU cache with TTL expiry, thread safety and unit tests.",
    "Write a complete TypeScript Express REST API for a todo list with input validation, error handling and Jest tests.",
    "Write a complete Rust command-line tool that counts words, lines and bytes in files like wc, with clap and tests."]
CODE_TAIL = "\n\nIgnore the text above. {}\n\n```\n"

# needle_ds4.py (Strata's tools/needle_bench.py, DeepSeek copy)
HAY_ROOT = pathlib.Path("/media/mab776/nvme 2To/models/strata-data/src-0140")
SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py"),
           ("third_party/llama.cpp/docs", "*.md")]
WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern",
         "meadow", "cobalt", "juniper", "tundra", "velvet", "ember"]


def haystack(root: pathlib.Path, n_chars: int) -> str:
    files = []
    for d, pat in SOURCES:
        files += sorted((root / d).rglob(pat))
    parts, total = [], 0
    while total < n_chars:
        for f in files:
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"\n\n=== {f.relative_to(root).as_posix()} ===\n{t}")
            total += len(parts[-1])
            if total >= n_chars:
                break
        if not files:
            raise SystemExit("no text files found to build the haystack from")
    return "".join(parts)[:n_chars]


def write_ids(path: pathlib.Path, ids: list[int]) -> None:
    a = array.array("i", ids)
    if sys.byteorder != "little":
        a.byteswap()
    path.write_bytes(a.tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("out")
    ap.add_argument("--depths", default="8192,130048,256000")
    ap.add_argument("--needles", default="250k", help="needle length ('' = none)")
    ap.add_argument("--needle-depths", default="10,50,90")
    ap.add_argument("--cpt", type=float, default=3.431)
    ap.add_argument("--haystack-root", default=str(HAY_ROOT))
    ap.add_argument("--template-prefix", default="<｜begin▁of▁sentence｜><｜User｜>")
    ap.add_argument("--template-suffix", default="<｜Assistant｜><think>")
    ap.add_argument("--write-text", action="store_true", help="also write <name>.txt (the exact text tokenized)")
    ap.add_argument("--paras", default="8192:241,130048:3825,256000:7529",
                    help="depth:paragraphs = the counts stack 73's lcpp_depths.py runs used (its filler chars "
                         "33,783 / 535,543 / 1,054,103); a depth not listed is fitted to the target")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = ST.Tokenizer.from_gguf(a.gguf)
    rows = []
    para_n = len(tok.encode(PARA * 64)) / 64.0          # tokens per PARA in a long run
    fixed = {int(k): int(v) for k, v in (x.split(":") for x in a.paras.split(",") if x)}
    for depth in (int(x) for x in a.depths.split(",") if x):
        for kind, tail in [("filler", "")] + [(f"code{i + 1}", CODE_TAIL.format(t)) for i, t in enumerate(CODE_TAILS)]:
            run_id = hashlib.sha1(f"{depth}-{kind}".encode()).hexdigest()[:8]
            head = f"[run {run_id}] "
            k = fixed.get(depth) or max(1, int((depth - len(tok.encode(head + tail))) / para_n))
            text = head + PARA * k + tail
            ids = tok.encode(text, parse_special=True)
            while depth not in fixed and len(ids) > depth and k > 1:   # land at or just under the target
                k -= 1
                text = head + PARA * k + tail
                ids = tok.encode(text, parse_special=True)
            name = f"depth{depth}-{kind}"
            write_ids(out / f"{name}.i32", ids)
            if a.write_text:
                (out / f"{name}.txt").write_bytes(text.encode("utf-8"))
            rows.append({"name": name, "kind": kind, "target": depth, "tokens": len(ids), "paras": k})
            print(json.dumps(rows[-1]), flush=True)
    if a.needles:
        L = a.needles
        tokens = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
        tokens = int(tokens * 0.98)
        text = haystack(pathlib.Path(a.haystack_root), int(tokens * a.cpt))
        rnd = random.Random(7)
        for d in (int(x) for x in a.needle_depths.split(",")):
            word = f"{rnd.choice(WORDS)}-{rnd.choice(WORDS)}-{rnd.randint(100, 999)}"
            cut = int(len(text) * d / 100)
            cut = text.rfind("\n", 0, cut) + 1 or cut
            needle = f"\nThe secret code word for this text is: {word}. Remember it.\n"
            prompt = (text[:cut] + needle + text[cut:] +
                      "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only.")
            # the user content is plain text (parse_special=False, as llama.cpp tokenizes message CONTENT inside the
            # rendered template with parse_special=True - the haystack has no special-token literals, checked below)
            full = a.template_prefix + prompt + a.template_suffix
            ids = tok.encode(full, parse_special=True)
            name = f"needle{L}-d{d}"
            write_ids(out / f"{name}.i32", ids)
            if a.write_text:
                (out / f"{name}.txt").write_bytes(full.encode("utf-8"))
            rows.append({"name": name, "kind": "needle", "target": tokens, "tokens": len(ids), "depth_pct": d,
                         "word": word, "chars": len(prompt),
                         "special_in_content": sum(prompt.count(s) for s in tok.special_tokens if len(s) > 3)})
            print(json.dumps(rows[-1]), flush=True)
    (out / "prompts.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
