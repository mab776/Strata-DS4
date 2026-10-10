#!/usr/bin/env python3
"""mab776 M3: token-id parity of tools/strata_tokenizer.py against llama.cpp's llama-tokenize.

    python3 tok_parity.py ours GGUF|TOKDIR INDIR OUTDIR   # our ids for every INDIR/*.txt (parse_special: *.nps.txt = off)
    python3 tok_parity.py compare INDIR OUTDIR            # OUTDIR/<name>.lcpp (llama-tokenize --ids) vs <name>.ours
    python3 tok_parity.py make-adversarial INDIR          # write the adversarial text set

llama-tokenize side (CPU, vocab-only, the same b11429 image as stack 73), see tok_parity.sh.  Files ending in .nps.txt
are tokenized with parse_special OFF on both sides (llama-tokenize --no-parse-special).  No BOS on either side
(the DS4 GGUF has add_bos_token false; the chat template writes the BOS text itself).
"""
from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))


def read_text(p: pathlib.Path) -> str:
    return p.read_bytes().decode("utf-8")      # exact bytes, no newline translation


def ours(gguf: str, indir: str, outdir: str) -> int:
    import strata_tokenizer as ST
    # a GGUF, or a tokenizer/ directory (extract()) = the server's path (Tokenizer.from_dir)
    tok = ST.Tokenizer.from_dir(gguf) if pathlib.Path(gguf).is_dir() else ST.Tokenizer.from_gguf(gguf)
    out = pathlib.Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(pathlib.Path(indir).glob("*.txt")):
        ps = not f.name.endswith(".nps.txt")
        ids = tok.encode(read_text(f), parse_special=ps)
        (out / (f.name[:-4] + ".ours")).write_text(json.dumps(ids))
    return 0


def parse_lcpp(s: str) -> list[int]:
    s = s.strip()
    i, j = s.find("["), s.rfind("]")
    if i < 0 or j < 0:
        raise ValueError("no id list in llama-tokenize output: %r" % s[:200])
    body = s[i + 1:j].strip()
    return [int(x) for x in body.split(",")] if body else []


def compare(indir: str, outdir: str) -> int:
    bad = total = toks = 0
    rows = []
    for f in sorted(pathlib.Path(indir).glob("*.txt")):
        name = f.name[:-4]
        a = pathlib.Path(outdir) / (name + ".lcpp")
        b = pathlib.Path(outdir) / (name + ".ours")
        if not a.exists() or not b.exists():
            print("MISSING %s (lcpp %s, ours %s)" % (name, a.exists(), b.exists()))
            bad += 1
            continue
        la, lb = parse_lcpp(a.read_text()), json.loads(b.read_text())
        total += 1
        toks += len(la)
        if la == lb:
            rows.append((name, len(la), "ok"))
            continue
        bad += 1
        k = next((i for i in range(min(len(la), len(lb))) if la[i] != lb[i]), min(len(la), len(lb)))
        print("DIFF %s: llama %d ids, ours %d ids, first difference at %d: llama %s ours %s"
              % (name, len(la), len(lb), k, la[max(0, k - 3):k + 5], lb[max(0, k - 3):k + 5]))
        rows.append((name, len(la), "DIFF@%d" % k))
    for r in rows:
        print("%-40s %8d  %s" % r)
    print("parity: %d files, %d ids (llama side), %d differ" % (total, toks, bad))
    return 0 if bad == 0 and total else 1


ADVERSARIAL = {
    "adv_code_py": "def fib(n: int) -> int:\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\n\nif __name__ == '__main__':\n    print([fib(i) for i in range(20)])\n",
    "adv_code_cpp": "#include <vector>\ntemplate <typename T> struct S { std::vector<T> v; T& operator[](size_t i) { return v[i]; } };\nint main(){ S<int> s; s.v = {1,2,3}; return s[1]-2; }  // ok\n",
    "adv_code_ts": "export const f = async (x?: number): Promise<string> => `v=${x ?? 0}`;\nconst re = /^(?:\\d{3}-)?\\d{4}$/u;\n",
    "adv_rust": "fn main() { let v: Vec<u64> = (1..=10).map(|x| x * x).collect(); println!(\"{:?}\", v); }\n",
    "adv_shell": "#!/bin/bash\nset -euo pipefail\nfor f in \"$@\"; do wc -l -- \"$f\" | awk '{print $1}'; done 2>&1 | tee /tmp/x.log\n",
    "adv_json": "{\"a\": [1, 2.5e-3, -7, true, null], \"b\": {\"c\": \"\\u00e9\\n\\t\"}, \"d\": \"x\\\\y\"}\n",
    "adv_digits": "0 01 012 0123 01234 012345 0123456 3.14159265358979 1e10 -42 +7 1,000,000 2026-10-09T21:15:00Z 0x1F 0b1010 ١٢٣٤٥ ௧௨௩ ⅠⅡⅢ ½ ²³\n",
    "adv_cjk": "中文测试：模型的分词器必须与 llama.cpp 完全一致。日本語のテキスト、カタカナ、ひらがな。한국어 텍스트도 있습니다。\n",
    "adv_mixed_script": "abc漢字def あいう123 ＡＢＣ　全角スペース ẞß İı ǅ ﬁ ᾐ Ω\n",
    "adv_arabic_hebrew": "مرحبا بالعالم 123 שלום עולם ‏RTL‏ mark\n",
    "adv_emoji": "👨‍👩‍👧‍👦 🏳️‍🌈 👍🏿 🇨🇦🇫🇷 ❤️‍🔥 🫠 ⌚⏰ 1️⃣ #️⃣\n",
    "adv_combining": "e\u0301 e\u0301\u0301 a\u0308o\u0308u\u0308 n\u0303 Z\u0335\u0321\u0322a\u0337l\u0336g\u0335o\u0338 \u0915\u093f \u0bb8\u0bcd\n",
    "adv_whitespace": "a  b   c\t\td\r\ne\r\rf\n\n\ng \u00a0h\u2003i\u3000j\u200bk\u2028l\u2029m\u0085n \x0b o\x0c p\n   \n\t\n",
    "adv_control": "\x00\x01\x02\x07\x08\x1b[31mred\x1b[0m\x7f\n",
    "adv_punct": "!!! ??? ... --- ___ *** ### @@@ $$$ %%% ^^^ &&& ((( ))) [[[ ]]] {{{ }}} <<< >>> ||| \\\\\\ /// ~~~ ``` '''\n\"Hello,\" she said—'world'… «quote» „Anführung“ 「括弧」\n",
    "adv_apostrophes": "it's IT'S It'S we'll WE'LL I'd you're they've isn't o'clock rock'n'roll 'quoted' l'été\n",
    "adv_markdown": "# Title\n\n- item **bold** _it_ `code`\n1. one\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```python\nprint('x')\n```\n> quote\n",
    "adv_urls": "https://user:pw@example.co.uk:8443/a/b.c?d=e&f=g#h ftp://x.y/z mailto:a@b.c file:///etc/hosts C:\\\\Windows\\\\System32\n",
    "adv_long_word": "Pneumonoultramicroscopicsilicovolcanoconiosis " * 20 + "\n",
    "adv_repeat": "ab" * 3000 + "\n" + "\n" * 50 + " " * 300 + "x\n",
    "adv_dsml": "\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Montréal</｜DSML｜parameter>\n<｜DSML｜parameter name=\"days\" string=\"false\">3</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
    "adv_chat_render": "<｜begin▁of▁sentence｜>You are helpful.<｜User｜>Hi there<｜Assistant｜><think>Let me think.</think>Hello!<｜end▁of▁sentence｜><｜User｜><tool_result>{\"ok\": true}</tool_result><｜Assistant｜><think>",
    "adv_specials_inline": "literal tokens in text: <think> </think> <｜User｜> <｜Assistant｜> <｜end▁of▁sentence｜> <|im_end|> <tool_call> [INST]\n",
}


def make_adversarial(indir: str) -> int:
    d = pathlib.Path(indir)
    d.mkdir(parents=True, exist_ok=True)
    for name, text in ADVERSARIAL.items():
        (d / (name + ".txt")).write_bytes(text.encode("utf-8"))
        # the same text with special-token parsing OFF (literal text everywhere)
        if "｜" in text or "<think>" in text or "<|" in text:
            (d / (name + ".nps.txt")).write_bytes(text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "ours":
        sys.exit(ours(*sys.argv[2:5]))
    if cmd == "compare":
        sys.exit(compare(*sys.argv[2:4]))
    if cmd == "make-adversarial":
        sys.exit(make_adversarial(sys.argv[2]))
    raise SystemExit(__doc__)
