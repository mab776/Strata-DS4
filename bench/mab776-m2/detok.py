#!/usr/bin/env python3
"""mab776 M2: decode a run's output ids (m2run.sh OUT/<label>.ids, one id per line) and check needles / show code.
    python3 detok.py TOKDIR_OR_GGUF IDS_FILE [--word W] [--head N]
Prints the reasoning/content split; with --word: FOUND/missed (content only, like needle_ds4.py)."""
import pathlib, sys
REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
import strata_tokenizer as ST
src, idsf = sys.argv[1], sys.argv[2]
word = sys.argv[sys.argv.index("--word") + 1] if "--word" in sys.argv else None
head = int(sys.argv[sys.argv.index("--head") + 1]) if "--head" in sys.argv else 1200
tok = ST.Tokenizer.from_dir(src) if pathlib.Path(src).is_dir() else ST.Tokenizer.from_gguf(src)
ids = [int(x) for x in pathlib.Path(idsf).read_text().split() if x.strip().lstrip("-").isdigit()]
text = tok.decode(ids)
reasoning, sep, content = text.partition("</think>")
if not sep:
    reasoning, content = "", text
print(f"{len(ids)} ids, reasoning {len(reasoning)} chars, content {len(content)} chars, closed think: {bool(sep)}")
if word:
    print(("FOUND" if word in content else "missed"), f"| word {word!r} in reasoning: {word in reasoning}")
print("--- content head ---")
print(content[:head])
