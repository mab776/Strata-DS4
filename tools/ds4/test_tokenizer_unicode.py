#!/usr/bin/env python3
"""tools/ds4/test_tokenizer_unicode.py - the pre-tokenizer reads llama.cpp's Unicode tables, not the `regex` module's.

llama.cpp resolves \\p{L} \\p{N} \\p{P} \\p{S} \\p{M} and \\s with its own tables (src/unicode-data.cpp, Unicode 15.0).
The python `regex` module ships newer ones, so a code point Unicode assigned since - a new emoji, a Latin addition, a
legacy-computing symbol, a mark, the riyal sign, a digit - was pre-split differently and got different ids.  The ids
in tests/ds4/tokenizer_unicode_golden.json are frozen from llama-tokenize b11429 for both pre-tokenizers this tree
ships, joyai-llm (DeepSeek-V4-Flash) and qwen35 (Qwen3.5 and later): 30 cases built on such code points - the
pre-fix tokenizer got 25 of them wrong on joyai-llm and 13 on qwen35 (27 in all) - and two controls of established
characters whose ids must not move.

Run:  DS4_GGUF=<DeepSeek-V4 gguf> QWEN_GGUF=<Qwen3.5+ gguf> python -m unittest tools/ds4/test_tokenizer_unicode.py
      (DS4_TOKENIZER / QWEN_TOKENIZER / STRATA_TOKENIZER may name an extracted tokenizer/ directory instead)
The table checks need nothing; each model's id check is skipped when its tokenizer is not on this machine.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "tools"))

import strata_tokenizer as ST  # noqa: E402

GOLDEN = REPO / "tests/ds4/tokenizer_unicode_golden.json"
DEFAULT_DS4 = "/media/mal/NVME1TB/Models/DeepSeek-V4-Flash-Q2-0731.gguf"   # tools/ds4/test_tokenizer_golden.py's


def _golden() -> dict:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


_TOKENIZERS: dict | None = None


def _tokenizers() -> dict:
    """{pre: Tokenizer} for every tokenizer the environment names (a GGUF, or an extracted tokenizer/ dir)."""
    global _TOKENIZERS
    if _TOKENIZERS is None:
        _TOKENIZERS = {}
        for var in ("DS4_GGUF", "DS4_TOKENIZER", "QWEN_GGUF", "QWEN_TOKENIZER", "STRATA_TOKENIZER"):
            p = pathlib.Path(os.environ.get(var) or (DEFAULT_DS4 if var == "DS4_GGUF" else ""))
            if not str(p) or str(p) == ".":
                continue
            if p.is_dir() and (p / "vocab.json").exists():
                tk = ST.Tokenizer.from_dir(p)
            elif p.is_file():
                tk = ST.Tokenizer.from_gguf(p)
            else:
                continue
            _TOKENIZERS.setdefault(tk.pre, tk)
    return _TOKENIZERS


def _byte_tokenizer(pre: str) -> ST.Tokenizer:
    """The 256 byte symbols and no merges: enough for _pre_split, no model needed."""
    return ST.Tokenizer([ST.BYTE_TO_UNICODE[b] for b in range(256)], [], None, pre)


def _in(cp: int, ranges) -> bool:
    return any(a <= cp <= b for a, b in ranges)


def _merge(ranges) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


class LlamaTables(unittest.TestCase):
    def test_module_matches_vendored_source(self):
        """tools/llama_unicode_classes.py is what the generator makes of the vendored llama.cpp."""
        src = REPO / "third_party/llama.cpp/src"
        if not (src / "unicode-data.cpp").exists():
            self.skipTest("no vendored llama.cpp source here")
        import gen_llama_unicode_classes as G
        text, _ = G.build(src)
        self.assertEqual(text, (REPO / "tools/llama_unicode_classes.py").read_text(encoding="utf-8"),
                         "stale: run tools/gen_llama_unicode_classes.py")

    def test_new_code_points_are_in_no_class(self):
        """Every code point the goldens exercise is UNDEFINED to llama.cpp: in no \\p{..}, not \\s, on both paths,
        and the collapsed text turns it into the 'other' representative."""
        cps = sorted({int(h, 16) for c in _golden()["cases"] for h in c["cps"]})
        self.assertGreater(len(cps), 250)
        for collapsed in (True, False):
            for name, ranges in ST._llama_classes(collapsed).items():
                hit = [("%04X" % cp) for cp in cps if _in(cp, ranges)]
                self.assertEqual(hit, [], "collapsed=%s class %s" % (collapsed, name))
        other = ST._REP + 2 * len(ST._CATEGORIES)
        t = ST._collapse_table()
        self.assertEqual([("%04X" % cp) for cp in cps if t[cp] != other], [])

    def test_established_code_points_keep_their_class(self):
        """Spot checks against unicode-data.cpp: the fix must not reclassify what llama.cpp always had."""
        cls = ST._llama_classes(True)
        for cp, name in ((0x41, "L"), (0x7E, "S"), (0x5F, "P"), (0x30, "N"), (0x2605, "S"), (0x0661, "N"),
                         (0x0301, "M"), (0x4E2D, "L"), (0x1FAE0, "S"), (0x00A0, "s"), (0x3000, "s"), (0x0085, "s")):
            self.assertTrue(_in(cp, cls[name]), "U+%04X should be %s" % (cp, name))
        self.assertFalse(_in(0x1C, cls["s"]), "U+001C is not whitespace to llama.cpp")

    def test_collapsed_text_resolves_like_the_code_points(self):
        """Each class over the collapsed alphabet (ASCII members + representatives, through the translate table)
        stands for exactly the code points _llama_classes gives it, on both paths."""
        t = ST._collapse_table()
        self.assertEqual(len(t), 0x110000)
        self.assertEqual(bytes(t[:0x80]), bytes(range(0x80)))
        runs: dict[int, list[tuple[int, int]]] = {}         # representative -> the code points collapsed to it
        start = 0x80
        for i in range(0x81, 0x110001):
            if i == 0x110000 or t[i] != t[start]:
                runs.setdefault(t[start], []).append((start, i - 1))
                start = i
        for collapsed in (True, False):
            explicit = ST._llama_classes(collapsed)
            for name, members in ST._collapsed_classes(collapsed).items():
                got = [(c, c) for c in members if c < 0x80]
                for r in members:
                    if r >= 0x80:
                        got += runs.get(r, [])
                self.assertEqual(_merge(got), _merge(explicit[name]), "collapsed=%s class %s" % (collapsed, name))

    def test_patterns_spell_out_every_class(self):
        for pats in ST.PRE_PATTERNS.values():
            for p in pats:
                src = ST._compile_pre(p)[0].pattern
                for esc in ("\\p{", "\\s", "\\S"):
                    self.assertNotIn(esc, src, "%r still leaves %s to the regex module" % (p[:40], esc))

    def test_refuses_table_dependent_constructs(self):
        classes = {k: ST._body(ST._to_ranges(v)) for k, v in ST._collapsed_classes(True).items()}
        for bad in (r"\d+", r"\w", r"\b", r"\p{Han}", r"\P{L}", r"(?i:a)", r"a.b", r"[^\S]"):
            with self.assertRaises(ValueError, msg=bad):
                ST._expand_classes(bad, classes)

    def test_pre_split_new_code_points(self):
        """llama.cpp's split around a few new code points, model-free."""
        ds = _byte_tokenizer("joyai-llm")
        self.assertEqual(ds._pre_split(" \U0001FAE9 x"), [" ", "\U0001FAE9", " x"])   # Unicode 16 emoji: UNDEFINED
        self.assertEqual(ds._pre_split(" \U00002605 x"), [" \U00002605", " x"])        # an established \p{S}
        self.assertEqual(ds._pre_split("100 \U000020C1"), ["100", " ", "\U000020C1"])  # the riyal sign (Unicode 17)
        self.assertEqual(ds._pre_split("abc\U00006F22\U00005B57def"), ["abc", "\U00006F22\U00005B57", "def"])
        qw = _byte_tokenizer("qwen35")
        self.assertEqual(qw._pre_split("a \U0001CCF0"), ["a", " \U0001CCF0"])          # Unicode 16 digit: not \p{N}
        self.assertEqual(qw._pre_split("a \U00000661"), ["a", " ", "\U00000661"])       # an established \p{N}
        self.assertEqual(qw._pre_split("\U0000A7CB's"), ["\U0000A7CB'", "s"])          # Unicode 16 letter: not \p{L}


class OracleIds(unittest.TestCase):
    """The frozen llama-tokenize ids, per pre-tokenizer."""

    def _check(self, pre: str):
        g = _golden()
        tk = _tokenizers().get(pre)
        if tk is None:
            self.skipTest("no %s tokenizer here (DS4_GGUF / QWEN_GGUF / *_TOKENIZER)" % pre)
        if len(tk.tokens) != g["models"][pre]["vocab_size"]:
            self.skipTest("%s tokenizer has %d tokens, the goldens %d" % (pre, len(tk.tokens),
                                                                       g["models"][pre]["vocab_size"]))
        for case in g["cases"]:
            with self.subTest(case=case["name"]):
                got = tk.encode(case["text"], parse_special=True)
                self.assertEqual(got, case["ids"][pre], "%s: %d ids, llama.cpp %d"
                                 % (case["name"], len(got), len(case["ids"][pre])))
                self.assertEqual(tk.decode(got), case["text"])

    def test_joyai_llm(self):
        self._check("joyai-llm")

    def test_qwen35(self):
        self._check("qwen35")


if __name__ == "__main__":
    unittest.main(verbosity=2)
