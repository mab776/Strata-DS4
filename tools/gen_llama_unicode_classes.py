#!/usr/bin/env python3
"""tools/gen_llama_unicode_classes.py - write tools/llama_unicode_classes.py from llama.cpp's own Unicode tables.

WHY.  The pre-tokenizer patterns say \\p{L} \\p{N} \\p{P} \\p{S} \\p{M} and \\s, and llama.cpp - the oracle - does not
resolve those with a regex engine's Unicode tables.  It resolves them with its own, generated once into
src/unicode-data.cpp (Unicode 15.0; the vendored tree and the b11429 oracle binary carry byte-identical tables).
The python `regex` module ships newer tables (Unicode 17 in 2026.9.x), so a code point assigned after 15.0 - a new
emoji, a Latin addition, a legacy-computing symbol, the riyal sign, a new digit or mark - lands in \\p{L}/\\p{S}/...
where llama.cpp sees an unassigned code point: a different pre-split, different ids.  This script folds llama.cpp's
tables into range lists, from which strata_tokenizer.py resolves every class (over a collapsed copy of the text, as
llama.cpp does) whatever `regex` version runs it.

WHAT IT READS.  third_party/llama.cpp/src/unicode-data.cpp (unicode_ranges_flags, unicode_set_whitespace,
unicode_map_lowercase) and src/unicode.cpp (k_ucat_map: what a \\p{..} also matches among ASCII characters, which the
std::regex path does not collapse).  It also fingerprints the unicode.cpp functions whose behaviour
strata_tokenizer.py reproduces with these tables (the collapse in unicode_regex_split, the qwen35 custom splitter,
the dispatch between them).  If one of them changes, the tables stop being the whole story and this script refuses
to write anything until a human has re-read the change and updated EXPECTED_CODE below.

Run:  python tools/gen_llama_unicode_classes.py            # rewrite tools/llama_unicode_classes.py
      python tools/gen_llama_unicode_classes.py --check    # exit 1 if the committed module differs
The runtime never needs the llama.cpp source: the generated module is committed.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "third_party" / "llama.cpp" / "src"
OUT = HERE / "llama_unicode_classes.py"

# unicode.h, struct unicode_cpt_flags: the category bits (exactly one per code point in unicode_ranges_flags)
UNDEFINED, NUMBER, LETTER, SEPARATOR, ACCENT_MARK, PUNCTUATION, SYMBOL, CONTROL = 1, 2, 4, 8, 16, 32, 64, 128
CATEGORY = {"N": NUMBER, "L": LETTER, "P": PUNCTUATION, "S": SYMBOL, "M": ACCENT_MARK}
FLAG_NAME = {"NUMBER": "N", "LETTER": "L", "PUNCTUATION": "P", "SYMBOL": "S", "ACCENT_MARK": "M"}
MAX_CODEPOINTS = 0x110000

# sha256 of each function below, comments stripped and whitespace collapsed, as reviewed against
# strata_tokenizer.py (third_party/llama.cpp at ggml pin 3cf03257; the b11429 oracle binary carries the same
# tables, k_ucat_map strings, regexes and custom splitters).  A mismatch means llama.cpp changed how it splits.
EXPECTED_CODE = {
    "unicode_cpt_flags_array": "ecd2bbf5fd007741a29586572b7379adc364bec0e19613d67ce637c4fc1a0b6d",
    "unicode_cpt_flags_from_cpt": "ab2c55087ecf11171171a5f95f08faf9a412ca8cd6d5eebb474c46571c1aae0b",
    "unicode_tolower": "48fbf05a72211176ea6b0c42df3f675a736e2913bd6652c940cca7c676205ea0",
    "unicode_regex_split_custom_qwen35": "07d86c3d169b5fc8fc0997c659d802eeb8034d3586dc1ef72f298c53560dbd94",
    "unicode_regex_split_stl": "9ec9d3fdb4ed2fa0690b1d6d81d2ae252bae164f42df88e3b20fcb3d3a01fb7f",
    "unicode_regex_split_custom": "9c20b689a54044f1a8ac24f25185e378fcbdd5bd7c40cb8a57040328e6e7488a",
    "unicode_regex_split": "0500b808c96da0b87b1734df742f9f26c36d1ab7cf5fe9e11ff6ba2ba3c5b8d0",
}


# ------------------------------------------------------------------ C++ source helpers
def _strip_comments(code: str) -> str:
    """`code` without // and /* */ comments; string and character literals are kept verbatim."""
    out, i, n = [], 0, len(code)
    while i < n:
        c = code[i]
        if code.startswith("//", i):
            j = code.find("\n", i)
            i = n if j < 0 else j
        elif code.startswith("/*", i):
            j = code.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c in "\"'":
            j = i + 1
            while code[j] != c:
                j += 2 if code[j] == "\\" else 1
            out.append(code[i:j + 1])
            i = j + 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _skip_literal(code: str, i: int) -> int:
    """Index just past the string/char literal starting at code[i]."""
    q, j = code[i], i + 1
    while code[j] != q:
        j += 2 if code[j] == "\\" else 1
    return j + 1


def _function(code: str, name: str) -> str:
    """The definition of function `name` in comment-free `code`: its line through the matching closing brace."""
    for m in re.finditer(r"\b%s\s*\(" % re.escape(name), code):
        i, depth = m.end(), 1
        while depth:                                    # the parameter list
            c = code[i]
            if c in "\"'":
                i = _skip_literal(code, i)
                continue
            depth += (c == "(") - (c == ")")
            i += 1
        while code[i].isspace():
            i += 1
        if code[i] != "{":                              # a declaration or a call
            continue
        depth, j = 0, i
        while True:
            c = code[j]
            if c in "\"'":
                j = _skip_literal(code, j)
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return code[code.rfind("\n", 0, m.start()) + 1:j + 1]
            j += 1
    raise SystemExit("unicode.cpp: no definition of %s()" % name)


def _fingerprint(text: str) -> str:
    return hashlib.sha256(" ".join(text.split()).encode("utf-8")).hexdigest()


def _c_unescape(lit: str) -> str:
    """A C string literal's body as the characters the compiler stores (\\xHH, \\\\, \\", \\n, \\r, \\t)."""
    out, i = [], 0
    while i < len(lit):
        if lit[i] != "\\":
            out.append(lit[i])
            i += 1
            continue
        e = lit[i + 1]
        if e == "x":
            j = i + 2
            while j < len(lit) and lit[j] in "0123456789abcdefABCDEF":
                j += 1
            out.append(chr(int(lit[i + 2:j], 16)))
            i = j
            continue
        out.append({"\\": "\\", "\"": "\"", "'": "'", "n": "\n", "r": "\r", "t": "\t"}[e])
        i += 2
    return "".join(out)


def _bracket_members(body: str) -> set[int]:
    """The characters an ECMAScript bracket-expression body matches (std::regex): literals, `\\x` identity escapes
    and `a-b` ranges.  k_ucat_map's bodies use nothing else - anything else is refused rather than guessed."""
    atoms, i = [], 0                                    # (char, was_escaped)
    while i < len(body):
        if body[i] == "\\":
            if body[i + 1].isalnum():
                raise SystemExit("k_ucat_map: unsupported escape \\%s" % body[i + 1])
            atoms.append((body[i + 1], True))
            i += 2
        else:
            atoms.append((body[i], False))
            i += 1
    out, k = set(), 0
    while k < len(atoms):
        if k + 2 < len(atoms) and atoms[k + 1] == ("-", False):
            out.update(range(ord(atoms[k][0]), ord(atoms[k + 2][0]) + 1))
            k += 3
        else:
            out.add(ord(atoms[k][0]))
            k += 1
    return out


# ------------------------------------------------------------------ ranges
def _to_ranges(cps) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for cp in sorted(cps):
        if out and out[-1][1] == cp - 1:
            out[-1][1] = cp
        else:
            out.append([cp, cp])
    return [(a, b) for a, b in out]


def _spec(ranges) -> str:
    return " ".join("%04X" % a if a == b else "%04X-%04X" % (a, b) for a, b in ranges)


def _wrap(spec: str, indent: str, width: int = 112) -> str:
    """A long range spec as implicitly concatenated string literals, one line each."""
    if not spec:
        return '""'
    lines, cur = [], ""
    for tok in spec.split(" "):
        add = (" " if cur else "") + tok
        if cur and len(indent) + len(cur) + len(add) + 3 > width:
            lines.append(cur + " ")
            cur = tok
        else:
            cur += add
    lines.append(cur)
    if len(lines) == 1:
        return '"%s"' % lines[0]
    return "(\n" + "\n".join('%s    "%s"' % (indent, ln) for ln in lines) + "\n%s)" % indent


# ------------------------------------------------------------------ main
def build(src: pathlib.Path) -> tuple[str, dict]:
    data_path, code_path = src / "unicode-data.cpp", src / "unicode.cpp"
    data = data_path.read_text(encoding="utf-8")
    code_raw = code_path.read_text(encoding="utf-8")
    code = _strip_comments(code_raw)

    stale = []
    for name, want in EXPECTED_CODE.items():
        got = _fingerprint(_function(code, name))
        if got != want:
            stale.append("  %s: %s (reviewed: %s)" % (name, got, want or "none"))
    if stale:
        raise SystemExit("unicode.cpp differs from the reviewed version in:\n%s\nRe-read those functions against "
                         "tools/strata_tokenizer.py (_llama_classes, LLAMA_CUSTOM_SPLIT), then update EXPECTED_CODE."
                         % "\n".join(stale))

    def block(name):
        b = data[data.index(name):]
        return b[b.index("{") + 1:b.index("};")]

    # unicode_ranges_flags: {start, flags}, each range running to the next start; the last entry is the sentinel
    starts = [(int(a, 16), int(f, 16)) for a, f in
              re.findall(r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}", block("unicode_ranges_flags"))]
    if starts[0][0] != 0 or starts[-1] != (MAX_CODEPOINTS, 0):
        raise SystemExit("unicode_ranges_flags: unexpected first/last entry %r %r" % (starts[0], starts[-1]))
    by_cat: dict[str, list[tuple[int, int]]] = {k: [] for k in CATEGORY}
    for (a, f), (b, _) in zip(starts, starts[1:]):
        if f == 0 or f & (f - 1) or f > CONTROL:
            raise SystemExit("unicode_ranges_flags: range %04X has flags %#x, not one category" % (a, f))
        for k, bit in CATEGORY.items():
            if f == bit:
                if by_cat[k] and by_cat[k][-1][1] == a - 1:
                    by_cat[k][-1] = (by_cat[k][-1][0], b - 1)
                else:
                    by_cat[k].append((a, b - 1))

    ws = sorted(int(x, 16) for x in re.findall(r"0x([0-9A-Fa-f]+)", block("unicode_set_whitespace")))
    # std::regex `\s` on the collapsed text is the C locale's isspace for ASCII; llama.cpp collapses every non-ASCII
    # whitespace to 0x0B.  strata_tokenizer.py relies on the table agreeing with isspace on ASCII.
    if [c for c in ws if c < 0x80] != [9, 10, 11, 12, 13, 32]:
        raise SystemExit("unicode_set_whitespace: ASCII members %r differ from isspace" % [c for c in ws if c < 0x80])

    # the qwen35 splitter tests contractions with unicode_tolower(next) == 's','t','m','d','r','e','v','l'; the
    # pattern spells them '[sS]' etc., which is the same only if no non-ASCII code point lowercases to one of them
    lower = [(int(a, 16), int(b, 16)) for a, b in
             re.findall(r"\{0x([0-9A-Fa-f]+), 0x([0-9A-Fa-f]+)\}", block("unicode_map_lowercase"))]
    odd = [(a, b) for a, b in lower if a >= 0x80 and b < 0x80 and chr(b) in "stmdrevl"]
    if odd:
        raise SystemExit("unicode_map_lowercase maps non-ASCII to a contraction letter: %r" % odd)

    # k_ucat_map: the ASCII characters each \p{..} stands for in the collapsed std::regex path
    m = code[code.index("k_ucat_map = {"):]
    m = m[:m.index("};")]
    ascii_ucat = {}
    for flag, lit in re.findall(r'\{\s*unicode_cpt_flags::(\w+),\s*"((?:[^"\\]|\\.)*)"\s*\}', m):
        ascii_ucat[FLAG_NAME[flag]] = sorted(_bracket_members(_c_unescape(lit)))
    if sorted(ascii_ucat) != sorted(CATEGORY):
        raise SystemExit("k_ucat_map: categories %r" % sorted(ascii_ucat))
    for k, members in ascii_ucat.items():                # it should agree with the table on ASCII; say so if not
        table = [c for a, b in by_cat[k] for c in range(a, b + 1) if c < 0x80]
        if members != table:
            print("note: k_ucat_map[%s] %r differs from unicode_ranges_flags on ASCII %r" % (k, members, table),
                  file=sys.stderr)

    meta = {
        "unicode-data.cpp": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "unicode.cpp": hashlib.sha256(code_path.read_bytes()).hexdigest(),
        "counts": {k: sum(b - a + 1 for a, b in r) for k, r in by_cat.items()},
    }
    ind = "    "
    lines = [
        '"""tools/llama_unicode_classes.py - GENERATED by tools/gen_llama_unicode_classes.py; do not edit by hand.',
        "",
        "llama.cpp's Unicode tables (third_party/llama.cpp/src/unicode-data.cpp, Unicode 15.0 as llama.cpp ships",
        "them; byte-identical to the tables compiled into the b11429 oracle) folded to what the pre-tokenizers read.",
        "strata_tokenizer.py builds every \\\\p{..} and \\\\s of its patterns from these, so the pre-split no longer",
        "depends on the Unicode version of the python `regex` module.  Ranges are inclusive hex, space separated.",
        '"""',
        "",
        "SOURCE = {",
        '    "unicode-data.cpp": "sha256:%s",' % meta["unicode-data.cpp"],
        '    "unicode.cpp": "sha256:%s",' % meta["unicode.cpp"],
        "}",
        "",
        "# unicode_ranges_flags, by the category flag a \\p{..} tests.  A code point under none of the five is",
        "# UNDEFINED (unassigned in Unicode 15.0), SEPARATOR or CONTROL to llama.cpp.",
        "RANGES = {",
    ]
    for k in ("N", "L", "P", "S", "M"):
        lines.append('    "%s": %s,  # %d code points' % (k, _wrap(_spec(by_cat[k]), ind), meta["counts"][k]))
    lines += [
        "}",
        "",
        "# unicode_set_whitespace: the code points whose is_whitespace flag is set.",
        "WHITESPACE = %s" % _wrap(_spec(_to_ranges(ws)), ""),
        "",
        "# unicode.cpp unicode_regex_split, k_ucat_map: the ASCII characters each \\p{..} also matches on",
        "# the std::regex path.  ASCII is not collapsed there: it stays itself, these ranges join the class.",
        "ASCII_UCAT = {",
    ]
    for k in ("N", "L", "P", "S", "M"):
        lines.append('    "%s": "%s",' % (k, _spec(_to_ranges(ascii_ucat[k]))))
    lines += ["}", ""]
    return "\n".join(lines), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=str(SRC), help="llama.cpp src/ directory (default: the vendored tree)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--check", action="store_true", help="compare with --out instead of writing it")
    ap.add_argument("--fingerprints", action="store_true", help="print the unicode.cpp fingerprints and exit")
    a = ap.parse_args()
    if a.fingerprints:
        code = _strip_comments((pathlib.Path(a.src) / "unicode.cpp").read_text(encoding="utf-8"))
        for name in EXPECTED_CODE:
            print('    "%s": "%s",' % (name, _fingerprint(_function(code, name))))
        return 0
    text, meta = build(pathlib.Path(a.src))
    out = pathlib.Path(a.out)
    if a.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print("%s is %s" % (out, "up to date" if same else "STALE: run tools/gen_llama_unicode_classes.py"))
        return 0 if same else 1
    out.write_text(text, encoding="utf-8", newline="\n")
    print("wrote %s: %s" % (out, ", ".join("%s %d" % kv for kv in meta["counts"].items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
