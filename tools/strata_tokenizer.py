"""tools/strata_tokenizer.py - the tokenizer, as the reference implementation of record.

WHY THIS EXISTS AND WHERE IT SITS.  The GGUF carries the whole tokenizer in metadata: `tokenizer.ggml.model`
= `gpt2`, 248,320 `tokens`, 247,587 `merges`, `token_type`, `pre` = `qwen35`, and a Jinja `chat_template`.
None of that needs the 35 GB of weights, so it is extracted into the pack's `tokenizer/` directory
(docs/pack-format.md §6) and this module is what reads it back.  A C++ port follows and is tested against
this one.

THE DECISIVE PROPERTY IS THE ROUND TRIP.  Byte-level BPE maps each BYTE to a printable unicode character so
that any UTF-8 input is representable with no UNK token.  Every failure mode of that mapping - the wrong
offset for a byte, a merge applied in the wrong order, a pre-tokenizer split that drops a character - still
produces plausible token ids.  `decode(encode(s)) == s` is the check that sees them, and it is exact because
the byte layer is lossless.  It is asserted over a corpus chosen to hit the boundaries: multi-byte UTF-8,
emoji (4-byte), combining marks, whitespace runs, and C0 control bytes.

mab776 merge (Strata v0.1.41 fb58e0db + Strata-DS4 85c3eab3): v0.1.41's file (piece cache, #1385 `_bpe`,
`control_tokens`) with DS4's `joyai-llm` pre-tokenizer (`JOYAI_LLM_PATTERNS`, `PRE_PATTERNS`, `_pre_split`) for
DeepSeek-V4-Flash; plus `from_dir()` (the pack's own `pre`, as Project Maya #27 does for GLM), special tokens
decoding to their own text (llama.cpp's token_to_piece), and `extract()` writing the patterns of the actual `pre`.

mab776: the patterns' \\p{..} and \\s are resolved with llama.cpp's own Unicode tables (llama_unicode_classes.py),
not the `regex` module's, on a collapsed copy of the text as llama.cpp does - see `_llama_classes`, `_compile_pre`.
"""
from __future__ import annotations

import functools
import itertools
import json
import pathlib
import sys

import regex

try:
    import llama_unicode_classes as _LUC
except ImportError:                                     # imported as tools.strata_tokenizer, or from elsewhere
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import llama_unicode_classes as _LUC

# ------------------------------------------------------------------ byte <-> unicode (GPT-2 byte-level BPE)
def bytes_to_unicode() -> dict[int, str]:
    """The GPT-2 byte encoder: 256 bytes -> 256 printable characters, reversibly.

    Bytes 33..126, 161..172 and 174..255 map to themselves; the remaining 68 (space, newline, the C0
    controls and the high range that would be invisible) are shifted into 256+n so that no byte is
    unprintable.  The shift is the whole trick and getting its ORDER wrong is invisible in the output ids.
    """
    bs = (list(range(0x21, 0x7F)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


BYTE_TO_UNICODE = bytes_to_unicode()
UNICODE_TO_BYTE = {v: k for k, v in BYTE_TO_UNICODE.items()}

# The `qwen35` pre-tokenizer, transcribed from the ORACLE rather than from the family resemblance:
# `.ref/llama.cpp/src/llama-vocab.cpp` L396, `case LLAMA_VOCAB_PRE_TYPE_QWEN35`.  The commented-out line
# above it is the `tokenizer.json` original, and llama.cpp's active version differs from it - it spells the
# contraction classes out instead of using `(?i:...)`, which is behaviourally the same.
#
# THE `\p{M}` IS THE WHOLE POINT OF THIS BEING TRANSCRIBED.  The QWEN3 pattern two cases earlier (L389) is
# the same shape with `\p{L}` where this has `[\p{L}\p{M}]` and without `\p{M}` in the punctuation negation.
# Writing the QWEN3 pattern for `qwen35` looks right, round-trips perfectly, and is wrong: a combining mark
# is `\p{M}`, so without it the mark gets swallowed into the following punctuation run and a token like `_j`
# never forms.  It cost 3 strings out of 1875 and only `tokenize_oracle_check` could see it.
QWEN35_PATTERN = (
    r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])"
    r"|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}"
    r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
    r"|\s*[\r\n]+"
    r"|\s+(?!\S)"
    r"|\s+"
)

# The `joyai-llm` pre-tokenizer (DeepSeek-V4-Flash and the DeepSeek3/Hunyuan family), transcribed from
# `.ref/llama.cpp src/llama-vocab.cpp` L320-326 (`case LLAMA_VOCAB_PRE_TYPE_JOYAI_LLM`).  UNLIKE qwen35 this is
# THREE regexes, and llama.cpp APPLIES THEM SEQUENTIALLY: each one further splits the pieces the previous one
# left (`unicode_regex_split` at `unicode.cpp:1291-1367` feeds the accumulating `bpe_offsets` into the next).
# The order is load-bearing.  `\p{N}{1,3}` must cut digit runs first (so `01234` is `012|34`); then
# `[一-龥぀-ゟ゠-ヿ]+` pulls Han / kana runs out of a larger `\p{L}+` (so `abc漢字def` is `abc|漢字|def`); the
# last pattern is the ordinary word splitter.  A single alternation of the three looks equivalent on ASCII and
# is not: it keeps `abc漢字def` whole.
JOYAI_LLM_PATTERNS = [
    r"\p{N}{1,3}",
    r"[一-龥぀-ゟ゠-ヿ]+",
    (r"[!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+"
     r"|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+"
     r"| ?[\p{P}\p{S}]+[\r\n]*"
     r"|\s*[\r\n]+"
     r"|\s+(?!\S)"
     r"|\s+"),
]

# pre type -> the pattern list llama.cpp applies, in order.  The `pre` string in GGUF metadata selects it.
# mab776: llama.cpp's case for joyai-llm also covers deepseek-v3, hunyuan-dense and hy_v4 (llama-vocab.cpp
# L319-327, pinned 3cf03257: same three regexes, none sets ignore_merges, all clean_spaces=false).
PRE_PATTERNS = {
    "qwen35": [QWEN35_PATTERN],
    "joyai-llm": JOYAI_LLM_PATTERNS,
    "deepseek-v3": JOYAI_LLM_PATTERNS,
    "hunyuan-dense": JOYAI_LLM_PATTERNS,
    "hy_v4": JOYAI_LLM_PATTERNS,
}

# ------------------------------------------------------------------ llama.cpp's character classes
# mab776: WHY THE PATTERNS ARE NOT COMPILED AS WRITTEN.  To the `regex` module `\p{L}` means "a letter in the Unicode
# version this `regex` release ships" (17.0 in 2026.9.x).  llama.cpp never asks a regex engine: it reads its own
# tables, generated once into src/unicode-data.cpp from Unicode 15.0.  Every code point assigned since - new emoji
# (U+1FAE9), Latin additions (U+A7CB, U+1C89), legacy-computing symbols (U+1CC00..), marks (U+1ACF, U+0897), the
# riyal sign (U+20C1), new digits - is \p{L}/\p{S}/... to `regex` and UNDEFINED to llama.cpp, so the pre-split moved
# and the ids with it (4 of 6 probe texts differed).  joyai-llm reads L/M, P/S, N and whitespace apart, so every new
# code point counts; qwen35 only tells L/M, N and whitespace from the rest, so its new letters, marks and digits do.
# So every \p{..} and \s is resolved with llama.cpp's tables instead: tools/llama_unicode_classes.py, generated from
# the vendored unicode-data.cpp, whose tables are byte-identical to those compiled into the b11429 oracle.  The split
# then matches the oracle on every code point (tools/ds4/test_tokenizer_unicode.py has the frozen ids).
#
# llama.cpp resolves the classes in TWO ways, picked by the pattern string itself (unicode.cpp unicode_regex_split):
#   * unicode_regex_split_custom(): a few exact pattern strings - qwen35's among them - never reach a regex engine;
#     a hand-written splitter reads unicode_cpt_flags_from_cpt() for every code point, ASCII included.  Our regex,
#     with each class taken straight from the table, splits exactly as unicode_regex_split_custom_qwen35 does: every
#     in-range code point has a non-zero flag, and no non-ASCII code point lowercases to a contraction letter (the
#     generator checks both and fingerprints the splitter, so a change upstream stops it).
#   * every other pattern with a \p{..} runs std::regex over a COLLAPSED copy of the text: ASCII stays itself, any
#     other code point becomes one byte - 0x0B if it is whitespace, else its category's byte for N/L/P/M/S, else
#     0xD0 - and \p{X} becomes [<X's byte><X's ASCII members, k_ucat_map>].  So there, a non-ASCII whitespace is in
#     no \p{..}, an ASCII character is in \p{X} only if k_ucat_map says so, and `\s` is std::regex's own (the
#     C-locale isspace on ASCII) plus the collapsed whitespace.  A pattern with no \p{..} (joyai's `[一-龥...]+`)
#     runs std::wregex on the code points themselves, non-ASCII whitespace turned into 0x0B: the same `\s`.
# With today's tables both views give the same sets; they stay separate because they are separate code upstream.
#
# HOW: llama.cpp's own collapse.  Spelled out as code point ranges (\p{L} alone is 660 of them) the classes made the
# `regex` module 7-9x slower.  Instead a custom-splitter or \p{..} pattern (ASCII literals only) runs on a collapsed
# copy of the text, where every non-ASCII code point is one representative per (category, whitespace) signature -
# _REP + 2 * category + whitespace, one str.translate through a 1.1 MB table - and each class is its ASCII members
# plus the representatives of the signatures it holds on that path.  A code point stays one code point, so the spans
# index the original text.  A pattern with neither (the Han/kana one) runs on the original text, as llama.cpp's
# std::wregex does.
LLAMA_CUSTOM_SPLIT = frozenset([QWEN35_PATTERN])         # unicode.cpp unicode_regex_split_custom(), the ones we use
_STD_REGEX_ASCII_SPACE = [(0x09, 0x0D), (0x20, 0x20)]   # std::regex `\s` on char: isspace() in the "C" locale
_CATEGORIES = "NLPSM"                                   # what a \p{..} names; any other code point is "other"
_REP = 0x80                                             # collapsed text: _REP + 2 * category index + whitespace


def _ranges(spec: str) -> list[tuple[int, int]]:
    out = []
    for tok in spec.split():
        a, _, b = tok.partition("-")
        out.append((int(a, 16), int(b or a, 16)))
    return out


def _to_ranges(cps) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for c in sorted(cps):
        if out and out[-1][1] == c - 1:
            out[-1] = (out[-1][0], c)
        else:
            out.append((c, c))
    return out


def _clip(ranges, lo: int, holes=()) -> list[tuple[int, int]]:
    """`ranges` from code point `lo` up, minus the code points in `holes`."""
    out = []
    for a, b in ranges:
        a = max(a, lo)
        for h in sorted(h for h in holes if a <= h <= b):
            if h > a:
                out.append((a, h - 1))
            a = h + 1
        if a <= b:
            out.append((a, b))
    return out


@functools.lru_cache(maxsize=None)
def _llama_classes(collapsed: bool) -> dict[str, tuple[tuple[int, int], ...]]:
    """{'L','N','P','S','M','s': code point ranges} as llama.cpp resolves \\p{L} ... \\p{M} and \\s - on its std::regex
    path (`collapsed`) or in its custom splitters (the flags of every code point, ASCII included)."""
    ws = _ranges(_LUC.WHITESPACE)
    if not collapsed:
        out = {k: _ranges(v) for k, v in _LUC.RANGES.items()}
        out["s"] = ws
    else:
        holes = {c for a, b in ws for c in range(a, b + 1)}
        out = {k: _ranges(_LUC.ASCII_UCAT[k]) + _clip(_ranges(v), 0x80, holes) for k, v in _LUC.RANGES.items()}
        out["s"] = _STD_REGEX_ASCII_SPACE + _clip(ws, 0x80)
    return {k: tuple(v) for k, v in out.items()}


@functools.lru_cache(maxsize=None)
def _collapse_table() -> bytearray:
    """The str.translate table of the collapsed text: an ASCII code point stays itself, any other becomes
    _REP + 2 * (its category's index in _CATEGORIES, len(_CATEGORIES) for none) + (1 if it is whitespace)."""
    t = bytearray(range(0x80)) + bytes([_REP + 2 * len(_CATEGORIES)]) * (0x110000 - 0x80)
    for ci, k in enumerate(_CATEGORIES):
        for a, b in _clip(_ranges(_LUC.RANGES[k]), 0x80):
            t[a:b + 1] = bytes([_REP + 2 * ci]) * (b - a + 1)
    for a, b in _clip(_ranges(_LUC.WHITESPACE), 0x80):
        for c in range(a, b + 1):
            t[c] |= 1
    return t


def _collapse(text: str) -> str:
    return text if text.isascii() else text.translate(_collapse_table())


@functools.lru_cache(maxsize=None)
def _collapsed_classes(collapsed: bool) -> dict[str, tuple[int, ...]]:
    """{class: the characters of the COLLAPSED text it matches}: its ASCII members, then the representative of every
    (category, whitespace) signature whose code points _llama_classes puts in it on that path."""
    out = {}
    for name, ranges in _llama_classes(collapsed).items():
        members = [c for a, b in ranges for c in range(a, min(b, 0x7F) + 1)]
        for ci in range(len(_CATEGORIES) + 1):
            for ws in (0, 1):
                if name == "s":
                    hit = ws == 1
                else:
                    hit = ci < len(_CATEGORIES) and _CATEGORIES[ci] == name and not (collapsed and ws)
                if hit:
                    members.append(_REP + 2 * ci + ws)
        out[name] = tuple(members)
    return out


def _esc(cp: int) -> str:
    return "\\x%02x" % cp if cp < 0x100 else "\\u%04x" % cp if cp < 0x10000 else "\\U%08x" % cp


def _body(ranges) -> str:
    return "".join(_esc(a) if a == b else _esc(a) + "-" + _esc(b) for a, b in ranges)


def _expand_classes(pattern: str, classes: dict[str, str]) -> str:
    """`pattern` with every \\p{X}, \\s and \\S replaced by the class body classes[X] / classes['s'].

    Refuses every other construct whose meaning comes from a Unicode table (\\d, \\w, \\b, other \\p{..}, inline
    flags such as case folding, an unescaped '.'), so nothing in a pre-tokenizer pattern can fall back silently to
    the Unicode version of the `regex` module."""
    out, i, n = [], 0, len(pattern)
    inside = first = False                              # inside [...]; just past '[' or '[^' (a ']' is literal there)
    while i < n:
        c = pattern[i]
        if c == "\\":
            e = pattern[i + 1]
            if e == "p" and pattern.startswith("{", i + 2):
                j = pattern.index("}", i)
                cls = classes.get(pattern[i + 3:j])
                if cls is None:
                    raise ValueError("pre-tokenizer pattern: no llama.cpp class for %s here" % pattern[i:j + 1])
                out.append(cls if inside else "[" + cls + "]")
                i = j + 1
            elif e == "s":
                out.append(classes["s"] if inside else "[" + classes["s"] + "]")
                i += 2
            elif e == "S" and not inside:
                out.append("[^" + classes["s"] + "]")
                i += 2
            elif e in "rnt" or not e.isalnum():
                out.append(pattern[i:i + 2])
                i += 2
            else:       # \d \w \b ... read Unicode tables; \x \u \U could name a code point the collapse replaced
                raise ValueError("pre-tokenizer pattern: escape \\%s is not supported" % e)
            first = False
            continue
        if not inside:
            if c == "." or (c == "(" and pattern.startswith("(?", i)
                            and not pattern.startswith(("(?:", "(?=", "(?!", "(?<=", "(?<!"), i)):
                raise ValueError("pre-tokenizer pattern: %r at %d depends on Unicode tables" % (pattern[i:i + 3], i))
            if c == "[":
                inside = first = True
                if pattern.startswith("[^", i):
                    out.append("[^")
                    i += 2
                    continue
        elif c == "]" and not first:
            inside = False
        else:
            first = False
        out.append(c)
        i += 1
    return "".join(out)


@functools.lru_cache(maxsize=None)
def _compile_pre(pattern: str):
    """(compiled, runs on the collapsed text?) for one pre-tokenizer pattern, dispatched as unicode_regex_split does:
    a custom splitter (the flags view) or a \\p{..} (the collapsed std::regex view) - both on the collapsed text - or
    neither: std::wregex on the code points, i.e. the original text with only \\s to resolve."""
    custom = pattern in LLAMA_CUSTOM_SPLIT
    if custom or "\\p{" in pattern:
        if not pattern.isascii():                       # llama.cpp throws on this mix too
            raise ValueError("pre-tokenizer pattern: \\p{..} together with non-ASCII characters")
        cls = _collapsed_classes(not custom)
        return regex.compile(_expand_classes(pattern, {k: _body(_to_ranges(v)) for k, v in cls.items()})), True
    return regex.compile(_expand_classes(pattern, {"s": _body(_llama_classes(True)["s"])})), False


class Tokenizer:
    def __init__(self, tokens: list[str], merges: list[str], token_types: list[int] | None = None,
                 pre: str = "qwen35", special_ids: dict[str, int] | None = None):
        self.tokens = tokens
        self.pre = pre
        self.token_types = token_types
        self.special_ids = special_ids or {}
        self.ids = {t: i for i, t in enumerate(tokens)}
        if len(self.ids) != len(tokens):
            raise ValueError("vocabulary has duplicate tokens: %d entries, %d unique"
                             % (len(tokens), len(self.ids)))
        # Merge rules as (left, right) -> rank.  A merge list that is not a total order over its pairs, or one
        # naming a token that is not in the vocabulary, would make BPE silently produce different ids than the
        # model was trained with, so both are checked here rather than discovered as a bad answer later.
        self.ranks: dict[tuple[str, str], int] = {}
        for i, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) != 2:
                raise ValueError("merge %d is not a pair: %r" % (i, m))
            if parts[0] not in self.ids or parts[1] not in self.ids:
                raise ValueError("merge %d names a token outside the vocabulary: %r" % (i, m))
            self.ranks[(parts[0], parts[1])] = i
        if pre not in PRE_PATTERNS:
            raise ValueError("unknown pre-tokenizer %r (known: %s)"
                             % (pre, ", ".join(sorted(PRE_PATTERNS))))
        self._pre_res = [_compile_pre(p) for p in PRE_PATTERNS[pre]]

        # The literals matched directly instead of being run through BPE.  GGUF token types: 3 = CONTROL,
        # 4 = USER_DEFINED.  The two classes behave DIFFERENTLY and llama.cpp's own tokenizer settled which:
        #
        #   * type 4 (USER_DEFINED: `<think>`, `<tool_call>`, `<tool_response>`, ...) is matched ALWAYS,
        #     with or without parse_special.
        #   * type 3 (CONTROL: `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`, ...) is matched ONLY when
        #     parse_special is set.
        #
        # Measured, not assumed: with parse_special=False the oracle still emitted `<tool_response>` as one
        # token and mine emitted four, and the three disagreements were exactly the type-4 cases while every
        # type-3 case agreed.  Treating both classes alike costs 3 strings in 1229 and silently changes a chat
        # prompt, because `<|im_end|>` decomposed into ordinary pieces is not the token the model expects.
        self.special_tokens: dict[str, int] = {}
        if token_types:
            for i, ty in enumerate(token_types):
                if ty in (3, 4):
                    self.special_tokens[tokens[i]] = i
        always = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 4]
        self.control_tokens = [t for t, i in self.special_tokens.items() if token_types and token_types[i] == 3]
        # Longest literal first, or `<|im_end|>` could match a shorter prefix of itself.  `regex.escape` so a
        # token containing regex metacharacters (several do: `<|`, `[`, `(`) is matched literally.
        self._always_re = self._alt(always)
        self._special_re = self._alt(list(self.special_tokens))
        # Initialize once, rather than allocating a default dict and looking it up
        # through __dict__.setdefault on each encoded segment or decoded token.
        self._piece_ids: dict[str, list[int]] = {}
        self._bytes_cache: dict[int, bytes] = {}

    @staticmethod
    def _alt(literals: list[str]):
        if not literals:
            return None
        return regex.compile("|".join(regex.escape(s) for s in sorted(literals, key=len, reverse=True)))

    # -------------------------------------------------------------- constructors
    @classmethod
    def from_dir(cls, path) -> "Tokenizer":
        """mab776: a pack's tokenizer/ directory (extract() below) WITH the pre-tokenizer and special ids its
        tokenizer.json names.  Built as Tokenizer(tokens, merges, types) - what serve/server.py did - a DeepSeek
        pack gets the qwen35 split (one digit a token, no Han/kana cut), the bug Project Maya #27 hit for GLM.
        No tokenizer.json (never written by extract): qwen35, as before."""
        t = pathlib.Path(path)
        vocab = json.loads((t / "vocab.json").read_text(encoding="utf-8"))
        tokens = [None] * len(vocab)
        for s, i in vocab.items():
            tokens[i] = s
        meta = json.loads((t / "tokenizer.json").read_text(encoding="utf-8")) if (t / "tokenizer.json").exists() else {}
        # token_type.json is required, as the v0.1.41 server required it: without it a half-written pack would start
        # with no special tokens (control tokens spelled out as text, stop ids from ordinary pieces).  A JSON null
        # (extract() on a GGUF without token types) still means "no types".
        if not (t / "token_type.json").exists():
            raise FileNotFoundError("%s is missing from the tokenizer directory; run setup again" % (t / "token_type.json"))
        types = json.loads((t / "token_type.json").read_text())
        return cls(tokens, (t / "merges.txt").read_text(encoding="utf-8").split("\n"), types,
                   meta.get("pre", "qwen35"), meta.get("special_ids") or None)

    @classmethod
    def from_gguf(cls, path) -> "Tokenizer":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from gguf_reader import GGUFFile
        md = GGUFFile(pathlib.Path(path)).metadata
        need = ["tokenizer.ggml.tokens", "tokenizer.ggml.merges"]
        missing = [k for k in need if k not in md]
        if missing:
            raise ValueError("GGUF is missing %s; it does not carry a tokenizer" % ", ".join(missing))
        model = md.get("tokenizer.ggml.model")
        if model != "gpt2":
            raise ValueError("expected a byte-level BPE tokenizer (model 'gpt2'), got %r" % model)
        special = {k: int(md[k]) for k in md
                   if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
        return cls(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                   md.get("tokenizer.ggml.pre", "qwen35"), special)

    # -------------------------------------------------------------- the algorithm
    def _bpe(self, word: str) -> list[str]:
        """Merge `word` (already byte-mapped) by LOWEST RANK first, repeatedly - not left to right.

        Applying merges in list order rather than rank order is the classic BPE bug: it produces a different
        segmentation and a plausible token count.

        A word longer than HEAP_MIN symbols (a long CJK run, a minified blob: one pre-token piece of thousands of
        bytes) goes to _bpe_heap, which applies the same merges in the same order without rescanning every pair
        after each merge (#268: that rescan is O(n^2) and took seconds on an 8K-character CJK run).
        """
        if len(word) > self.HEAP_MIN:
            return self._bpe_heap(word)
        # `self` is read ONCE, here.  #1385: an interpreter (CPython 3.14.4) was seen handing this frame an int
        # for `self` partway through the scan (`'int' object has no attribute 'ranks'`).  Nothing in this class
        # can do that (no cache, decorator, slots or callback; a thread hammering test cannot make it happen),
        # so the scan below works on a local and no longer re-reads `self` once per symbol pair.
        ranks_get = self.ranks.get
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for i in range(len(parts) - 1):
                r = ranks_get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            parts[best:best + 2] = [parts[best] + parts[best + 1]]
        return parts

    HEAP_MIN = 64       # words up to this many symbols keep the scan above, unchanged

    def _bpe_heap(self, word: str) -> list[str]:
        """_bpe's result in O(n log n): the symbols as a linked list, every adjacent pair with a rank in a heap
        keyed (rank, position).  Popping the lowest rank and then the lowest position is exactly the order the
        scan picks (its strict `<` keeps the leftmost of equal ranks), and a symbol's position is where it starts
        in the word, which only its left neighbour's merge can change - so a popped pair whose two symbols are
        no longer the ones it was pushed with is stale and skipped."""
        import heapq
        parts = list(word)
        n = len(parts)
        nxt = list(range(1, n)) + [-1]
        prv = list(range(-1, n - 1))
        ranks = self.ranks
        heap = []
        for i in range(n - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None:
                heap.append((r, i, parts[i], parts[i + 1]))
        heapq.heapify(heap)
        while heap:
            r, i, left, right = heapq.heappop(heap)
            j = nxt[i]
            if parts[i] != left or j < 0 or parts[j] != right:
                continue                                # stale: one of its symbols has merged since
            parts[i] = left + right
            parts[j] = None                             # j is gone: i takes its place in the list
            k = nxt[j]
            nxt[i] = k
            if k >= 0:
                prv[k] = i
                r2 = ranks.get((parts[i], parts[k]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, i, parts[i], parts[k]))
            p = prv[i]
            if p >= 0:
                r2 = ranks.get((parts[p], parts[i]))
                if r2 is not None:
                    heapq.heappush(heap, (r2, p, parts[p], parts[i]))
        return [s for s in parts if s is not None]

    def _pre_split(self, text: str) -> list[str]:
        """The pre-tokenizer pieces of `text`: each pattern in `_pre_res` splits the pieces the patterns before
        it left, exactly as llama.cpp's `unicode_regex_split` refines `bpe_offsets` one regex at a time.

        mab776: a pattern runs on the collapsed copy of `text` or on `text` itself (_compile_pre), the same length
        either way; pos/endpos bound each piece as slicing did (a lookahead does not see past endpos)."""
        if len(self._pre_res) == 1 and self._pre_res[0][0].groups == 0:
            # mab776 fast path (one pattern = qwen35): C-level findall instead of the span loop when the matches tile
            # the whole text - then gap-keeping and findall give the same pieces; collapsing keeps lengths, so the
            # pieces are cut from `text` at the same offsets.  Anything else falls through to the general loop.
            pat, on_collapsed = self._pre_res[0]
            src = _collapse(text) if on_collapsed else text
            pieces = [p for p in pat.findall(src) if p]
            ends = list(itertools.accumulate(map(len, pieces)))
            if (ends[-1] if ends else 0) == len(text):
                if src is text or src == text:
                    return pieces
                return [text[a:b] for a, b in zip([0] + ends[:-1], ends)]
        spans = [(0, len(text))]
        collapsed = None
        for pat, on_collapsed in self._pre_res:
            if on_collapsed and collapsed is None:
                collapsed = _collapse(text)
            src = collapsed if on_collapsed else text
            nxt: list[tuple[int, int]] = []
            for a, b in spans:
                last = a
                for m in pat.finditer(src, a, b):
                    if m.start() > last:
                        nxt.append((last, m.start()))
                    if m.end() > m.start():
                        nxt.append((m.start(), m.end()))
                    last = m.end()
                if last < b:
                    nxt.append((last, b))
            spans = nxt
        return [text[a:b] for a, b in spans]

    PIECE_CACHE_MAX = 200_000    # pre-tokenizer pieces remembered (an agent resends its whole history every turn)

    def _encode_plain(self, text: str) -> list[int]:
        out: list[int] = []
        # A piece's ids depend on the piece alone, so repeated pieces (most of a resent conversation) are looked up
        # instead of merged again.  The ids are the ones _bpe gives: this only skips the work.
        cache = self.__dict__.setdefault("_piece_ids", {})
        # xolotl: _pre_split, not the dense lineage's `self._re.findall` - the fork replaced `_re` with the
        # per-`pre` pattern list `_pre_res` (joyai-llm = 3 sequential regexes); for qwen35's single pattern the two
        # give the same pieces (the pattern covers every character, so there are no unmatched gaps).
        for piece in self._pre_split(text):
            got = cache.get(piece)
            if got is None:
                mapped = "".join(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))
                got = []
                for tok in self._bpe(mapped):
                    i = self.ids.get(tok)
                    if i is None:
                        raise KeyError("BPE produced a token outside the vocabulary: %r" % tok)
                    got.append(i)
                if len(cache) < self.PIECE_CACHE_MAX:
                    cache[piece] = got
            out.extend(got)
        return out

    def _encode_matching(self, text: str, pat, plain=()) -> list[int]:
        """Encode `text`, emitting any literal `pat` matches as single tokens and BPE-ing the rest.

        The split happens on the RAW text, before the byte mapping, because a special token's string is a
        literal to match rather than bytes to decompose.  Everything between the matches is tokenized
        normally - which is why a near-miss like `<|im_star` still costs ordinary tokens.  A match that starts
        inside one of the `plain` (start, end) spans is left to the text around it (#537).
        """
        if pat is None:
            return self._encode_plain(text)
        out: list[int] = []
        pos = 0
        # Matches arrive in text order. For reusable span sequences, consume each
        # span once instead of scanning the entire list for every literal match.
        # Sorting a copy handles unsorted/overlapping input without mutating it.
        # Other iterables keep the original consumption semantics below.
        span_sequence = isinstance(plain, (list, tuple)) and bool(plain)
        if span_sequence:
            spans = iter(sorted(plain))
            upcoming = next(spans, None)
            plain_end = 0
        for m in pat.finditer(text):
            start = m.start()
            if span_sequence:
                while upcoming is not None and upcoming[0] <= start:
                    plain_end = max(plain_end, upcoming[1])
                    upcoming = next(spans, None)
                if start < plain_end:
                    continue
            elif plain and any(a <= start < b for a, b in plain):
                continue
            if start > pos:
                out.extend(self._encode_plain(text[pos:start]))
            out.append(self.special_tokens[m.group(0)])
            pos = m.end()
        if pos < len(text):
            out.extend(self._encode_plain(text[pos:]))
        return out

    def encode(self, text: str, parse_special: bool = False, plain=()) -> list[int]:
        """Tokenize `text`.

        `parse_special` controls only the type-3 CONTROL literals such as `<|im_end|>`; the type-4
        USER_DEFINED ones such as `<think>` are matched either way.  See the note in `__init__`.
        `plain`: (start, end) spans of `text` that are ordinary text even where they spell a literal - a
        `</think>` quoted in a message (#537) - and are tokenized with the text around them.
        """
        return self._encode_matching(text, self._special_re if parse_special else self._always_re, plain)

    def token_bytes(self, i: int) -> bytes:
        """The raw bytes of one token (a multi-byte character can be split across tokens).

        mab776: a special token (type 3/4) is its own text, as llama.cpp's token_to_piece gives it - DeepSeek's
        `<｜end▁of▁sentence｜>` and `｜DSML｜` hold characters outside the byte alphabet and raised KeyError here.
        Qwen's specials are printable ASCII, which the byte alphabet maps to itself: their bytes do not change."""
        cache = self.__dict__.setdefault("_bytes_cache", {})
        b = cache.get(i)
        if b is None:
            if i < 0 or i >= len(self.tokens):
                raise IndexError("token id %d is outside the vocabulary (%d)" % (i, len(self.tokens)))
            if self.token_types and self.token_types[i] in (3, 4):
                b = cache[i] = self.tokens[i].encode("utf-8")
                return b
            raw = bytearray()
            for ch in self.tokens[i]:
                v = UNICODE_TO_BYTE.get(ch)
                if v is None:
                    raise KeyError("token %d contains a character outside the byte alphabet: %r" % (i, ch))
                raw.append(v)
            b = cache[i] = bytes(raw)
        return b

    def decode(self, ids: list[int], errors: str = "replace") -> str:
        return b"".join(self.token_bytes(i) for i in ids).decode("utf-8", errors=errors)


# ------------------------------------------------------------------ the pack's tokenizer/ directory
def extract(gguf_path, out_dir) -> dict:
    """Write the tokenizer into `<out_dir>/tokenizer/` so the engine never opens the weight shards for it."""
    out = pathlib.Path(out_dir) / "tokenizer"
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer.from_gguf(gguf_path)
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    md = GGUFFile(pathlib.Path(gguf_path)).metadata
    pats = PRE_PATTERNS[tk.pre]
    cfg = {
        "model": "gpt2",
        "pre": tk.pre,
        "vocab_size": len(tk.tokens),
        "n_merges": len(tk.ranks),
        "special_ids": tk.special_ids,
        # mab776: the GGUF's own flag (DeepSeek-V4-Flash: False, its chat template writes bos_token itself)
        "add_bos_token": bool(md.get("tokenizer.ggml.add_bos_token", False)),
        # The pattern is SHIPPED, not recomputed by the reader: it is transcribed from llama.cpp for the
        # declared `pre` type, and a C++ port that re-derived it would be free to get `\p{M}` wrong again.
        # mab776: the patterns of THIS pre, applied in order (joyai-llm: three; qwen35: one).
        "pre_patterns": list(pats),
        "pre_pattern": pats[0] if len(pats) == 1 else None,
        "pre_pattern_source": (".ref/llama.cpp src/llama-vocab.cpp L396 (LLAMA_VOCAB_PRE_TYPE_QWEN35)"
                               if tk.pre == "qwen35" else
                               "llama.cpp 3cf03257 src/llama-vocab.cpp L319-327 (LLAMA_VOCAB_PRE_TYPE_%s; three "
                               "regexes applied in sequence, unicode.cpp unicode_regex_split)"
                               % tk.pre.upper().replace("-", "_")),
    }
    (out / "vocab.json").write_text(json.dumps(tk.ids, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("\n".join("%s %s" % k for k, _ in
                                              sorted(tk.ranks.items(), key=lambda kv: kv[1])), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(tk.token_types), encoding="utf-8")
    (out / "tokenizer.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    # the model's own chat template: fine-tunes change it (Swift 1.5 differs from Qwen3.8-Flash-Next's)
    tpl = md.get("tokenizer.chat_template")
    if tpl:
        (out / "chat_template.jinja").write_text(tpl, encoding="utf-8", newline="\n")
    return cfg


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true", help="round-trip a corpus and report")
    args = ap.parse_args()
    cfg = extract(args.gguf, args.out)
    print("tokenizer/: vocab %d, merges %d, pre %s, specials %s"
          % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"], cfg["special_ids"]))
    if args.check:
        tk = Tokenizer.from_gguf(args.gguf)
        corpus = ["", "Hello, world!", "  leading and trailing  ", "a\n\n\nb",
                  "def f(x):\n\treturn x  # comment\n", "\u4f60\u597d\uff0c\u4e16\u754c", "\u0645\u0631\u062d\u0628\u0627",
                  "\U0001f600\U0001f680\U0001f1fa\U0001f1f8", "e\u0301\u0301 combining", "\x00\x01\x7f control",
                  "\u00a0non-breaking\u00a0space", "1234567890", "MixedCASE_and-dashes", "\r\n\r\n", "\u2028\u2029",
                  "x" * 5000]
        bad = 0
        for s in corpus:
            ids = tk.encode(s)
            back = tk.decode(ids)
            if back != s:
                print("  *** ROUND TRIP FAILED *** %r -> %d ids -> %r" % (s[:40], len(ids), back[:40]))
                bad += 1
        print("round trip: %d strings, %d failed" % (len(corpus), bad))
        return 0 if bad == 0 else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
