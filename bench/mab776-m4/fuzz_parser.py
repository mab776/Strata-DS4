"""Adversarial fuzz of serve/deepseek.py's DSML parser.
1. random texts (markers, partial markers, newlines, words, valid/invalid pieces): whole == random chunkings == char by
   char (several finish reasons, thinking on/off, tools/no tools); no-call outputs lose and add no character;
2. round trip: random valid replies built in the template's DSML form (random typed values incl. unicode, quotes,
   newlines, markup-looking text) parse back to exactly the calls, reasoning and content that were written;
3. real DS4 token boundaries (STRATA_DS4_TOKENIZER): detokenized pieces == whole, plus every cut between pieces.
Prints counts."""
import json
import os
import random
import sys
import time

sys.path.insert(0, "/src/tools")
sys.path.insert(0, "/src")
from serve.deepseek import (DSML, END, INVOKE_CLOSE, INVOKE_OPEN, PARAM_CLOSE, PARAM_OPEN, START,  # noqa: E402
                            THINK_END, THINK_START, DeepSeekOutputParser)
from serve.test_deepseek import TOOLS, block, invoke, parse, summary  # noqa: E402

N = int(os.environ.get("FUZZ_N", "3000"))
rng = random.Random(int(os.environ.get("FUZZ_SEED", "20261010")))
FRAGS = ["abc", "word ", " ", "\n", "\n\n", "\n\n\n", "\t", THINK_END, THINK_START, START, END, START[:3], START[:9],
         END[:5], END[:12], DSML, "<", ">", '"', "</", "｜", INVOKE_OPEN + 'write">', INVOKE_OPEN + 'get_weather">\n',
         INVOKE_OPEN + 'nope">', INVOKE_OPEN, INVOKE_OPEN[:8], PARAM_OPEN + 'a" string="true">',
         PARAM_OPEN + 'n" string="false">', PARAM_OPEN + 'a"', PARAM_OPEN + 'city" string="true">', PARAM_OPEN[:11],
         "5", "[1, 2]", '{"k": "v"}', "x y", PARAM_CLOSE, PARAM_CLOSE[:6], INVOKE_CLOSE, INVOKE_CLOSE[:9],
         invoke("write", a="v"), invoke("get_weather", city="Oslo"), block(invoke("write", a="1")),
         block(invoke("get_weather", city="P"), invoke("write", path="p", content="c")), "é", "台北", "😀", "​"]
stats = {"texts": 0, "parses": 0, "with_calls": 0, "no_loss_checked": 0, "roundtrip": 0, "roundtrip_calls": 0,
         "token_texts": 0, "token_parses": 0}


def cuts(text):
    yield list(text)
    for _ in range(6):
        if len(text) < 2:
            yield [text]
            continue
        k = rng.randint(1, min(40, len(text) - 1))
        c = sorted(rng.sample(range(1, len(text)), k))
        yield [text[a:b] for a, b in zip([0] + c, c + [len(text)])]


t0 = time.time()
for _ in range(N):
    text = "".join(rng.choice(FRAGS) for _ in range(rng.randint(1, 30)))
    stats["texts"] += 1
    for thinking in (True, False):
        for tools in (TOOLS, None):
            for reason in ("stop", "length"):
                whole = parse(text, thinking, tools, reason)
                for pieces in cuts(text):
                    got = parse(pieces, thinking, tools, reason)
                    stats["parses"] += 1
                    if got != whole:
                        print("MISMATCH", repr(text), pieces, thinking, tools is not None, reason, got, whole)
                        sys.exit(1)
                ev = whole[0]
                if any(k == "call" for k, _ in ev):
                    stats["with_calls"] += 1
                    continue
                r = "".join(t for k, t in ev if k == "reasoning")
                c = "".join(t for k, t in ev if k == "content")
                if not thinking:
                    want = ("", text)
                elif THINK_END in text:
                    i = text.index(THINK_END)
                    want = (text[:i], text[i + len(THINK_END):].lstrip("\n"))
                else:
                    want = (text, "")
                if (r, c) != want:
                    print("LOSS", repr(text), thinking, tools is not None, reason, (r, c), want)
                    sys.exit(1)
                stats["no_loss_checked"] += 1
print("fuzz done", time.time() - t0, "s", flush=True)

# 2. round trip of valid replies
WORDS = ["a", "Montréal", "台北 \"true\"", "x\ny", " lead", "trail ", "<", ">", '"q"', "\\", "é😀", "</", DSML,
         END, INVOKE_CLOSE, PARAM_CLOSE + " not an end", START, "<think>", "</think>", "{", "[1,2]"]


def value():
    kind = rng.randint(0, 6)
    if kind == 0:
        return rng.randint(-5, 10 ** 6)
    if kind == 1:
        return rng.choice([True, False, None, 1.5])
    if kind == 2:
        return [value() for _ in range(rng.randint(0, 3))]
    if kind == 3:
        return {rng.choice(["k", "é", "a b"]): value() for _ in range(rng.randint(0, 2))}
    return "".join(rng.choice(WORDS) for _ in range(rng.randint(0, 6)))


def safe_string(v):
    """A string value parses back only when no PARAM_CLOSE in it is followed by the next element's opener."""
    return isinstance(v, str) and not any(v[i + len(PARAM_CLOSE):].lstrip(" \t\r\n").startswith((PARAM_OPEN,
                                          INVOKE_CLOSE)) or not v[i + len(PARAM_CLOSE):].strip(" \t\r\n")
                                          for i in range(len(v)) if v.startswith(PARAM_CLOSE, i))


for _ in range(N):
    calls = []
    for _ in range(rng.randint(1, 3)):
        args = {}
        for _ in range(rng.randint(0, 4)):
            v = value()
            if isinstance(v, str) and not safe_string(v):
                v = v.replace(PARAM_CLOSE, "")
            args[rng.choice(["path", "content", "n", "flags", "city", "x y"])] = v
        calls.append((rng.choice(["write", "get_weather", "search_files"]), args))
    reasoning = "".join(rng.choice(["Think ", "more.", "\n", "台北", "<", "x"]) for _ in range(rng.randint(0, 8)))
    reasoning = reasoning.replace(THINK_END, "").rstrip("\n") or "Plan."
    content = "".join(rng.choice(["Sure", " ", "\n", "ok.", "台北"]) for _ in range(rng.randint(0, 5)))
    text = reasoning + THINK_END + content + ("\n\n" if content else "") + block(*[invoke(n, **a) for n, a in calls])
    got, _ = parse(text)
    want = [("reasoning", reasoning)] + ([("content", content.lstrip("\n"))] if content.lstrip("\n") else []) + \
        [("call", c) for c in calls]
    # JSON round trip of the written values (1.5 stays 1.5, tuples are lists)
    want = json.loads(json.dumps(want))
    if json.loads(json.dumps(got)) != want:
        print("ROUNDTRIP", repr(text), got, want)
        sys.exit(1)
    for pieces in cuts(text):
        if parse(pieces)[0] != got:
            print("ROUNDTRIP-CHUNK", repr(text), pieces)
            sys.exit(1)
    stats["roundtrip"] += 1
    stats["roundtrip_calls"] += len(calls)
print("round trip done", time.time() - t0, "s", flush=True)

# 3. real token boundaries
tdir = os.environ.get("STRATA_DS4_TOKENIZER")
if tdir:
    import strata_tokenizer as ST
    from serve.server import Detokenizer
    tok = ST.Tokenizer.from_dir(tdir)
    for _ in range(max(200, N // 10)):
        text = "".join(rng.choice(FRAGS + WORDS) for _ in range(rng.randint(1, 25)))
        ids = tok.encode(text, parse_special=True)
        d = Detokenizer(tok)
        pieces = [d.push(i) for i in ids]
        if "".join(pieces) != text:
            continue                                      # a text the tokenizer normalizes: skip
        stats["token_texts"] += 1
        for thinking in (True, False):
            whole = parse(text, thinking)
            if parse(pieces, thinking) != whole:
                print("TOKEN MISMATCH", repr(text), pieces)
                sys.exit(1)
            for i in range(0, len(pieces) + 1, max(1, len(pieces) // 8)):
                if parse(["".join(pieces[:i]), "".join(pieces[i:])], thinking) != whole:
                    print("TOKEN SPLIT", repr(text), i)
                    sys.exit(1)
                stats["token_parses"] += 1
print("tokens done", time.time() - t0, "s")
print(json.dumps(stats))
