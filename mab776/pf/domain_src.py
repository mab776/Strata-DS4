#!/usr/bin/env python3
"""mab776: the domain corpora of domain_test.py, and their 4K-token PPL samples (two disjoint parts each: A = the
domain's own routing count, B = the PPL text).  Writes ram-128gb/v41/pf/ppl/dom-<domain>-{A,B}.i32; prints counts only.
  domain_src.py"""
import glob, os, sys
import numpy as np

sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-ds41/tools")
from ple_bench import corpus  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
import json  # noqa: E402

C = "/home/mab776/.claude/jobs/6a7c0663/tmp/corpora"
P = "/home/mab776/Documents/portainer"
TOK = "/media/mab776/nvme 2To/models/deepseek-v41/gsq/tokenizer"
OUT = "/home/mab776/Documents/portainer/ram-128gb/v41/pf/ppl"


def gutenberg(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    a, b = t.find("*** START OF"), t.find("*** END OF")
    if a >= 0:
        t = t[t.find("\n", a) + 1:]
    if b >= 0:
        t = t[:t.find("*** END OF")]
    return t


def files(pattern_list, exclude=()):
    parts = []
    for pat in pattern_list:
        for f in sorted(glob.glob(pat, recursive=True)):
            if any(x in f for x in exclude) or os.path.getsize(f) > 200000:
                continue
            try:
                parts.append(open(f, encoding="utf-8", errors="replace").read())
            except OSError:
                pass
    return "\n\n".join(parts)


def sources():
    return {
        "code": corpus()[5000000:],
        "powershell": files(["/home/mab776/Documents/golf/**/*.ps1"], exclude=(".venv",)),
        "ops": files([P + "/*/docker-compose*.yml", P + "/*/*.sh", P + "/caddy-config/Caddyfile"],
                     exclude=("strata-ds4", "project-maya", "strata-ds41", ".bak")),
        "french": files([P + "/menu-planner/*.md", P + "/menu-planner/data/*.md"]),
        "fr-literary": gutenberg(C + "/pg14155.txt"),
        "poetry": gutenberg(C + "/pg6099.txt") + "\n\n" + gutenberg(C + "/pg1041.txt"),
        "en-prose": gutenberg(C + "/pg1342.txt"),
    }


if __name__ == "__main__":
    vocab = json.loads(open(TOK + "/vocab.json", encoding="utf-8").read())
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    merges = open(TOK + "/merges.txt", encoding="utf-8").read().split("\n")
    types = json.loads(open(TOK + "/token_type.json").read())
    tok = ST.Tokenizer(tokens, merges, types, pre="joyai-llm")
    for dom, text in sources().items():
        ids = tok.encode(text[:400000])
        n = len(ids)
        if n < 12288 + 4096:
            print(dom, "too short:", n, "tokens")
            continue
        a = np.array(ids[:16384 if n >= 16384 + 4096 + 1024 else 12288], dtype=np.int32)   # A: the domain's count
        b = np.array(ids[-4096:], dtype=np.int32)                                           # B: the PPL text
        a.tofile(f"{OUT}/dom-{dom}-A.i32")
        b.tofile(f"{OUT}/dom-{dom}-B.i32")
        print(dom, "tokens", n, "A", len(a), "B", len(b))
