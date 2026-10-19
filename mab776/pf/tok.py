#!/usr/bin/env python3
"""mab776 overnight: tokenize a text file with V4.1's tokenizer (strata_tokenizer + the joyai-llm pre-split, as
serve_ds41.py) and write the first N ids as int32 - the --ids-file of a one-shot --ppl run.   tok.py TEXT N OUT"""
import json, sys
from pathlib import Path
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-ds41/tools")
import strata_tokenizer as ST
import array
T = Path("/media/mab776/nvme 2To/models/deepseek-v41/gsq/tokenizer")
vocab = json.loads((T / "vocab.json").read_text(encoding="utf-8"))
tokens = [None] * len(vocab)
for t, i in vocab.items():
    tokens[i] = t
tok = ST.Tokenizer(tokens, (T / "merges.txt").read_text(encoding="utf-8").split("\n"), json.loads((T / "token_type.json").read_text()), pre="joyai-llm")
ids = tok.encode(Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace"))[: int(sys.argv[2])]
array.array("i", ids).tofile(open(sys.argv[3], "wb"))
print(len(ids), "ids ->", sys.argv[3])
