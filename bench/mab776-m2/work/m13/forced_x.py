#!/usr/bin/env python3
"""M13 cross-arm numerics: the same teacher-forced token sequence in two engine configurations.
   forced_x.py URL DEPTH ref LABEL   - a plain greedy run's tokens become the oracle (kept in the logs dir) and its rows the
                                       reference (logs/m13ref.rows); the engine must run with DS4_DUMP_TOKLOGITS=/data/logs/tl
   forced_x.py URL DEPTH cmp LABEL   - forced onto that oracle (DS4_SERVE_TEST_ORACLE + DS4_SERVE_TEST_FORCE): plain, K=1,
                                       K=2 rows vs the reference: KL per row (row by row: the oomd lesson), top-1 changes"""
import json, os, shutil, sys, urllib.request
import numpy as np
url, depth, mode, label = sys.argv[1].rstrip("/"), int(sys.argv[2]), sys.argv[3], sys.argv[4]
LOGS = "/media/mab776/nvme 2To/models/strata-ds4/logs"
V = 129280
PARA = ("The quick brown fox jumps over the lazy dog while 12 engineers debug a "
        "distributed cache at 3am, logging metric 4567 and tracing span 89ab. ")
TAIL = ("\n\nIgnore the text above. Write a complete, well-commented Python module implementing an LRU cache with TTL "
        "expiry, thread safety and unit tests.\n\n```\n")
prompt = "[run forced-m13] " + PARA * (int(depth * 4.08 / len(PARA)) + 1) + TAIL
def ask(k):
    body = {"model": "deepseek-v4-flash", "max_tokens": 384, "temperature": 0, "stream": False,
            "messages": [{"role": "user", "content": prompt}], "strata_tune": {"mtp_k": k, "lookup": 0}}
    urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                                  {"Content-Type": "application/json"}), timeout=7200).read()
def newest():
    fs = [f for f in os.listdir(LOGS) if f.startswith("tl.")]
    return max(fs, key=lambda f: int(f.split(".")[1]))
ref_rows, oracle = os.path.join(LOGS, "m13ref.rows"), os.path.join(LOGS, "oracle.txt")
if mode == "ref":
    ask(0)
    f = os.path.join(LOGS, newest())
    toks = []
    R = np.memmap(f, dtype=np.float32, mode="r").reshape(-1, V)
    for r in range(R.shape[0]): toks.append(int(R[r].argmax()))
    open(oracle, "w").write("\n".join(map(str, toks)) + "\n")
    shutil.move(f, ref_rows)
    print(f"== {label}: reference of {len(toks)} rows at depth {depth} kept")
    sys.exit(0)
A = np.memmap(ref_rows, dtype=np.float32, mode="r").reshape(-1, V)
def lsm(x):
    x = x.astype(np.float64); x -= x.max(); return x - np.log(np.exp(x).sum())
print(f"== {label} vs the reference (depth {depth}, {A.shape[0]} rows)")
for name, k in (("plain", 0), ("K1", 1), ("K2", 2)):
    ask(k)
    f = os.path.join(LOGS, newest())
    B = np.memmap(f, dtype=np.float32, mode="r").reshape(-1, V)
    m = min(len(A), len(B)); kl = np.empty(m); top = 0
    for r in range(m):
        a = lsm(A[r]); b = lsm(B[r]); kl[r] = (np.exp(a) * (a - b)).sum(); top += int(a.argmax() != b.argmax())
    print(f"  {name:6s}: KL mean {kl.mean():.5f} median {np.median(kl):.5f} p95 {np.percentile(kl, 95):.5f} "
          f"max {kl.max():.4f}, top-1 differs {top}/{m}", flush=True)
    del B
    os.remove(f)
