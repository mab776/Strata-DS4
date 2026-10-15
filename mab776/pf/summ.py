#!/usr/bin/env python3
"""mab776 overnight: one table row per (variant, depth) from ram-128gb/v41/pf/*.jsonl (+ the chunk prof / prompt lines of
each variant's .log).   summ.py [label-prefix ...]"""
import glob, json, os, re, sys
O = "/home/mab776/Documents/portainer/ram-128gb/v41/pf"
pre = sys.argv[1:]
rows = []
for f in sorted(glob.glob(f"{O}/*.jsonl"), key=os.path.getmtime):
    L = os.path.basename(f)[:-6]
    if pre and not any(L.startswith(p) for p in pre):
        continue
    for ln in open(f):
        try:
            r = json.loads(ln)
        except Exception:
            continue
        if "turn" in r:
            rows.append((L, f"turn{r['turn']}+{r['new']}", r.get("read"), r.get("prefill_s"), r.get("prefill_tps"), r.get("decode_tps")))
        else:
            rows.append((L, str(r.get("depth")), r.get("depth"), r.get("prefill_s"), r.get("prefill_tps"), r.get("decode_tps")))
print(f"{'variant':<18} {'prompt':>12} {'read':>7} {'prefill_s':>9} {'tok/s':>7} {'decode':>7}")
for L, d, rd, ps, pt, dt in rows:
    print(f"{L:<18} {d:>12} {rd if rd is not None else '':>7} {ps if ps is not None else '':>9} {pt if pt is not None else '':>7} {dt if dt is not None else '':>7}")
for f in sorted(glob.glob(f"{O}/*.log"), key=os.path.getmtime):
    L = os.path.basename(f)[:-4]
    if L in ("sweep", "q1") or L.endswith(".engine") or (pre and not any(L.startswith(p) for p in pre)):
        continue
    for ln in open(f):
        if "chunk prof" in ln or "prompt phases" in ln or "prompt cache:" in ln:
            print(f"  {L}: {ln.strip()[:260]}")
