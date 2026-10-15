#!/usr/bin/env python3
"""mab776 overnight: the report's variant table (markdown) from ram-128gb/v41/pf/<variant>.{jsonl,log} + the KL of each
variant's klgate dump vs n0-base.   table.py variant [variant ...]"""
import json, os, re, subprocess, sys
O = "/home/mab776/Documents/portainer/ram-128gb/v41/pf"
KL = "/home/mab776/Documents/portainer/strata-ds41/tools/v41/logits_kl.py"
print("| variant | 8.7k prefill s (tok/s) | 32k / 123k prefill s (tok/s) | decode 8k / long | ctr RAM max | min host avail | KL vs base (top-1) |")
print("|---|---|---|---|---|---|---|")
for v in sys.argv[1:]:
    rows = [json.loads(l) for l in open(f"{O}/{v}.jsonl") if l.startswith("{")] if os.path.exists(f"{O}/{v}.jsonl") else []
    by = {}
    for r in rows:
        if "turn" in r:
            continue
        d = r.get("depth") or 0
        by["8k" if d < 12000 else "long"] = r
    def cell(r):
        return f"{r['prefill_s']} ({r['prefill_tps']})" if r else "-"
    dec = f"{(by.get('8k') or {}).get('decode_tps', '-')} / {(by.get('long') or {}).get('decode_tps', '-')}"
    ctr, av = [], []
    if os.path.exists(f"{O}/{v}.log"):
        for ln in open(f"{O}/{v}.log"):
            m = re.search(r"(\d+) avail GiB.*ctr ([\d.]+)GiB", ln)
            if m:
                av.append(int(m.group(1))); ctr.append(float(m.group(2)))
    kl = "-"
    a, b = f"{O}/dump/n0-base/req001.tail.f32", f"{O}/dump/{v}/req001.tail.f32"
    if v != "n0-base" and os.path.exists(a) and os.path.exists(b):
        out = subprocess.run(["python3", KL, a, b], capture_output=True, text=True).stdout
        m = re.search(r"KL mean ([\d.e+-]+).*top-1 agreement ([\d.]+)", out)
        if m:
            kl = f"{float(m.group(1)):.1e} ({m.group(2)})"
    long = by.get("long")
    print(f"| {v} | {cell(by.get('8k'))} | {cell(long)}{' @' + str(long['depth']) if long else ''} | {dec} | "
          f"{max(ctr) if ctr else '-'} GiB | {min(av) if av else '-'} GiB | {kl} |")
