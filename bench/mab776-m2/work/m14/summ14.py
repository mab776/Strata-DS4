#!/usr/bin/env python3
"""M14: mean verify pass ms (and tok/s) per arm label / depth / K from an A/B log section.  summ14.py LOG [section-substring]"""
import re, sys, collections
log = open(sys.argv[1]).read()
if len(sys.argv) > 2:
    i = log.index(sys.argv[2]); j = log.find("\n#### ", i + 1); log = log[i: j if j > 0 else None]
arm = None; d = collections.defaultdict(list)
for l in log.splitlines():
    m = re.match(r"### (\S+) \d", l) or re.match(r"### (pf-b \S+) \d", l)
    if l.startswith("### pf-b"): arm = " ".join(l.split()[1:3])
    elif m: arm = m.group(1)
    m = re.search(r"depth\s+(\d+) K=(\d) decode\s+([\d.]+) tok/s, ([\d.]+) tok/pass, pass ([\d.]+) ms", l)
    if m and arm: d[(m.group(1), m.group(2), arm)].append((float(m.group(5)), float(m.group(3))))
for k in sorted(d, key=lambda k: (int(k[0]), k[1], k[2])):
    v = d[k]; print("depth %6s K=%s %-26s pass %6.2f ms  decode %5.1f tok/s  (n=%d)" % (k[0], k[1], k[2], sum(x[0] for x in v) / len(v), sum(x[1] for x in v) / len(v), len(v)))
