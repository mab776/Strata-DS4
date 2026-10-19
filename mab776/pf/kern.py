#!/usr/bin/env python3
"""mab776 overnight: top kernels / memcpy of an nsys stats export (prof.sh's csv files), grouped by a coarse family.
   kern.py ram-128gb/v41/pf/prof/<label>   (reads <label>_cuda_gpu_kern_sum.csv, _cuda_gpu_mem_time_sum.csv)"""
import csv, re, sys
P = sys.argv[1]
def rd(n):
    try:
        return list(csv.DictReader(open(f"{P}_{n}.csv")))
    except FileNotFoundError:
        return []
K = rd("cuda_gpu_kern_sum")
tot = sum(float(r["Total Time (ns)"]) for r in K)
fam = {}
def family(n):
    n = n.lower()
    for k, v in (("lightning", "indexer"), ("topk", "top-k"), ("top_k", "top-k"), ("argsort", "argsort"), ("mmq", "mmq (dense+expert)"),
                 ("mul_mat_q", "mmq (dense+expert)"), ("native_gu", "expert grouped gu"), ("native_down", "expert grouped down"),
                 ("grouped", "expert grouped"), ("prom", "expert grouped (prom)"), ("ptq1", "ptq1 transcode"), ("flash_attn", "flash-attn"),
                 ("fattn", "flash-attn"), ("mul_mat_vec", "mmvq/mmvf"), ("dsv4_hc", "hc ops"), ("dsv41", "dsv41 fused"), ("get_rows", "get_rows"),
                 ("set_rows", "set_rows"), ("cpy", "cpy"), ("quantize", "quantize"), ("rms_norm", "norm"), ("norm", "norm"),
                 ("bin_bcast", "elementwise"), ("unary", "elementwise"), ("scale", "elementwise"), ("swiglu", "swiglu"), ("glu", "swiglu"),
                 ("softmax", "softmax"), ("rope", "rope"), ("concat", "concat"), ("pool", "pool"), ("weighted_rows", "expert sum")):
        if k in n:
            return v
    return "other"
for r in K:
    f = family(r["Name"])
    fam[f] = fam.get(f, 0) + float(r["Total Time (ns)"])
print(f"GPU kernel time {tot/1e9:.2f} s")
for f, t in sorted(fam.items(), key=lambda x: -x[1]):
    print(f"  {f:<26} {t/1e9:7.2f} s  {100*t/tot:5.1f} %")
print("top 25 kernels:")
for r in sorted(K, key=lambda r: -float(r["Total Time (ns)"]))[:25]:
    print(f"  {float(r['Total Time (ns)'])/1e9:7.3f} s {int(r['Instances']):>8} x {float(r['Avg (ns)'])/1e3:9.1f} us  {r['Name'][:110]}")
M = rd("cuda_gpu_mem_time_sum")
for r in M:
    print(f"  mem {r.get('Operation','?'):<34} {float(r['Total Time (ns)'])/1e9:7.3f} s {r.get('Count', r.get('Instances',''))}")
S = rd("cuda_gpu_mem_size_sum")
for r in S:
    print(f"  size {r.get('Operation','?'):<33} {float(r.get('Total (MB)', 0))/1e3:9.1f} GB")
