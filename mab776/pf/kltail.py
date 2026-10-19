#!/usr/bin/env python3
"""mab776: KL of two tail dumps whose row counts differ (A = the last na rows, B = the last nb >= na rows): A vs B's last na rows.
   kltail.py A.tail.f32 B.tail.f32"""
import sys, numpy as np
V = 129280
A = np.memmap(sys.argv[1], dtype=np.float32, mode="r").reshape(-1, V)
B = np.memmap(sys.argv[2], dtype=np.float32, mode="r").reshape(-1, V)
n = min(len(A), len(B)); A = A[len(A) - n:]; B = B[len(B) - n:]
def ls(x):
    x = np.asarray(x, dtype=np.float64); m = x.max(1, keepdims=True); z = x - m
    return z - np.log(np.exp(z).sum(1, keepdims=True))
kl, t1 = [], 0
for r in range(0, n, 64):
    a, b = ls(A[r:r+64]), ls(B[r:r+64])
    kl.append((np.exp(a) * (a - b)).sum(1)); t1 += int((a.argmax(1) == b.argmax(1)).sum())
kl = np.concatenate(kl)
print(f"{n} rows: KL mean {kl.mean():.3e} p99 {np.quantile(kl, .99):.3e} max {kl.max():.3e}; top-1 {t1 / n:.4f}")
