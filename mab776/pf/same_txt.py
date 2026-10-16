#!/usr/bin/env python3
"""mab776: identity check between two arms' domain_test.py replies (OUT=<dir>/<label> -> <label>.<i>.txt): per request
SAME / DIFFERS (+ the first differing character offset).  Prints no text.
  same_txt.py DIR LABEL_A LABEL_B"""
import glob, os, sys
d, a, b = sys.argv[1:4]
n_same = n = 0
for fa in sorted(glob.glob(f"{d}/{a}.*.txt"), key=lambda p: int(p.rsplit(".", 2)[-2])):
    i = fa.rsplit(".", 2)[-2]
    fb = f"{d}/{b}.{i}.txt"
    if not os.path.exists(fb):
        print(i, "missing in", b)
        continue
    ta, tb = open(fa, errors="replace").read(), open(fb, errors="replace").read()
    n += 1
    if ta == tb:
        n_same += 1
        print(i, "SAME", len(ta))
    else:
        k = next((j for j in range(min(len(ta), len(tb))) if ta[j] != tb[j]), min(len(ta), len(tb)))
        print(i, "DIFFERS at char", k, "of", len(ta), "/", len(tb))
print(f"{n_same}/{n} identical")
