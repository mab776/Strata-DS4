#!/usr/bin/env python3
# mkcfg.py <name> [--flag value | --flag ...] : variant of mab776/strata-ds41.json -> mab776/pf/<name>.json
# a flag already in args gets its value replaced; "--flag=-" removes it
import json, sys, os
W = "/home/mab776/Documents/portainer/strata-ds41"
c = json.load(open(f"{W}/mab776/strata-ds41.json"))
a = c["args"]; name = sys.argv[1]; rest = sys.argv[2:]; i = 0
while i < len(rest):
    k = rest[i]
    if k.endswith("=-"):
        k = k[:-2]
        if k in a:
            j = a.index(k); del a[j]
            if j < len(a) and not a[j].startswith("--"): del a[j]
        i += 1; continue
    v = rest[i + 1] if i + 1 < len(rest) and not rest[i + 1].startswith("--") else None
    if k in a:
        j = a.index(k)
        if v is not None: a[j + 1] = v
    else:
        a += [k] + ([v] if v is not None else [])
    i += 2 if v is not None else 1
json.dump(c, open(f"{W}/mab776/pf/{name}.json", "w"), indent=1)
print(f"mab776/pf/{name}.json", " ".join(a[a.index("--ctx"):]))
