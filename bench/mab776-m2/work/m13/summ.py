import re, sys
# per A/B block: mean pass ms by (arm value, K) over the ABBA rounds
cur = None; d = {}; order = []
for l in open(sys.argv[1]):
    m = re.match(r'#### ab (\S+)', l)
    if m: cur = m.group(1); d[cur] = {}; order.append(cur); continue
    if l.startswith('#### final'): cur = None; continue
    m = re.match(r'### (\S+)=(\d) ', l)
    if m and cur: arm = m.group(2); continue
    m = re.search(r'depth\s+(\d+) K=(\d) decode\s+([\d.]+) tok/s, ([\d.]+) tok/pass, pass ([\d.]+) ms', l)
    if m and cur:
        d[cur].setdefault((arm, m.group(1), m.group(2)), []).append(float(m.group(5)))
for v in order:
    keys = sorted({(k[1], k[2]) for k in d[v]})
    for dep, K in keys:
        a = d[v].get(('0', dep, K), []); b = d[v].get(('1', dep, K), [])
        if a and b:
            ma, mb = sum(a)/len(a), sum(b)/len(b)
            print(f"{v:16s} {dep:>6} K={K}: pass ms off {ma:6.2f} (n={len(a)}) -> on {mb:6.2f} (n={len(b)})  {100*(mb-ma)/ma:+.1f} %")
