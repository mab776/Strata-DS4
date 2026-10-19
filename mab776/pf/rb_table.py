#!/usr/bin/env python3
"""mab776: the route-bias A/B table (rb_ab.sh + ppl2.sh p5.txt + rb_lcb.sh results) as markdown.
  rb_table.py [ARM ...]   (default A B C D)"""
import glob, json, os, re, statistics, sys

O = "/home/mab776/Documents/portainer/ram-128gb/v41/pf"
D = "/home/mab776/Documents/portainer/strata-coder/bench/ab-gsq-20260926"
arms = sys.argv[1:] or ["A", "B", "C", "D"]
name = {"A": "A 0.5/0.3 (served)", "B": "B exact 0/0", "C": "C 0.2/0.1", "D": "D file-only 0/0.3", "B0": "B, overlap off", "B2": "B again"}
ppl_arm = {"A": "A", "B": "exact", "C": "C", "D": "D"}
doms = ["code", "powershell", "ops", "french", "poetry"]


def rows(path):
    out = []
    if os.path.exists(path):
        for l in open(path):
            try:
                out.append(json.loads(l))
            except ValueError:
                pass
    return out


def ppl(label):
    p = f"{O}/ppl/{label}.log"
    if not os.path.exists(p):
        return None
    m = re.search(r"^PPL ([\d.]+)", open(p).read(), re.M)
    return float(m.group(1)) if m else None


def f(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


cols = {}
for a in arms:
    r = rows(f"{O}/rb-{a}.jsonl")
    dep = {}
    for x in r:
        if "depth" in x and "turn" not in x and "run" not in x and x.get("depth", 0) > 5000:
            dep["8.7k" if x["depth"] < 50000 else "123k"] = x
    turns = {x["new"]: x for x in r if "turn" in x}
    dom = rows(f"{O}/rb-{a}.dom.jsonl")
    snf = [x for x in rows(f"{O}/rb-{a}.snf.jsonl") if "rec" in x and "error" not in x and x.get("decode_tps")]
    c = {}
    for k in ("8.7k", "123k"):
        x = dep.get(k)
        c[f"decode {k} tok/s"] = f(x and x.get("decode_tps"))
        c[f"prefill {k} s"] = f(x and x.get("prefill_s"))
        c[f"hit rate {k} %"] = f(x and x.get("hit_rate") and 100 * x["hit_rate"])
    for n in (300, 1500):
        x = turns.get(n)
        c[f"turn +{n} wall s"] = f(x and x.get("wall_s"), 2)
    by = {x["domain"]: x for x in dom}
    for d in ["code", "powershell", "ops", "french", "fr-literary", "poetry"]:
        x = by.get(d)
        c[f"decode {d} tok/s"] = f(x and x.get("decode_tps"))
    sw = [by[k]["decode_tps"] for k in ("switch:powershell", "switch:french", "switch:powershell2") if k in by]
    c["decode PS-FR-PS session tok/s"] = f(statistics.mean(sw) if sw else None)
    c["decode 9 domain requests, mean"] = f(statistics.mean(x["decode_tps"] for x in dom) if dom else None)
    c["GPU-resident share, domains %"] = f(statistics.mean(x["fast"] for x in dom if "fast" in x) if dom else None)
    c["file-tier share, domains %"] = f(statistics.mean(x["file"] for x in dom if "file" in x) if dom else None, 2)
    c["decode Marc's traffic tok/s (n)"] = (f(statistics.mean(x["decode_tps"] for x in snf)) + f" ({len(snf)})") if snf else "-"
    c["GPU-resident share, Marc's traffic %"] = f(statistics.mean(x["fast"] for x in snf if "fast" in x) if snf else None)
    for d in doms:
        c[f"PPL {d}"] = f(ppl(f"p5-{d}-{ppl_arm.get(a, '')}"), 4) if a in ppl_arm else "-"
    if a == "A":
        for d in doms:
            c[f"PPL {d}"] += f" (adapted {f(ppl(f'p5-{d}-Aadapt'), 4)})"
    g = f"{D}/grade-v41-rb{a}-32k-t1-med.json"
    gen = f"{D}/gen-v41-rb{a}-32k-t1-med.jsonl"
    if os.path.exists(g) and os.path.exists(gen):
        gr, ge = json.load(open(g)), {json.loads(l)["id"]: json.loads(l) for l in open(gen)}
        ok = sum(1 for i in ge if gr.get(i, {}).get("ok"))
        tok = sum(x["tokens"] for x in ge.values())
        sec = sum(x["secs"] for x in ge.values())
        c["LCB-13 solved"] = f"{ok}/{len(ge)}"
        c["LCB-13 tokens per solved"] = f"{tok / max(ok, 1):,.0f}"
        c["LCB-13 min per solved"] = f"{sec / 60 / max(ok, 1):.1f}"
    cols[a] = c
keys = []
for c in cols.values():
    for k in c:
        if k not in keys:
            keys.append(k)
print("| | " + " | ".join(name.get(a, a) for a in arms) + " |")
print("|---|" + "---|" * len(arms))
for k in keys:
    print(f"| {k} | " + " | ".join(cols[a].get(k, "-") for a in arms) + " |")
