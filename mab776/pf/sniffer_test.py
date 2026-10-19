#!/usr/bin/env python3
"""mab776: Marc's real traffic (llm-sniffer captures) through a local test engine - BLIND: this script reads the bodies,
Claude does not.  Prints metrics only: record index, body length, prompt tokens, decode tok/s, the expert-tier split
and DSpark tokens/pass; a bucket label (language / content type, classified here) only when >= 5 picked requests
share it, else "real-traffic".  No text, excerpts, keywords or matched strings; errors print their type only.
  sniffer_test.py URL LABEL CONTAINER [N_SESSIONS]"""
import collections, glob, gzip, hashlib, json, re, subprocess, sys, urllib.request
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
from strata_bench import last_request  # noqa: E402

url, label, ctr = sys.argv[1], sys.argv[2], sys.argv[3]
n_sessions = int(sys.argv[4]) if len(sys.argv) > 4 else 6
MAX_CHARS = 260000

FR = set("le la les un une des est et pour avec que qui dans sur pas je tu nous vous il elle ce cette mais ou donc car "
         "faire fait peux veux merci bonjour quoi comment pourquoi".split())
EN = set("the a an is and for with that which in on not i you we he she this but or so because do does can want please "
         "hello what how why".split())


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def classify(body):
    msgs = body.get("messages") or []
    users = " ".join(text_of(m.get("content")) for m in msgs if m.get("role") == "user")
    words = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", users.lower()[:200000])
    fr = sum(w in FR for w in words)
    en = sum(w in EN for w in words)
    lang = "fr" if fr > en else "en"
    allt = " ".join(text_of(m.get("content")) for m in msgs)[:400000]
    score = {
        "powershell": len(re.findall(r"\$[A-Za-z_]\w*\s*=|\b(Get|Set|New|Invoke|Write)-[A-Z]\w+|\.ps1\b", allt)),
        "yaml": len(re.findall(r"(?m)^\s*[a-z_]+:\s*$|^\s*-\s+[a-z_]+:", allt)),
        "bash": len(re.findall(r"(?m)^\s*(sudo|docker|cd|ls|grep|echo|export)\s|\$\(|\bfi\b|\bdone\b", allt)),
        "python": len(re.findall(r"(?m)^\s*(def|import|from|class)\s|self\.", allt)),
        "c-like": len(re.findall(r"#include|std::|->|;\s*$", allt)),
    }
    kind = max(score, key=score.get) if max(score.values()) > 20 else "prose"
    return lang + "/" + kind


recs = []
for f in sorted(glob.glob("/home/mab776/Documents/portainer/llm-sniffer/data/*.jsonl.gz")):
    with gzip.open(f, "rt", errors="replace") as h:
        for line in h:
            try:
                d = json.loads(line)
            except Exception:
                continue
            b = d.get("body") or {}
            if d.get("path") != "/v1/chat/completions" or not b.get("messages"):
                continue
            first = next((text_of(m.get("content")) for m in b["messages"] if m.get("role") == "user"), "")
            key = hashlib.sha1(first[:4000].encode("utf-8", "replace")).hexdigest()
            recs.append((key, len(json.dumps(b)), b))
sess = collections.defaultdict(list)
for i, (key, n, b) in enumerate(recs):
    sess[key].append((i, n, b))
order = sorted(sess.values(), key=len, reverse=True)
picked = []
for reqs in order:
    ok = [r for r in reqs if r[1] <= MAX_CHARS]
    if not ok:
        continue
    picked.append(ok[int(len(ok) * 0.6)])
    if len(picked) >= n_sessions:
        break
labels = [classify(b) for _, _, b in picked]
cnt = collections.Counter(labels)
print(json.dumps({"label": label, "records": len(recs), "sessions": len(sess), "picked": len(picked),
                  "buckets": {k: v for k, v in cnt.items() if v >= 5} or {"real-traffic": len(picked)}}), flush=True)


def call(path, body=None):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)


def tier():
    log = subprocess.run(["docker", "exec", ctr, "sh", "-c", "tail -n 60 /tmp/strata-ds41.log"],
                         capture_output=True, text=True).stdout
    r = {}
    dec = [l for l in log.splitlines() if re.match(r"^serve: \d+ tokens in", l)]
    if dec:
        m = re.search(r"experts: (\d+) hit, (\d+) cpu, (\d+) pcie, (\d+) file(?:, (\d+) gpu1)?", dec[-1])
        if m:
            h, c, p, f, g = (int(x or 0) for x in m.groups())
            t = max(1, h + c + p + f + g)
            r.update(hit=round(100 * h / t, 1), gpu1=round(100 * g / t, 1), pcie=round(100 * p / t, 1),
                     file=round(100 * f / t, 2), fast=round(100 * (h + g) / t, 1))
    dsp = [l for l in log.splitlines() if l.startswith("serve: DSpark ")]
    if dsp:
        m = re.search(r"([\d.]+) tokens/pass \(all", dsp[-1])
        if m:
            r["tok_pass"] = float(m.group(1))
    return r


for (i, n, b), lab in zip(picked, labels):
    row = {"label": label, "rec": i, "body_chars": n, "bucket": lab if cnt[lab] >= 5 else "real-traffic"}
    req = {"messages": b["messages"], "max_tokens": 384, "temperature": 0, "stream": False}
    if b.get("tools"):
        req["tools"] = b["tools"]
    try:
        call("/v1/chat/completions", req)
        last = last_request(call("/metrics"))
        row.update(prompt_tok=last.get("prompt_tokens"), gen=last.get("output_tokens"), decode_tps=last.get("decode_tok_s"))
        row.update(tier())
    except Exception as e:
        row["error"] = type(e).__name__
    print(json.dumps(row), flush=True)
