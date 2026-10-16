#!/usr/bin/env python3
"""mab776: decode + expert-tier split per DOMAIN (code / PowerShell / ops / French / French literary / poetry), then a
session that switches domain (PowerShell -> French -> PowerShell).  Greedy, fresh conversations.  Prints metrics only
(no prompt or reply text); replies go to OUT.<i>.txt when OUT is set (identity checks between arms).
  domain_test.py URL LABEL CONTAINER"""
import glob, json, os, re, subprocess, sys
sys.path.insert(0, "/home/mab776/Documents/portainer/strata-coder/bench")
from ple_bench import corpus  # noqa: E402
from strata_bench import call, last_request  # noqa: E402

url, label, ctr = sys.argv[1], sys.argv[2], sys.argv[3]
out = os.environ.get("OUT")
N = int(os.environ.get("DOMAIN_CHARS", "24000"))


def gutenberg(path):
    t = open(path, encoding="utf-8", errors="replace").read()
    a, b = t.find("*** START OF"), t.find("*** END OF")
    if a >= 0:
        t = t[t.find("\n", a) + 1:]
    if b >= 0:
        t = t[:t.find("*** END OF")]
    return t


def files(pattern_list, exclude=()):
    parts = []
    for pat in pattern_list:
        for f in sorted(glob.glob(pat, recursive=True)):
            if any(x in f for x in exclude) or os.path.getsize(f) > 200000:
                continue
            try:
                parts.append(open(f, encoding="utf-8", errors="replace").read())
            except OSError:
                pass
    return "\n\n".join(parts)


C = "/home/mab776/.claude/jobs/6a7c0663/tmp/corpora"
P = "/home/mab776/Documents/portainer"
src = {
    "code": corpus()[5000000:],
    "powershell": files(["/home/mab776/Documents/golf/**/*.ps1"], exclude=(".venv",)),
    "ops": files([P + "/*/docker-compose*.yml", P + "/*/*.sh", P + "/caddy-config/Caddyfile"],
                 exclude=("strata-ds4", "project-maya", "strata-ds41", ".bak")),
    "french": files([P + "/menu-planner/*.md", P + "/menu-planner/data/*.md"]),
    "fr-literary": gutenberg(C + "/pg14155.txt"),
    "poetry": gutenberg(C + "/pg6099.txt") + "\n\n" + gutenberg(C + "/pg1041.txt"),
}
ask = {
    "code": "Continue this source code: write the next functions of the file in the same style.",
    "powershell": "Write a new PowerShell script in the same style as the scripts above: it checks a list of remote "
                  "machines, reports each one's service status in a table, and restarts the services that stopped.",
    "ops": "Write a new docker-compose.yml in the same style for a small web service with a healthcheck, a named "
           "volume and the shared network, then the bash deploy script that creates it through the Portainer API.",
    "french": "Réponds en français : en suivant ces règles, propose le menu de la semaine prochaine (soupers du lundi "
              "au dimanche), avec la liste d'épicerie et ce qu'il faut sortir du congélateur.",
    "fr-literary": "Continue ce texte dans le même style, en français, sur plusieurs paragraphes.",
    "poetry": "Write new poems in the same style as the ones above.",
}


def tier_line():
    """The engine log's last decode line and DSpark line for this request (numbers only)."""
    log = subprocess.run(["docker", "exec", ctr, "sh", "-c", "tail -n 60 /tmp/strata-ds41.log"],
                         capture_output=True, text=True).stdout
    dec = [l for l in log.splitlines() if re.match(r"^serve: \d+ tokens in", l)]
    dsp = [l for l in log.splitlines() if l.startswith("serve: DSpark ")]
    r = {}
    if dec:
        m = re.search(r"experts: (\d+) hit, (\d+) cpu, (\d+) pcie, (\d+) file(?:, (\d+) gpu1)?", dec[-1])
        if m:
            h, c, p, f, g = (int(x or 0) for x in m.groups())
            tot = max(1, h + c + p + f + g)
            r.update(hit=round(100 * h / tot, 1), gpu1=round(100 * g / tot, 1), pcie=round(100 * p / tot, 1),
                     file=round(100 * f / tot, 2), fast=round(100 * (h + g) / tot, 1))
        m = re.search(r"(\d+) fills", dec[-1])
        if m:
            r["fills"] = int(m.group(1))
    if dsp:
        m = re.search(r"([\d.]+) tokens/pass \(all", dsp[-1])
        if m:
            r["tok_pass"] = float(m.group(1))
    return r


i = 0


def ask_one(dom, msgs):
    global i
    r = call(url, "/v1/chat/completions", {"messages": msgs, "max_tokens": 384, "temperature": 0, "stream": False})
    m = last_request(call(url, "/metrics"))
    msg = r["choices"][0]["message"]
    text = (msg.get("reasoning_content") or "") + "\x00" + (msg.get("content") or "")
    if out:
        open(f"{out}.{i}.txt", "w").write(text)
    row = {"label": label, "i": i, "domain": dom, "prompt_tok": m["prompt_tokens"], "reused": m.get("reused"),
           "gen": m.get("output_tokens"), "decode_tps": m.get("decode_tok_s")}
    row.update(tier_line())
    print(json.dumps(row), flush=True)
    i += 1
    return msg.get("content") or "ok"


for dom in ["code", "powershell", "ops", "french", "fr-literary", "poetry"]:
    t = src[dom]
    ask_one(dom, [{"role": "user", "content": t[:N] + "\n\n" + ask[dom]}])
# one conversation that changes domain twice
msgs = [{"role": "user", "content": src["powershell"][N:2 * N] + "\n\n" + ask["powershell"]}]
msgs.append({"role": "assistant", "content": ask_one("switch:powershell", msgs)})
msgs.append({"role": "user", "content": src["french"][N // 2:N // 2 + N // 2] + "\n\n" + ask["french"]})
msgs.append({"role": "assistant", "content": ask_one("switch:french", msgs)})
msgs.append({"role": "user", "content": src["powershell"][2 * N:2 * N + N // 2] + "\n\n" + ask["powershell"]})
ask_one("switch:powershell2", msgs)
