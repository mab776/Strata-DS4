#!/bin/bash
# M9 LCB noise check, stack 73 side (Marc 10-11, 2nd ask): the same 6 disagreeing problems sampled 2 more times on
# llama.cpp stack 73 (same settings: 32k, T 1.0, top_p 1.0, no top_k) into gen-ds4-iq2m-32k-t1-resample-{a,b}, graded,
# and lcb-resample-table.txt rebuilt with a "stack 73 k/3" column.  Runs inside the stack 73 window of m9_swe2.sh
# (after Strata's SWE block 1, before 73's SWE issues).  Official pass@1 stays the first sample on both engines.
B=/home/mab776/Documents/portainer/strata-coder/bench; D=$B/ab-gsq-20260926; N=ds4s-m12-32k-t1; R=ds4-iq2m-32k-t1
IDS=abc391_d,abc394_c,abc394_g,abc400_d,abc400_g,arc192_e
cd /home/mab776/Documents/portainer/strata-ds4/bench/m9
[ "$(docker inspect -f '{{.State.Status}}' ds4-lcpp 2>/dev/null)" = running ] || { echo "== $(date '+%F %T') 73 resample: ds4-lcpp not running - skipped"; exit 1; }
# 16:41 first try: llama.cpp answers /v1/models while still loading and 503s every chat request -> wait for a real reply
for i in $(seq 1 120); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' -m 60 localhost:8330/v1/chat/completions -H 'Content-Type: application/json' \
      -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ] && break; sleep 15; done
echo "== $(date '+%F %T') 73 resample ids: $IDS"
for s in a b; do
  for try in 1 2; do LCB_IDS=$IDS LCB_DIR=$D LCB_MAXT=32768 LCB_TEMP=1.0 LCB_TOPP=1.0 LCB_TOPK=0 \
      python3 $B/lcb_eval.py gen http://127.0.0.1:8330 $R-resample-$s >> lcb-gen-$R-resample.log 2>&1; done
  [ "$(wc -l < $D/gen-$R-resample-$s.jsonl 2>/dev/null || echo 0)" -ge 6 ] || { echo "== $(date '+%F %T') 73 resample $s incomplete - stopping"; exit 1; }
  LCB_DIR=$D python3 $B/lcb_eval.py grade $R-resample-$s > lcb-grade-$R-resample-$s.log 2>&1
done
python3 - "$D" "$N" "$R" "$IDS" <<'PY' > lcb-resample-table.tmp && mv lcb-resample-table.tmp lcb-resample-table.txt; cat lcb-resample-table.txt
import json, sys
D, N, R, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
g = lambda n: json.load(open(f"{D}/grade-{n}.json"))
a, r = g(N), g(R)
sa, sb, ra, rb = g(N + "-resample-a"), g(N + "-resample-b"), g(R + "-resample-a"), g(R + "-resample-b")
f = lambda x: "pass" if x else "fail"
ok = lambda d, i: bool(d.get(i, {}).get("ok"))
print("| problem | diff | Strata 1st | Strata a | Strata b | Strata k/3 | 73 1st | 73 a | 73 b | stack 73 k/3 |")
print("|---|---|---|---|---|---|---|---|---|---|")
for i in ids:
    s = [ok(a, i), ok(sa, i), ok(sb, i)]; t = [ok(r, i), ok(ra, i), ok(rb, i)]
    print(f"| {i} | {a[i]['diff']} | " + " | ".join(map(f, s)) + f" | {sum(s)}/3 | " + " | ".join(map(f, t)) + f" | {sum(t)}/3 |")
print(f"\npass@1 (first sample, official): Strata {sum(v['ok'] for v in a.values())}/{len(a)}, stack 73 {sum(v['ok'] for v in r.values())}/{len(r)}")
PY
[ -s lcb-resample-table.txt ] && echo "== $(date '+%F %T') 73 resample done"
