#!/bin/bash
# M9 LCB noise check (Marc 10-11): every problem where Strata's first sample and stack 73's run disagree, sampled 2 more
# times on Strata (32k, T 1.0, top_p 1.0) into gen-ds4s-m12-32k-t1-resample-{a,b}; graded; table of k/3 per problem.
B=/home/mab776/Documents/portainer/strata-coder/bench; D=$B/ab-gsq-20260926; N=ds4s-m12-32k-t1; R=ds4-iq2m-32k-t1
cd /home/mab776/Documents/portainer/strata-ds4/bench/m9
IDS=$(python3 -c "
import json; a=json.load(open('$D/grade-$N.json')); b=json.load(open('$D/grade-$R.json'))
print(','.join(sorted(k for k in a if k in b and a[k]['ok'] != b[k]['ok'])))")
echo "== $(date '+%F %T') resample ids: $IDS"
[ -z "$IDS" ] && exit 0
for s in a b; do
  for try in 1 2; do LCB_IDS=$IDS LCB_DIR=$D LCB_MAXT=32768 LCB_TEMP=1.0 LCB_TOPP=1.0 LCB_TOPK=0 \
      python3 $B/lcb_eval.py gen http://127.0.0.1:8330 $N-resample-$s >> lcb-gen-$N-resample.log 2>&1; done
  LCB_DIR=$D python3 $B/lcb_eval.py grade $N-resample-$s > lcb-grade-$N-resample-$s.log 2>&1
done
python3 - "$D" "$N" "$R" "$IDS" <<'PY' | tee lcb-resample-table.txt
import json, sys
D, N, R, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
g = lambda n: json.load(open(f"{D}/grade-{n}.json"))
a, r, s1, s2 = g(N), g(R), g(N + "-resample-a"), g(N + "-resample-b")
print("| problem | diff | stack 73 | Strata 1st | resample a | resample b | Strata k/3 |")
print("|---|---|---|---|---|---|---|")
for i in ids:
    v = [a[i]["ok"], s1.get(i, {}).get("ok"), s2.get(i, {}).get("ok")]
    f = lambda x: "pass" if x else "fail" + (" (cap)" if False else "")
    print(f"| {i} | {a[i]['diff']} | {f(r[i]['ok'])} | {f(v[0])} | {f(v[1])} | {f(v[2])} | {sum(bool(x) for x in v)}/3 |")
tot = sum(v['ok'] for v in a.values()); print(f"\nStrata pass@1 (first sample): {tot}/{len(a)}; stack 73: {sum(v['ok'] for v in r.values())}/{len(r)}")
PY
echo "== $(date '+%F %T') resample done"
