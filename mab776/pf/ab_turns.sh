#!/usr/bin/env bash
# mab776: agent-turn A/B - one throwaway container per config (mab776/pf/tmp-cfg/<cfg>.json), the turn bench (9.5k fresh,
# then +300 / +1,500 user tokens) REPS times with fresh nonces, results -> ram-128gb/v41/pf/<label>.jsonl
#   ab_turns.sh LABEL CFG [REPS=2]
set -u
L=$1; C=$2; R=${3:-2}
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; U=http://127.0.0.1:8332
rm -f /home/mab776/v41-engram/park-test/* 2>/dev/null
(cd $W && CFG=mab776/pf/tmp-cfg/$C.json GPUS=0,1 DEPTHS=" " KEEP=1 bash mab776/pf/run.sh $L \
   -v /home/mab776/v41-engram/park-test:/home/mab776/v41-engram/park-test) > /dev/null || { echo "$L: run.sh failed"; exit 1; }
for r in $(seq 1 $R); do
  timeout 1200 python3 $W/mab776/pf/turn_bench.py $U --nonce t$L-$r-$(date +%s) --label $L 2>&1 | tail -3 >> $O/$L.jsonl
done
docker exec ds41-pf sh -c 'grep -E "wave layer|prompt phases|error|ERR" /tmp/strata-ds41.log' | cut -c1-200 > $O/$L.phases.log
docker rm -f ds41-pf >/dev/null
python3 - $O/$L.jsonl <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
for t in (0, 1, 2):
    xs = [r for r in rows if r.get('turn') == t]
    print(f"turn {t} (read {xs[0]['read'] if xs else '-'}): prefill " + " / ".join(f"{r['prefill_s']:.2f}" for r in xs) + " s, wall " + " / ".join(f"{r['wall_s']:.1f}" for r in xs) + " s")
PY
grep -E "wave layer" $O/$L.phases.log | tail -1
