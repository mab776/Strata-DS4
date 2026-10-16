#!/usr/bin/env bash
# mab776: LCB-13 (ram-128gb/v41/quality.sh's 13 ids, 32k cap, T 1.0 / top_p 1.0, reasoning effort medium = 62 - the
# served default since 2026-10-15) on one route-bias arm's throwaway container (mab776/pf/tmp-cfg/rb-<arm>.json)
# -> strata-coder/bench/ab-gsq-20260926/{gen,grade}-v41-rb<arm>-32k-t1-med.*
#   rb_lcb.sh ARM [ARM ...]     (stops if the mode page leaves OFF)
set -u
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
B=/home/mab776/Documents/portainer/strata-coder/bench; D=$B/ab-gsq-20260926
IDS=abc388_d,abc389_d,abc394_d,abc396_d,abc399_c,abc400_c,abc391_g,abc396_f,abc394_g,abc400_g,arc192_e,abc391_d,abc400_d
mode() { curl -s -m 5 localhost:8500/api/status | python3 -c "import sys,json;print(json.load(sys.stdin).get('mode'))" 2>/dev/null; }
for A in "$@"; do
  L=rb-$A; N=v41-rb$A-32k-t1-med
  [ "$(mode)" = off ] || { echo "[$(date +%T)] mode is $(mode) - stopping before $N"; exit 3; }
  echo "[$(date +%T)] $N start"
  (cd $W && CFG=mab776/pf/tmp-cfg/$L.json GPUS=0,1 DEPTHS=" " TURNS=0 KEEP=1 bash mab776/pf/run.sh $L-lcb) || { echo "$N: start failed"; docker rm -f ds41-pf >/dev/null 2>&1; continue; }
  LCB_IDS=$IDS LCB_DIR=$D LCB_MAXT=32768 LCB_TEMP=1.0 LCB_TOPP=1.0 LCB_TOPK=0 LCB_EFFORT=medium \
    python3 $B/lcb_eval.py gen http://127.0.0.1:8332 $N 2>&1 | tee -a $O/$L.lcb.log
  docker exec ds41-pf cat /tmp/strata-ds41.log > $O/$L-lcb.engine.log 2>/dev/null
  docker rm -f ds41-pf >/dev/null 2>&1
  LCB_DIR=$D python3 $B/lcb_eval.py grade $N 2>&1 | tail -3 | tee -a $O/$L.lcb.log
  echo "[$(date +%T)] $N done"
done
