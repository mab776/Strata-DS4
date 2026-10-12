#!/bin/bash
# M9 (Marc 2026-10-11): SWE-bench Verified-mini, DeepSeek-V4-Flash IQ2_M on Strata-DS4 (stack 74, deployed M12 config)
# vs llama.cpp (stack 73), same harness / overlay (ds4.yaml) / temperature 1.0 / default effort.  PAIRED BLOCKS of
# 10 issues; block 1 runs on Strata, then the same 10 on stack 73 (the mode page's DeepSeek engine switch = start /
# stop only); later blocks Strata only (Marc 10-11: stack 73 at ~27 tok/s would need 10-12 h for all 50).  Resumable (preds.json).  Ends on Strata, then grades.
cd "$(dirname "$0")"
export HF_HOME=$PWD/.hf MSWEA_COST_TRACKING=ignore_errors HF_HUB_OFFLINE=1
NAME=m9-1011; LOG=night-$NAME.log; BLOCK=${BLOCK:-10}
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
engine() {   # $1 = strata | lcpp: make it the running DeepSeek engine and wait until it answers
  local want=$1 c; c=$([ $want = strata ] && echo strata-ds4 || echo ds4-lcpp)
  [ "$(curl -s -m 10 localhost:8500/api/status | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['mode'],d['ds_variant'])")" = "deepseek $want" ] && \
    [ "$(docker inspect -f '{{.State.Status}}' $c 2>/dev/null)" = running ] || {
      log "engine -> $want"; curl -s -m 600 -X POST localhost:8500/api/ds-variant/$want > /dev/null
      [ "$(curl -s -m 10 localhost:8500/api/status | python3 -c "import sys,json;print(json.load(sys.stdin)['mode'])")" = deepseek ] || \
        curl -s -m 600 -X POST localhost:8500/api/mode/deepseek > /dev/null; }
  for i in $(seq 1 90); do curl -s -m 5 localhost:8330/v1/models > /dev/null 2>&1 && [ "$(docker inspect -f '{{.State.Status}}' $c)" = running ] && break; sleep 10; done
  for i in $(seq 1 120); do   # 10-11 16:41: llama.cpp answers /v1/models while loading (503 on chat) - wait for a real reply
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 60 localhost:8330/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ] && break; sleep 15; done
  sleep 10; log "engine $want ready ($(docker inspect -f '{{.State.Status}}' $c))"
}
done_in() { [ -f "$1/preds.json" ] && python3 -c "import json,sys;sys.exit(0 if '$2' in json.load(open('$1/preds.json')) else 1)"; }
# m9_swe2 (10-11 16:20): restarted from m9_swe.sh to add the stack 73 LCB resample at the start of the 73 window (Marc)
# (Strata LCB resample ran in the first chain, 14:15-15:39)
mapfile -t ORDER < order.txt
for ((b = 0; b < ${#ORDER[@]}; b += BLOCK)); do
  engs=strata; [ $b -eq 0 ] && engs="strata lcpp"   # Marc 10-11: stack 73 only on the first paired block of 10
  for eng in $engs; do
    dir=runs/$NAME/$eng-t1.0; mkdir -p $dir
    todo=(); for iid in "${ORDER[@]:b:BLOCK}"; do done_in $dir $iid || todo+=($iid); done
    [ ${#todo[@]} -eq 0 ] && continue
    engine $eng
    [ $eng = lcpp ] && ! grep -q "73 resample done" /home/mab776/Documents/portainer/strata-ds4/bench/m9/m9_chain.log && \
      /home/mab776/Documents/portainer/strata-ds4/bench/m9/m9_resample73.sh >> /home/mab776/Documents/portainer/strata-ds4/bench/m9/m9_chain.log 2>&1
    for iid in "${todo[@]}"; do
      [ -e STOP-M9 ] && { log "STOP-M9 - ending"; break 3; }
      log "run $iid $eng"; s=$(date +%s)
      # 10-11 22:21: sphinx-9320 ballooned mini-extra to 3.1 GB (a huge command output held in memory) and the chain
      # unit's own 3G cap killed the whole chain -> each issue now runs in its own capped scope; a blow-up = that issue fails
      systemd-run --user --scope -q -p MemoryMax=4608M -p MemorySwapMax=0 timeout -s INT 3300 .venv/bin/mini-extra swebench --subset MariusHobbhahn/swe-bench-verified-mini --split test \
        --filter "^${iid}\$" -o $dir -w 1 -c swebench.yaml -c ds4.yaml -c model.model_kwargs.temperature=1.0 \
        >> runs/$NAME/mini-$eng.log 2>&1
      st=$(python3 -c "
import json
try: print(json.load(open('$dir/$iid/$iid.traj.json'))['info']['exit_status'])
except Exception as e: print('no-traj')" 2>/dev/null)
      [ $st = no-traj ] && journalctl --user --since "-$(( $(date +%s)-s+5 )) s" --no-pager | grep -q 'oom-kill' && st="no-traj (memory blow-up: OOM-killed at 4.5G)"
      log "  -> $st in $(( ($(date +%s)-s)/60 )) min"
      docker ps -q --filter name=minisweagent- | xargs -r docker rm -f >/dev/null 2>&1
    done
  done
done
engine strata
log "agent phase done - grading (1 worker)"
./swe_grade.sh $NAME >> $LOG 2>&1
log "all done"
