#!/usr/bin/env bash
# mab776 overnight: the FINAL check of the served config file itself (mab776/strata-ds41.json, no -e overrides; the
# engine = build-ds4-gpu, the wrapper default): 8k + 127k (+ turns), the engine's env, tools probe, needles at 246k.
set -u
W=/home/mab776/Documents/portainer/strata-ds41; V=/home/mab776/Documents/portainer/ram-128gb/v41; O=$V/pf
log() { echo "[$(date +%T)] $*" | tee -a $O/sweep.log; }
cd $W && KEEP=1 DUMP=1 GEN=256 TURNS=1 DEPTHS=8000:3000000,127000:15000000 CFG=mab776/strata-ds41.json bash mab776/pf/run.sh ${FIN:-fin-final} 2>&1 | tail -8 | tee -a $O/sweep.log
pid=$(docker exec ds41-pf sh -c 'pgrep -x ds41_generate | head -1')
docker exec ds41-pf sh -c "tr '\0' '\n' < /proc/$pid/environ | grep -E '^(GGML_|STRATA_)' | sort" > $O/${FIN:-fin-final}.env 2>&1
log "engine env: $(wc -l < $O/${FIN:-fin-final}.env) STRATA_/GGML_ vars ($(grep -c -E 'MIRROR|PSTAGE|ROLLBACK|DSPARK_DEPTH=5' $O/${FIN:-fin-final}.env) of the key ones)"
python3 $V/tools_probe.py http://127.0.0.1:8332 > $O/${FIN:-fin}-tools.log 2>&1
log "tools: $(tail -1 $O/${FIN:-fin}-tools.log)"
NEEDLE_ROOT="/media/mab776/nvme 2To/models/strata-data/src-01431" timeout 2400 python3 /home/mab776/Documents/portainer/deepseek-v4/needle_ds4.py \
   --url http://127.0.0.1:8332 --lengths 246k --depths 10,50,90 --timeout 1800 --out $O/${FIN:-fin}-needles.json > $O/${FIN:-fin}-needles.log 2>&1
log "needles: $(tail -1 $O/${FIN:-fin}-needles.log) | $(grep -c FOUND $O/${FIN:-fin}-needles.log) found"
docker exec ds41-pf sh -c 'grep -E "^serve: prompt [0-9]+ tokens" /tmp/strata-ds41.log' | tail -4 >> $O/${FIN:-fin-final}.log
docker stop -t 60 ds41-pf >/dev/null; docker rm ds41-pf >/dev/null 2>&1
log "fin done"
