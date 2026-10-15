#!/usr/bin/env bash
# mab776 overnight, first GPU window (after QUALITY_DONE): tools re-test on ds41-test, then the unit gates and the
# baseline profile in throwaway containers.  Log: ram-128gb/v41/pf/q1.log
set -u
W=/home/mab776/Documents/portainer/strata-ds41; V=/home/mab776/Documents/portainer/ram-128gb/v41; O=$V/pf; mkdir -p $O
B=/home/mab776/Documents/portainer/strata-coder/bench
log() { echo "[$(date +%T)] $*" | tee -a $O/q1.log; }
until grep -q QUALITY_DONE $V/quality.log; do sleep 30; done
log "QUALITY_DONE seen"
# 1. tools probe on the restarted ds41-test (serve_ds41.py FUNC_START fix; engine binary = build-ds4-gpu, every new path off)
docker restart ds41-test >/dev/null
t0=$(date +%s)
until curl -sf -m 5 http://127.0.0.1:8331/v1/models >/dev/null && python3 $B/strata_bench.py 200 --offset 900000 --url http://127.0.0.1:8331 --nonce tw$t0 --gen 8 >/dev/null 2>&1; do
  sleep 10; [ $(( $(date +%s) - t0 )) -gt 600 ] && { log "ds41-test not ready"; break; }
done
log "ds41-test ready after $(( $(date +%s) - t0 )) s"
python3 $V/tools_probe.py http://127.0.0.1:8331 > $V/tools-after-fix.log 2>&1
log "tools: $(tail -1 $V/tools-after-fix.log)"
docker stop -t 60 ds41-test >/dev/null
log "ds41-test stopped"
# 2. unit gates (GPU 0): the tier's chunk path vs the F32 reference, old / strided prestage; the prompt cache twice-gate
unit() {   # unit <tag> [ENV=V ...]
  local tag=$1; shift; local e=(); for x in "$@"; do e+=(-e "$x"); done
  log "unit $tag: $*"
  docker run --rm --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 "${e[@]}" --memory 40g --memory-swap 40g --ulimit memlock=-1:-1 \
    -v "/media/mab776/nvme 2To/models/deepseek-v41/gsq:/m:ro" -v $W:$W:ro -w $W strata-ds4:cu13 \
    $W/build-n1/test_ds41_moe_gpu /m/V41-GSQ.gguf --gpu --layers 0,1,2,3 --tokens 4 --chunk 1024 > $O/unit-$tag.log 2>&1
  log "  rc=$? FAIL lines: $(grep -c FAIL $O/unit-$tag.log); last: $(tail -1 $O/unit-$tag.log)"
}
unit old STRATA_PSTAGE_STRIDED=0
unit strided STRATA_PSTAGE_STRIDED=1
unit pc2x STRATA_PSTAGE_STRIDED=1 TEST_CHUNK_TWICE=1 STRATA_PROMPT_CACHE_GIB=4 STRATA_PROMPT_CACHE_MIN=256 STRATA_PROMPT_CACHE_KEEP_GIB=2
# 3. baseline profile (32k prompt, nsys window mid-prefill), every new path off
cd $W && CFG=mab776/strata-ds41.json bash mab776/pf/prof.sh p0-base 32000 25 12 -e DS41_BUILD=build-n1 -e MIMO_CHUNK_PROF=1 2>&1 | tee -a $O/q1.log
log "q1 done"
