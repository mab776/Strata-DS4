#!/usr/bin/env bash
# mab776 overnight: quick unit gates (64-row chunks: the F32 reference is the slow part), baseline profile, then screens 1 + 2
set -u
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
log() { echo "[$(date +%T)] $*" | tee -a $O/q1.log; }
unit() {   # unit <tag> <build> [ENV=V ...]
  local tag=$1 b=$2; shift 2; local e=(); for x in "$@"; do e+=(-e "$x"); done
  log "unit $tag ($b): $*"
  timeout 900 docker run --rm --name ds41-unit --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0 "${e[@]}" --memory 40g --memory-swap 40g --ulimit memlock=-1:-1 \
    -v "/media/mab776/nvme 2To/models/deepseek-v41/gsq:/m:ro" -v $W:$W:ro -w $W strata-ds4:cu13 \
    $W/$b/test_ds41_moe_gpu /m/V41-GSQ.gguf --gpu --layers 1,2,3 --tokens 2 --chunk 64 > $O/unit-$tag.log 2>&1
  log "  rc=$? FAIL lines: $(grep -c FAIL $O/unit-$tag.log); last: $(tail -1 $O/unit-$tag.log | cut -c1-150)"
  docker rm -f ds41-unit >/dev/null 2>&1
}
unit old build-n3 STRATA_PSTAGE_STRIDED=0
unit strided build-n3 STRATA_PSTAGE_STRIDED=1
unit strided-lazy build-n3 STRATA_PSTAGE_STRIDED=1 STRATA_LAZY_XQ=1
unit pc2x build-n3 STRATA_PSTAGE_STRIDED=1 TEST_CHUNK_TWICE=1 STRATA_PROMPT_CACHE_GIB=4 STRATA_PROMPT_CACHE_MIN=32 STRATA_PROMPT_CACHE_KEEP_GIB=2
cd $W && CFG=mab776/strata-ds41.json bash mab776/pf/prof.sh p0-base 32000 25 12 -e DS41_BUILD=build-n1 -e MIMO_CHUNK_PROF=1 2>&1 | tee -a $O/q1.log
log "q2 profile done; screen 1"
bash $W/mab776/pf/sweep.sh $W/mab776/pf/s1.txt
bash $W/mab776/pf/sweep.sh $W/mab776/pf/s2.txt
log "q2 done"
