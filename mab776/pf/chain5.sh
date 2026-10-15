#!/usr/bin/env bash
# after chain4 (the 127k confirmation): a profile of the current best (32k prompt, window mid-prefill)
while ps -eo cmd | grep -q "^bash chain4.sh"; do sleep 15; done
cd /home/mab776/Documents/portainer/strata-ds41 && CFG=mab776/pf/pb32k.json bash mab776/pf/prof.sh p1-best 32000 12 10 -e DS41_BUILD=build-n3 -e GGML_TOPK_RADIX=1 -e STRATA_PSTAGE_STRIDED=1 -e STRATA_FN_DEV=1 -e STRATA_SPAN_ASYNC=1 -e STRATA_KALL16=1 -e STRATA_LAZY_XQ=1 -e STRATA_PROMPT_CACHE_GIB=24 -e STRATA_CHUNK_BUDGET_GB=6 -e STRATA_BLOB_MIRROR=/home/mab776/Documents/portainer/strata-ds41/mab776/mirror/V41-GSQ.experts.mirror -e STRATA_BLOB_MIRROR_SHARE=0.5 >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log 2>&1
echo "[$(date +%T)] chain5 done" >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log
