#!/usr/bin/env bash
# after q2 (screens 1 + 2): profile of the current best, then the 127k confirmation (s5), then the ppl gate (p1)
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; PF=/home/mab776/Documents/portainer/strata-ds41/mab776/pf
while ps -eo cmd | grep -q "^bash q2.sh"; do sleep 15; done
cd /home/mab776/Documents/portainer/strata-ds41 && CFG=mab776/pf/pb32k.json bash $PF/prof.sh p1-best 32000 12 10 -e DS41_BUILD=build-n3 -e GGML_TOPK_RADIX=1 -e STRATA_PSTAGE_STRIDED=1 -e STRATA_FN_DEV=1 -e STRATA_SPAN_ASYNC=1 -e STRATA_KALL16=1 -e STRATA_LAZY_XQ=1 -e STRATA_PROMPT_CACHE_GIB=24 -e STRATA_CHUNK_BUDGET_GB=6 -e STRATA_BLOB_MIRROR=/home/mab776/Documents/portainer/strata-ds41/mab776/mirror/V41-GSQ.experts.mirror -e STRATA_BLOB_MIRROR_SHARE=0.5 > $O/prof-p1.out 2>&1
echo "[$(date +%T)] profile p1-best done" >> $O/sweep.log
bash $PF/sweep.sh $PF/s5.txt
bash $PF/ppl.sh $PF/p1.txt
echo "[$(date +%T)] chain6 done" >> $O/sweep.log
