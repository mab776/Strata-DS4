#!/bin/bash
D=/home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m14
echo "#### kl $(date +%T)"; $D/kl14.sh 130048
for v in DS4_FUSE_PREDICT GGML_CUDA_RANK_ARGSORT; do
  [ -e $D/STOP ] && break
  echo "#### ab $v $(date +%T)"; $D/ab14.sh $v 8192
done
[ -e $D/STOP ] || { echo "#### pf-b sweep $(date +%T)"; $D/pfb.sh; }
echo "#### chain done $(date +%T)"
