#!/bin/bash
D=/home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m13
echo "#### kl $(date +%T)"; $D/kl.sh 130048
for v in DS4_HC_MIX DS4_HC_PRENORM DS4_LEAN DS4_SH_FOLD DS4_ROUTE_PAR; do
  [ -e $D/STOP ] && break
  echo "#### ab $v $(date +%T)"; $D/ab.sh $v 8192
done
[ -e $D/STOP ] || { echo "#### final $(date +%T)"; $D/final.sh; }
echo "#### chain done $(date +%T)"
