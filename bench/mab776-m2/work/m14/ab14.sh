#!/bin/bash
# M14 server A/B of ONE env switch: VAR=0 (old path) vs 1, ABBA, same prompt per arm, greedy K=1 then K=2 twice.
#   ab14.sh VAR [DEPTHS=8192]      Ends on the canonical config.
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m14/lib14.sh
VAR=$1; DEPTHS=${2:-8192}
for v in 0 1 1 0; do
  [ -e "$STOP" ] && { echo "### STOP file - ending early"; break; }
  arm "{\"$VAR\":\"$v\"}"; up || exit 1
  echo "### $VAR=$v $(date +%T) env: $(envline)"
  NONCE=m14-ab ksweep "$DEPTHS" 1:0,2:0,2:0 512
done
restore
