#!/bin/bash
# M14: upstream's --pf-b sweep on our box - the prefetch budget 0.7 (ours) / 1.43 (upstream's default) / 0 (off),
# mirrored order, same prompts, greedy K=1 + K=2 x2, 8k and 127k.   Ends on the canonical config.
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m14/lib14.sh
for b in 0.7 1.43 0 0 1.43 0.7; do
  [ -e "$STOP" ] && { echo "### STOP"; break; }
  arm '{}' $b; up || exit 1
  echo "### pf-b $b $(date +%T) env: $(envline)"
  NONCE=m14-pfb ksweep 8192,130048 1:0,2:0,2:0 512
done
restore
