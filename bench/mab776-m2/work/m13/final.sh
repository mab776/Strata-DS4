#!/bin/bash
# M13 table: every M13 switch off (the old paths: BEFORE) vs the new engine's defaults (AFTER), ABBA, same prompts,
# greedy K=1 + K=2 x2 from the same restored state, 8k / 127k / 250k; prefill = each arm's read of each depth.
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m13/lib.sh
OFF='{"DS4_HC_MIX":"0","DS4_HC_PRENORM":"0","DS4_LEAN":"0","DS4_SH_FOLD":"0","DS4_ROUTE_PAR":"0","DS4_PF_PINNED":"0"}'
for arm in before after after before; do
  [ -e "$STOP" ] && { echo "### STOP"; break; }
  if [ $arm = before ]; then arm_env "$OFF"; else arm_env '{"DS4_SH_FOLD":"0"}'; fi   # after = the shipped set (SH_FOLD rejected)
  up || exit 1
  echo "### $arm $(date +%T) env: $(envline)"
  NONCE=m13-final ksweep 8192,130048,250000 1:0,2:0,2:0 512
done
restore
