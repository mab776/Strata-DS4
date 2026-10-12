#!/bin/bash
# M13 probe: (1) the new engine, every M13 switch at its default (on) + DS4_CHECK_ROUTE=1 + DS4_DBG_GRAPH=1: the card's
# route groups vs the host's, the graphs' node counts; (2) every M13 decode switch off + DS4_DBG_GRAPH=1: the old counts.
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m13/lib.sh
OFF='{"DS4_HC_MIX":"0","DS4_HC_PRENORM":"0","DS4_LEAN":"0","DS4_SH_FOLD":"0","DS4_ROUTE_PAR":"0","DS4_PF_PINNED":"0"}'
for arm in on off; do
  [ -e "$STOP" ] && break
  if [ $arm = on ]; then arm_env '{"DS4_CHECK_ROUTE":"1","DS4_DBG_GRAPH":"1"}'; else arm_env "$(python3 -c "import json;d=json.loads('$OFF');d['DS4_DBG_GRAPH']='1';print(json.dumps(d))")"; fi
  L0=$(wc -l < "$LOG")
  up || exit 1
  echo "### probe $arm $(date +%T) $(docker exec strata-ds4 cat /app/VERSION | cut -c1-40) env: $(envline)"
  NONCE=m13-probe ksweep 8192 2:0,2:7,1:0 256
  tail -n +$((L0 + 1)) "$LOG" | grep -a '\[route-chk\]' | tail -3
  echo "  attention graph nodes per (layer, pass size) variant - total over the variants seen:"
  tail -n +$((L0 + 1)) "$LOG" | grep -a '\[graph\] attn layer' | awk '{n[$7]++; s+=$7} END {printf "  %d variants, %d nodes in all, mean %.1f\n", NR, s, s/NR}'
  tail -n +$((L0 + 1)) "$LOG" | grep -a '\[graph\] attn layer 3 \|\[graph\] attn layer 4 ' | cut -c1-60 | head -8
done
restore
