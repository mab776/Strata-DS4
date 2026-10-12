#!/bin/bash
# M13 numerics gate for DS4_HC_MIX (not bit-identical): teacher-forced rows at 127k, reference = HC_MIX=0; then
# HC_MIX=0 again (the restart noise floor), HC_MIX=1, and the whole M13 set off (the old engine's paths) for scale.
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m13/lib.sh
D=${1:-130048}
DUMP='"DS4_DUMP_TOKLOGITS":"/data/logs/tl"'
FORCE='"DS4_SERVE_TEST_ORACLE":"/data/logs/oracle.txt","DS4_SERVE_TEST_FORCE":"1"'
run_py() { systemd-run --user --scope -q -p MemoryMax=3G python3 $W/m13/forced_x.py $U $D "$@"; }
arm_env "{$DUMP,\"DS4_HC_MIX\":\"0\"}"; up || exit 1; echo "### ref HC_MIX=0 $(date +%T) env: $(envline)"
run_py ref mix0-ref
for a in "mix0-again:{$DUMP,$FORCE,\"DS4_HC_MIX\":\"0\"}" "mix1:{$DUMP,$FORCE}" \
         "m13-all-off:{$DUMP,$FORCE,\"DS4_HC_MIX\":\"0\",\"DS4_HC_PRENORM\":\"0\",\"DS4_LEAN\":\"0\",\"DS4_SH_FOLD\":\"0\",\"DS4_ROUTE_PAR\":\"0\",\"DS4_PF_PINNED\":\"0\"}"; do
  [ -e "$STOP" ] && break
  arm_env "${a#*:}"; up || exit 1; echo "### ${a%%:*} $(date +%T) env: $(envline)"
  run_py cmp "${a%%:*}"
done
rm -f "/media/mab776/nvme 2To/models/strata-ds4/logs/m13ref.rows" "/media/mab776/nvme 2To/models/strata-ds4/logs/oracle.txt"
restore
