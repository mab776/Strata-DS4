#!/bin/bash
# M14 numerics gate: teacher-forced rows at 127k.  Reference = both M14 switches off; then off again (the restart
# noise floor), then the M14 defaults (warp top-k router + fused predict; the predict is only a prefetch hint).
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m14/lib14.sh
D=${1:-130048}
DUMP='"DS4_DUMP_TOKLOGITS":"/data/logs/tl"'
FORCE='"DS4_SERVE_TEST_ORACLE":"/data/logs/oracle.txt","DS4_SERVE_TEST_FORCE":"1"'
OFF='"GGML_CUDA_RANK_ARGSORT":"0","DS4_FUSE_PREDICT":"0"'
run_py() { systemd-run --user --scope -q -p MemoryMax=3G python3 $W/m13/forced_x.py $U $D "$@"; }
arm "{$DUMP,$OFF}"; up || exit 1; echo "### ref (M14 off) $(date +%T) env: $(envline)"
run_py ref off-ref
for a in "off-again:{$DUMP,$FORCE,$OFF}" "m14-on:{$DUMP,$FORCE}"; do
  [ -e "$STOP" ] && break
  arm "${a#*:}"; up || exit 1; echo "### ${a%%:*} $(date +%T) env: $(envline)"
  run_py cmp "${a%%:*}"
done
rm -f "/media/mab776/nvme 2To/models/strata-ds4/logs/m13ref.rows" "/media/mab776/nvme 2To/models/strata-ds4/logs/oracle.txt" "/media/mab776/nvme 2To/models/strata-ds4/logs/tl".*
restore
