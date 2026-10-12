#!/bin/bash
# M13 server A/B of ONE env switch: VAR=0 (old path) vs VAR=1, ABBA (2 rounds), the same prompt in every arm (fixed
# nonce), greedy: K=1 then K=2 twice from the same restored state (pass ms is the engine's own number; tok/s moves with
# draft acceptance).   ab.sh VAR [DEPTHS=8192] [EXTRA_JSON='{}']   Ends on the canonical config.
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m13/lib.sh
VAR=$1; DEPTHS=${2:-8192}; EXTRA=${3:-'{}'}
for v in 0 1 1 0; do
  [ -e "$STOP" ] && { echo "### STOP file - ending early"; break; }
  arm_env "$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); d[sys.argv[2]]=sys.argv[3]; print(json.dumps(d))" "$EXTRA" "$VAR" "$v")"
  up || exit 1
  echo "### $VAR=$v $(date +%T) env: $(envline)"
  NONCE=m13-ab ksweep "$DEPTHS" 1:0,2:0,2:0 512
done
restore
