#!/usr/bin/env bash
# mab776: the route-bias A/B (Marc 2026-10-15) - one throwaway container per arm (mab776/pf/tmp-cfg/rb-<arm>.json):
# decode at 8.7k / 123k (GEN 256) + agent turns (run.sh), the per-domain decode (domain_test.py, replies to rbtxt/ for
# the identity check) and Marc's real traffic (sniffer_test.py, blind: metrics only).  Stops if the mode page leaves OFF.
#   rb_ab.sh ARM [ARM ...]      (SNIFF=0: skip the sniffer part)
set -u
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
mkdir -p $O/rbtxt
mode() { curl -s -m 5 localhost:8500/api/status | python3 -c "import sys,json;print(json.load(sys.stdin).get('mode'))" 2>/dev/null; }
for A in "$@"; do
  L=rb-$A
  [ "$(mode)" = off ] || { echo "[$(date +%T)] mode is $(mode) - stopping before $L"; exit 3; }
  echo "[$(date +%T)] $L start"
  (cd $W && CFG=mab776/pf/tmp-cfg/$L.json GPUS=0,1 GEN=256 TURNS=1 KEEP=1 bash mab776/pf/run.sh $L) || { echo "$L run.sh failed"; docker rm -f ds41-pf >/dev/null 2>&1; continue; }
  [ "$(mode)" = off ] || { echo "[$(date +%T)] mode is $(mode) - stopping in $L"; docker rm -f ds41-pf >/dev/null 2>&1; exit 3; }
  rm -f $O/rbtxt/$L.*.txt
  OUT=$O/rbtxt/$L timeout 1800 python3 $W/mab776/pf/domain_test.py http://127.0.0.1:8332 $L ds41-pf > $O/$L.dom.jsonl 2> $O/$L.dom.err
  echo "[$(date +%T)] $L domain: $(wc -l < $O/$L.dom.jsonl) rows"
  docker exec ds41-pf cat /tmp/strata-ds41.log > $O/$L.engine.log 2>/dev/null   # before the sniffer part (never after it)
  if [ "${SNIFF:-1}" = 1 ]; then
    [ "$(mode)" = off ] || { echo "[$(date +%T)] mode is $(mode) - stopping in $L"; docker rm -f ds41-pf >/dev/null 2>&1; exit 3; }
    timeout 3600 python3 $W/mab776/pf/sniffer_test.py http://127.0.0.1:8332 $L ds41-pf 6 > $O/$L.snf.jsonl 2> $O/$L.snf.err
    echo "[$(date +%T)] $L sniffer: $(wc -l < $O/$L.snf.jsonl) rows, stderr $(wc -l < $O/$L.snf.err) lines"
  fi
  docker rm -f ds41-pf >/dev/null 2>&1
  echo "[$(date +%T)] $L done"
done
