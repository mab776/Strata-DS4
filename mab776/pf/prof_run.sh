#!/usr/bin/env bash
# mab776: the prefill profile run - one throwaway container (mab776/pf/tmp-cfg/<cfg>.json, add MIMO_CHUNK_PROF /
# STRATA_DENSE_PROF there): the 8.7k / 123k bench + turns (run.sh), four more +300 agent turns, the container's memory
# peak; then the engine log's prompt phases + chunk / dense profile lines.
#   prof_run.sh LABEL CFG
set -u
L=$1; C=$2
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
(cd $W && CFG=mab776/pf/tmp-cfg/$C.json GPUS=0,1 GEN=128 TURNS=1 KEEP=1 bash mab776/pf/run.sh $L \
   -v /home/mab776/v41-engram/park-test:/home/mab776/v41-engram/park-test >/dev/null 2>&1) || { echo "$L: run.sh failed"; exit 1; }
timeout 900 python3 $W/mab776/pf/turn_bench.py http://127.0.0.1:8332 --nonce rep-$L-$(date +%s) --label $L-rep \
  --turns 300,300,300,300 --base 8000 --offset 9000000 >> $O/$L.jsonl 2>&1
echo "peak: $(docker exec ds41-pf sh -c 'cat /sys/fs/cgroup/memory.peak 2>/dev/null || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes' | awk '{printf "%.1f GiB", $1/1073741824}') | host avail $(free -g | awk 'NR==2{print $7}') GiB"
docker exec ds41-pf cat /tmp/strata-ds41.log > $O/$L.engine.log
docker rm -f ds41-pf >/dev/null
python3 - "$O/$L.jsonl" <<'PY'
import json, sys
for l in open(sys.argv[1]):
    try: d = json.loads(l)
    except ValueError: continue
    if "turn" in d:
        print(f"  turn +{d['new']}: {d['wall_s']} s (prefill {d.get('prefill_tps')} tok/s)")
    elif d.get("depth", 0) > 5000:
        print(f"  {d['depth']}: prefill {d['prefill_tps']} tok/s ({d['prefill_s']} s), decode {d['decode_tps']} tok/s")
PY
grep -E "^serve: prompt phases" $O/$L.engine.log | cut -c1-140
