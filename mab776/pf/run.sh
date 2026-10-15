#!/usr/bin/env bash
# mab776 overnight prefill sweep: one engine variant in a throwaway container (NOT a stack), bench it, stop it.
#   GPUS=0|0,1  CFG=mab776/pf/<x>.json  MEM=104g  DEPTHS="8000:3000000 127000:15000000"  ./run.sh <label> [-e K=V ...]
# results -> ram-128gb/v41/pf/<label>.jsonl + .log (prompt phases, chunk prof, mem)
set -u
L=$1; shift
W=/home/mab776/Documents/portainer/strata-ds41
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; mkdir -p $O
B=/home/mab776/Documents/portainer/strata-coder/bench
CFG=${CFG:-mab776/strata-ds41.json}; MEM=${MEM:-104g}; GPUS=${GPUS:-0}; PORT=${PORT:-8332}
DEPTHS=${DEPTHS:-"8000:3000000 127000:15000000"}; DEPTHS=${DEPTHS//,/ }
U=http://127.0.0.1:$PORT
docker rm -f ds41-pf >/dev/null 2>&1
# DUMP=1: the engine dumps every request's last 512 prompt rows of logits (STRATA_SERVE_DUMP_TAIL) to $O/dump/$L, and the
# FIRST depth uses the fixed nonce "klgate" (the same prompt in every variant -> logits_kl.py between variants)
dump=(); [ "${DUMP:-0}" = 1 ] && { mkdir -p $O/dump/$L; rm -f $O/dump/$L/*; dump=(-e STRATA_SERVE_DUMP=/dump -e STRATA_SERVE_DUMP_TAIL=512 -v $O/dump/$L:/dump); }
docker run -d --name ds41-pf --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUS -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e HOME=/home/mab776 -e STRATA_TEXT_LOG=/tmp/strata-ds41-text.log "${dump[@]}" "$@" \
  --memory $MEM --memory-swap $MEM --oom-score-adj 500 --ulimit memlock=-1:-1 --ipc host --security-opt label=disable \
  -p $PORT:8150 -w $W \
  -v "/media/mab776/nvme 2To/models/strata-data/src-01431:/home/mab776/AI/Strata:ro" \
  -v "/media/mab776/nvme 2To/models/deepseek-v41/gsq:/media/mab776/nvme 2To/models/deepseek-v41/gsq:ro" \
  -v "/media/mab776/nvme 2To/models/deepseek-v41/engram-a:/media/mab776/nvme 2To/models/deepseek-v41/engram-a:ro" \
  -v /home/mab776/v41-engram/engram-b:/home/mab776/v41-engram/engram-b:ro \
  -v $W:$W:ro strata-ds4:cu13 python tools/v41/serve/serve_ds41.py --engine strata --config $CFG --port 8150 >/dev/null || exit 1
t0=$(date +%s)
until curl -sf -m 5 $U/v1/models >/dev/null && python3 $B/strata_bench.py 300 --offset 900000 --url $U --nonce w$L-$t0 --gen 16 >/dev/null 2>&1; do
  sleep 10; [ $(( $(date +%s) - t0 )) -gt 600 ] && { echo "$L: not ready"; docker logs --tail 30 ds41-pf; docker exec ds41-pf tail -30 /tmp/strata-ds41.log; exit 1; }
  docker ps -q -f name=ds41-pf | grep -q . || { echo "$L: died"; exit 1; }
done
echo "$L ready after $(( $(date +%s) - t0 )) s" | tee -a $O/$L.log
first=1
for d in $DEPTHS; do
  nonce=$L-$t0; [ "${DUMP:-0}" = 1 ] && [ $first = 1 ] && nonce=klgate; first=0
  python3 $B/strata_bench.py ${d%%:*} --offset ${d##*:} --ratio 3.1 --url $U --nonce $nonce --label $L --gen ${GEN:-128} 2>&1 | tail -1 | tee -a $O/$L.jsonl
  echo "   RAM $(free -g | awk 'NR==2{print $3" used / "$7" avail GiB"}') | VRAM $(nvidia-smi --query-gpu=memory.used --format=csv,noheader | paste -sd/) | ctr $(docker stats --no-stream --format '{{.MemUsage}}' ds41-pf)" | tee -a $O/$L.log
done
[ "${TURNS:-0}" = 1 ] && python3 $W/mab776/pf/turn_bench.py $U --nonce t$L-$t0 --label $L 2>&1 | tee -a $O/$L.jsonl
docker exec ds41-pf sh -c 'grep -E "prompt phases|^serve: prompt|chunk prof|chunk prefetch|prompt cache|chunk prestage|VRAM|tier:|enc-skip|error|ERR" /tmp/strata-ds41.log' >> $O/$L.log
docker exec ds41-pf cat /tmp/strata-ds41.log > $O/$L.engine.log 2>/dev/null
[ "${KEEP:-0}" = 1 ] || docker rm -f ds41-pf >/dev/null
echo "$L done"
