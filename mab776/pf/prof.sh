#!/usr/bin/env bash
# mab776 overnight: nsys trace of a V4.1 PREFILL in a throwaway container (ds41-pf, NOT a stack): the server runs under
# `nsys launch`, one fresh prompt of NTOK tokens is sent, the capture covers [START_S, START_S + DUR_S] of it.
#   CFG=mab776/pf/<x>.json  prof.sh LABEL [NTOK=32000] [START_S=20] [DUR_S=12] [-e K=V ...]
# report -> ram-128gb/v41/pf/prof/LABEL.nsys-rep (+ csv summaries via nsys stats)
set -u
L=$1; NTOK=${2:-32000}; ST=${3:-20}; DUR=${4:-12}; shift 4 || shift $#
W=/home/mab776/Documents/portainer/strata-ds41
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf/prof; mkdir -p $O
B=/home/mab776/Documents/portainer/strata-coder/bench
CFG=${CFG:-mab776/strata-ds41.json}; MEM=${MEM:-104g}; GPUS=${GPUS:-0}; PORT=${PORT:-8332}
U=http://127.0.0.1:$PORT
NS=/nsys/target-linux-x64/nsys
docker rm -f ds41-pf >/dev/null 2>&1
docker run -d --name ds41-pf --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUS -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e HOME=/home/mab776 "$@" \
  --memory $MEM --memory-swap $MEM --oom-score-adj 500 --ulimit memlock=-1:-1 --ipc host --security-opt label=disable \
  -p $PORT:8150 -w $W \
  -v "/media/mab776/nvme 2To/models/strata-data/src-01431:/home/mab776/AI/Strata:ro" \
  -v "/media/mab776/nvme 2To/models/deepseek-v41/gsq:/media/mab776/nvme 2To/models/deepseek-v41/gsq:ro" \
  -v "/media/mab776/nvme 2To/models/deepseek-v41/engram-a:/media/mab776/nvme 2To/models/deepseek-v41/engram-a:ro" \
  -v /home/mab776/v41-engram/engram-b:/home/mab776/v41-engram/engram-b:ro \
  -v $W:$W:ro -v $O:/out -v /opt/nvidia/nsight-systems/2025.3.2:/nsys:ro strata-ds4:cu13 \
  $NS launch --trace=cuda,nvtx python tools/v41/serve/serve_ds41.py --engine strata --config $CFG --port 8150 >/dev/null || exit 1
t0=$(date +%s)
until curl -sf -m 5 $U/v1/models >/dev/null && python3 $B/strata_bench.py 300 --offset 900000 --url $U --nonce pw$L-$t0 --gen 16 >/dev/null 2>&1; do
  sleep 10; [ $(( $(date +%s) - t0 )) -gt 600 ] && { echo "$L: not ready"; docker logs --tail 30 ds41-pf; exit 1; }
  docker ps -q -f name=ds41-pf | grep -q . || { echo "$L: died"; exit 1; }
done
echo "$L ready after $(( $(date +%s) - t0 )) s"
python3 $B/strata_bench.py $NTOK --offset 5000000 --ratio 3.1 --url $U --nonce p$L-$t0 --label $L --gen 16 > $O/$L.bench 2>&1 &
RP=$!
sleep $ST
docker exec ds41-pf $NS start --sample=none --cpuctxsw=none --output=/out/$L --force-overwrite=true && echo "capture started $(date +%T)"
nvidia-smi dmon -s ut -d 1 -c $DUR > $O/$L.dmon 2>&1
docker exec ds41-pf $NS stop && echo "capture stopped $(date +%T)"
wait $RP; tail -1 $O/$L.bench
docker exec ds41-pf sh -c 'grep -E "prompt phases|^serve: prompt|chunk prof|chunk prefetch|prompt cache|VRAM|tier:" /tmp/strata-ds41.log' > $O/$L.log
[ "${KEEP:-0}" = 1 ] || { docker stop -t 30 ds41-pf >/dev/null; docker rm ds41-pf >/dev/null 2>&1; }
docker run --rm -e NVIDIA_VISIBLE_DEVICES=none -u $(id -u):$(id -g) -e HOME=/tmp -v $O:/p -v /opt/nvidia/nsight-systems/2025.3.2:/nsys:ro \
  nvidia/cuda:13.0.2-devel-ubuntu24.04 sh -c "cd /p && $NS stats -q --force-export=true \
  -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum -f csv -o /p/$L $L.nsys-rep" > /dev/null 2>&1
ls $O/${L}*
echo "$L prof done"
