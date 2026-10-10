#!/bin/bash
# mab776 M2: one ds4_generate run on GPU 0 in a RAM-capped container, with peaks sampled every 2 s.
#   m2run.sh LABEL IDS_FILE N_PREDICT [extra ds4_generate args...]
# env: OUT (results dir, default bench/mab776-m2/out), MEM (container cap GiB, default MemAvailable - 9),
#      ARENA (GiB, default MEM - 6), SKIP=1|0 (--arena-skip-resident), CHUNK=4096|...|0 (0 = decode-loop prefill),
#      PCIE (default 0.55), BUILD (default build-ds4-cu13), IMAGE (default nvidia/cuda:13.0.2-devel-ubuntu24.04),
#      MARGIN (default 2.0), MMQ=1|0 (--chunk-mmq), PRESTAGE=1|0 (--chunk-prestage)
# Refuses to start when the GPUs are not idle (another engine / mode page). Never touches any other container.
set -u
L=$1; IDS=$2; N=$3; shift 3
R=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$R/bench/mab776-m2/out}; mkdir -p "$OUT"
BLOBDIR="/media/mab776/nvme 2To/models/ollama/models/blobs"
BLOBF=sha256-a79d3b8c742f240dc947758bbeb46d266e9328ace6a3b71704ab4bf2a8610d68
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
if [ "$used" -gt 2000 ]; then echo "GPUs busy ($used MiB used) - not starting"; exit 3; fi
AV=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
MEM=${MEM:-$((AV - 9))}; ARENA=${ARENA:-$((MEM - 6))}
CH=${CHUNK:-4096}; PF=(--prefill-chunk "$CH"); [ "${MMQ:-1}" = 1 ] && PF+=(--chunk-mmq); [ "${PRESTAGE:-1}" = 1 ] && PF+=(--chunk-prestage)
[ "$CH" = 0 ] && PF=()
SK=(); [ "${SKIP:-1}" = 1 ] && SK=(--arena-skip-resident)
BUILD=${BUILD:-build-ds4-cu13}; IMAGE=${IMAGE:-nvidia/cuda:13.0.2-devel-ubuntu24.04}
IDSD=$(cd "$(dirname "$IDS")" && pwd); IDSF=$(basename "$IDS")
docker rm -f ds4-m2 >/dev/null 2>&1
echo "== $L: $(date '+%F %T') MEM ${MEM}g ARENA ${ARENA} SKIP ${SKIP:-1} CHUNK $CH PCIE ${PCIE:-0.55} $BUILD $*" | tee "$OUT/$L.log"
docker run -d --name ds4-m2 --gpus '"device=0"' -e NVIDIA_VISIBLE_DEVICES=0 -e MIMO_CHUNK_PROF=1 -e GGML_NO_BACKTRACE=1 \
  --ipc host --memory "${MEM}g" --memory-swap "${MEM}g" --ulimit memlock=-1 \
  -v "$R":/src:ro -v "$BLOBDIR":/m:ro -v "$IDSD":/w:ro -w /src "$IMAGE" \
  "$BUILD/ds4_generate" -m "/m/$BLOBF" --ids-file "/w/$IDSF" -n "$N" --temp 0 --stop 1 \
    --ctx 262144 --comp-type iq4_nl --icomp-q8 --slots auto --vram-margin "${MARGIN:-2.0}" \
    --pcie "${PCIE:-0.55}" --vram-lru --pf-b 0.7 --arena-gib "$ARENA" "${SK[@]}" "${PF[@]}" \
    --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin "$@" >/dev/null || exit 1
CID=$(docker inspect -f '{{.Id}}' ds4-m2); PID=$(docker inspect -f '{{.State.Pid}}' ds4-m2)
CG=/sys/fs/cgroup/system.slice/docker-$CID.scope
t0=$(date +%s)
while [ "$(docker inspect -f '{{.State.Running}}' ds4-m2 2>/dev/null)" = true ]; do
  echo "$(( $(date +%s) - t0 )) $(awk '/VmHWM/{print $2}' /proc/$PID/status 2>/dev/null || echo 0) $(cat $CG/memory.peak 2>/dev/null || echo 0)" \
       "$(awk '/MemAvailable/{print $2}' /proc/meminfo) $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader,nounits)" \
       >> "$OUT/$L.mem"
  sleep 2
done
docker logs ds4-m2 > "$OUT/$L.ids" 2>> "$OUT/$L.log"
echo "exit=$(docker inspect -f '{{.State.ExitCode}}' ds4-m2) oom=$(docker inspect -f '{{.State.OOMKilled}}' ds4-m2) wall=$(( $(date +%s) - t0 ))s" >> "$OUT/$L.log"
docker rm ds4-m2 >/dev/null
grep -E '^(prefill|decode :|experts/token|decode ms/token|tier moves|VRAM free|slots auto|tier:|prefill chunks|ppl:|chunk prof|tier ms|context cap|exit=)|error|failed|refused|abort' "$OUT/$L.log"
awk '{if($2>h)h=$2; if($3>p)p=$3; if(m==""||$4<m)m=$4; if($5>v)v=$5} END{printf "peak VmHWM %.1f GiB, cgroup peak %.1f GiB, min host avail %.1f GiB, peak VRAM %d MiB\n",h/1048576,p/2^30,m/1048576,v}' "$OUT/$L.mem" | tee -a "$OUT/$L.log"
