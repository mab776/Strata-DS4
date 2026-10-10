#!/bin/bash
# mab776 M2: one ds4_generate run on GPU 0 in a RAM-capped container, with peaks sampled every 2 s.
#   m2run.sh LABEL IDS_FILE N_PREDICT [extra ds4_generate args...]
# env: OUT (results dir, default bench/mab776-m2/out), MEM (container cap GiB, default MemAvailable - 9),
#      ARENA (GiB, default MEM - 6), SKIP=1|0 (--arena-skip-resident), CHUNK=4096|...|0 (0 = decode-loop prefill),
#      PCIE (default 0.55), BUILD (default build-ds4-cu13), IMAGE (default nvidia/cuda:13.0.2-devel-ubuntu24.04),
#      MARGIN (default 2.0), MMQ=1|0 (--chunk-mmq), PRESTAGE=1|0 (--chunk-prestage), SEED_FIRST=1 (cache seeded at load),
#      MMQ_FP4=1 (STRATA_MMQ_FP4=1: MXFP4 experts on MMQ with FP4 activations instead of MMVQ)
#      M7a: G1=1 (GPU 1 as a static expert tier: both cards visible, PCI bus order, --gpu1 1), G1_GIB (cap),
#      GROW=G (--vram-grow G: seeded at load on the MARGIN budget, grown after the prompt to leave G GiB; MARGIN ~8.5),
#      XENV="VAR=val ..." (extra container env: DS4_CHECK_G1=1, DS4_CHECK_GPU=1, DS4_TIER_PROF=1)
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
MEM=${MEM:-$((AV - 10))}; ARENA=${ARENA:-$((MEM - 6))}
# keep >= 8 GiB free on the host even if the caller asked for more (other jobs share the box)
[ "$MEM" -gt $((AV - 10)) ] && MEM=$((AV - 10)); [ "$ARENA" -gt $((MEM - 5)) ] && ARENA=$((MEM - 5))
CH=${CHUNK:-4096}; PF=(--prefill-chunk "$CH"); [ "${MMQ:-1}" = 1 ] && PF+=(--chunk-mmq); [ "${PRESTAGE:-1}" = 1 ] && PF+=(--chunk-prestage)
[ "$CH" = 0 ] && PF=()
SK=(); [ "${SKIP:-1}" = 1 ] && SK=(--arena-skip-resident)
BUILD=${BUILD:-build-ds4-cu13}; IMAGE=${IMAGE:-nvidia/cuda:13.0.2-devel-ubuntu24.04}
IDSD=$(cd "$(dirname "$IDS")" && pwd); IDSF=$(basename "$IDS")
GPUS='"device=0"'; GENV=(-e NVIDIA_VISIBLE_DEVICES=0); G1A=()
if [ "${G1:-0}" = 1 ]; then
  GPUS='"device=0,1"'; GENV=(-e NVIDIA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID); G1A=(--gpu1 1)
  [ -n "${G1_GIB:-}" ] && G1A+=(--gpu1-gib "$G1_GIB")
fi
[ -n "${GROW:-}" ] && G1A+=(--vram-grow "$GROW")
XE=(); for kv in ${XENV:-}; do XE+=(-e "$kv"); done   # extra container env, e.g. XENV="DS4_CHECK_G1=1 DS4_TIER_PROF=1"
MTPDIR=${MTPDIR:-"/media/mab776/nvme 2To/models/deepseek-v4-draft/mtp"}   # M7b: mounted at /mtp (--mtp /mtp/<file>)
docker rm -f ds4-m2 >/dev/null 2>&1
echo "== $L: $(date '+%F %T') MEM ${MEM}g ARENA ${ARENA} SKIP ${SKIP:-1} CHUNK $CH PCIE ${PCIE:-0.55} G1 ${G1:-0} GROW ${GROW:--} MARGIN ${MARGIN:-2.0} $BUILD $*" | tee "$OUT/$L.log"
docker run -d --name ds4-m2 --gpus "$GPUS" "${GENV[@]}" "${XE[@]}" -e MIMO_CHUNK_PROF=1 -e GGML_NO_BACKTRACE=1 ${SEED_FIRST:+-e DS4_SEED_FIRST=1} ${MMQ_FP4:+-e STRATA_MMQ_FP4=$MMQ_FP4} \
  --ipc host --memory "${MEM}g" --memory-swap "${MEM}g" --ulimit memlock=-1 \
  -v "$R":/src:ro -v "$BLOBDIR":/m:ro -v "$IDSD":/w:ro -v "$MTPDIR":/mtp:ro -w /src "$IMAGE" \
  "$BUILD/ds4_generate" -m "/m/$BLOBF" --ids-file "/w/$IDSF" -n "$N" --temp 0 --stop 1 \
    --ctx 262144 --comp-type iq4_nl --icomp-q8 --slots auto --vram-margin "${MARGIN:-2.0}" \
    --pcie "${PCIE:-0.55}" --vram-lru --pf-b 0.7 --arena-gib "$ARENA" "${SK[@]}" "${PF[@]}" \
    --profile bench/ds4-2026-10-05/route-probe/ds4routes.bin "${G1A[@]}" "$@" >/dev/null || exit 1
CID=$(docker inspect -f '{{.Id}}' ds4-m2); PID=$(docker inspect -f '{{.State.Pid}}' ds4-m2)
CG=/sys/fs/cgroup/system.slice/docker-$CID.scope
t0=$(date +%s)
while [ "$(docker inspect -f '{{.State.Running}}' ds4-m2 2>/dev/null)" = true ]; do
  echo "$(( $(date +%s) - t0 )) $(awk '/VmHWM/{v=$2} END{print v+0}' /proc/$PID/status 2>/dev/null) $(cat $CG/memory.peak 2>/dev/null || echo 0)" \
       "$(awk '/MemAvailable/{print $2}' /proc/meminfo) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')" \
       >> "$OUT/$L.mem"
  a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$a" -lt 6 ]; then low=$((${low:-0} + 1)); else low=0; fi
  if [ "${low:-0}" -ge 3 ]; then echo "host MemAvailable ${a} GiB < 6 for 6 s - stopping ds4-m2" | tee -a "$OUT/$L.log"; docker stop -t 5 ds4-m2 >/dev/null; fi
  sleep 2
done
docker logs ds4-m2 > "$OUT/$L.ids" 2>> "$OUT/$L.log"
echo "exit=$(docker inspect -f '{{.State.ExitCode}}' ds4-m2) oom=$(docker inspect -f '{{.State.OOMKilled}}' ds4-m2) wall=$(( $(date +%s) - t0 ))s" >> "$OUT/$L.log"
docker rm ds4-m2 >/dev/null
grep -E '^(prefill|decode :|experts/token|decode ms/token|tier moves|VRAM free|slots auto|tier:|prefill chunks|ppl:|chunk prof|tier ms|context cap|exit=|gpu1|vram grow|grow_cache|\[g1-chk\] GPU|MTP|verify)|error|failed|refused|abort|DIFFER' "$OUT/$L.log"
awk 'NF>=5 && $4>0 {if($2>h)h=$2; if($3>p)p=$3; if(m==""||$4<m)m=$4; if($5>v)v=$5; if(NF>=6 && $6>v1)v1=$6} END{printf "peak VmHWM %.1f GiB, cgroup peak %.1f GiB, min host avail %.1f GiB, peak VRAM %d MiB (GPU 1 %d MiB)\n",h/1048576,p/2^30,m/1048576,v,v1}' "$OUT/$L.mem" | tee -a "$OUT/$L.log"
