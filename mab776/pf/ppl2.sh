#!/usr/bin/env bash
# mab776: ppl.sh with per-line GPUs, arena and extra engine args (the GPU 1 tier gate).  ppl2.sh LISTFILE
#   lines: <label> <build> <ids.i32> [GPUS=0,1] [ARENA=76.5] [ARGS=--peer-device,1] [ENV=V ...]
# results -> ram-128gb/v41/pf/ppl/<label>.log (the PPL line)
set -u
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; P=$O/ppl
G="/media/mab776/nvme 2To/models/deepseek-v41/gsq"
while read -r L B IDS REST; do
  [ -z "${L:-}" ] && continue; case "$L" in \#*) continue;; esac
  e=(); gpus=0; arena=88; xa=()
  for x in $REST; do
    case "$x" in
      GPUS=*) gpus=${x#GPUS=};; ARENA=*) arena=${x#ARENA=};; ARGS=*) IFS=, read -r -a xa <<< "${x#ARGS=}";;
      *) e+=(-e "$x");;
    esac
  done
  echo "[$(date +%T)] ppl $L ($B $IDS gpus $gpus arena $arena ${xa[*]:-}) ${e[*]:-}" | tee -a $O/sweep.log
  timeout 900 docker run --rm --name ds41-ppl --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$gpus -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    -e STRATA_NO_LARGEPAGES=1 "${e[@]}" \
    --memory 104g --memory-swap 104g --oom-score-adj 500 --ulimit memlock=-1:-1 --ulimit core=0 --ipc host \
    -v "$G:$G:ro" -v "/media/mab776/nvme 2To/models/deepseek-v41/engram-a:/media/mab776/nvme 2To/models/deepseek-v41/engram-a:ro" \
    -v /home/mab776/v41-engram/engram-b:/home/mab776/v41-engram/engram-b:ro -v $W:$W:ro -v $P:$P strata-ds4:cu13 \
    $W/$B/ds41_generate -m "$G/V41-GSQ.gguf" --backend cuda --experts gpu \
      --engram "/media/mab776/nvme 2To/models/deepseek-v41/engram-a:/home/mab776/v41-engram/engram-b" --rank "$G/rank.bin" \
      --ctx 8192 --arena-adapt --kv-fp4 --arena-gib $arena "${xa[@]}" --ids-file $P/$IDS --ppl -n 1 --quiet > $P/$L.log 2>&1
  echo "[$(date +%T)]   $L rc=$? $(grep -E '^PPL' $P/$L.log)" | tee -a $O/sweep.log
  docker rm -f ds41-ppl >/dev/null 2>&1
done < "$1"
