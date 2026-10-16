#!/usr/bin/env bash
# mab776: nsys timeline of the decode's verify passes (one-shot ds41_generate, greedy DSpark, the served engine flags,
# GPUs 0+1) - the capture starts after load + prompt (--delay) and lasts --duration seconds.  nsys = the host's
# /opt/nvidia/nsight-systems/2025.3.2, mounted (the image has none).
#   nsys_decode.sh LABEL BUILD [ENV=V ...]       -> ram-128gb/v41/pf/nsys/LABEL.nsys-rep + LABEL.log
set -u
L=$1; BLD=$2; shift 2
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf/nsys; mkdir -p $O
G="/media/mab776/nvme 2To/models/deepseek-v41/gsq"
e=(); for x in "$@"; do e+=(-e "$x"); done
docker rm -f ds41-nsys >/dev/null 2>&1
timeout 900 docker run --rm --name ds41-nsys --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=0,1 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e STRATA_NO_LARGEPAGES=1 -e GGML_NO_BACKTRACE=1 \
  -e GGML_TOPK_RADIX=1 -e STRATA_PSTAGE_STRIDED=1 -e STRATA_FN_DEV=1 -e STRATA_SPAN_ASYNC=1 -e STRATA_KALL16=1 \
  -e STRATA_LAZY_XQ=1 -e GGML_CUDA_CPY_ROWS=1 -e GGML_LI_WMMA_FP4=1 -e STRATA_TC_KEEP=1 -e GGML_LI_MMA=1 \
  -e STRATA_CHUNK_BUDGET_GB=6 -e STRATA_LOOKUP=7 -e STRATA_NAN_GUARD=1 -e STRATA_D3_OVERLAP=1 \
  -e STRATA_DSPARK="$G/V41-MTP-Vision.gguf" -e STRATA_DSPARK_KEEP=$W/tools/v41/serve/dspark_expert_counts.txt:64 \
  -e STRATA_DSPARK_DEPTH=5 -e STRATA_DSPARK_CONF=0.6 "${e[@]}" \
  --memory 104g --memory-swap 104g --oom-score-adj 500 --ulimit memlock=-1:-1 --ulimit core=0 --ipc host \
  -v "$G:$G:ro" -v "/media/mab776/nvme 2To/models/deepseek-v41/engram-a:/media/mab776/nvme 2To/models/deepseek-v41/engram-a:ro" \
  -v /home/mab776/v41-engram/engram-b:/home/mab776/v41-engram/engram-b:ro -v $W:$W:ro -v $O:$O -v $O/../ppl:$O/../ppl:ro \
  -v /opt/nvidia/nsight-systems/2025.3.2:/opt/nvidia/nsight-systems/2025.3.2:ro strata-ds4:cu13 \
  /opt/nvidia/nsight-systems/2025.3.2/bin/nsys profile -o $O/$L --force-overwrite true -t cuda --sample=none --cpuctxsw=none --cuda-graph-trace=node \
    --delay ${DELAY:-110} --duration ${DUR:-8} \
  $W/$BLD/ds41_generate -m "$G/V41-GSQ.gguf" --backend cuda --experts gpu \
    --engram "/media/mab776/nvme 2To/models/deepseek-v41/engram-a:/home/mab776/v41-engram/engram-b" --rank "$G/rank.bin" \
    --ctx 8192 --arena-adapt --kv-fp4 --route-bias 0.5 --route-bias-file 0.3 --arena-gib 76.5 --prompt-batch 32768 \
    --peer-device 1 --temp 0 -n ${GEN:-1536} --ids-file /home/mab776/Documents/portainer/ram-128gb/v41/pf/ppl/dom-code-B.i32 \
    > $O/$L.log 2>&1
echo "$L rc=$? $(grep -E '^decode' $O/$L.log | tail -1 | cut -c1-160)"
