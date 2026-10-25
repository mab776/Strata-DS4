#!/usr/bin/env bash
# mab776 (10-25): the ISTA-DASLab IQ2_XXS + Engram throwaway gate (NOT a stack). Mirrors mab776/pf/run.sh +
# cand_run.sh with the mounts the ISTA set needs:
#   - the ENGINE comes from strata-ds4/build-dd (the post-merge Xolotl 281afdad build); the container's cwd is
#     strata-ds4 (serve + tools/v41 from this tree), config = mab776/pf/tmp-cfg/ista1025.json (exe -> this tree's
#     ds41-engine-ista.sh).  The ds41 worktree must be at the same commit (it only matters for its own runs).
#   - the whole models drive is mounted ro (covers ista/ shards + Engram, gsq/ MTP-mmproj-tokenizer, vision-venv/ref)
#   - /home/mab776/v41-engram rw (park-ista lives there); NO /mirror (the 56 GiB mirror is the GSQ model's).
# Gate order (the V4.1 gate, re-baselined: new engine AND new model, so today's md5 becomes the baseline):
#   one-shot greedy md5 (x2 = determinism) -> 8.7k + 123k bench -> turns -> tools x3 -> needles 246k x3 + 480k -> RAM margin
#   USE_MD5=1 adds a third one-shot (the draft-independence check wants two runs at least).
#   GPUS=0,1 MEM=104g PORT=8332 ./ista_run.sh LABEL ; KEEP=1 leaves ds41-ista running after the bench phase.
set -u
L=${1:?usage: ista_run.sh LABEL}; shift 2>/dev/null || true
W=/home/mab776/Documents/portainer/strata-ds4
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; mkdir -p $O
B=/home/mab776/Documents/portainer/strata-coder/bench
V=/home/mab776/Documents/portainer/ram-128gb/v41
CFG=${CFG:-mab776/pf/tmp-cfg/ista1025.json}; MEM=${MEM:-104g}; GPUS=${GPUS:-0,1}; PORT=${PORT:-8332}
U=http://127.0.0.1:$PORT
docker rm -f ds41-ista >/dev/null 2>&1
docker run -d --name ds41-ista --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=$GPUS -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e HOME=/home/mab776 -e STRATA_TEXT_LOG=/tmp/strata-ds41-text.log "$@" \
  --memory $MEM --memory-swap $MEM --oom-score-adj 500 --ulimit memlock=-1:-1 --ulimit core=0 --ipc host --security-opt label=disable \
  -p $PORT:8150 -w $W \
  -v "/media/mab776/nvme 2To/models/strata-data/src-01431:/home/mab776/AI/Strata:ro" \
  -v "/media/mab776/nvme 2To/models:/media/mab776/nvme 2To/models:ro" \
  -v /home/mab776/v41-engram:/home/mab776/v41-engram \
  -v $W:$W:ro strata-ds4:cu13 python tools/v41/serve/serve_ds41.py --engine strata --config $CFG --port 8150 >/dev/null || exit 1
# memory sampler (every 5 s): the cgroup's anon/shmem/kernel + host available (the OOM rule: >= 4 GiB margin under $MEM)
( sleep 30; while docker ps -q -f name=ds41-ista | grep -q .; do
    docker exec ds41-ista sh -c 'awk "/^anon |^file |^kernel |^shmem /{printf \"%s %.2f \", \$1, \$2/1073741824}" /sys/fs/cgroup/memory.stat' 2>/dev/null
    echo "host_avail $(free -g | awk 'NR==2{print $7}')"; sleep 5; done ) > $O/$L.mem.log 2>&1 &
t0=$(date +%s)
until curl -sf -m 5 $U/v1/models >/dev/null && python3 $B/strata_bench.py 300 --offset 900000 --url $U --nonce w$L-$t0 --gen 16 >/dev/null 2>&1; do
  sleep 10; [ $(( $(date +%s) - t0 )) -gt 1500 ] && { echo "$L: not ready"; docker logs --tail 30 ds41-ista; docker exec ds41-ista tail -30 /tmp/strata-ds41.log; exit 1; }
  docker ps -q -f name=ds41-ista | grep -q . || { echo "$L: died"; docker logs --tail 40 ds41-ista; exit 1; }
done
echo "$L ready after $(( $(date +%s) - t0 )) s" | tee -a $O/$L.log
# 1) one-shot greedy md5 (x2, x3 with USE_MD5=1): determinism now, the baseline for every later engine change
n=$(( ${USE_MD5:-0} ? 3 : 2 ))
for i in $(seq $n); do
  curl -s $U/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"A railway has 9 stations. Write a haiku about the third one."}],"temperature":0,"max_tokens":64,"reasoning_effort":"low"}' \
    | python3 -c 'import sys,json,hashlib; d=json.load(sys.stdin); print("oneshot raw md5", hashlib.md5(d["choices"][0]["message"].get("content","").encode()).hexdigest()[:12])' >> $O/$L.md5.log
done
echo "md5: $(tr '\n' ' ' < $O/$L.md5.log)"
# 2) prefill + decode at the two gate windows; 3) agent turns
python3 $B/strata_bench.py 8700 --offset 900000 --ratio 3.1 --url $U --nonce $L-$t0 --label $L --gen ${GEN:-256} 2>&1 | tail -1 | tee -a $O/$L.jsonl
python3 $B/strata_bench.py 123000 --offset 9000000 --ratio 3.1 --url $U --nonce $L-$t0 --label $L --gen ${GEN:-256} 2>&1 | tail -1 | tee -a $O/$L.jsonl
[ "${TURNS:-1}" = 1 ] && timeout 1200 python3 $W/mab776/pf/turn_bench.py $U --nonce t$L-$t0 --label $L 2>&1 | tee -a $O/$L.jsonl
# 4) tools x3
for i in 1 2 3; do python3 $V/tools_probe.py $U > $O/$L.tools.$i.log 2>&1; echo "tools $i: $(tail -1 $O/$L.tools.$i.log)"; done
# 5) needles: 246k x 3 depths + the 480k long-context proof
NEEDLE_ROOT="/media/mab776/nvme 2To/models/strata-data/src-01431" timeout 3000 python3 /home/mab776/Documents/portainer/deepseek-v4/needle_ds4.py \
  --url $U --lengths 246k --depths 10,50,90 --timeout 3600 --out $O/$L.needles.json > $O/$L.needles.log 2>&1
echo "needles 246k: $(grep -E 'found' $O/$L.needles.log | tail -1)"
NEEDLE_ROOT="/media/mab776/nvme 2To/models/strata-data/src-01431" timeout 3000 python3 /home/mab776/Documents/portainer/deepseek-v4/needle_ds4.py \
  --url $U --lengths 480k --depths 50 --timeout 3600 --out $O/$L.needles480.json > $O/$L.needles480.log 2>&1
echo "needles 480k: $(grep -E 'found' $O/$L.needles480.log | tail -1)"
docker exec ds41-ista sh -c 'grep -E "error|ERR|nonfinite|illegal" /tmp/strata-ds41.log | tail -5'
docker exec ds41-ista cat /tmp/strata-ds41.log > $O/$L.engine.log 2>/dev/null
docker exec ds41-ista cat /tmp/strata-ds41-text.log > $O/$L.text.log 2>/dev/null
[ "${KEEP:-0}" = 1 ] || docker rm -f ds41-ista >/dev/null
awk -v mem=104 '{a=s=k=0; for(i=1;i<NF;i++){if($i=="anon")a=$(i+1); if($i=="shmem")s=$(i+1); if($i=="kernel")k=$(i+1); if($i=="host_avail")h=$(i+1)} t=a+s+k; if(t>mx){mx=t; ma=a; ms=s; mk=k}; if(hm==""||h<hm)hm=h}
     END{printf "memory: peak anon+shmem+kernel %.2f GiB (anon %.2f, shmem %.2f, kernel %.2f) -> margin %.2f GiB of %s; host avail min %s GiB\n", mx, ma, ms, mk, mem-mx, mem, hm}' $O/$L.mem.log
echo "$L gate done"
