#!/usr/bin/env bash
# mab776: a deploy candidate's gate in a throwaway container (mab776/pf/tmp-cfg/<cfg>.json): the 8.7k / 123k bench +
# turns (run.sh), tools probe, needles at 246k x 3 depths, the default reasoning effort (a request naming none -> the
# config's level: the server's startup line), a regenerate, the memory peak (anon + shmem + kernel = what the OOM killer counts).  Prints a summary.
#   cand_run.sh LABEL CFG
set -u
L=$1; C=$2
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; V=/home/mab776/Documents/portainer/ram-128gb/v41
U=http://127.0.0.1:8332
# memory sampler (every 5 s while the container lives): the cgroup's anon / file bytes and the host's available RAM
( sleep 30; while docker ps -q -f name=ds41-pf | grep -q .; do
    docker exec ds41-pf sh -c 'awk "/^anon |^file |^kernel |^shmem /{printf \"%s %.2f \", \$1, \$2/1073741824}" /sys/fs/cgroup/memory.stat' 2>/dev/null
    echo "host_avail $(free -g | awk 'NR==2{print $7}')"; sleep 5; done ) > $O/$L.mem.log 2>&1 &
(cd $W && CFG=mab776/pf/tmp-cfg/$C.json GPUS=0,1 GEN=256 TURNS=1 KEEP=1 bash mab776/pf/run.sh $L \
   -v /home/mab776/v41-engram/park-test:/home/mab776/v41-engram/park-test) || { echo "$L: run.sh failed"; exit 1; }
python3 $V/tools_probe.py $U > $O/$L.tools.log 2>&1; echo "tools: $(tail -1 $O/$L.tools.log)"
echo "effort: $(docker logs ds41-pf 2>&1 | grep -m1 'thinking level' | cut -c1-60)"
NEEDLE_ROOT="/media/mab776/nvme 2To/models/strata-data/src-01431" timeout 3000 python3 /home/mab776/Documents/portainer/deepseek-v4/needle_ds4.py \
  --url $U --lengths 246k --depths 10,50,90 --timeout 3600 --out $O/$L.needles.json > $O/$L.needles.log 2>&1
echo "needles: $(grep -E 'found' $O/$L.needles.log | tail -1)"
REGEN_OUT= timeout 900 python3 $W/mab776/pf/regen_test.py $U $L > $O/$L.regen.jsonl 2>&1; echo "regen: $(cut -c1-160 $O/$L.regen.jsonl | tail -3 | tr '\n' ' ')"
docker exec ds41-pf sh -c 'grep -E "error|ERR|nonfinite|illegal|park" /tmp/strata-ds41.log | tail -5'
docker exec ds41-pf cat /tmp/strata-ds41.log > $O/$L.engine.log 2>/dev/null
docker rm -f ds41-pf >/dev/null
awk '{a=s=k=0; for(i=1;i<NF;i++){if($i=="anon")a=$(i+1); if($i=="shmem")s=$(i+1); if($i=="kernel")k=$(i+1); if($i=="host_avail")h=$(i+1)} t=a+s+k; if(t>mx){mx=t; ma=a; ms=s; mk=k}; if(hm==""||h<hm)hm=h}
     END{printf "memory: peak anon+shmem+kernel %.2f GiB (anon %.2f, shmem %.2f, kernel %.2f) -> margin %.2f GiB of 104; host avail min %s GiB\n", mx, ma, ms, mk, 104-mx, hm}' $O/$L.mem.log
echo "$L gate done"
