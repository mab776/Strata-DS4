#!/usr/bin/env bash
# mab776: memory probe for a deploy candidate - start it in the throwaway container (run.sh; BENCH=1 = its bench first), sample the
# cgroup every 2 s (anon / shmem = pinned buffers / kernel: what the OOM killer counts; file includes shmem), run the
# needles one length at a time (LENGTH:DEPTHS), print each length's peak of anon + shmem + kernel and the margin to the cap.
#   mem_probe.sh LABEL CFG ["504k:50 246k:10,50,90"] [CAP_GIB=104]      BENCH=1: run.sh's 8.7k / 123k bench + turns first
set -u
L=$1; C=$2; LENS=${3:-"504k:50 246k:50"}; CAP=${4:-104}
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; U=http://127.0.0.1:8332
rm -f /home/mab776/v41-engram/park-test/* 2>/dev/null
echo bench > $O/$L.phase
( sleep 20; while docker ps -q -f name=ds41-pf | grep -q .; do
    m=$(docker exec ds41-pf sh -c 'awk "/^anon |^file |^kernel |^shmem /{printf \"%s %.2f \", \$1, \$2/1073741824}" /sys/fs/cgroup/memory.stat' 2>/dev/null)
    echo "$m host_avail $(free -m | awk 'NR==2{printf "%.1f", $7/1024}') phase $(cat $O/$L.phase 2>/dev/null)"; sleep 2; done ) > $O/$L.mem.log 2>&1 &
bd=(DEPTHS=" "); [ "${BENCH:-0}" = 1 ] && bd=(GEN=256 TURNS=1)
(cd $W && env "${bd[@]}" CFG=mab776/pf/tmp-cfg/$C.json GPUS=0,1 KEEP=1 bash mab776/pf/run.sh $L \
   -v /home/mab776/v41-engram/park-test:/home/mab776/v41-engram/park-test) || { echo "$L: run.sh failed"; exit 1; }

for ld in $LENS; do
  len=${ld%%:*}; dep=${ld##*:}; nd=$(echo $dep | tr , "\n" | wc -l)
  echo $len > $O/$L.phase
  NEEDLE_ROOT="/media/mab776/nvme 2To/models/strata-data/src-01431" timeout 3000 python3 /home/mab776/Documents/portainer/deepseek-v4/needle_ds4.py \
    --url $U --lengths $len --depths $dep --timeout 3600 --out $O/$L.$len.needle.json >> $O/$L.needles.log 2>&1
  echo "needles $len: $(grep -E "^ *$len" $O/$L.needles.log | tail -$nd | grep -c FOUND) found of $nd: $(grep -E "^ *$len" $O/$L.needles.log | tail -3 | cut -c1-90 | tr "\n" " ")"
  sleep 4
done
echo done > $O/$L.phase; sleep 3
docker exec ds41-pf sh -c 'grep -E "error|ERR|nonfinite|illegal|out of memory" /tmp/strata-ds41.log | tail -5; cat /sys/fs/cgroup/memory.events | tr "\n" " "'
docker logs ds41-pf 2>&1 | grep -E "stopped unexpectedly|exit code" | tail -3
docker rm -f ds41-pf >/dev/null
for ld in bench $LENS; do
  len=${ld%%:*}
  awk -v P=$len -v CAP=$CAP '{a=s=k=0; for(i=1;i<NF;i++){if($i=="anon")a=$(i+1); if($i=="shmem")s=$(i+1); if($i=="kernel")k=$(i+1); if($i=="phase")ph=$(i+1); if($i=="host_avail")h=$(i+1)}
       if(ph==P){t=a+s+k; if(t>mx){mx=t; ma=a; ms=s; mk=k}; if(hm==""||h<hm)hm=h}}
       END{printf "%s: peak anon+shmem+kernel %.2f GiB (anon %.2f, shmem %.2f, kernel %.2f) -> margin %.2f GiB of %d; host avail min %s GiB\n", P, mx, ma, ms, mk, CAP-mx, CAP, hm}' $O/$L.mem.log
done
rm -f $O/$L.phase
