#!/usr/bin/env bash
# mab776: a needle against the LIVE stack 78 (container strata-ds41, host :8330) with the container's cgroup sampled
# every 2 s (anon / shmem / kernel), so the deployed config's memory margin is measured where it runs.
#   live_needle.sh LABEL [LENGTH=504k] [DEPTH=50]      CPT=<chars per token> sets the haystack size (3.372 -> 504k = ~505k tokens)
set -u
L=$1; LEN=${2:-504k}; DEP=${3:-50}; CAP=104
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
rm -f $O/$L.stop
( while docker ps -q -f name=strata-ds41 | grep -q . && [ ! -f $O/$L.stop ]; do
    m=$(docker exec strata-ds41 sh -c 'awk "/^anon |^file |^kernel |^shmem /{printf \"%s %.2f \", \$1, \$2/1073741824}" /sys/fs/cgroup/memory.stat' 2>/dev/null)
    echo "$m host_avail $(free -m | awk 'NR==2{printf "%.1f", $7/1024}')"; sleep 2; done ) > $O/$L.mem.log 2>&1 &
NEEDLE_ROOT="/media/mab776/nvme 2To/models/strata-data/src-01431" timeout 3000 python3 /home/mab776/Documents/portainer/deepseek-v4/needle_ds4.py \
  --url http://127.0.0.1:8330 --lengths $LEN --depths $DEP --timeout 3600 --out $O/$L.needle.json > $O/$L.needles.log 2>&1
touch $O/$L.stop; sleep 3
cat $O/$L.needles.log
awk -v CAP=$CAP '{a=s=k=0; for(i=1;i<NF;i++){if($i=="anon")a=$(i+1); if($i=="shmem")s=$(i+1); if($i=="kernel")k=$(i+1); if($i=="host_avail")h=$(i+1)}
     t=a+s+k; if(t>mx){mx=t; ma=a; ms=s; mk=k}; if(hm==""||h<hm)hm=h}
     END{printf "live: peak anon+shmem+kernel %.2f GiB (anon %.2f, shmem %.2f, kernel %.2f) -> margin %.2f GiB of %d; host avail min %s GiB\n", mx, ma, ms, mk, CAP-mx, CAP, hm}' $O/$L.mem.log
rm -f $O/$L.stop
