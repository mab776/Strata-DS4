#!/bin/bash
# mab776: sample a container's cgroup memory, GPU 0/1 VRAM and host MemAvailable once a second until the container
# exits; print the peaks (and the minimum MemAvailable) as one JSON line.   usage: peaks.sh CONTAINER [OUT.json]
c="$1"; out="${2:-/dev/stdout}"
id=$(docker inspect -f '{{.Id}}' "$c" 2>/dev/null) || { echo '{"error":"no container"}' > "$out"; exit 1; }
cg=/sys/fs/cgroup/system.slice/docker-$id.scope
peak_mem=0; peak_g0=0; peak_g1=0; min_avail=999999999
while docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null | grep -q true; do
  m=$(cat "$cg/memory.current" 2>/dev/null || echo 0)
  read g0 g1 < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')
  a=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ "$m" -gt "$peak_mem" ] && peak_mem=$m
  [ "${g0:-0}" -gt "$peak_g0" ] && peak_g0=$g0
  [ "${g1:-0}" -gt "$peak_g1" ] && peak_g1=$g1
  [ "$a" -lt "$min_avail" ] && min_avail=$a
  sleep 1
done
printf '{"peak_cgroup_gib":%.2f,"peak_gpu0_mib":%d,"peak_gpu1_mib":%d,"min_host_avail_gib":%.2f}\n' \
  "$(echo "$peak_mem/1073741824" | bc -l)" "$peak_g0" "$peak_g1" "$(echo "$min_avail/1048576" | bc -l)" > "$out"
