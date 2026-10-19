#!/usr/bin/env bash
# mab776 overnight: build the second-drive blob mirror of the file tier (the coldest experts by rank.bin) on the ROOT drive
# (WD SN7100; the models drive is the Samsung 990 EVO) - upstream's --build-mirror, then --check-mirror (byte compare).
# Waits for screen 1's last (repeat-baseline) variant to start, so only that run's timing is disturbed.
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
G="/media/mab776/nvme 2To/models/deepseek-v41/gsq"
until grep -q "variant n0b-base" $O/sweep.log; do sleep 10; done
echo "[$(date +%T)] mirror build start" >> $O/sweep.log
docker run --rm --name ds41-mirror --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=none --memory 6g --memory-swap 6g \
  -v "$G:$G:ro" -v $W:$W strata-ds4:cu13 \
  $W/build-n3/ds41_generate -m "$G/V41-GSQ.gguf" --rank "$G/rank.bin" --build-mirror $W/mab776/mirror/V41-GSQ.experts.mirror --mirror-gib 38 \
  > $O/mirror-build.log 2>&1
echo "[$(date +%T)] mirror build rc=$? $(tail -1 $O/mirror-build.log)" >> $O/sweep.log
docker run --rm --name ds41-mirror --runtime nvidia -e NVIDIA_VISIBLE_DEVICES=none --memory 6g --memory-swap 6g \
  -v "$G:$G:ro" -v $W:$W:ro strata-ds4:cu13 \
  $W/build-n3/ds41_generate -m "$G/V41-GSQ.gguf" --check-mirror $W/mab776/mirror/V41-GSQ.experts.mirror > $O/mirror-check.log 2>&1
echo "[$(date +%T)] mirror check rc=$? $(tail -1 $O/mirror-check.log)" >> $O/sweep.log
