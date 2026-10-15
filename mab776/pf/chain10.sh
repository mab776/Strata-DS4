#!/usr/bin/env bash
# mab776 overnight: 127k of the best, decode screen, agent turns (cores disabled in the test containers since 03:05)
PF=/home/mab776/Documents/portainer/strata-ds41/mab776/pf
bash $PF/sweep2.sh $PF/s7.txt
bash $PF/sweep2.sh $PF/s4.txt
bash $PF/sweep2.sh $PF/s3.txt
echo "[$(date +%T)] chain10 done" >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log
