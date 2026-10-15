#!/usr/bin/env bash
# mab776 overnight: q1 (tools, units, profile) then screen 1
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
while kill -0 1439736 2>/dev/null; do sleep 15; done
bash /home/mab776/Documents/portainer/strata-ds41/mab776/pf/sweep.sh /home/mab776/Documents/portainer/strata-ds41/mab776/pf/s1.txt
echo "[$(date +%T)] chain1 done" >> $O/sweep.log
