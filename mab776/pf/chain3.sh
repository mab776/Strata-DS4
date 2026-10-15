#!/usr/bin/env bash
# after q2 (screens 1 + 2): the ppl gate
while ps -eo cmd | grep -q "^bash q2.sh"; do sleep 15; done
bash /home/mab776/Documents/portainer/strata-ds41/mab776/pf/ppl.sh /home/mab776/Documents/portainer/strata-ds41/mab776/pf/p1.txt
echo "[$(date +%T)] chain3 done" >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log
