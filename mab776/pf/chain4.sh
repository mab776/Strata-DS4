#!/usr/bin/env bash
# after chain3 (ppl gate): screen 5
while ps -eo cmd | grep -q "^bash chain3.sh"; do sleep 15; done
bash /home/mab776/Documents/portainer/strata-ds41/mab776/pf/sweep.sh /home/mab776/Documents/portainer/strata-ds41/mab776/pf/s5.txt
echo "[$(date +%T)] chain4 done" >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log
