#!/usr/bin/env bash
# mab776 overnight: screen 2 right after chain1 (screen 1)
while pgrep -f "bash chain1.sh" >/dev/null; do sleep 15; done
bash /home/mab776/Documents/portainer/strata-ds41/mab776/pf/sweep.sh /home/mab776/Documents/portainer/strata-ds41/mab776/pf/s2.txt
echo "[$(date +%T)] chain2 done" >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log
