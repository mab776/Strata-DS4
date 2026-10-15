#!/usr/bin/env bash
# after chain7: screen 6
PF=/home/mab776/Documents/portainer/strata-ds41/mab776/pf
while ps -eo cmd | grep -q "^bash chain7.sh"; do sleep 15; done
bash $PF/sweep2.sh $PF/s6.txt
echo "[$(date +%T)] chain8 done" >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log
