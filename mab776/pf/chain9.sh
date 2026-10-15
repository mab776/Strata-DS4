#!/usr/bin/env bash
# after chain6 (profile, 127k, ppl): prefill screen 6, then decode (s4), then agent turns (s3)
PF=/home/mab776/Documents/portainer/strata-ds41/mab776/pf
while ps -eo cmd | grep -q "^bash chain6.sh"; do sleep 15; done
bash $PF/sweep2.sh $PF/s6.txt
bash $PF/sweep2.sh $PF/s4.txt
bash $PF/sweep2.sh $PF/s3.txt
echo "[$(date +%T)] chain9 done" >> /home/mab776/Documents/portainer/ram-128gb/v41/pf/sweep.log
