#!/bin/bash
# M9 end (Marc 10-11 22:40): release the GPUs as soon as the SWE agent phase is done - grading is CPU/docker only.
# Mode page "off" stops stack 74 (and every other managed GPU service); other sessions take the GPUs afterwards.
S=/home/mab776/Documents/portainer/strata-coder/bench/swe/night-m9-1011.log; L=/home/mab776/Documents/portainer/strata-ds4/bench/m9/m9_chain.log
n0=$(grep -c 'agent phase done' $S)
until [ "$(grep -c 'agent phase done' $S)" -gt "$n0" ]; do sleep 30; done
curl -s -m 600 -X POST localhost:8500/api/mode/off > /dev/null
sleep 20; echo "== $(date '+%F %T') GPUs released: mode page -> off (strata-ds4 $(docker inspect -f '{{.State.Status}}' strata-ds4))" >> $L
