#!/usr/bin/env bash
# mab776: a deploy candidate's gate in a throwaway container (mab776/pf/tmp-cfg/<cfg>.json): the 8.7k / 123k bench +
# turns (run.sh), tools probe, needles at 246k x 3 depths, the default reasoning effort (a request naming none -> the
# config's level, "high" -> 75; read from the engine's text log), a regenerate.  Prints a summary.
#   cand_run.sh LABEL CFG
set -u
L=$1; C=$2
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf; V=/home/mab776/Documents/portainer/ram-128gb/v41
U=http://127.0.0.1:8332
(cd $W && CFG=mab776/pf/tmp-cfg/$C.json GPUS=0,1 GEN=256 TURNS=1 KEEP=1 bash mab776/pf/run.sh $L \
   -v /home/mab776/v41-engram/park-test:/home/mab776/v41-engram/park-test) || { echo "$L: run.sh failed"; exit 1; }
python3 $V/tools_probe.py $U > $O/$L.tools.log 2>&1; echo "tools: $(tail -1 $O/$L.tools.log)"
for e in none high; do
  b='{"messages":[{"role":"user","content":"Say OK."}],"max_tokens":4,"temperature":0,"stream":false'
  [ $e = high ] && b="$b"',"reasoning_effort":"high"'
  curl -s -m 300 $U/v1/chat/completions -H 'Content-Type: application/json' -d "$b}" >/dev/null
  echo "effort $e -> $(docker exec ds41-pf sh -c 'grep -o "Reasoning Effort: [0-9]*" /tmp/strata-ds41-text.log | tail -1')"
done
NEEDLE_ROOT="/media/mab776/nvme 2To/models/strata-data/src-01431" timeout 3000 python3 /home/mab776/Documents/portainer/deepseek-v4/needle_ds4.py \
  --url $U --lengths 246k --depths 10,50,90 --timeout 3600 --out $O/$L.needles.json > $O/$L.needles.log 2>&1
echo "needles: $(grep -E 'found' $O/$L.needles.log | tail -1)"
REGEN_OUT= timeout 900 python3 $W/mab776/pf/regen_test.py $U $L > $O/$L.regen.jsonl 2>&1; echo "regen: $(cut -c1-160 $O/$L.regen.jsonl | tail -3 | tr '\n' ' ')"
docker exec ds41-pf sh -c 'grep -E "error|ERR|nonfinite|illegal|park" /tmp/strata-ds41.log | tail -5'
docker exec ds41-pf cat /tmp/strata-ds41.log > $O/$L.engine.log 2>/dev/null
docker rm -f ds41-pf >/dev/null
echo "$L gate done"
