#!/bin/bash
# M9 chain (2026-10-11): wait for the Strata LCB generation (39 answers), grade + compare it with stack 73's run, then
# the SWE paired blocks (strata-coder/bench/swe/m9_swe.sh).  Host-side jobs run in this capped scope.
B=/home/mab776/Documents/portainer/strata-coder/bench; D=$B/ab-gsq-20260926; N=ds4s-m12-32k-t1
cd /home/mab776/Documents/portainer/strata-ds4/bench/m9
say() { echo "== $(date '+%F %T') $*"; }
until [ -f $D/gen-$N.jsonl ] && [ "$(wc -l < $D/gen-$N.jsonl)" -ge 39 ]; do
  pgrep -f "lcb_eval.py gen http://127.0.0.1:8330 $N" >/dev/null || { say "LCB gen not running and < 39 answers - restarting it";
    LCB_DIR=$D LCB_MAXT=32768 LCB_TEMP=1.0 LCB_TOPP=1.0 LCB_TOPK=0 python3 $B/lcb_eval.py gen http://127.0.0.1:8330 $N >> lcb-gen-$N.log 2>&1; }
  sleep 60
done
say "LCB: 39 answers - grading"
LCB_DIR=$D python3 $B/lcb_eval.py grade $N > lcb-grade-$N.log 2>&1
LCB_DIR=$D python3 $B/lcb_eval.py compare ds4-iq2m-32k-t1 $N > lcb-compare-$N-vs-ds4-iq2m-32k-t1.txt 2>&1
tail -6 lcb-compare-$N-vs-ds4-iq2m-32k-t1.txt
say "SWE paired blocks"
/home/mab776/Documents/portainer/strata-coder/bench/swe/m9_swe.sh
say "M9 chain done"
