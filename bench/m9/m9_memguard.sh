#!/bin/bash
# M9 memory guard (10-11 21:10, after systemd-oomd killed the claude-portainer RC tmux scope at 20:46:53 while stack 73's
# llama-server grew to ~51 GB during its SWE block): log MemAvailable + full PSI every 30 s; if available < 2.5 GiB or
# full avg10 > 20 %, interrupt the running SWE issue and stop the chain at the next boundary (STOP-M9).
L=/home/mab776/Documents/portainer/strata-ds4/bench/m9/memguard.log
while systemctl --user -q is-active m9-chain4; do
  a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo); p=$(awk -F'[= ]' '/^full/{print int($3)}' /proc/pressure/memory)
  echo "$(date +%T) avail ${a}M psi ${p}" >> $L
  if [ $a -lt 2560 ] || [ $p -gt 20 ]; then
    echo "$(date +%T) CRITICAL - STOP-M9 + INT to mini-extra" >> $L
    touch /home/mab776/Documents/portainer/strata-coder/bench/swe/STOP-M9
    for pid in $(pgrep -f 'bin/mini-extra swebench'); do kill -INT $pid; done; exit 1
  fi
  sleep 30
done
