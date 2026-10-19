#!/usr/bin/env bash
# mab776 overnight: KL of variant A's dumped prompt-tail logits vs B's (DUMP=1 runs: req001 = the fixed-nonce 8k prompt)
#   kl.sh A B [req001]
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf/dump; R=${3:-req001}
python3 /home/mab776/Documents/portainer/strata-ds41/tools/v41/logits_kl.py $O/$1/$R.tail.f32 $O/$2/$R.tail.f32 2>&1 | tail -4
