#!/usr/bin/env bash
# mab776 overnight: KL of every dumped variant's klgate prompt (req001) vs a reference variant (default n0-base)
O=/home/mab776/Documents/portainer/ram-128gb/v41/pf/dump; REF=${1:-n0-base}; shift
for d in $O/*/; do
  v=$(basename $d); [ "$v" = "$REF" ] && continue
  [ -n "${1:-}" ] && ! echo "$v" | grep -qE "$1" && continue
  [ -f $d/req001.tail.f32 ] || continue
  echo "$v vs $REF: $(python3 /home/mab776/Documents/portainer/strata-ds41/tools/v41/logits_kl.py $O/$REF/req001.tail.f32 $d/req001.tail.f32 2>&1 | tail -1)"
done
