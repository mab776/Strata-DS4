#!/usr/bin/env bash
# mab776 overnight: run a list of variants through run.sh, one throwaway container each (ds41-pf), sequentially.
#   sweep.sh LISTFILE     lines: <label> <cfg.json> [DEPTHS=..] [TURNS=1] [-e K=V ...]   ('#' = comment)
# Stops early (and leaves the GPUs free) if host MemAvailable drops under 8 GiB between variants.
set -u
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf
log() { echo "[$(date +%T)] $*" | tee -a $O/sweep.log; }
while read -r L CFG REST; do
  [ -z "${L:-}" ] && continue; case "$L" in \#*) continue;; esac
  av=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$av" -lt 8 ] && { log "STOP: MemAvailable $av GiB < 8"; break; }
  envs=(); dargs=()
  for x in $REST; do case "$x" in DEPTHS=*|TURNS=*|GEN=*|MEM=*|GPUS=*|DUMP=*|KEEP=*) envs+=("$x");; *) dargs+=("$x");; esac; done
  log "variant $L ($CFG) ${envs[*]:-} ${dargs[*]:-}"
  (cd $W && env "${envs[@]}" CFG=mab776/pf/$CFG bash mab776/pf/run.sh $L "${dargs[@]}") 2>&1 | tail -6 | tee -a $O/sweep.log
done < "$1"
docker rm -f ds41-pf >/dev/null 2>&1
log "sweep $1 done"
