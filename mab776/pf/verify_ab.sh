#!/usr/bin/env bash
# mab776: verify-pass exactness A/B - one-shot greedy decodes (nsys_decode.sh NSYS=0) of the same prompt; prints each arm's
# token-stream md5, tokens/pass and its first token that differs from the FIRST arm's stream.
#   verify_ab.sh BUILD "label:ENV=V,ENV=V" ...     (an arm's env: comma-separated; NO_DSPARK=1 / DSPARK=... allowed)
set -u
BLD=$1; shift
W=/home/mab776/Documents/portainer/strata-ds41; O=/home/mab776/Documents/portainer/ram-128gb/v41/pf/nsys
BASE="STRATA_SPLIT_FORK=1 STRATA_ENGRAM_AHEAD=1 STRATA_PV_GRAPHS=1 STRATA_ASYNC_FINISH=1 STRATA_D3_PARTS=128 STRATA_WAVE_LAYER=1 STRATA_PRESTAGE_GIB=0"
ref=""
for arm in "$@"; do
  L=${arm%%:*}; envs=${arm#*:}; [ "$envs" = "$arm" ] && envs=""
  sh_env=(); dk_env=()
  IFS=, read -r -a kvs <<< "$envs"
  for kv in "${kvs[@]}"; do [ -z "$kv" ] && continue; case "$kv" in NO_DSPARK=*|DSPARK=*|VMARGIN=*) sh_env+=("$kv");; *) dk_env+=("$kv");; esac; done
  [ "$(curl -s -m 5 localhost:8500/api/status | python3 -c "import sys,json;print(json.load(sys.stdin).get('mode'))")" = off ] || { echo "mode not off - stop"; exit 3; }
  (cd $W && env NSYS=0 ARENA=82.5 "${sh_env[@]}" bash mab776/pf/nsys_decode.sh $L $BLD $BASE "${dk_env[@]}" >/dev/null 2>&1)
  grep -E '^[0-9]+$' $O/$L.log > $O/$L.tok
  [ -z "$ref" ] && ref=$O/$L.tok
  python3 - "$ref" "$O/$L.tok" "$L" "$O/$L.log" <<'PY'
import sys, hashlib, re
a = open(sys.argv[1]).read().split(); b = open(sys.argv[2]).read().split()
md5 = hashlib.md5(("\n".join(b) + "\n").encode()).hexdigest()[:12]
d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
log = open(sys.argv[4]).read()
m = re.search(r"DSpark: (\d+) verify passes, ([\d.]+) accepted drafts/pass = ([\d.]+) tokens/pass", log)
v = re.search(r"verify ([\d.]+) ms/pass", log); dec = re.search(r"^decode: .*= ([\d.]+) tok/s", log, re.M)
err = re.search(r"(nonfinite|non-finite|Aborted|error)", log)
print(f"{sys.argv[3]:14s} n={len(b):5d} md5 {md5}  first diff vs ref: {d if d is not None else '-'}  "
      f"tok/pass {m.group(3) if m else '1'}  verify {v.group(1) if v else '-'} ms  decode {dec.group(1) if dec else '?'} tok/s{'  ERR ' + err.group(1) if err else ''}")
PY
done
