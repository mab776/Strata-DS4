#!/usr/bin/env bash
# mab776 overnight: make a measured variant the served config.  Backs up mab776/strata-ds41.json and mab776/ds41-engine.sh
# as .bak-20261015-<tag>, then writes the variant's json args into strata-ds41.json and its env flags as wrapper defaults.
#   apply.sh <tag> <variant cfg.json in mab776/pf> "ENV=V ENV=V ..."
set -eu
W=/home/mab776/Documents/portainer/strata-ds41/mab776
TAG=$1; CFG=$2; ENVS=$3
cp -n $W/strata-ds41.json $W/strata-ds41.json.bak-20261015-$TAG
cp -n $W/ds41-engine.sh $W/ds41-engine.sh.bak-20261015-$TAG
python3 - "$W/pf/$CFG" "$W/strata-ds41.json" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
v, c = json.load(open(src)), json.load(open(dst))
c["args"] = v["args"]
json.dump(c, open(dst, "w"), indent=1)
print("args:", " ".join(c["args"][c["args"].index("--ctx"):]))
PY
python3 - "$W/ds41-engine.sh" "$ENVS" <<'PY'
import sys
p, envs = sys.argv[1], sys.argv[2].split()
s = open(p).read()
marker = "# mab776 overnight 2026-10-15: measured defaults (each = an opt-in engine path; set =0 / unset to go back)\n"
block = marker + "".join(f'export {k}="${{{k}:-{v}}}"\n' for k, v in (e.split("=", 1) for e in envs))
if marker in s:   # replace a previous block
    a = s.index(marker); b = s.index("exec ", a)
    s = s[:a] + block + s[b:]
else:
    a = s.index('exec "$PWD')
    s = s[:a] + block + s[a:]
open(p, "w").write(s)
print(block)
PY
