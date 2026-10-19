#!/usr/bin/env bash
# mab776 overnight: make a measured variant the served config.  Backs up mab776/strata-ds41.json as
# .bak-20261015-<tag>, then writes the variant's engine args and its env flags (the config's "env": the Strata server
# hands it to the engine, after the container's environment) into strata-ds41.json.
#   apply.sh <tag> <variant cfg.json in mab776/pf> "ENV=V ENV=V ..."
set -eu
W=/home/mab776/Documents/portainer/strata-ds41/mab776
TAG=$1; CFG=$2; ENVS=$3
cp -n $W/strata-ds41.json $W/strata-ds41.json.bak-20261015-$TAG
python3 - "$W/pf/$CFG" "$W/strata-ds41.json" "$ENVS" <<'PY'
import json, sys
src, dst, envs = sys.argv[1], sys.argv[2], sys.argv[3].split()
v, c = json.load(open(src)), json.load(open(dst))
c["args"] = v["args"]
c["env"] = dict(e.split("=", 1) for e in envs)
c["_comment_mab776"] = ("2026-10-15 overnight (ram-128gb/v41/OVERNIGHT-REPORT.md): --prompt-batch 32768 + the env block = "
                        "opt-in engine paths, each measured; drop a key (or set it to 0) to go back to the old path. "
                        "STRATA_BLOB_MIRROR needs mab776/mirror/V41-GSQ.experts.mirror (rebuild: see the report).")
json.dump(c, open(dst, "w"), indent=1)
print("args:", " ".join(c["args"][c["args"].index("--ctx"):]))
print("env:", json.dumps(c["env"], indent=1))
PY
