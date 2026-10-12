# M13 window helpers (sourced): arm configs written over the deploy snapshot's ds4-server.json, restarts of stack 74's
# container, the canonical config restored by restore().
W=/home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work; T=$W/m6reg; U=http://127.0.0.1:8330
LOG="/media/mab776/nvme 2To/models/strata-ds4/logs/ds4-engine.log"
CANON=/home/mab776/Documents/portainer/deepseek-strata/ds4-server.json
RUNCFG="/media/mab776/nvme 2To/models/strata-ds4/run/ds4-server.json"
STOP=$W/m13/STOP
# arm_env 'JSON of env overrides' (a value "" removes the key)
arm_env() { python3 - "$1" "$CANON" "$RUNCFG" <<'PY'
import json, sys
over, canon, run = sys.argv[1:4]
c = json.load(open(canon))
env = dict(c["env"])
for k, v in json.loads(over).items():
    if v == "": env.pop(k, None)
    else: env[k] = v
c["env"] = env
json.dump(c, open(run, "w"), indent=1)
PY
}
up() {
  docker restart strata-ds4 >/dev/null
  for i in $(seq 1 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' -m 60 $U/v1/chat/completions -H 'Content-Type: application/json' \
        -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":1}')" = 200 ] && return 0
    sleep 5
  done
  echo "### server never answered"; docker logs --tail 20 strata-ds4; return 1
}
envline() { docker exec strata-ds4 sh -c 'tr "\0" "\n" < /proc/$(pgrep -f ds4_serve | head -1)/environ' | grep -E '^(DS4|GGML_MMVF|GGML_LI)' | sort | tr '\n' ' '; }
restore() { cp "$CANON" "$RUNCFG"; up; echo "### restored (canonical config) $(date +%T)"; }
# ksweep VAR-label depths seq: one sweep, printed compactly
ksweep() { KS_NONCE=${NONCE:-m13} python3 $T/ksweep2.py $U "$LOG" "$1" "$2" ${3:-512} 2>&1 | python3 -c "
import sys, json
for l in sys.stdin:
    try: x = json.loads(l)
    except Exception: print('  ' + l.rstrip()); continue
    if x.get('i', 0) == 0: print('  depth %6d read at %s tok/s' % (x['depth'], x.get('prefill'))); continue
    print('  depth %6d K=%d decode %5.1f tok/s, %s tok/pass, pass %s ms, md5 %s | GPU0 %s GPU1 %s CPU %s PCIe %s' % (x['depth'], x['k'], x.get('decode', 0), x.get('tok_pass', '?'), x.get('pass_ms', '?'), x.get('md5'), x.get('gpu0'), x.get('gpu1'), x.get('cpu'), x.get('pcie')))"; }
