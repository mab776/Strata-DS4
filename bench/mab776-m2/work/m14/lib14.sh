# M14 window helpers (sourced): the M13 kit (lib.sh) + arm configs that may also set --pf-b, and an env line that
# shows the ggml switch too.
. /home/mab776/Documents/portainer/strata-ds4/bench/mab776-m2/work/m13/lib.sh
STOP=$W/m14/STOP
# arm 'JSON of env overrides' [pf-b value]  (a value "" removes the key; no pf-b = the canonical one)
arm() { python3 - "$1" "${2:-}" "$CANON" "$RUNCFG" <<'PY'
import json, sys
over, pfb, canon, run = sys.argv[1:5]
c = json.load(open(canon))
env = dict(c["env"])
for k, v in json.loads(over).items():
    if v == "": env.pop(k, None)
    else: env[k] = v
c["env"] = env
if pfb:
    a = c["args"]; a[a.index("--pf-b") + 1] = pfb
json.dump(c, open(run, "w"), indent=1)
PY
}
envline() { docker exec strata-ds4 sh -c 'p=$(pgrep -f ds4_serve | head -1); tr "\0" "\n" < /proc/$p/environ | grep -E "^(DS4|GGML_MMVF|GGML_LI|GGML_CUDA)" | sort | tr "\n" " "; tr "\0" " " < /proc/$p/cmdline | grep -o -- "--pf-b [0-9.]*"'; }
