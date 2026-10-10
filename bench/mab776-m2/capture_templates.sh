#!/bin/bash
# mab776 M2/M4: byte-exact stack-73 chat renders WITHOUT touching stack 73: a CPU-only twin llama-server from the SAME
# image digest (b11429), same GGUF, same chat flags (only --jinja), GPUs hidden.  llama.cpp MAP_POPULATEs the whole
# 85 GB GGUF on load (no flag avoids it) -> ~1-3 min of NVMe reads under the 16g cgroup cap, nothing else.
#   capture_templates.sh OUTDIR [NEEDLE_TXT_DIR]
# OUTDIR/<conv>.render.json = /apply-template answer for bench/mab776-m2/conv/<conv>.json; props.json = build info;
# NEEDLE_TXT_DIR/needle*.txt (make_prompts.py --write-text output) are re-rendered from their user content and checked
# byte-for-byte + by /tokenize count.
set -uo pipefail
OUT=$(realpath -m "$1"); mkdir -p "$OUT"; ND="${2:-}"
HERE=$(cd "$(dirname "$0")" && pwd)
IMG=ghcr.io/ggml-org/llama.cpp@sha256:b17f82642e3298346e3e88ea822563fc9f04bf5ea0eb1942d6f33f4be6c23a7a
M="/media/mab776/nvme 2To/models/ollama/models/blobs/sha256-a79d3b8c742f240dc947758bbeb46d266e9328ace6a3b71704ab4bf2a8610d68"
docker rm -f ds4-tmpl >/dev/null 2>&1
t0=$(date +%s)
docker run -d --name ds4-tmpl -e NVIDIA_VISIBLE_DEVICES=none --memory 16g --memory-swap 16g -p 127.0.0.1:8331:8080 \
  -v "$M":/models/ds4.gguf:ro --entrypoint /app/llama-server "$IMG" \
  -m /models/ds4.gguf --alias deepseek-v4-flash --host 0.0.0.0 --port 8080 --jinja --parallel 1 \
  -dev none -ngl 0 --load-mode mmap --no-repack --fit off --no-warmup -c 2048 -b 64 -ub 64 --cache-ram 0 -t 4 >/dev/null
for i in $(seq 1 120); do
  curl -fsS -m 3 127.0.0.1:8331/health >/dev/null 2>&1 && break
  docker inspect -f '{{.State.Running}}' ds4-tmpl 2>/dev/null | grep -q true || { echo "twin died:"; docker logs --tail 30 ds4-tmpl; exit 1; }
  sleep 5
done
echo "twin up after $(( $(date +%s) - t0 )) s"
curl -s 127.0.0.1:8331/props > "$OUT/props.json"
python3 -c "import json;d=json.load(open('$OUT/props.json'));print('build_info:',d.get('build_info'))"
for f in "$HERE"/conv/*.json; do
  n=$(basename "$f" .json)
  curl -s 127.0.0.1:8331/apply-template -H 'Content-Type: application/json' --data-binary @"$f" > "$OUT/$n.render.json"
  python3 - "$OUT/$n.render.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d.get("prompt")
print("%-40s %s" % (sys.argv[1].rsplit("/", 1)[-1], ("%d chars" % len(p)) if p is not None else ("ERROR " + json.dumps(d)[:200])))
PY
done
if [ -n "$ND" ]; then
  python3 - "$ND" "$OUT" <<'PY'
import json, pathlib, sys, urllib.request
nd, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
PRE, SUF = "<｜begin▁of▁sentence｜><｜User｜>", "<｜Assistant｜><think>"
def post(path, body):
    r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8331" + path, json.dumps(body).encode(),
                                                      {"Content-Type": "application/json"}), timeout=600)
    return json.loads(r.read())
for f in sorted(nd.glob("needle*.txt")):
    full = f.read_bytes().decode("utf-8")
    assert full.startswith(PRE) and full.endswith(SUF), f
    user = full[len(PRE):-len(SUF)]
    render = post("/apply-template", {"messages": [{"role": "user", "content": user}]})["prompt"]
    toks = post("/tokenize", {"content": render, "add_special": True, "parse_special": True})["tokens"]
    (out / (f.stem + ".render.txt")).write_bytes(render.encode("utf-8"))
    print("%s: render == ours %s (%d vs %d chars), llama.cpp /tokenize %d ids" %
          (f.stem, render == full, len(render), len(full), len(toks)))
PY
fi
docker stop ds4-tmpl >/dev/null; docker rm ds4-tmpl >/dev/null
echo "done in $(( $(date +%s) - t0 )) s"
