#!/bin/bash
# mab776 M4: the adversarial checks behind the M4 commit, re-runnable (CPU only, 2 CPUs, no GPU, no network).
#   DS4_TOK=<DS4 pack tokenizer dir> QWEN_TOK=<Qwen pack tokenizer dir> bench/mab776-m4/verify.sh
# 1. render_check.py   the 14 stack-73 captures through the normalizer + DeepSeekTemplate (13 equal, c06b by design)
# 2. fuzz_parser.py    DSML parser: random chunkings vs whole text, round trips, real DS4 token boundaries (FUZZ_N)
# 3. qwen_compare.py   Qwen-visible outputs of this tree vs mab776-main (expect only openai_ids/pua_with_tags to differ)
# 4. literal_timing.py literal-tag cost old vs new at ~100k / ~250k tokens (DS4 and Qwen vocabularies; same ids)
set -eu
R=$(cd "$(dirname "$0")/../.." && pwd)
DS4_TOK=${DS4_TOK:-$R/bench/mab776-m2/work/ds4tok/tokenizer}
QWEN_TOK=${QWEN_TOK:?set QWEN_TOK to a Qwen pack tokenizer directory}
OLD=$(mktemp -d); trap 'rm -rf "$OLD"' EXIT
git -C "$R" archive mab776-main serve tools/strata_tokenizer.py tools/gguf_reader.py tools/llama_unicode_classes.py | tar -x -C "$OLD"
run() { docker run --rm --network none -e NVIDIA_VISIBLE_DEVICES=none --cpus 2 --cpu-shares 64 --memory 6g \
          -u "$(id -u):$(id -g)" -e HOME=/tmp -e STRATA_DS4_TOKENIZER=/tok -e FUZZ_N="${FUZZ_N:-3000}" \
          -v "$R":/src:ro -v "$DS4_TOK":/tok:ro -v "$QWEN_TOK":/qtok:ro -v "$OLD":/old:ro -w /tmp \
          strata-ds4-pytest:py312 "$@"; }
run python -B /src/bench/mab776-m4/render_check.py
run python -B /src/bench/mab776-m4/fuzz_parser.py | tail -1
for v in old src; do run python -B /src/bench/mab776-m4/qwen_compare.py /$v /qtok | sed -n '/^{/,$p' > "$OLD/qwen_$v.json"; done
python3 -c "import json,sys; a,b=(json.load(open(sys.argv[i])) for i in (1,2)); print('qwen differs:', sorted(k for k in a if a[k]!=b.get(k)))" "$OLD/qwen_old.json" "$OLD/qwen_src.json"
for vocab in /tok /qtok; do for v in old src; do
  echo "$vocab $v: $(run python -B /src/bench/mab776-m4/literal_timing.py /$v $vocab | tail -1 | python3 -c 'import json,sys; d=json.load(sys.stdin); print({k: {f: d[k][f] for f in ("tokens", "ids", "mark_ms", "unmark_ms", "encode_prompt_ms")} for k in ("104k", "256k")})')"
done; done
