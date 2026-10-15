#!/usr/bin/env bash
# mab776: upstream's tools/v41/serve/ds41-engine.sh for a container: same env defaults, minus flock / systemd-run /
# the ComfyUI check (the container's --memory is the RAM cap). Paths = the host's, mounted at the same place.
set -u
cd /home/mab776/Documents/portainer/strata-ds41
export GGML_NO_BACKTRACE=1
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
# DSpark: upstream uses a Q2_K requant of the sidecar (24 GB card); we load the original (MXFP4 experts)
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-/media/mab776/nvme 2To/models/deepseek-v41/gsq/V41-MTP-Vision.gguf}"
export STRATA_DSPARK_KEEP="${STRATA_DSPARK_KEEP:-$PWD/tools/v41/serve/dspark_expert_counts.txt:64}"
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
[ -z "$STRATA_SERVE_DSPARK" ] && unset STRATA_SERVE_DSPARK
exec "$PWD/${DS41_BUILD:-build-ds4-gpu}/ds41_generate" "$@"
