#!/usr/bin/env bash
# mab776 (10-24): ds41-engine.sh for the ISTA-DASLab GSQ-RCO throwaway test. Diffs from the served wrapper:
#  - DS41_BUILD=build-dd            -> the post-merge (Xolotl 281afdad) engine build directory
#  - STRATA_DSPARK_KEEP dropped     -> the expert-counts keep list is a GSQ-model artifact; unmeasured for ISTA,
#                                      default behaviour instead (re-add with a measured file if the keep list wins)
#  - STRATA_SERVE_LOOKUP(_GREEDY)   -> the SERVE-side key the merged engine reads (the served config's STRATA_LOOKUP=14
#                                      maps to nothing in this tree - fix the served config at the next stack update)
set -u
cd /home/mab776/Documents/portainer/strata-ds4
export GGML_NO_BACKTRACE=1
export STRATA_NO_LARGEPAGES="${STRATA_NO_LARGEPAGES:-1}"
export STRATA_SERVE_STEP_MAX="${STRATA_SERVE_STEP_MAX:-96}"
export STRATA_SERVE_PASS_BIAS="${STRATA_SERVE_PASS_BIAS:-0}"
export STRATA_SERVE_DSPARK="${STRATA_SERVE_DSPARK-/media/mab776/nvme 2To/models/deepseek-v41/gsq/V41-MTP-Vision.gguf}"
export STRATA_DSPARK_DEPTH="${STRATA_DSPARK_DEPTH:-4}"
export STRATA_DSPARK_CONF="${STRATA_DSPARK_CONF:-0.6}"
[ -z "$STRATA_SERVE_DSPARK" ] && unset STRATA_SERVE_DSPARK
exec "$PWD/${DS41_BUILD:-build-dd}/ds41_generate" "$@"
