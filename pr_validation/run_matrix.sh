#!/usr/bin/env bash
# The PR #38609 matrix, on Qwen3-30B-A3B-FP8 (fp8 lets DeepEP run its normal
# mode; the bf16 Qwen1.5-MoE only supports low_latency, which the unquantized
# masked runner then refuses).
#
# Axes, all stamped into every artifact name:
#   hook-present / hook-removed  does the model class define
#                                get_model_config_for_expert_location
#   order-base / order-fixed     is the metadata read before or after the
#                                `ep_dispatch_algorithm is None` early return
#   epalg-none / epalg-static    is an EPLB dispatch algorithm configured
set -u
R=/opt/dlami/nvme/pr38609/run_arm.sh
export MODEL_DIR=/opt/dlami/nvme/models/Qwen3-30B-A3B-FP8
DEEPEP="--moe-a2a-backend deepep --ep-size 2"
S=qwen3moe-fp8_tp2_ep2_a2a-deepep-auto

bash $R wt-e2e              ${S}_hook-present_order-base_epalg-none    $DEEPEP
bash $R wt-e2e-nohook       ${S}_hook-removed_order-base_epalg-none    $DEEPEP
bash $R wt-e2e-fixed        ${S}_hook-present_order-fixed_epalg-none   $DEEPEP
bash $R wt-e2e-nohook-fixed ${S}_hook-removed_order-fixed_epalg-none   $DEEPEP
bash $R wt-e2e              ${S}_hook-present_order-base_epalg-static  $DEEPEP --ep-dispatch-algorithm static
bash $R wt-e2e-fixed        ${S}_hook-present_order-fixed_epalg-static $DEEPEP --ep-dispatch-algorithm static
bash $R wt-e2e-nohook-fixed ${S}_hook-removed_order-fixed_epalg-static $DEEPEP --ep-dispatch-algorithm static
