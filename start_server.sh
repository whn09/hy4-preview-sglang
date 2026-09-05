#!/bin/bash
# In-container: one Hy4-preview server on one B300 node.
#
# ONE script serves all three arms, because they differ by three flags and
# nothing else -- a second copy would drift:
#   PD_ROLE unset    -> the cookbook's verified single-node cell (10_launch_standalone.sh)
#   PD_ROLE=prefill  -> prefill side of a 1P1D pair          (20_launch_prefill.sh)
#   PD_ROLE=decode   -> decode side of a 1P1D pair           (21_launch_decode.sh)
#
# The model path points at local NVMe weights instead of an HF-hub pull.
#
# What is deliberately NOT passed, because the runtime resolves it for HYV4 and a
# flag here would cancel that resolution:
#   --quantization       : read from config.json quantization_config (MXFP8)
#   --attention-backend  : DSA (flashmla_sparse + FP8 indexer cache) is auto-picked
#   --enable-*-cp / --pp : HYV4ForCausalLM rejects both before allocation
#
# MoE parallelism (--ep-size / --moe-a2a-backend / --deepep-mode / DP attention)
# is NOT spelled out here either -- it comes from build_moe_args() in
# env_common.sh, which resolves EP the same way sglang does and refuses the
# backends that cannot run this architecture.
set -euo pipefail

source /host/hy4-preview-sglang/env_common.sh
setup_runtime_env
build_multinode_args
build_pd_args
build_moe_args
# NCCL GIN, which every DeepEP arm needs -- including a single-node one, because
# deep_ep asserts a GIN backend exists even when nothing crosses the wire. Runs
# here rather than in the launcher because /sys/class/infiniband is a property of
# the container, and this is the process that consumes the variables.
export_gin_envs

# ---- MTP (NextN) speculative decoding ----
# One draft layer (model.mtp_layers.0, ~10B params) ships in BOTH checkpoints, so
# there is no separate draft path to point at -- unlike K3's DSPARK, which needs
# --speculative-draft-model-path. steps=3 / topk=1 / draft-tokens=4 is the
# cookbook preset. Note the request budget becomes
# prompt + max_tokens + 4 <= context length while this is on.
#
# Under PD both sides get the same spec args. That is not a guess about which side
# drafts: SGLang resolves the draft geometry from server_args on each side
# independently, and a prefill that does not know about the MTP layer hands the
# decode side a KV layout it cannot verify against. Keep them symmetric, and keep
# them in the results tag -- SPEC is an axis of every PD number too.
SPEC_ARGS=()
if [[ "${SPEC:-on}" == "on" ]]; then
    SPEC_ARGS=(
        --speculative-algorithm NEXTN
        --speculative-num-steps 3
        --speculative-eagle-topk 1
        --speculative-num-draft-tokens 4
    )
fi

# ---- MXFP8 kernel stack ----
# deep_gemm is the validated HYV4 MXFP8 path for both the MoE runner and the
# plain FP8 GEMMs. The runtime also DEFAULTS both to deep_gemm for HYV4, so these
# two flags are documentation as much as configuration -- but they are what the
# verified cell passes, so pass them. They are meaningless on BF16 weights.
GEMM_ARGS=()
if [[ "${QUANT:-mxfp8}" == "mxfp8" ]]; then
    GEMM_ARGS=(--moe-runner-backend deep_gemm --fp8-gemm-backend deep_gemm)
fi

# ---- optional overrides (all unset in the verified cell) ----
CTX_ARGS=();    [[ -n "${CONTEXT_LEN:-}"      ]] && CTX_ARGS=(--context-length "$CONTEXT_LEN")
MEM_ARGS=();    [[ -n "${MEM_FRACTION:-}"     ]] && MEM_ARGS=(--mem-fraction-static "$MEM_FRACTION")
RUN_ARGS=();    [[ -n "${MAX_RUNNING:-}"      ]] && RUN_ARGS=(--max-running-requests "$MAX_RUNNING")
CHUNK_ARGS=();  [[ -n "${CHUNKED_PREFILL:-}"  ]] && CHUNK_ARGS=(--chunked-prefill-size "$CHUNKED_PREFILL")
# Decode CUDA-graph capture is ON by default and long-soak validation of it on
# Hy4 is still open upstream; this is the documented fallback, not a tuning knob.
CG_ARGS=();     [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]] && CG_ARGS=(--disable-cuda-graph)
# Off makes seeded benchmark runs repeatable (no cross-run prefix reuse). Leave it
# ON for real serving.
RADIX_ARGS=();  [[ "${DISABLE_RADIX:-0}" == "1" ]] && RADIX_ARGS=(--disable-radix-cache)

echo "=== Hy4-preview: quant=${QUANT} tp=${TP_SIZE} nodes=${NNODES}/rank${NODE_RANK}" \
     "profile=${PROFILE} spec=${SPEC} ctx=${CONTEXT_LEN:-default}" \
     "pd=${PD_ROLE:-off}${PD_ROLE:+/${TRANSFER_BACKEND:-mooncake}} ==="
# EP/a2a are echoed as what WILL run, not as what was requested: EP_EFF is the
# post-resolution degree (a2a-spanning backends force EP = TP upstream, and this
# kit resolves it the same way so the two never diverge).
echo "    moe: a2a=${A2A_BACKEND} ep=${EP_EFF} moe_tp=$(( TP_SIZE / EP_EFF ))" \
     "deepep_mode=$([[ $A2A_BACKEND == deepep ]] && echo "$DEEPEP_MODE" || echo n/a)" \
     "dp_attn=${DP_ATTN:-off} -> ${MOE_ARGS[*]:-<none, pure TP>}"
if [[ "$A2A_BACKEND" == deepep* ]]; then
    echo "    gin: NCCL_GIN_TYPE=${NCCL_GIN_TYPE:-unset} NCCL_SYM_GIN_KERNELS_ENABLE=${NCCL_SYM_GIN_KERNELS_ENABLE:-unset}" \
         "NCCL_IB_HCA=${NCCL_IB_HCA:-unpinned} v2_cap=${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-n/a}"
fi
if [[ -n "${PD_ROLE:-}" ]]; then
    echo "    MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-} MC_MAX_CONCURRENT_REG_MR=${MC_MAX_CONCURRENT_REG_MR:-}" \
         "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}" \
         "NIXL_BACKEND=${SGLANG_DISAGGREGATION_NIXL_BACKEND:-n/a}"
fi

# `sglang serve` (the 0.5.18 CLI the cookbook uses) rather than
# `python3 -m sglang.launch_server`; same server, and staying on the cookbook's
# entry point means a flag it documents cannot be missing here.
#
# --tp-size, not the cookbook's `--tp`: this build has no `--tp` option at all, it
# only works because argparse accepts unambiguous prefixes. That silently breaks
# the day another `--tp*` flag is added, so spell it out.
exec sglang serve \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --tp-size "$TP_SIZE" \
    "${GEMM_ARGS[@]}" \
    ${MOE_ARGS[@]+"${MOE_ARGS[@]}"} \
    --reasoning-parser auto \
    --tool-call-parser auto \
    "${SPEC_ARGS[@]}" \
    "${CTX_ARGS[@]}" \
    "${MEM_ARGS[@]}" \
    "${RUN_ARGS[@]}" \
    "${CHUNK_ARGS[@]}" \
    "${CG_ARGS[@]}" \
    "${RADIX_ARGS[@]}" \
    ${MULTINODE_ARGS[@]+"${MULTINODE_ARGS[@]}"} \
    ${PD_ARGS[@]+"${PD_ARGS[@]}"} \
    --host 0.0.0.0 \
    --port "$PORT" \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --dist-timeout 7200
