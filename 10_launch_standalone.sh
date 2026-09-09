#!/bin/bash
# HOST-side launcher: start the single-node Hy4-preview container on one B300.
#
#   bash 10_launch_standalone.sh                          # MXFP8 TP8, MTP on
#   PROFILE=high-throughput bash 10_launch_standalone.sh  # MXFP8 TP8, MTP off
#   QUANT=bf16 bash 10_launch_standalone.sh               # BF16 TP8, MTP on
#   TP_SIZE=4 bash 10_launch_standalone.sh                # the old TP4 cell, half a node
#
# MoE parallelism (both arms are TP8, so EP=8 is the whole node):
#   EP_SIZE=8 bash 10_launch_standalone.sh                # EP, no all-to-all library
#   A2A_BACKEND=deepep bash 10_launch_standalone.sh       # DeepEP; forces EP = TP
#   DP_ATTN=8 EP_SIZE=8 bash 10_launch_standalone.sh      # + attention DP
#   IMAGE=hy4-preview-v2:latest A2A_BACKEND=deepep_v2 CHUNKED_PREFILL=2048 \
#     bash 10_launch_standalone.sh    # DeepEP v2; PATCHED image, chunk <= V2_CAP.
#                                     # Give the v1 arm the SAME CHUNKED_PREFILL
#                                     # or the pair also differs in chunk size.
#
# Cross-node (TP16 BF16 over EFA -- not a cookbook-verified cell): same command
# on both hosts but NODE_RANK, and DIST_INIT_ADDR is rank 0's IP on both. Only
# rank 0 binds :$PORT.
#   host A: QUANT=bf16 NNODES=2 NODE_RANK=0 TP_SIZE=16 DIST_INIT_ADDR=$A_IP bash 10_launch_standalone.sh
#   host B: QUANT=bf16 NNODES=2 NODE_RANK=1 TP_SIZE=16 DIST_INIT_ADDR=$A_IP bash 10_launch_standalone.sh
#
# Follow with: docker logs -f hy4-preview
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-hy4-preview}"
build_cache_args
require_weights
# Resolves EP_EFF / MOE_ARGS / MOE_TAG and refuses an impossible A2A_BACKEND
# before the ~10 min weight load. The container recomputes the same values from
# the same env vars, so this is a fail-fast gate, not a second source of truth.
build_moe_args
[[ "$A2A_BACKEND" == "deepep" ]] && require_deepep_image "$IMAGE"
# deepep_v2 needs three source patches, one of which is a numerics fix, so
# the image is verified by a marker rather than trusted.
[[ "$A2A_BACKEND" == "deepep_v2" ]] && require_deepep_v2_image "$IMAGE"
# Cross-node, DeepEP has to reach the NIC, and on EFA that means NCCL GIN. The
# thin v2 image is the stock image plus 3 .py files, so it has neither NCCL 2.31
# nor the ofi-nccl plugin -- single-node only.
if (( NNODES > 1 )) && [[ "$A2A_BACKEND" == deepep* ]]; then
    require_gin_capable_image "$IMAGE"
fi
# EVERY cross-node arm needs the OFI plugin, not just the DeepEP ones: plain NCCL
# cannot reach an EFA NIC without it and silently uses TCP over ENA instead. Fills
# EFA_ARGS from the host stack when the image has none. (Cost of not having this,
# measured 2026-09-09 on P5EN-3/4: a TP16 a2a=none arm with 0 bytes on EFA.)
build_efa_args

# Pin the ranks rather than exposing all 8 GPUs even at TP8: the device set then
# belongs to the container's config instead of being an accident of which GPUs
# sglang happened to enumerate, and at TP_SIZE=4 it leaves GPUs 4-7 genuinely
# free for a second instance.
GPUS_PER_NODE=$(( TP_SIZE / NNODES ))
if [[ -z "${GPU_LIST:-}" ]]; then
    GPU_LIST=$(seq -s, 0 $(( GPUS_PER_NODE - 1 )))
fi

docker rm -f "$NAME" 2>/dev/null || true

# ... and only now, with this kit's own container gone, ask whether the GPUs are
# actually free. Order matters: before the rm -f, our own previous run would
# always look like a foreign tenant.
[[ "${ALLOW_BUSY_GPUS:-0}" == "1" ]] || require_free_gpus "$GPU_LIST"

# Variables forwarded ONLY when actually set, because for several of these an
# empty string is not the same as unset: `NCCL_GIN_TYPE=` makes NCCL parse ""
# rather than auto-detect, and export_gin_envs() decides by emptiness too. So a
# blanket `-e VAR="${VAR:-}"` would silently pin every one of these to "".
#   NCCL_GIN_TYPE / _IB_HCA / _SYM_GIN_KERNELS_ENABLE : override export_gin_envs's
#     choice by hand (it resolves them from NNODES + the device list otherwise).
#   EP_JIT_* : DeepEP's own JIT knobs. EP_JIT_PRINT_COMPILER_COMMAND and
#     EP_JIT_DEBUG are the only way to see WHY a JIT build failed -- the C++ side
#     asserts `exit_code == 0` at kernel_runtime.hpp:33 and prints nothing else.
#   EP_JIT_CACHE_DIR : point it at a bind mount to keep cubins across runs, or per
#     image to avoid the header-hash collision (deep_ep hashes includes, not
#     header CONTENT, so two images sharing a cache dir can read each other's
#     cubins and a header-only change measures as a no-op).
PASSTHRU_ARGS=()
#   SGLANG_DEEPGEMM_STANDARD_LAYOUT : auto|masked|compact. THE knob that makes a
#     non-DeepEP arm comparable to a DeepEP one at the kernel level. With a2a=none
#     the standard pre-permute picks the MASKED layout by a memory budget, while
#     every DeepEP-normal arm goes through the COMPACT (ep_scatter + contiguous
#     grouped GEMM) path -- so an a2a=none reference is not a control for the
#     compact path unless it is pinned to `compact` here.
for v in DEEPEP_V2_MODE V2_CAP V1_CAP DISPATCH_DTYPE CUDA_HOME \
         SGLANG_DEEPGEMM_STANDARD_LAYOUT SGLANG_HY4_DBG_SCATTER \
         SGLANG_HY4_DBG_MOE_DUMP \
         NCCL_GIN_TYPE NCCL_IB_HCA NCCL_SYM_GIN_KERNELS_ENABLE \
         NCCL_DEBUG NCCL_DEBUG_SUBSYS \
         EP_JIT_DEBUG EP_JIT_PRINT_COMPILER_COMMAND EP_JIT_CACHE_DIR; do
    [[ -n "${!v:-}" ]] && PASSTHRU_ARGS+=(-e "$v=${!v}")
done

# --net=host: needed at NNODES>1 for EFA/ENA device discovery, and it keeps
#   :$PORT reachable without a published-port hop.
# --device=/dev/infiniband: EFA, and NOT only for multi-node. A DeepEP arm needs
#   it even at NNODES=1: deep_ep asserts that NCCL resolved a GIN backend
#   (csrc/kernels/backend/nccl.cu:87) whether or not anything crosses the wire,
#   and with no device visible NCCL reports GIN type NONE and the assert fires.
#   For non-DeepEP arms it is unused, cheap, and its absence is what makes a later
#   2-node attempt fail with "no network".
# --shm-size=64g: 169 shards / 760GB move through /dev/shm during load.
# --cap-add SYS_NICE: without it every rank logs "User lacks permission to set NUMA
#   affinity, skipping NUMA node configuration for GPU" and runs with whatever
#   NUMA placement it inherits. The K3 kit got this for free from --privileged;
#   this kit does not use --privileged, so ask for the one capability.
# --init: TP leaves unreaped children, and without an init as PID 1 `docker rm -f`
#   fails with "PID ... is zombie and can not be killed" -- which then aborts the
#   NEXT launch under set -e. (Hit on the K3 kit 2026-09-04.)

# --privileged + /dev/gdrdrv, for DeepEP arms only.
#
# GIN type 5 (EFA_GDA) works by having the GPU write a 64-byte WQE and ring the
# NIC's doorbell in MMIO itself (reference_efa_gda_mechanism). Mapping that
# doorbell BAR into the GPU's address space needs more than --device on the uverbs
# nodes, and when it fails NOTHING says "permission": NCCL simply reports the RAIL
# team's GIN type as NONE, and deep_ep aborts at
# csrc/kernels/backend/nccl.cu:101 with text about "a network configuration
# issue". Measured 2026-09-05 on B300-3/4: identical images, identical
# NCCL_GIN_TYPE=5 / _SYM_GIN_KERNELS_ENABLE=0 / _IB_HCA=rdmap, TP16 NCCL comm
# healthy across the two nodes -- and railedGinType NONE until this was added.
# The K3 kit reached the same conclusion from the other direction and made every
# arm privileged so that "a GIN init failure could be a permissions artefact"
# stopped being a live hypothesis.
#
# Note the ASYMMETRY that makes this easy to miss: a single-node DeepEP arm keeps
# its whole a2a on NVLink, so it never touches the doorbell and comes up
# privileged or not. This only bites at NNODES>1, i.e. in the hybrid mode that
# reads props.railedGinType instead of props.ginType.
#
# Scoped to deepep* rather than applied blanket, so the non-DeepEP reference arm
# keeps the smaller capability set and stays comparable to what the cookbook runs.
PRIV_ARGS=()
if [[ "$A2A_BACKEND" == deepep* ]]; then
    PRIV_ARGS+=(--privileged)
    # --privileged DEFEATS --gpus "device=...". Privileged mode bypasses the
    # device cgroup, so every /dev/nvidia* is visible and the container enumerates
    # ALL 8 GPUs regardless of the device list -- measured 2026-09-08 on B300-1:
    # `--privileged --gpus "device=4,5" nvidia-smi -L` lists 8, the same command
    # without --privileged lists 2. Harmless at TP8 (the list is all of them
    # anyway), fatal as soon as the node is shared: a TP4 DeepEP arm asked for
    # GPUs 4-7 put its ranks on physical 0-3 and OOMed on top of the TP4
    # reference arm already loading there (torch reported "GPU 3 ... 34 MiB free"
    # while claiming only 76 GiB for itself -- the missing 191 GiB was the OTHER
    # container). Pin the ordinals by hand, since the cgroup will not.
    PRIV_ARGS+=(-e "CUDA_VISIBLE_DEVICES=$GPU_LIST")
    # GDRCopy. Absent unless the DKMS module is loaded and /dev/gdrdrv was
    # mknod'd by hand (gdrdrv has no udev rule -- feedback_gdrdrv_after_kernel_upgrade);
    # NCCL falls back without it, so this is an optimization, not a requirement.
    [[ -e /dev/gdrdrv ]] && PRIV_ARGS+=(--device=/dev/gdrdrv)
fi

docker run -d --name "$NAME" \
    --init \
    --gpus "\"device=${GPU_LIST}\"" \
    --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_NICE \
    --device=/dev/infiniband \
    ${PRIV_ARGS[@]+"${PRIV_ARGS[@]}"} \
    ${EFA_ARGS[@]+"${EFA_ARGS[@]}"} \
    --shm-size=64g \
    -v "$HOST_MODEL_DIR/$MODEL_DIRNAME:$MODEL_PATH:ro" \
    "${CACHE_ARGS[@]}" \
    -v "$SCRIPT_DIR_HOST:/host/hy4-preview-sglang:ro" \
    -e QUANT="$QUANT" \
    -e PROFILE="$PROFILE" \
    -e SPEC="$SPEC" \
    -e TOPO="${TOPO:-tp${TP_SIZE}x${NNODES}node${MOE_TAG}${NET_TAG}}" \
    -e BENCH_GPUS="${BENCH_GPUS:-$TP_SIZE}" \
    -e TP_SIZE="$TP_SIZE" \
    -e A2A_BACKEND="$A2A_BACKEND" \
    -e EP_SIZE="${EP_SIZE:-}" \
    -e DEEPEP_MODE="$DEEPEP_MODE" \
    -e DP_ATTN="${DP_ATTN:-}" \
    -e ALLOW_UNVALIDATED_A2A="${ALLOW_UNVALIDATED_A2A:-0}" \
    -e V2_CAP="$V2_CAP" \
    -e SPEC_A2A_BACKEND="${SPEC_A2A_BACKEND:-}" \
    "${PASSTHRU_ARGS[@]}" \
    -e MODEL_PATH="$MODEL_PATH" \
    -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    -e CONTEXT_LEN="${CONTEXT_LEN:-}" \
    -e MEM_FRACTION="${MEM_FRACTION:-}" \
    -e MAX_RUNNING="${MAX_RUNNING:-}" \
    -e CHUNKED_PREFILL="${CHUNKED_PREFILL:-}" \
    -e DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}" \
    -e DISABLE_ATTN_TP_GATHER="${DISABLE_ATTN_TP_GATHER:-0}" \
    -e DISABLE_RADIX="${DISABLE_RADIX:-0}" \
    -e NNODES="$NNODES" -e NODE_RANK="$NODE_RANK" \
    -e DIST_INIT_ADDR="$DIST_INIT_ADDR" -e DIST_INIT_PORT="$DIST_INIT_PORT" \
    -e PRIMARY_IFACE="$PRIMARY_IFACE" \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -e PORT="$PORT" \
    --entrypoint bash \
    "$IMAGE" \
    /host/hy4-preview-sglang/start_server.sh

echo "launched '$NAME': quant=$QUANT tp=$TP_SIZE gpus=$GPU_LIST profile=$PROFILE spec=$SPEC image=$IMAGE"
echo "  moe   : a2a=$A2A_BACKEND ep=$EP_EFF moe_tp=$(( TP_SIZE / EP_EFF )) dp_attn=${DP_ATTN:-off}$([[ "$A2A_BACKEND" == "deepep_v2" ]] && echo " v2_cap=$V2_CAP")"
echo "  topo  : ${TOPO:-tp${TP_SIZE}x${NNODES}node${MOE_TAG}${NET_TAG}}"
if (( NNODES > 1 )); then
    echo "  tp=$TP_SIZE over $NNODES nodes, this host is node-rank $NODE_RANK, rendezvous ${DIST_INIT_ADDR}:${DIST_INIT_PORT}"
    echo "  (rank != 0 never binds :$PORT -- do not wait for 'server is fired up' there)"
fi
echo "logs  : docker logs -f $NAME"
echo "health: curl -s localhost:${PORT}/health_generate"
