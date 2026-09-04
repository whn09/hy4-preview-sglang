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

# Pin the ranks rather than exposing all 8 GPUs even at TP8: the device set then
# belongs to the container's config instead of being an accident of which GPUs
# sglang happened to enumerate, and at TP_SIZE=4 it leaves GPUs 4-7 genuinely
# free for a second instance.
GPUS_PER_NODE=$(( TP_SIZE / NNODES ))
if [[ -z "${GPU_LIST:-}" ]]; then
    GPU_LIST=$(seq -s, 0 $(( GPUS_PER_NODE - 1 )))
fi

docker rm -f "$NAME" 2>/dev/null || true

# --net=host: needed at NNODES>1 for EFA/ENA device discovery, and it keeps
#   :$PORT reachable without a published-port hop.
# --device=/dev/infiniband: EFA. Unused at NNODES=1, cheap, and its absence is
#   what makes a later 2-node attempt fail with "no network".
# --shm-size=64g: 169 shards / 760GB move through /dev/shm during load.
# --cap-add SYS_NICE: without it every rank logs "User lacks permission to set NUMA
#   affinity, skipping NUMA node configuration for GPU" and runs with whatever
#   NUMA placement it inherits. The K3 kit got this for free from --privileged;
#   this kit does not use --privileged, so ask for the one capability.
# --init: TP leaves unreaped children, and without an init as PID 1 `docker rm -f`
#   fails with "PID ... is zombie and can not be killed" -- which then aborts the
#   NEXT launch under set -e. (Hit on the K3 kit 2026-09-04.)
docker run -d --name "$NAME" \
    --init \
    --gpus "\"device=${GPU_LIST}\"" \
    --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_NICE \
    --device=/dev/infiniband \
    --shm-size=64g \
    -v "$HOST_MODEL_DIR/$MODEL_DIRNAME:$MODEL_PATH:ro" \
    "${CACHE_ARGS[@]}" \
    -v "$SCRIPT_DIR_HOST:/host/hy4-preview-sglang:ro" \
    -e QUANT="$QUANT" \
    -e PROFILE="$PROFILE" \
    -e SPEC="$SPEC" \
    -e TOPO="${TOPO:-tp${TP_SIZE}x${NNODES}node${MOE_TAG}}" \
    -e BENCH_GPUS="${BENCH_GPUS:-$TP_SIZE}" \
    -e TP_SIZE="$TP_SIZE" \
    -e A2A_BACKEND="$A2A_BACKEND" \
    -e EP_SIZE="${EP_SIZE:-}" \
    -e DEEPEP_MODE="$DEEPEP_MODE" \
    -e DP_ATTN="${DP_ATTN:-}" \
    -e ALLOW_UNVALIDATED_A2A="${ALLOW_UNVALIDATED_A2A:-0}" \
    -e V2_CAP="$V2_CAP" \
    -e MODEL_PATH="$MODEL_PATH" \
    -e SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
    -e CONTEXT_LEN="${CONTEXT_LEN:-}" \
    -e MEM_FRACTION="${MEM_FRACTION:-}" \
    -e MAX_RUNNING="${MAX_RUNNING:-}" \
    -e CHUNKED_PREFILL="${CHUNKED_PREFILL:-}" \
    -e DISABLE_CUDA_GRAPH="${DISABLE_CUDA_GRAPH:-0}" \
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
echo "  topo  : ${TOPO:-tp${TP_SIZE}x${NNODES}node${MOE_TAG}}"
if (( NNODES > 1 )); then
    echo "  tp=$TP_SIZE over $NNODES nodes, this host is node-rank $NODE_RANK, rendezvous ${DIST_INIT_ADDR}:${DIST_INIT_PORT}"
    echo "  (rank != 0 never binds :$PORT -- do not wait for 'server is fired up' there)"
fi
echo "logs  : docker logs -f $NAME"
echo "health: curl -s localhost:${PORT}/health_generate"
