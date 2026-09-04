#!/bin/bash
# HOST-side launcher: start the single-node Hy4-preview container on one B300.
#
#   bash 10_launch_standalone.sh                          # MXFP8 TP4, MTP on
#   PROFILE=high-throughput bash 10_launch_standalone.sh  # MXFP8 TP4, MTP off
#   QUANT=bf16 bash 10_launch_standalone.sh               # BF16 TP8, MTP on
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

# TP4 uses half the node, so pin the ranks instead of exposing all 8 GPUs: it
# leaves 4-7 genuinely free (a second instance, or a K3 arm) and it makes the
# device set part of the container's config rather than an accident of which
# GPUs sglang happened to enumerate.
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
    -e TOPO="${TOPO:-tp${TP_SIZE}x${NNODES}node}" \
    -e BENCH_GPUS="${BENCH_GPUS:-$TP_SIZE}" \
    -e TP_SIZE="$TP_SIZE" \
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
if (( NNODES > 1 )); then
    echo "  tp=$TP_SIZE over $NNODES nodes, this host is node-rank $NODE_RANK, rendezvous ${DIST_INIT_ADDR}:${DIST_INIT_PORT}"
    echo "  (rank != 0 never binds :$PORT -- do not wait for 'server is fired up' there)"
fi
echo "logs  : docker logs -f $NAME"
echo "health: curl -s localhost:${PORT}/health_generate"
