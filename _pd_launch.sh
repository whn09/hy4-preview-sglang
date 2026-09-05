#!/bin/bash
# HOST-side launcher shared by 20_launch_prefill.sh and 21_launch_decode.sh.
# Not an entry point -- it needs PD_ROLE and NAME from its caller.
#
# Deliberately ONE file instead of the K3 kit's two near-identical launchers: the
# prefill and decode containers differ only in --disaggregation-mode and which
# host they run on, and the K3 pair has already drifted (one grew a retry loop
# around `docker rm -f`, the other did not).
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

: "${PD_ROLE:?_pd_launch.sh needs PD_ROLE}"
: "${NAME:?_pd_launch.sh needs NAME}"

# The PD arms run the EFA image, not the cookbook's. See Dockerfile for why the
# stock image cannot move a KV cache over EFA.
IMAGE="$PD_IMAGE"
[[ "${TRANSFER_BACKEND:-mooncake}" == "mooncake" ]] && require_efa_image "$IMAGE"

build_cache_args
build_gdr_args
require_weights
# EP/a2a resolution + fail-fast. The two sides of a pair must match on this as
# well as on TRANSFER_BACKEND and SPEC: the MoE geometry is per-server, but a
# prefill and a decode with different expert layouts produce KV the other side
# cannot verify against, and nothing negotiates it at handshake time.
build_moe_args
[[ "$A2A_BACKEND" == "deepep" ]] && require_deepep_image "$IMAGE"
# deepep_v2 needs three source patches, one of which is a numerics fix, so
# the image is verified by a marker rather than trusted.
[[ "$A2A_BACKEND" == "deepep_v2" ]] && require_deepep_v2_image "$IMAGE"

# Pin the ranks rather than exposing all 8 GPUs: the device set becomes part of
# the container's config instead of an accident of enumeration order, and at
# TP_SIZE=4 GPUs 4-7 stay genuinely free.
GPUS_PER_NODE=$(( TP_SIZE / NNODES ))
if [[ -z "${GPU_LIST:-}" ]]; then
    GPU_LIST=$(seq -s, 0 $(( GPUS_PER_NODE - 1 )))
fi

# Unmapping ~760 GB of model volumes can outlast a single `rm -f`, leaving the
# container Exited-but-present and the next `docker run` failing with "container
# name already in use". Retry until the name is actually free -- and retry the
# REMOVAL, not an inspect, because inspect succeeds on an Exited container.
for _ in $(seq 1 60); do
    docker inspect "$NAME" >/dev/null 2>&1 || break
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    sleep 2
done
if docker inspect "$NAME" >/dev/null 2>&1; then
    echo "ERROR: could not remove existing container '$NAME'" >&2
    exit 1
fi

# Foreign tenant on these GPUs? A PD arm is the worst place to find out, because
# the prefill node can load fine while the decode node OOMs and the only visible
# symptom is a router that says the peer is "not accepting connections".
[[ "${ALLOW_BUSY_GPUS:-0}" == "1" ]] || require_free_gpus "$GPU_LIST"

# TOPO is the benchmark's topology axis, read back out of the container by
# 91_bench.sh. BENCH_GPUS is the tok/s/GPU denominator and it is NOT $TP_SIZE
# here: a 1P1D pair occupies TP_SIZE GPUs on each of two hosts. Getting this
# wrong is how a PD arm gets credited with double its real per-GPU throughput.
TOPO="${TOPO:-pd1p1d-tp${TP_SIZE}-${TRANSFER_BACKEND:-mooncake}${MOE_TAG}}"
BENCH_GPUS="${BENCH_GPUS:-$(( TP_SIZE * 2 ))}"

docker run -d --name "$NAME" \
    --init \
    --gpus "\"device=${GPU_LIST}\"" \
    --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --cap-add SYS_NICE \
    --device=/dev/infiniband ${GDR_ARGS[@]+"${GDR_ARGS[@]}"} \
    --shm-size=64g \
    -v "$HOST_MODEL_DIR/$MODEL_DIRNAME:$MODEL_PATH:ro" \
    "${CACHE_ARGS[@]}" \
    -v "$SCRIPT_DIR_HOST:/host/hy4-preview-sglang:ro" \
    -e PD_ROLE="$PD_ROLE" \
    -e TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}" \
    -e BOOTSTRAP_PORT="$BOOTSTRAP_PORT" \
    -e IB_DEVICE="${IB_DEVICE:-}" \
    ${MC_MAX_CONCURRENT_REG_MR+-e MC_MAX_CONCURRENT_REG_MR} \
    ${MOONCAKE_PROTOCOL+-e MOONCAKE_PROTOCOL} \
    ${PYTORCH_CUDA_ALLOC_CONF+-e PYTORCH_CUDA_ALLOC_CONF} \
    -e QUANT="$QUANT" \
    -e PROFILE="$PROFILE" \
    -e SPEC="$SPEC" \
    -e TOPO="$TOPO" \
    -e BENCH_GPUS="$BENCH_GPUS" \
    -e TP_SIZE="$TP_SIZE" \
    -e A2A_BACKEND="$A2A_BACKEND" \
    -e EP_SIZE="${EP_SIZE:-}" \
    -e DEEPEP_MODE="$DEEPEP_MODE" \
    -e DP_ATTN="${DP_ATTN:-}" \
    -e ALLOW_UNVALIDATED_A2A="${ALLOW_UNVALIDATED_A2A:-0}" \
    -e V2_CAP="$V2_CAP" \
    -e SPEC_A2A_BACKEND="${SPEC_A2A_BACKEND:-}" \
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

echo "launched '$NAME': role=$PD_ROLE backend=${TRANSFER_BACKEND:-mooncake} quant=$QUANT tp=$TP_SIZE gpus=$GPU_LIST spec=$SPEC"
echo "  image : $IMAGE"
echo "  moe   : a2a=$A2A_BACKEND ep=$EP_EFF moe_tp=$(( TP_SIZE / EP_EFF )) dp_attn=${DP_ATTN:-off}$([[ "$A2A_BACKEND" == "deepep_v2" ]] && echo " v2_cap=$V2_CAP")"
echo "  topo  : $TOPO   (tok/s/GPU denominator = $BENCH_GPUS)"
echo "  http  : 0.0.0.0:${PORT}   bootstrap: ${BOOTSTRAP_PORT}"
echo "  logs  : docker logs -f $NAME"
