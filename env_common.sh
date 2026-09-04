#!/bin/bash
# Shared config for all Hy4-preview scripts. Sourced by the host launcher AND by
# the in-container start script, so keep it side-effect free (no docker calls, no
# python) above the function definitions.
#
# Reference: SGLang cookbook "Hy4 preview"
#   docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx
#   docs/src/snippets/configs/tencent/hy4-preview.jsx   (the verified cells)
# Every default below is one of that config's *verified* B300 cells; the knobs
# that are NOT in a verified cell (DeepEP, DP-attention, PD, HiCache) are
# deliberately absent from this kit rather than half-wired.

# ---- paths (host) ----
NVME="${NVME:-/opt/dlami/nvme}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-$NVME/models}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$NVME/cache}"
SCRIPT_DIR_HOST="${SCRIPT_DIR_HOST:-/home/ubuntu/hy4-preview-sglang}"

# The cookbook's own image. Unlike the K3 kit there is no local build: Hy4 needs
# no patches, no Mooncake wheel and no DeepEP -- the verified recipes are pure TP
# on one node, so the stock image IS the tested artefact. Pin the digest below
# once a measurement campaign starts: `hy4-preview` is a moving tag, and a
# re-pull would silently change what a results/ log refers to.
IMAGE="${IMAGE:-lmsysorg/sglang:hy4-preview}"

# ---- quantization picks the topology, not just the weights ----
# 770B total / 49B active. MXFP8 ~760GB -> ~190GB/rank at TP4; BF16 ~1.5TB ->
# ~190GB/rank at TP8. Both fit a B300 (288GB) node; on 141/192GB parts the BF16
# arm is a 2-node TP16 recipe instead (not our hardware).
#
# MXFP8 needs SM100+. It self-describes through the checkpoint's ModelOpt
# hf_quant_config (UE8M0 group-32 weight scales, dynamic activations), so there
# is deliberately no --quantization flag anywhere in this kit -- passing one
# would override the checkpoint's own recipe.
QUANT="${QUANT:-mxfp8}"
case "$QUANT" in
    mxfp8)
        MODEL_REPO="tencent/Hy4-preview-FP8"
        MODEL_DIRNAME="Hy4-preview-FP8"
        DEFAULT_TP=4
        ;;
    bf16)
        MODEL_REPO="tencent/Hy4-preview"
        MODEL_DIRNAME="Hy4-preview"
        DEFAULT_TP=8
        ;;
    *)
        echo "ERROR: QUANT must be mxfp8 or bf16 (got '$QUANT')" >&2; exit 1 ;;
esac

TP_SIZE="${TP_SIZE:-$DEFAULT_TP}"
MODEL_PATH="${MODEL_PATH:-/models/$MODEL_DIRNAME}"
# Serve under the HF repo id even though the weights are local: the benchmark and
# the OpenAI clients in the cookbook address the model by that name, and
# bench_serving's --model is matched against this string.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_REPO}"

# ---- serving profile ----
# The cookbook's two operating points differ ONLY in whether the built-in MTP
# (NextN) draft layer is used:
#   low-latency     : NEXTN steps=3 topk=1 draft-tokens=4  (best at bs=1)
#   high-throughput : no speculation (at saturation draft+verify costs more than
#                     it saves)
# Anything else -- attention backend, quantization, CUDA-graph capture -- is
# auto-resolved by the runtime for HYV4 and must stay unset.
PROFILE="${PROFILE:-low-latency}"
case "$PROFILE" in
    low-latency)     SPEC=on  ;;
    high-throughput) SPEC=off ;;
    *) echo "ERROR: PROFILE must be low-latency or high-throughput (got '$PROFILE')" >&2; exit 1 ;;
esac
# Explicit SPEC= wins over the profile, for a spec-on/spec-off A/B at one profile.
SPEC="${SPEC_OVERRIDE:-$SPEC}"

PORT="${PORT:-30000}"

# ---- PD disaggregation (1P1D) ----
# NOT a cookbook-verified cell -- the cookbook only publishes single-node TP
# recipes. This is our own arm, and everything about it is stated rather than
# implied.
#
# Topology lives in ONE place, 22_launch_router.sh, which takes any number of
# --prefill/--decode peers; the 20_/21_ launchers do not know how many peers
# exist. Defaults are the 1P1D pair the user asked for: B300-1 prefills,
# B300-2 decodes. Private (ENA) IPs -- these change on a stop/start, so read
# them from `hostname -I` rather than trusting this line after a restart.
PREFILL_IPS="${PREFILL_IPS:-172.31.30.164}"
DECODE_IPS="${DECODE_IPS:-172.31.28.101}"
# The prefill side's KV-handshake port. Distinct from $PORT (HTTP): the router
# gets both, and the decode side connects to this one directly.
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ROUTER_PORT="${ROUTER_PORT:-8000}"

# mooncake (default) or nixl. Both can do EFA on this hardware, but only from
# the image built by ./Dockerfile:
#   mooncake -> needs the ...-efa-... wheel. The stock image's
#               mooncake-transfer-engine-cuda13 0.3.12.post1 has NO EFA
#               transport at all (engine.so does not link libfabric; its
#               transports are RdmaTransport/TcpTransport/IbgdaDeviceTransport),
#               and its ibverbs path segfaults on EFA, so it silently ends up on
#               TcpTransport over ENA.
#   nixl     -> libplugin_LIBFABRIC.so IS in the stock image, but the bundled
#               libfabric is too old for it ("version `FABRIC_1.7' not found"),
#               and mounting the host's fails on rdma-core skew
#               ("libefa.so.1: version `EFA_1.7' not found"). With this image's
#               EFA installer it loads, creates a backend, and registers GPU
#               memory -- verified on B300-1, 2026-09-04.
# require_efa_image() below refuses to launch the mooncake arm on an image whose
# wheel cannot do EFA, because that failure mode is a slow success, not an error.
TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}"

# The PD image, which is NOT the image the single-node arms run. See Dockerfile.
PD_IMAGE="${PD_IMAGE:-hy4-preview-efa:latest}"

# ---- multi-node (H200/B200 BF16 shape; optional on B300) ----
# Single node covers every verified B300 cell, so this stays at 1. It exists
# because a 2-node TP16 BF16 arm is the only way to put Hy4's TP all-reduce on
# EFA, which is the interesting cross-node measurement on this hardware.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
DIST_INIT_ADDR="${DIST_INIT_ADDR:-}"
DIST_INIT_PORT="${DIST_INIT_PORT:-29500}"

# Primary (ENA) interface, for GLOO/NCCL bootstrap under NNODES>1. Resolved on
# whichever side sources this file; harmless at NNODES=1.
PRIMARY_IFACE="${PRIMARY_IFACE:-$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -1)}"

# ---- JIT / autotune caches to persist on the host ----
# deep_gemm is BOTH the MoE runner and the FP8 GEMM backend here, so a cold
# container JITs the whole MoE stack before the first token. Mount these or every
# launch pays it again (and a "cold JIT" stall reads exactly like a hang -- see
# the K3 kit's TileLang note).
CACHE_MOUNTS=(
    "/root/.cache/deep_gemm=deep_gemm"
    "/root/.cache/flashinfer=flashinfer"
    "/root/.cache/torch=torch"
    "/root/.cache/tvm-ffi=tvm-ffi"
    "/root/.cache/sglang=sglang"
    "/root/.triton=triton"
    "/root/.nv/ComputeCache=nv_compute"
)

# Fills CACHE_ARGS with docker -v flags, creating the host dirs first.
build_cache_args() {
    local entry cpath hsub
    CACHE_ARGS=()
    for entry in "${CACHE_MOUNTS[@]}" "$@"; do
        cpath="${entry%%=*}"; hsub="${entry#*=}"
        mkdir -p "$HOST_CACHE_DIR/$hsub"
        CACHE_ARGS+=(-v "$HOST_CACHE_DIR/$hsub:$cpath")
    done
}

# Fills GDR_ARGS with the /dev/gdrdrv device flag when the host has the gdrdrv
# module loaded. Mooncake's EFA path registers device memory and expects GDRCopy;
# without the device it logs "Failed to open gdr handle" and falls back to a
# slower host-memory path rather than failing. Conditional because the module is
# NOT loaded on every host -- DKMS has to be rebuilt after a kernel upgrade and
# there is no udev rule to recreate the node -- and a missing --device makes
# `docker run` fail outright.
build_gdr_args() {
    GDR_ARGS=()
    if [[ -c /dev/gdrdrv ]]; then
        GDR_ARGS+=(--device=/dev/gdrdrv)
    else
        echo "WARN: /dev/gdrdrv missing -- Mooncake/NCCL will run without GDRCopy." >&2
        echo "      Check: lsmod | grep gdrdrv ; sudo mknod /dev/gdrdrv c \$(awk '/gdrdrv/{print \$1}' /proc/devices) 0" >&2
    fi
}

# ---- in-container runtime env ----
setup_runtime_env() {
    export PYTHONUNBUFFERED=1
    # ~760GB / 169 shards (1.5TB BF16): the default load timeout is not enough.
    export SGLANG_LOAD_TIMEOUT="${SGLANG_LOAD_TIMEOUT:-7200}"

    if (( NNODES > 1 )) || [[ -n "${PD_ROLE:-}" ]]; then
        # p6-b300 exposes 18 EFA HCAs plus the ENA NIC. Exclude loopback and the
        # docker bridge or NCCL bootstraps on 172.17.x and never connects.
        export FI_PROVIDER="${FI_PROVIDER:-efa}"
        export FI_EFA_USE_DEVICE_RDMA=1
        export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker}"
        export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$PRIMARY_IFACE}"
    fi
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

    # ---- PD-only: KV transfer over EFA ----
    if [[ -n "${PD_ROLE:-}" ]]; then
        # Mooncake's protocol selector. "efa" is not the default; without it the
        # engine picks its ibverbs RDMA path, which on EFA hardware segfaults.
        export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-efa}"

        # Bounds Mooncake's MR-registration thread fan-out. FEWER threads is
        # FASTER here, because registration serializes on the EFA provider's
        # per-domain lock -- measured on K3's 1376 buffers: 128 -> 130.1 s,
        # 32 -> 51.0 s, 8 -> 20.7 s, 4 -> 24.8 s. Hy4 registers a different
        # number of buffers (MLA + a separate FP8 indexer pool), so re-sweep
        # before quoting 8 as tuned rather than inherited.
        export MC_MAX_CONCURRENT_REG_MR="${MC_MAX_CONCURRENT_REG_MR:-8}"

        # THIS ONE IS NOT OPTIONAL. With expandable_segments:True every KV block
        # transfer fails:
        #   efa_context.cpp:1175] fi_read/fi_write failed: Invalid argument
        # and the only user-visible symptom is "mooncake session ... is not
        # alive", because one failure blacklists the session. Registration is
        # fine (zero fi_mr_regattr failures, all rails up) -- the problem is that
        # expandable_segments backs a tensor with a growable VA reservation whose
        # physical mapping is remapped as it grows, so the dmabuf-derived MR
        # registered at startup no longer describes the memory at transfer time
        # and libfabric rejects the RMA with EINVAL. Measured 2026-09-03 on
        # 2x p6-b300; transfer_engine_bench does 105.43 GB/s at the same block
        # size between the same two nodes, so it is not EFA.
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

        # NIXL alternative (TRANSFER_BACKEND=nixl). SGLang reads the plugin name
        # from this env (srt/disaggregation/nixl/conn.py); its default is UCX,
        # which has no EFA support and would quietly run over TCP.
        if [[ "${TRANSFER_BACKEND:-mooncake}" == "nixl" ]]; then
            export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-LIBFABRIC}"
        fi
    fi
}

# Fills PD_ARGS. Empty when PD_ROLE is unset, so the same start script serves the
# single-node and the disaggregated arms.
build_pd_args() {
    PD_ARGS=()
    [[ -z "${PD_ROLE:-}" ]] && return 0
    case "$PD_ROLE" in
        prefill|decode) : ;;
        *) echo "ERROR: PD_ROLE must be prefill or decode (got '$PD_ROLE')" >&2; exit 1 ;;
    esac
    PD_ARGS=(
        --disaggregation-mode "$PD_ROLE"
        --disaggregation-transfer-backend "${TRANSFER_BACKEND:-mooncake}"
        --disaggregation-bootstrap-port "${BOOTSTRAP_PORT:-8998}"
    )
    # --disaggregation-ib-device is deliberately NOT set: its default (None)
    # triggers Mooncake's own device detection, which on this host enumerates the
    # EFA rdmap devices. Pass IB_DEVICE= only to pin a subset, and note that the
    # flag's help text is written for mlx5_* names.
    [[ -n "${IB_DEVICE:-}" ]] && PD_ARGS+=(--disaggregation-ib-device "$IB_DEVICE")
    return 0
}

# Fills MULTINODE_ARGS. Fails fast rather than hanging for --dist-timeout when
# the rendezvous address is missing or TP does not divide over the nodes.
build_multinode_args() {
    MULTINODE_ARGS=()
    (( NNODES <= 1 )) && return 0
    if [[ -z "$DIST_INIT_ADDR" ]]; then
        echo "ERROR: NNODES=$NNODES needs DIST_INIT_ADDR (rank 0's IP, same value on every node)" >&2
        exit 1
    fi
    if (( TP_SIZE % NNODES != 0 )); then
        echo "ERROR: TP_SIZE=$TP_SIZE does not divide over NNODES=$NNODES" >&2
        exit 1
    fi
    MULTINODE_ARGS=(
        --nnodes "$NNODES"
        --node-rank "$NODE_RANK"
        --dist-init-addr "${DIST_INIT_ADDR}:${DIST_INIT_PORT}"
    )
}

# Fails the launch when the weights are absent instead of letting sglang treat
# /models/... as an HF repo id 20 minutes later. Host-side only.
require_weights() {
    local dir="$HOST_MODEL_DIR/$MODEL_DIRNAME"
    if [[ ! -f "$dir/config.json" ]]; then
        echo "ERROR: no weights at $dir (run: bash 00_download_models.sh $QUANT)" >&2
        exit 1
    fi
    # A truncated download is the failure mode that costs an hour: the shard
    # count comes from the index, so compare it against what is on disk.
    local want have
    want=$(python3 -c 'import json,sys,collections;print(len(set(json.load(open(sys.argv[1]))["weight_map"].values())))' \
            "$dir/model.safetensors.index.json" 2>/dev/null || echo 0)
    have=$(ls "$dir"/*.safetensors 2>/dev/null | wc -l | tr -d ' ')
    if (( want > 0 && have != want )); then
        echo "ERROR: $dir has $have/$want safetensors shards -- download incomplete." >&2
        exit 1
    fi
}

# One env var out of a RUNNING container's config -- the only honest source for a
# benchmark tag, because PROFILE/QUANT are env vars and never appear in the
# server's own `server_args=` line. Reading this shell's variables instead would
# label a spec-off server "low-latency" whenever 91_bench.sh is invoked without
# the same PROFILE= that launched it.
# Assert the image really has an EFA-capable Mooncake BEFORE a 10-minute weight
# load. This gate exists because the failure it catches is a slow success, not an
# error: a non-EFA wheel passes SGLang's PD warmup and then either kills the first
# real request or quietly runs the whole benchmark over TcpTransport.
#
# Three things in the K3 original were wrong and every one of them made the gate
# LIE; they are fixed here, so do not "simplify" any of them back:
#   1. The extension is `engine.so`, NOT `engine.cpython-<abi>.so` -- that name
#      came from the old cmake source build, and the glob matched nothing, so
#      `strings` errored on a literal asterisk and a known-good image was
#      rejected as "no EFA support".
#   2. `grep -ciE efa` is vacuous: "efa" is a substring of "d-efa-ult". It returns
#      389 on the NON-EFA wheel, i.e. it PASSES on exactly the build this check
#      exists to reject. The discriminator is that the EFA build links libfabric
#      and carries its entry points (ibverbs symbols are in both wheels).
#   3. `bindCudaContextIfNeeded` was our FORK's function name; the upstream wheel
#      carrying the same fix does not use it. Grep for the driver-API calls the
#      fix is made of instead.
require_efa_image() {
    local img="$1"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "ERROR: image '$img' not found. Build it first:" >&2
        echo "  docker build -t hy4-preview-efa:latest -f Dockerfile ." >&2
        exit 1
    fi
    local rc=0 out
    #   4. This container gets NO --gpus, so libcuda.so.1 is not injected. Locating
    #      the package with `import mooncake` can therefore fail on a PERFECTLY GOOD
    #      image (the extension load needs the driver), which the case-arms below
    #      would report as NO_ENGINE_SO. find_spec locates without executing, and
    #      every check that follows is ldd/strings -- nothing needs a driver.
    out=$(docker run --rm --entrypoint bash "$img" -c '
        set -eu
        MC=$(python3 -c "import importlib.util as u;print(list(u.find_spec(\"mooncake\").submodule_search_locations)[0])")
        E="$MC/engine.so"
        test -f "$E"                                              || { echo "NO_ENGINE_SO $MC"; exit 1; }
        ldd "$E" | grep -q libfabric                              || { echo "NO_LIBFABRIC";    exit 1; }
        test "$(strings "$E" | grep -cE "^fi_(getinfo|mr_reg)" || true)" -ge 2 \
                                                                  || { echo "NO_FI_SYMS";      exit 1; }
        strings "$E" | grep -q cuDevicePrimaryCtxRetain           || { echo "NO_CTXFIX";       exit 1; }
        strings "$E" | grep -q MC_MAX_CONCURRENT_REG_MR           || { echo "NO_REGCAP";       exit 1; }
        echo OK
    ' 2>&1) || rc=$?
    case "$out" in
        *OK*) : ;;
        *NO_ENGINE_SO*)
            echo "ERROR: '$img' has no mooncake engine.so; the wheel layout changed." >&2
            echo "       $out" >&2; exit 1 ;;
        *NO_LIBFABRIC*)
            echo "ERROR: mooncake in '$img' does not link libfabric -- this is the" >&2
            echo "       NON-EFA wheel (the stock hy4-preview image ships exactly" >&2
            echo "       that). Build ./Dockerfile, or set TRANSFER_BACKEND=nixl." >&2
            exit 1 ;;
        *NO_FI_SYMS*)
            echo "ERROR: mooncake in '$img' links libfabric but carries no libfabric" >&2
            echo "       entry points. Suspect a stripped or partial build." >&2; exit 1 ;;
        *NO_CTXFIX*)
            echo "ERROR: mooncake in '$img' lacks the GPU-MR CUDA-context fix." >&2
            echo "       Without it GPU KV registration fails 'Operation not" >&2
            echo "       supported', startup and PD warmup still PASS, and the" >&2
            echo "       FIRST REAL REQUEST dies." >&2; exit 1 ;;
        *NO_REGCAP*)
            echo "ERROR: mooncake in '$img' has no MC_MAX_CONCURRENT_REG_MR, so the" >&2
            echo "       MR-registration fan-out is unbounded (138 s of startup)." >&2; exit 1 ;;
        *)
            echo "ERROR: mooncake EFA check on '$img' failed (rc=$rc):" >&2
            echo "       $out" >&2; exit 1 ;;
    esac
}

read_cenv() {
    local name="$1" var="$2" fallback="${3:-}" v
    v=$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$name" 2>/dev/null \
        | grep "^${var}=" | head -1 | cut -d= -f2-)
    echo "${v:-${fallback:-unknown}}"
}
