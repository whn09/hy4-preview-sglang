#!/bin/bash
# Capture one sparse layer's MoE boundary tensors from one arm, for offline diff.
#
#   bash 98_moe_dump.sh none     # a2a=none, pinned to the compact layout
#   bash 98_moe_dump.sh deepep   # a2a=deepep normal, bf16 dispatch
#   python3 98_moe_dump_cmp.py results/dumps/moe_dump-hy4-{normal,deepep}-*rank0.pt
#
# Needs the debug image (patches/_dbg_moe_dump.py applied):
#   docker run --name dbgbuild --entrypoint bash -v "$PWD:/host:ro" \
#     hy4-nightly-deepep:latest -c 'python3 /host/patches/_dbg_moe_dump.py' \
#     && docker commit dbgbuild hy4-nightly-deepep-dbg:latest && docker rm dbgbuild
#
# The two arms must differ in the all-to-all and NOTHING else, so both are TP8
# EP8, MTP off, and the a2a=none arm is pinned to the compact grouped-GEMM layout
# -- unpinned it picks the MASKED layout by a memory budget and stops being a
# control for the path DeepEP takes.
set -euo pipefail
cd "$(dirname "$0")"

ARM="${1:-none}"
TAG="${TAG:-hy4}"
IMAGE="${IMAGE:-hy4-nightly-deepep-dbg:latest}"
PORT="${PORT:-30000}"
NAME="hy4-dump-$ARM"
PROMPT="${PROMPT:-The capital of France is}"

# DISABLE_CUDA_GRAPH=1 is MANDATORY here, not a tuning choice. Prefill CUDA
# graphs are on by default ('prefill': {'backend': 'breakable'}), so a served
# prefill REPLAYS a graph captured during startup and runs no Python at all --
# the dump's arm-file trigger can never fire, and the run looks like the
# instrumentation was never applied (measured 2026-09-08: helper present in
# site-packages, env set, /tmp/hy4_dbg_go created, request served, zero dumps).
# The DeepEP arm is eager regardless, and an eager a2a=none arm was separately
# confirmed coherent, so this does not weaken the comparison.
COMMON=(QUANT=mxfp8 TP_SIZE=8 EP_SIZE=8 SPEC_OVERRIDE=off MEM_FRACTION=0.85
        MAX_RUNNING=128 DISABLE_CUDA_GRAPH=1
        "IMAGE=$IMAGE" "NAME=$NAME" "SGLANG_HY4_DBG_MOE_DUMP=$TAG")
case "$ARM" in
    none)   ARM_ENV=(A2A_BACKEND=none SGLANG_DEEPGEMM_STANDARD_LAYOUT=compact) ;;
    # DEEPEP_MODE=normal, not the `auto` default: auto sends decode through the
    # LOW-LATENCY dispatcher, whose masked runner has the same BF16 hole this
    # kit's patch closed in the normal pre-permute -- measured 2026-09-08, the
    # arm dies at deep_gemm.py:718 `act_sf_last = hidden_states_scale.shape[-1]`
    # with hidden_states_scale None. Every serving arm quoted so far was
    # mode=normal, so this also keeps the dump comparable to them.
    deepep) ARM_ENV=(A2A_BACKEND=deepep DISPATCH_DTYPE=bf16 DEEPEP_MODE=normal) ;;
    *) echo "ERROR: ARM must be none or deepep (got '$ARM')" >&2; exit 1 ;;
esac

env "${COMMON[@]}" "${ARM_ENV[@]}" TOPO="moedump-${TAG}-${ARM}" \
    bash 10_launch_standalone.sh

for _ in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "localhost:$PORT/health_generate" || true)
    [[ "$code" == "200" ]] && break
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true \
        || { echo "FATAL: $NAME died"; docker logs "$NAME" 2>&1 | tail -40; exit 1; }
    sleep 10
done
[[ "${code:-}" == "200" ]] || { echo "FATAL: never became healthy"; exit 1; }

# Arm the dump only now. Every forward before this point is warmup or CUDA-graph
# capture on dummy tokens, and the health probe above generates too -- dumping
# either would compare two batches of garbage and look like a clean match.
docker exec "$NAME" touch /tmp/hy4_dbg_go
# Ask the server for its own model id. A wrong one is a 400 that this script
# would otherwise swallow, and a request that never runs a forward produces no
# dump for a reason that has nothing to do with the instrumentation.
MODEL_ID=$(curl -s -m 10 "localhost:$PORT/v1/models" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
# max_tokens=1: one prefill, no decode forwards, so the dumped layer is reached
# exactly once and both arms dump the same batch shape.
RESP=$(curl -s -m 120 "localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"prompt":sys.argv[2],"max_tokens":1,"temperature":0}))' "$MODEL_ID" "$PROMPT")")
grep -q '"text"' <<<"$RESP" || { echo "FATAL: request failed: $RESP"; exit 1; }
sleep 3

mkdir -p results/dumps
docker exec "$NAME" bash -c 'ls /tmp/moe_dump-*.pt' | while read -r f; do
    docker cp "$NAME:$f" "results/dumps/$(basename "$f")"
done
docker logs "$NAME" 2>&1 | grep HY4_DBG_MOE_DUMP | sort -u
ls -la results/dumps/
