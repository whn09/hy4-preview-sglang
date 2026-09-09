#!/bin/bash
# Run 96_logprob_parity.py's two arms on ONE node, half a node each.
#
#   bash 99_parity_pair.sh                       # ref a2a=none vs test a2a=deepep
#   TEST_A2A=deepep_v2 bash 99_parity_pair.sh    # ... vs deepep_v2 (patched image)
#
# 96_logprob_parity.py wants two live endpoints, and its documented invocation
# (`--ref B300-4:30000 --test B300-3:30000`) is two hosts. With one B300 available
# the only way to hold both arms up at once is TP4 on GPUs 0-3 and TP4 on GPUs
# 4-7 -- the "half a node" cell 10_launch_standalone.sh already documents. TP4 is
# not the serving shape, so this gates CORRECTNESS only; never quote throughput
# from these two containers, they are sharing a node's NVLink and host bandwidth.
#
# Both arms are pure EP at their own TP, MTP off, and the reference is pinned to
# SGLANG_DEEPGEMM_STANDARD_LAYOUT=compact -- unpinned it picks the MASKED
# grouped-GEMM layout by a memory budget and stops being a control for the path
# every DeepEP arm takes.
set -euo pipefail
cd "$(dirname "$0")"

TP="${TP:-4}"
TEST_A2A="${TEST_A2A:-deepep}"
REF_PORT="${REF_PORT:-30000}"
TEST_PORT="${TEST_PORT:-30001}"
IMAGE="${IMAGE:-hy4-nightly-deepep:latest}"
# The fix under test. See the DISABLE_ATTN_TP_GATHER block in start_server.sh:
# without it, a2a!=none + DP-attention-off makes the scheduler hand the MoE an
# attn-TP SHARD-local num_token_non_padded while hunyuan_v4.py hands it the full
# padded width, and the gate mask throws away all but 1/attn_tp_size of the
# batch. Set to 0 to reproduce the failure.
ATG="${DISABLE_ATTN_TP_GATHER:-1}"
# NOISE FLOOR CONTROL. `TEST_A2A=none TEST_CG_OFF=1 bash 99_parity_pair.sh` runs
# the SAME config on both halves of the node and differs only in the axes a DeepEP
# arm cannot avoid differing in: the GPU set (0-3 vs 4-7) and CUDA graphs
# (DEEPEP_MODE=normal disables capture, so the DeepEP arm is always eager while
# the reference replays graphs). Without that number, a nonzero |dlogprob| on the
# DeepEP arm cannot be attributed -- 96_logprob_parity.py's 0.05 tolerance is a
# tolerance, not a measurement, and reduction order alone can exceed it.
CG_OFF=(); [[ "${TEST_CG_OFF:-0}" == "1" ]] && CG_OFF=(DISABLE_CUDA_GRAPH=1)

COMMON=(QUANT=mxfp8 SPEC_OVERRIDE=off "TP_SIZE=$TP" "EP_SIZE=$TP"
        MEM_FRACTION=0.85 MAX_RUNNING=64
        # Both arms MUST agree on this: 96_logprob_parity.py enforces
        # chunked_prefill_size, and an unset value resolves differently per arm.
        CHUNKED_PREFILL=2048
        "IMAGE=$IMAGE")

# Reuse an arm that is already up and answering. Each arm costs ~10 min of weight
# load, and iterating on the TEST arm (a flag, a patch) should not re-pay for the
# reference. REUSE=0 forces both to be rebuilt.
already_up() {
    [[ "${REUSE:-1}" == "1" ]] || return 1
    docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -q true || return 1
    [[ "$(curl -s -o /dev/null -w '%{http_code}' -m 5 "localhost:$2/health_generate" || true)" == "200" ]]
}

if already_up hy4-parity-ref "$REF_PORT"; then
    echo "=== ref arm: reusing the healthy hy4-parity-ref on :$REF_PORT"
else
echo "=== ref arm: a2a=none tp=$TP gpus 0-$((TP-1)) port $REF_PORT"
env "${COMMON[@]}" A2A_BACKEND=none \
    SGLANG_DEEPGEMM_STANDARD_LAYOUT=compact \
    GPU_LIST="$(seq -s, 0 $((TP-1)))" \
    NAME=hy4-parity-ref PORT="$REF_PORT" TOPO="parity-ref-none-tp$TP" \
    bash 10_launch_standalone.sh
fi

# The dispatch axes, defaulted to the pair that is known to SERVE and pinned so the
# filename carries them. They are not free choices:
#   TEST_DISPATCH=bf16 is the only granularity matching [1,32] weights, but bf16
#     carries no activation scale, so the masked (low-latency) runner dies at
#     moe_runner/deep_gemm.py:718 -- hence TEST_MODE=normal.
#   TEST_DISPATCH=fp8 + TEST_MODE=auto is the FAST arm (graphs, low-latency decode)
#     and the only one that could ever serve deepep_v2, whose dispatch is masked-only.
#     It was condemned as "fluent nonsense" at 0/4 -- but that was measured BEFORE
#     blocker 7 was known, so it conflates an fp8 scale misread with the 7/8-token
#     drop. Re-run it with ATG=1 before believing either verdict.
TEST_MODE="${TEST_MODE:-normal}"
TEST_DISPATCH="${TEST_DISPATCH:-bf16}"

echo "=== test arm: a2a=$TEST_A2A tp=$TP gpus $TP-$((2*TP-1)) port $TEST_PORT atg=$ATG mode=$TEST_MODE disp=$TEST_DISPATCH"
env "${COMMON[@]}" A2A_BACKEND="$TEST_A2A" DEEPEP_MODE="$TEST_MODE" DISPATCH_DTYPE="$TEST_DISPATCH" \
    DISABLE_ATTN_TP_GATHER="$ATG" ${CG_OFF[@]+"${CG_OFF[@]}"} \
    SGLANG_DEEPGEMM_STANDARD_LAYOUT="${TEST_LAYOUT:-compact}" \
    GPU_LIST="$(seq -s, "$TP" $((2*TP-1)))" \
    NAME=hy4-parity-test PORT="$TEST_PORT" TOPO="parity-test-$TEST_A2A-tp$TP-$TEST_MODE-$TEST_DISPATCH" \
    bash 10_launch_standalone.sh

for pair in "hy4-parity-ref:$REF_PORT" "hy4-parity-test:$TEST_PORT"; do
    name="${pair%:*}"; port="${pair#*:}"
    for _ in $(seq 1 120); do
        code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "localhost:$port/health_generate" || true)
        [[ "$code" == "200" ]] && break
        docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true \
            || { echo "FATAL: $name died"; docker logs "$name" 2>&1 | tail -40; exit 1; }
        sleep 10
    done
    [[ "${code:-}" == "200" ]] || { echo "FATAL: $name never became healthy"; exit 1; }
    echo "$name healthy on :$port"
done

mkdir -p results
OUT="results/parity-tp${TP}-none_vs_${TEST_A2A}-atg${ATG}-mode${TEST_MODE}-disp${TEST_DISPATCH}-cgoff${TEST_CG_OFF:-0}.txt"
NF=(); [[ "$TEST_A2A" == none ]] && NF=(--noise-floor)
python3 96_logprob_parity.py --ref "localhost:$REF_PORT" --test "localhost:$TEST_PORT" \
    ${NF[@]+"${NF[@]}"} 2>&1 | tee "$OUT"
echo "wrote $OUT"
