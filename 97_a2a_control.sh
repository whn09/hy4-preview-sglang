#!/bin/bash
# Does DeepEP-normal produce the same tokens as no all-to-all, on a model that is
# NOT Hy4?
#
#   bash 97_a2a_control.sh none
#   bash 97_a2a_control.sh deepep
#   diff results/a2a_control-<model>-none.txt results/a2a_control-<model>-deepep.txt
#
# WHY THIS EXISTS. Every DeepEP arm on the MXFP8 Hy4 checkpoint fails the logprob
# gate (96_logprob_parity.py: 0/4, worst top-5 |dlogprob| 7.48), and by 2026-09-08
# the obvious explanations are eliminated: the compact/ep_scatter grouped-GEMM
# chain is coherent at group 32 with a2a=none (SGLANG_DEEPGEMM_STANDARD_LAYOUT=
# compact), eager mode is coherent (DISABLE_CUDA_GRAPH=1), and the scatter places
# every assignment exactly once (patches/_dbg_scatter_stats.py: recv=6,
# placed=6/48 per rank x 8 ranks = 48, non-local marked -1, ids local 0..31).
#
# That leaves two very different possibilities, and they need opposite work:
#   * DeepEP-normal is broken for THIS image/build in general  -> not our patches,
#     and no Hy4 DeepEP number can exist until the image is fixed;
#   * it is specific to Hy4/MXFP8              -> our patch set is incomplete.
# A second model separates them. Qwen3-30B-A3B-FP8 is a block-128 FP8 MoE, so
# quant_info.use_mxfp8 is False and mr_dg_normal.diff's act_gran_k resolves to
# 128 -- byte-identical to upstream. Any divergence here is upstream's.
#
# Deliberately NOT built on env_common.sh: that file resolves Hy4 paths, Hy4
# quantization and the Hy4 profile, and a control whose job is to be a different
# model should not inherit them.
set -euo pipefail
cd "$(dirname "$0")"

A2A="${1:-none}"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/models/Qwen3-30B-A3B-FP8}"
IMAGE="${IMAGE:-hy4-nightly-deepep:latest}"
TP="${TP:-8}"
PORT="${PORT:-30000}"
# fp8 is upstream's default dispatch payload; the Hy4 arm is forced onto bf16
# because there is no MXFP8 dispatch on CUDA. That makes dispatch dtype an axis,
# and an untagged rerun would overwrite the arm it is meant to be compared with
# (feedback_stamp_every_variable_in_filenames).
DISPATCH_DTYPE="${DISPATCH_DTYPE:-fp8}"
NAME="a2a-control-$A2A"
MODEL_TAG="$(basename "$MODEL_DIR")"
OUT="results/a2a_control-${MODEL_TAG}-${A2A}$([[ "$A2A" == none ]] || echo "-disp${DISPATCH_DTYPE}").txt"

test -d "$MODEL_DIR" || { echo "FATAL: no weights at $MODEL_DIR" >&2; exit 1; }

# Pure EP on BOTH arms, so the only difference is the all-to-all library.
# It is also the only legal shape here: Qwen3-30B-A3B has moe_intermediate_size
# 768, and MoE-TP over 8 ranks gives 96, which is not a multiple of the FP8
# weight_block_size_n=128 -- measured 2026-09-08, moe_ep_setup.py:144 refuses the
# ep-less arm outright. A model whose intermediate size survives /8 would launch
# and quietly compare EP+DeepEP against MoE-TP, two different kernels.
A2A_ARGS=(--ep-size "$TP")
if [[ "$A2A" != "none" ]]; then
    # deepep-mode normal on purpose: that is the mode the Hy4 arm runs, and the
    # low-latency mode goes through a different pre-permute entirely.
    A2A_ARGS+=(--moe-a2a-backend "$A2A" --deepep-mode normal
               --deepep-dispatcher-output-dtype "$DISPATCH_DTYPE")
fi

# Same reason as the Hy4 arms: with a2a=none the standard pre-permute picks the
# MASKED grouped-GEMM layout by a memory budget while DeepEP-normal always takes
# the COMPACT one, so an unpinned reference is not a control for the compact path.
LAYOUT="${SGLANG_DEEPGEMM_STANDARD_LAYOUT:-compact}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" --init \
    --gpus "\"device=$(seq -s, 0 $((TP-1)))\"" \
    --net=host --ipc=host --privileged \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband --shm-size=32g \
    -v "$MODEL_DIR:/model:ro" \
    -e SGLANG_DEEPGEMM_STANDARD_LAYOUT="$LAYOUT" \
    --entrypoint python3 "$IMAGE" \
    -m sglang.launch_server \
    --model-path /model \
    --tp-size "$TP" \
    --host 0.0.0.0 --port "$PORT" \
    --mem-fraction-static 0.80 \
    "${A2A_ARGS[@]}" >/dev/null

echo "launched $NAME (a2a=$A2A tp=$TP model=$MODEL_TAG image=$IMAGE)"
for _ in $(seq 1 90); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "localhost:$PORT/health_generate" || true)
    [[ "$code" == "200" ]] && break
    docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -q true \
        || { echo "FATAL: $NAME died before serving"; docker logs "$NAME" 2>&1 | tail -30; exit 1; }
    sleep 10
done
[[ "${code:-}" == "200" ]] || { echo "FATAL: never became healthy"; exit 1; }

mkdir -p results
# Greedy, so the two arms are comparable token for token. Short and few: this is a
# correctness control, not a benchmark -- a repetition collapse shows up inside 32
# tokens and every Hy4 failure so far did.
: > "$OUT"
{
  echo "### model=$MODEL_TAG a2a=$A2A tp=$TP ep=$TP layout=$LAYOUT image=$IMAGE"
  echo "### deepep_mode=$([[ "$A2A" == none ]] && echo n/a || echo normal) dispatch_dtype=$([[ "$A2A" == none ]] && echo n/a || echo "$DISPATCH_DTYPE")"
} >> "$OUT"
for p in "The capital of France is" \
         "def quicksort(arr):" \
         "In one sentence, why is the sky blue?"; do
    python3 - "$PORT" "$p" >> "$OUT" <<'PY'
import json, sys, urllib.request
port, prompt = sys.argv[1], sys.argv[2]
req = urllib.request.Request(
    f"http://localhost:{port}/v1/completions",
    data=json.dumps({"model": "/model", "prompt": prompt,
                     "max_tokens": 32, "temperature": 0}).encode(),
    headers={"Content-Type": "application/json"})
text = json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["text"]
print(f"PROMPT {prompt!r}\n  -> {text!r}")
PY
done
cat "$OUT"
echo "wrote $OUT"
