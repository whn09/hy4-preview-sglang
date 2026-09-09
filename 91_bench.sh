#!/bin/bash
# Throughput / latency benchmark via sglang.bench_serving.
#
#   bash 91_bench.sh                                    # 1k/1k, concurrency 32
#   ISL=4096 OSL=512 CONCURRENCY=64 bash 91_bench.sh
#   NUM_PROMPTS=32 CONCURRENCY=1 bash 91_bench.sh        # the MTP operating point
#
# Raw output is tee'd to results/<TAG>.log and the JSON summary written alongside.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

SRV="${SRV:-hy4-preview}"
ENDPOINT="${ENDPOINT:-localhost:$PORT}"
HOST="${ENDPOINT%%:*}"
BPORT="${ENDPOINT##*:}"

# Fail here rather than let read_cenv fall back, because the fallback is SILENT
# and every axis below is read through it: with the wrong SRV name a DeepEP arm
# benchmarks as "tp8x1node ... moe_a2a=none" -- a well-formed log about the wrong
# configuration, which is worse than no log at all. Measured 2026-09-08: the
# nightly DeepEP arm runs in a container called hy4-nightly-deepep, and a run
# left at the default name produced exactly that mislabelled file.
#
# Only when the endpoint is local: with a remote ENDPOINT there is genuinely no
# container here to read, and the header's "unknown" cells are then honest.
if [[ "$HOST" == "localhost" || "$HOST" == "127.0.0.1" ]] && ! docker inspect "$SRV" >/dev/null 2>&1; then
    echo "FATAL: no container named '$SRV' on this host, but ENDPOINT is local." >&2
    echo "       Every axis in the filename comes from that container's env and" >&2
    echo "       read_cenv falls back silently, so this run would be labelled with" >&2
    echo "       the wrong topology and backend. Containers here:" >&2
    docker ps -a --format '         {{.Names}}  ({{.Image}})' >&2
    echo "       Re-run with SRV=<name>." >&2
    exit 2
fi
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
CONCURRENCY="${CONCURRENCY:-32}"
# The cookbook's own ladder: enough requests that the measured window is the
# steady state and not the ramp. Overridable, but then say so in the filename --
# NUM_PROMPTS is in TAG below precisely because a c16/n32 run and a c16/n64 run
# have different denominators and must not share a log.
case "$CONCURRENCY" in
    1)    DEF_PROMPTS=32   ;;
    16)   DEF_PROMPTS=32   ;;
    64)   DEF_PROMPTS=128  ;;
    256)  DEF_PROMPTS=512  ;;
    1024) DEF_PROMPTS=2048 ;;
    *)    DEF_PROMPTS=$(( CONCURRENCY * 4 )) ;;
esac
NUM_PROMPTS="${NUM_PROMPTS:-$DEF_PROMPTS}"

# Everything that changes the number must be in the filename, and it is read from
# the RUNNING CONTAINER rather than from this shell: `bash 91_bench.sh` invoked
# without the same PROFILE=/QUANT=/TP_SIZE= that launched the server would
# otherwise stamp a spec-off run "low-latency" and delete the spec-on log with the
# same name. "unknown" means the container was not reachable from here (e.g. a
# remote ENDPOINT) -- treat such a row as unlabelled.
RUN_QUANT="$(read_cenv "$SRV" QUANT "${QUANT:-}")"
RUN_PROF="$(read_cenv "$SRV" PROFILE "${PROFILE:-}")"
RUN_SPEC="$(read_cenv "$SRV" SPEC "${SPEC:-}")"
RUN_TP="$(read_cenv "$SRV" TP_SIZE "${TP_SIZE:-}")"
RUN_NN="$(read_cenv "$SRV" NNODES "${NNODES:-1}")"
RUN_MODEL="$(read_cenv "$SRV" SERVED_MODEL_NAME "$SERVED_MODEL_NAME")"
RUN_MPATH="$(read_cenv "$SRV" MODEL_PATH "$MODEL_PATH")"
# Topology is an axis, so it is in the FILENAME, not just the header. Without it a
# 1P1D run and a single-node run at the same quant/tp/profile/conc collide on one
# name and the second silently deletes the first.
RUN_TOPO="$(read_cenv "$SRV" TOPO "${TOPO:-tp${RUN_TP}x${RUN_NN}node}")"
# The tok/s/GPU denominator, taken from the container rather than computed here: a
# 1P1D pair occupies TP_SIZE GPUs on EACH of two hosts, so TP alone would credit
# it with double its real per-GPU throughput.
RUN_GPUS="$(read_cenv "$SRV" BENCH_GPUS "${BENCH_GPUS:-$RUN_TP}")"
# MoE geometry, for the header. It is already inside RUN_TOPO (build_moe_args
# appends MOE_TAG to TOPO) so the FILENAME is safe without these; they are here so
# a log states the EP degree in words instead of leaving it to be decoded from a
# tag fragment. EP_SIZE is the REQUEST -- read a2a too, because with any
# a2a-spanning backend the EP that ran is tp_size regardless of what was asked.
RUN_A2A="$(read_cenv "$SRV" A2A_BACKEND "${A2A_BACKEND:-none}")"
RUN_EP="$(read_cenv "$SRV" EP_SIZE "${EP_SIZE:-}")"
[[ "$RUN_A2A" != "none" && "$RUN_A2A" != "unknown" ]] && RUN_EP="$RUN_TP"
RUN_DPATTN="$(read_cenv "$SRV" DP_ATTN "${DP_ATTN:-}")"
# deepep_v2 only, and it is an axis: the capacity bounds the decode CUDA graph
# and sizes the ElasticBuffer, so two v2 rows at different caps are not
# comparable. Read the env var the SERVER got, not the launcher knob.
RUN_V2CAP="$(read_cenv "$SRV" SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK "")"
# The chunk is coupled to that capacity (the v2 budget check refuses a chunk larger
# than the cap), so it belongs in the header next to it: "default" means the
# runtime picked it -- 16384 on this GPU -- not that it does not matter.
RUN_CHUNK="$(read_cenv "$SRV" CHUNKED_PREFILL "${CHUNKED_PREFILL:-}")"
# The admission cap bounds offered concurrency, so a c=256 run behind max_running=48
# is a 48-slot run: report it next to the concurrency it was asked for.
RUN_MAXRUN="$(read_cenv "$SRV" MAX_RUNNING "${MAX_RUNNING:-}")"

# The bench client runs in the SERVER's image, not the kit's default $IMAGE: the
# tokenizer and bench_serving then match the build under test, and a nightly-image
# arm does not silently pull a second 20 GB image just to drive it.
BENCH_IMAGE="${BENCH_IMAGE:-$(docker inspect -f '{{.Config.Image}}' "$SRV" 2>/dev/null || echo "$IMAGE")}"

RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR_HOST/results}"
TAG="${TAG:-${RUN_QUANT}-${RUN_TOPO}-${RUN_PROF}-spec${RUN_SPEC}-isl${ISL}-osl${OSL}-c${CONCURRENCY}-n${NUM_PROMPTS}}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${TAG}.log"
JSON="$RESULTS_DIR/${TAG}.json"

# ONE FILE = ONE RUN. bench_serving's --output-file APPENDS (it is JSONL and that
# is upstream behaviour), while the .log below is truncated -- so re-running the
# same TAG used to leave a .json with two rows and a .log describing only the
# second. That is how the 2026-09-09 p5en c=1 point ended up holding both a TCP
# and an EFA run under one name, a 2.63x difference recoverable only by
# reconstructing container start times
# (results/bf16-tp16x2node-TCPFALLBACK-README.md).
#
# Rotate the previous PAIR into results/superseded/ instead: nothing is deleted,
# the pair stays matched, and gen_bench_table.py's non-recursive glob("*.log")
# does not see it -- so a superseded run cannot reappear as a duplicate row with
# identical axes. Give the rerun a distinguishing TAG if you want it published.
if [[ -f "$LOG" || -f "$JSON" ]]; then
    _sup="$RESULTS_DIR/superseded"
    mkdir -p "$_sup"
    _n=1
    while [[ -e "$_sup/${TAG}.run${_n}.log" || -e "$_sup/${TAG}.run${_n}.json" ]]; do
        _n=$(( _n + 1 ))
    done
    [[ -f "$LOG" ]]  && mv "$LOG"  "$_sup/${TAG}.run${_n}.log"
    [[ -f "$JSON" ]] && mv "$JSON" "$_sup/${TAG}.run${_n}.json"
    echo "note : a previous run under this tag was moved to superseded/${TAG}.run${_n}.*" >&2
fi

echo "bench: ${ENDPOINT}  isl=${ISL} osl=${OSL} n=${NUM_PROMPTS} conc=${CONCURRENCY}"
echo "log  : ${LOG}"

{
  echo "### tag=${TAG}"
  echo "### endpoint=${ENDPOINT} image=$(docker inspect -f '{{.Config.Image}}' "$SRV" 2>/dev/null || echo unknown)"
  echo "### quant=${RUN_QUANT} tp=${RUN_TP} nnodes=${RUN_NN} profile=${RUN_PROF} spec=${RUN_SPEC}"
  echo "### topo=${RUN_TOPO} gpus=${RUN_GPUS} backend=$(read_cenv "$SRV" TRANSFER_BACKEND none)"
  echo "### moe_a2a=${RUN_A2A} ep=${RUN_EP:-1} dp_attn=${RUN_DPATTN:-off} v2_cap=${RUN_V2CAP:-n/a} chunk=${RUN_CHUNK:-default}"
  echo "### isl=${ISL} osl=${OSL} num_prompts=${NUM_PROMPTS} concurrency=${CONCURRENCY} max_running=${RUN_MAXRUN:-auto}"
  echo "### started=$(date -u +%FT%TZ)"
} > "$LOG"

# --flush-cache: the random dataset is seeded, so a second run replays the same
# prompts and hits the radix cache from the first -- inflating output throughput
# and collapsing TTFT. For a fully cache-free measurement also launch with
# DISABLE_RADIX=1.
#
# --tokenizer must point at the LOCAL weights. The server reports its model_path
# as /models/Hy4-preview-FP8 and bench_serving would otherwise try to resolve that
# as an HF repo id ("Repo id must be in the form 'repo_name' or
# 'namespace/repo_name'").
docker run --rm --name "hy4-bench-$$" --net=host \
    -v "$HOST_MODEL_DIR/$MODEL_DIRNAME:$RUN_MPATH:ro" \
    -v "$RESULTS_DIR:/results" \
    --entrypoint python3 "$BENCH_IMAGE" \
    -m sglang.bench_serving \
    --backend sglang-oai \
    --host "$HOST" --port "$BPORT" \
    --model "$RUN_MODEL" \
    --tokenizer "$RUN_MPATH" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --flush-cache \
    --output-file "/results/${TAG}.json" 2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "### finished=$(date -u +%FT%TZ) rc=${rc}" >> "$LOG"
[[ -f "$JSON" ]] && echo "json : ${JSON}"
exit "$rc"
