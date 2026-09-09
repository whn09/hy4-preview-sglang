#!/usr/bin/env bash
# One arm of the PR #38609 end-to-end matrix.
#
# Usage: run_arm.sh <tree> <tag> [extra sglang flags...]
#   <tree>  a worktree under /opt/dlami/nvme (e.g. wt-e2e-nohook-fixed)
#   <tag>   stamped into every artifact name; must name every axis
#
# Writes  logs/<tag>.log         the server log
#         results/<tag>.json     {"outcome": ..., "completions": [...]} or the failure
set -u

TREE=$1; TAG=$2; shift 2
NVME=/opt/dlami/nvme
MODEL=${MODEL_DIR:-$NVME/models/Qwen1.5-MoE-A2.7B-Chat}
PORT=30000
mkdir -p $NVME/logs $NVME/results
LOG=$NVME/logs/$TAG.log
OUT=$NVME/results/$TAG.json

docker exec e2e bash -lc "pkill -f launch_server; sleep 5" >/dev/null 2>&1
docker exec -d -e PYTHONPATH=$NVME/$TREE/python \
  -e SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=1024 e2e bash -lc \
  "cd $NVME/$TREE && python3 -m sglang.launch_server --model-path $MODEL \
   --tp 2 --port $PORT --mem-fraction-static 0.80 --log-level info \
   --chunked-prefill-size 256 --cuda-graph-max-bs-decode 8 --cuda-graph-max-bs-prefill 256 $* \
   > $LOG 2>&1"

# Wait for either a healthy server or a dead launcher.
for i in $(seq 1 90); do
  sleep 10
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' localhost:$PORT/health_generate || true)
  [ "$code" = "200" ] && break
  # Anchor on the python process: the `bash -lc` wrapper's own cmdline
  # contains "launch_server" too, so an unanchored pgrep never reports death.
  alive=$(docker exec e2e bash -lc "pgrep -cf '^python3 -m sglang.launch_server' || true")
  if [ "${alive:-0}" = "0" ]; then break; fi
done

if [ "$code" = "200" ]; then
  docker exec -e TAG="$TAG" -e OUT="$OUT" -e PORT="$PORT" e2e \
    python3 $NVME/pr38609/probe.py
  docker exec e2e bash -lc "pkill -f launch_server" >/dev/null 2>&1
else
  python3 - "$LOG" "$OUT" "$TAG" <<'PY'
import json, sys, re
log, out, tag = sys.argv[1:4]
text = open(log, errors="replace").read()
# The last exception line is the one a user sees.
errs = re.findall(r"^(?:\w+Error|AssertionError|Exception)\b.*$", text, re.M)
lines = [l for l in text.splitlines() if l.strip()]
json.dump(
    {
        "tag": tag,
        "outcome": "crashed",
        "exception": errs[-1] if errs else "",
        "exception_is_bare": bool(errs) and errs[-1].strip() == "AssertionError",
        "log_tail": lines[-12:],
    },
    open(out, "w"),
    indent=2,
)
print(f"{tag}: crashed -> {errs[-1] if errs else '(no exception line)'}")
PY
fi
