#!/bin/bash
# Smoke test against a running endpoint. Everything here is plain curl so it works
# from the host with no python deps.
#
#   bash 90_smoke_test.sh
#   ENDPOINT=localhost:30000 bash 90_smoke_test.sh
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

ENDPOINT="${ENDPOINT:-localhost:$PORT}"
MODEL="$(read_cenv "${NAME:-hy4-preview}" SERVED_MODEL_NAME "$SERVED_MODEL_NAME")"

echo "=== /health ==="
curl -sf "http://${ENDPOINT}/health" && echo " OK" || echo " FAIL"

echo "=== /get_model_info ==="
curl -s "http://${ENDPOINT}/get_model_info" | python3 -m json.tool 2>/dev/null || echo "(no model info)"

# no_think first: it is the cheap probe. Hy4 defaults reasoning_effort to `high`,
# so a bare request spends thousands of tokens thinking before it answers, and a
# too-small max_tokens then returns an EMPTY content with the whole budget spent
# in reasoning_content -- which looks like a broken model rather than a truncated
# reply. Note no_think is NOT a standard OpenAI tier: it has to go through
# chat_template_kwargs, not the top-level reasoning_effort field.
echo
echo "=== chat completion (no_think) ==="
curl -s "http://${ENDPOINT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"What is 127 * 34? Answer with just the number.\"}],
      \"chat_template_kwargs\": {\"reasoning_effort\": \"no_think\"},
      \"max_tokens\": 256,
      \"temperature\": 0
    }" | python3 -m json.tool

echo
echo "=== chat completion (reasoning_effort=high) ==="
curl -s "http://${ENDPOINT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Solve step by step: what is 15% of 240?\"}],
      \"reasoning_effort\": \"high\",
      \"max_tokens\": 2048
    }" | python3 -c '
import json, sys
d = json.load(sys.stdin)
m = d["choices"][0]["message"]
print("--- reasoning_content ---"); print((m.get("reasoning_content") or "(none)")[:1200])
print("--- content ---");           print(m.get("content"))
print("--- usage ---", d.get("usage"))
'

echo
echo "=== streaming (first 10 chunks) ==="
curl -sN "http://${ENDPOINT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"${MODEL}\",
      \"messages\": [{\"role\": \"user\", \"content\": \"Count from 1 to 20.\"}],
      \"chat_template_kwargs\": {\"reasoning_effort\": \"no_think\"},
      \"max_tokens\": 128,
      \"stream\": true
    }" | head -10
