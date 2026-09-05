#!/bin/bash
# Build hy4-preview-v2-efa:latest = the deepep_v2 patches applied ON TOP of the
# EFA image, which is what a CROSS-NODE or PD deepep_v2 arm needs.
#
#   bash 05_pull_pd_image.sh      # the base, from ECR (~75 s in-region)
#   bash 06_build_v2_efa_image.sh # ~1 min, no compiler runs
#
# Why a second v2 image exists at all. `hy4-preview-v2:latest` is deliberately
# thin -- the stock image plus three .py files -- so that a v1-vs-v2 comparison
# has exactly one variable. That is the right image for a SINGLE-NODE arm, where
# DeepEP moves expert tokens over NVLink and no NIC is involved.
#
# It is the wrong image the moment a second node appears. Cross-node DeepEP
# reaches the NIC through NCCL GIN, and the stock image cannot: measured
# 2026-09-05, it ships pip NCCL 2.29.7 (GIN needs >= 2.31), no
# libnccl-net-ofi.so and no /opt/amazon/efa. The failure is not a clean error --
# NCCL falls back to TCP over ENA, so you get a hang in the first MoE layer or a
# "cross-node" number that is really a socket number. env_common.sh's
# require_gin_capable_image() refuses that combination up front.
#
# The cost of this image is that it is NOT the thin one: relative to
# hy4-preview-v2 it also changes NCCL (2.29.7 -> 2.31.2), adds the EFA stack and
# swaps the Mooncake wheel. So a v1-vs-v2 comparison run on THIS image must use
# the EFA image (not the stock one) for its v1 arm, or the pair differs in NCCL
# as well as in backend.
set -euo pipefail

cd "$(dirname "$0")"

BASE="${BASE:-hy4-preview-efa:latest}"
TAG="${TAG:-hy4-preview-v2-efa:latest}"

if ! docker image inspect "$BASE" >/dev/null 2>&1; then
    echo "ERROR: base image '$BASE' is not present." >&2
    echo "       bash 05_pull_pd_image.sh   # pulls it from ECR and self-verifies" >&2
    exit 1
fi

echo "base : $BASE"
echo "tag  : $TAG"
docker build -t "$TAG" --build-arg BASE_IMAGE="$BASE" -f Dockerfile.deepep_v2 .

# Verify BOTH properties, because this image is the intersection of two and
# either one silently missing is a wrong measurement rather than a failure.
echo
echo "---- verifying $TAG ----"
docker run --rm --entrypoint bash "$TAG" -c '
set -eu
fail=0
cat /etc/hy4-deepep-v2-patched
V=$(python3 -c "import importlib.metadata as m; print(m.version(\"nvidia-nccl-cu13\"))")
echo "pip NCCL          : $V"
case "$V" in 2.3[1-9]*|2.[4-9][0-9]*|[3-9].*) ;; *) echo "  ^ TOO OLD for GIN (need >= 2.31)"; fail=1;; esac
if [ -f /opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so ]; then echo "ofi-nccl plugin   : present"
else echo "ofi-nccl plugin   : MISSING"; fail=1; fi
if [ -f /opt/amazon/efa/lib/libfabric.so.1 ]; then echo "libfabric         : $(/opt/amazon/efa/bin/fi_info --version 2>/dev/null | head -1)"
else echo "libfabric         : MISSING"; fail=1; fi
# The Mooncake wheel matters only for a PD arm on this image, but check it here
# rather than 10 minutes into one: "efa" is a substring of "default", so grep the
# libfabric symbols, never `grep -ci efa`.
if python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))" >/dev/null 2>&1; then
  MC=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))")
  if strings "$MC"/engine*.so 2>/dev/null | grep -q fi_mr_reg; then echo "mooncake          : EFA (fi_mr_reg present)"
  else echo "mooncake          : NOT EFA-capable"; fail=1; fi
fi
[ "$fail" = 0 ] || { echo; echo "FATAL: image is not usable for a cross-node deepep_v2 arm."; exit 1; }
echo
echo "image OK: deepep_v2 patches + NCCL GIN + EFA"
'

echo
echo "use it with:"
echo "  IMAGE=$TAG A2A_BACKEND=deepep_v2 CHUNKED_PREFILL=2048 \\"
echo "    NNODES=2 NODE_RANK=<0|1> TP_SIZE=16 DIST_INIT_ADDR=<rank0 ip> \\"
echo "    bash 10_launch_standalone.sh"
