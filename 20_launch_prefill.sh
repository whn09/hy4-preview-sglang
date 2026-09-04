#!/bin/bash
# HOST-side launcher: the PREFILL side of a 1P1D pair. Run on B300-1.
#
#   bash 20_launch_prefill.sh                            # MXFP8 TP4, MTP on, mooncake
#   PROFILE=high-throughput bash 20_launch_prefill.sh    # MTP off
#   TRANSFER_BACKEND=nixl bash 20_launch_prefill.sh      # NIXL/LIBFABRIC instead
#
# Needs the EFA image (`docker build -t hy4-preview-efa:latest -f Dockerfile .`)
# and the weights on THIS host -- the KV cache crosses the wire, the weights do
# not. Bring up prefill and decode in any order, then 22_launch_router.sh last.
set -euo pipefail
cd "$(dirname "$0")"
PD_ROLE=prefill NAME="${NAME:-hy4-prefill}" exec bash ./_pd_launch.sh
