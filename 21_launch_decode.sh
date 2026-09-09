#!/bin/bash
# HOST-side launcher: the DECODE side of a 1P1D pair. Run on B300-2.
#
#   bash 21_launch_decode.sh                             # MXFP8 TP8, MTP on, mooncake
#   PROFILE=high-throughput bash 21_launch_decode.sh     # MTP off
#   TRANSFER_BACKEND=nixl bash 21_launch_decode.sh       # must MATCH the prefill side
#
# A side can span more than one node -- on four p5en, 1P1D means TP16 across two
# nodes PER SIDE. Run this on both decode nodes with NODE_RANK 0 and 1 and a
# DIST_INIT_ADDR that is THIS side's rank 0, NOT the prefill side's: the two sides
# are independent NCCL groups and sharing one address makes all four nodes try to
# form a single TP32 world and hang in the rendezvous.
#   QUANT=bf16 TP_SIZE=16 NNODES=2 NODE_RANK=0 DIST_INIT_ADDR=$D_RANK0 \
#     MEM_FRACTION=0.90 bash 21_launch_decode.sh
#
# TP_SIZE, SPEC, TRANSFER_BACKEND and the MoE geometry (A2A_BACKEND / EP_SIZE /
# DP_ATTN) must be the same on both sides. None of them is negotiated at
# handshake time: a mismatch shows up as a stalled request or a blacklisted
# mooncake session, not as a startup error.
#
# Note on admission: with MTP on, SGLang resets max_running_requests to 48 unless
# it is set explicitly (the same trap as K3/DSPARK). On the decode side that IS
# the concurrency ceiling of the whole pair, so a c=256 benchmark against this
# arm is measuring a 48-slot queue. Set MAX_RUNNING= deliberately, and put it in
# the results tag when you do.
set -euo pipefail
cd "$(dirname "$0")"
PD_ROLE=decode NAME="${NAME:-hy4-decode}" exec bash ./_pd_launch.sh
