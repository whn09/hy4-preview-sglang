#!/bin/bash
# HOST-side launcher: the DECODE side of a 1P1D pair. Run on B300-2.
#
#   bash 21_launch_decode.sh                             # MXFP8 TP4, MTP on, mooncake
#   PROFILE=high-throughput bash 21_launch_decode.sh     # MTP off
#   TRANSFER_BACKEND=nixl bash 21_launch_decode.sh       # must MATCH the prefill side
#
# TRANSFER_BACKEND and SPEC must be the same on both sides. They are not
# negotiated at handshake time: a mismatch shows up as a stalled request or a
# blacklisted mooncake session, not as a startup error.
#
# Note on admission: with MTP on, SGLang resets max_running_requests to 48 unless
# it is set explicitly (the same trap as K3/DSPARK). On the decode side that IS
# the concurrency ceiling of the whole pair, so a c=256 benchmark against this
# arm is measuring a 48-slot queue. Set MAX_RUNNING= deliberately, and put it in
# the results tag when you do.
set -euo pipefail
cd "$(dirname "$0")"
PD_ROLE=decode NAME="${NAME:-hy4-decode}" exec bash ./_pd_launch.sh
