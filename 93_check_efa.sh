#!/bin/bash
# Is the cross-node traffic actually on EFA? Run this WHILE the server is
# generating (a bench in flight, or any decode loop) -- it measures rates, so on
# an idle server everything reads zero and it says so.
#
# Why this exists: on 2026-09-09 a 2-node TP16 BF16 arm on P5EN-3/4 came up
# healthy, served correct output, and put its entire TP all-reduce on TCP. The
# only visible symptom was the throughput (53.64 tok/s at bs=1). Nothing in the
# sglang log, the NCCL log at NCCL_DEBUG=WARN, or `docker ps` says which
# transport won -- but the NIC counters do, and they cannot be argued with.
#
#   bash 93_check_efa.sh          # 5 s window
#   WINDOW=20 bash 93_check_efa.sh
#
# Exit 1 when the ENA interface is busy and EFA is flat, i.e. the fallback case.
set -euo pipefail

WINDOW="${WINDOW:-5}"

efa_tx_total() {
    local d sum=0 v
    for d in /sys/class/infiniband/*/ports/1/hw_counters; do
        [[ -r "$d/tx_bytes" ]] || continue
        v=$(cat "$d/tx_bytes")
        sum=$(( sum + v ))
    done
    echo "$sum"
}

IFACE="${PRIMARY_IFACE:-$(ip -o -4 route show to default 2>/dev/null | awk '{print $5}' | head -1)}"
NICS=$(ls -d /sys/class/infiniband/* 2>/dev/null | wc -l)
if (( NICS == 0 )); then
    echo "no EFA devices under /sys/class/infiniband -- this host has no EFA at all." >&2
    exit 1
fi

efa0=$(efa_tx_total)
ena0=$(cat "/sys/class/net/$IFACE/statistics/tx_bytes")
sleep "$WINDOW"
efa1=$(efa_tx_total)
ena1=$(cat "/sys/class/net/$IFACE/statistics/tx_bytes")

efa_mb=$(( (efa1 - efa0) / WINDOW / 1048576 ))
ena_mb=$(( (ena1 - ena0) / WINDOW / 1048576 ))

echo "over ${WINDOW}s:  EFA tx = ${efa_mb} MB/s (aggregate over $NICS NICs)   ${IFACE} tx = ${ena_mb} MB/s"

# 8 MB/s of ENA is roughly what an idle host's ssh/metadata chatter reaches; below
# that there is no traffic to attribute and the answer is "run a bench first".
if (( efa_mb == 0 && ena_mb < 8 )); then
    echo "VERDICT: idle. Nothing is crossing the wire -- start a generation and re-run." >&2
    exit 0
fi
if (( efa_mb == 0 )); then
    echo "VERDICT: TCP FALLBACK. ${ena_mb} MB/s on $IFACE and nothing on EFA." >&2
    echo "  NCCL has no aws-ofi-nccl plugin, so it chose NET/Socket. --device=/dev/infiniband" >&2
    echo "  is not sufficient. Relaunch: build_efa_args (env_common.sh) mounts the host's" >&2
    echo "  stack when the image lacks one, and 10_launch_standalone.sh now calls it for" >&2
    echo "  every NNODES>1 arm. Confirm the transport inside the container with:" >&2
    echo "    NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET ... then grep -m1 'NET/' the log" >&2
    exit 1
fi
echo "VERDICT: on EFA."
