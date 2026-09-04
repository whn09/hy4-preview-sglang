#!/bin/bash
# Concurrency ladder at one operating point. Relaunch the server between the two
# PROFILEs -- MTP on/off is a launch-time decision, so a sweep cannot cross it.
#
#   bash 92_sweep.sh                         # c = 1 16 64 256 at 1k/1k
#   CONCS="1 16" ISL=4096 OSL=512 bash 92_sweep.sh
#
# The two arms the cookbook actually distinguishes:
#   PROFILE=low-latency      -> read the c=1 row  (TPOT is the point)
#   PROFILE=high-throughput  -> read the c=256 row (tok/s/GPU is the point)
set -uo pipefail

cd "$(dirname "$0")"

CONCS="${CONCS:-1 16 64 256}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"

for c in $CONCS; do
    echo "############ concurrency=$c ############"
    CONCURRENCY="$c" ISL="$ISL" OSL="$OSL" bash 91_bench.sh || echo "  (c=$c failed, continuing)"
    # A failed arm must not silently vanish from the table: 91_bench.sh writes the
    # rc into its own log header, and gen_bench_table.py skips rows whose rc != 0.
    sleep 5
done

echo
echo "==> table:"
python3 gen_bench_table.py 2>/dev/null || echo "(run gen_bench_table.py once results exist)"
