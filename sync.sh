#!/bin/bash
# Push scripts to the test host(s), and pull benchmark results back.
#
#   bash sync.sh push     # local -> hosts (scripts only)
#   bash sync.sh pull     # hosts -> local results/
#   bash sync.sh          # push then pull
#
# NEVER use a bare `rsync --delete` here: $REMOTE also holds results/, which only
# exists on the hosts (91_bench.sh writes it there), so a --delete push silently
# wipes every benchmark log. The excludes below are the whole safety mechanism.
set -euo pipefail

cd "$(dirname "$0")"
# All four nodes are ours as of the 2026-09-05 relaunch (the K3 arms that owned
# B300-1/B300-2 are gone). Note that P6-B300-* in ~/.ssh/config are a COLLEAGUE's
# machines, not ours -- only the B300-N aliases.
HOSTS="${HOSTS:-B300-1 B300-2 B300-3 B300-4}"
REMOTE="${REMOTE:-/home/ubuntu/hy4-preview-sglang}"

push() {
    for h in $HOSTS; do
        ssh "$h" "mkdir -p $REMOTE"
        rsync -az --exclude 'results/' --exclude '.git/' --exclude '__pycache__/' ./ "$h:$REMOTE/"
        echo "pushed -> $h"
    done
}

pull() {
    mkdir -p results
    for h in $HOSTS; do
        rsync -az "$h:$REMOTE/results/" results/ 2>/dev/null && echo "pulled <- $h" \
            || echo "no results on $h"
    done
}

case "${1:-both}" in
    push) push ;;
    pull) pull ;;
    both) push; pull ;;
    *) echo "usage: $0 [push|pull]" >&2; exit 1 ;;
esac
