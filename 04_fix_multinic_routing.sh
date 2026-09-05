#!/bin/bash
# HOST-side, run once per node AFTER EVERY BOOT, before any cross-node arm.
#
#   bash 04_fix_multinic_routing.sh          # fix this node
#   bash 04_fix_multinic_routing.sh --check   # report only, exit 1 if broken
#
# WHY THIS EXISTS
# ---------------
# A p6-b300 launched with the multi-NIC path gets **17 ENA "interface" ENIs in the
# same subnet** (device index 0-16), on top of the 16 efa-only ones. ec2-net-utils
# adds a policy-routing rule for each SECONDARY IP (tables 101-116) but leaves the
# PRIMARY IP to the main table -- and in the main table the 16 secondary
# `172.31.16.0/20 dev enpXXX proto kernel` routes have metric 0 while the primary's
# has metric 100. So the lowest-metric route to any same-subnet peer points at a
# SECONDARY interface.
#
# The result is asymmetric routing that AWS silently drops: a packet sourced from
# the primary IP leaves through a secondary ENI, that ENI's source/destination
# check rejects a source IP it does not own, and the frame never reaches the wire.
# Measured on B300-1/B300-2 2026-09-05 with tcpdump on the receiver:
#
#   enp71s0  In  IP 172.31.17.128 > 172.31.29.80: ICMP echo request   <- arrives
#   enp170s0 Out IP 172.31.29.80 > 172.31.17.128: ICMP echo reply     <- wrong NIC
#
# i.e. requests arrive, replies vanish. 100% packet loss between two instances in
# one subnet whose security group already allows all traffic from itself.
#
# HOW IT MISLEADS YOU
# -------------------
# Every AWS-side check passes, so the instinct is to keep checking AWS: the SG has
# the self-referencing `IpProtocol: -1` rule, both nodes are in one subnet, the
# NACL is default, `hostname -I` is right, ssh works and `/health` returns 200 on
# each node LOCALLY. What fails is only node-to-node, so it reads as a security
# group that was not reattached after a relaunch. 22_launch_router.sh reports
# "172.31.x.y:30000 is not accepting connections", which reads as "the server is
# still loading" -- and a cold Hy4 does take ~10 min, so the wrong explanation is
# always available. Distinguish it in one command: ping the peer. If ICMP fails
# too, it is this, not the server.
#
# THE FIX
# -------
# One source-based rule per node, pointing traffic FROM the primary IP at the
# primary interface. Table 100 is used because ec2-net-utils owns 101-116.
# Non-persistent by design (nothing writes to /etc): re-run it after every boot.
set -euo pipefail

CHECK_ONLY=0
[[ "${1:-}" == "--check" ]] && CHECK_ONLY=1

# The primary interface is the one carrying the default route the OS actually
# prefers -- read it rather than hard-coding enp71s0, which is not guaranteed.
IFACE="$(ip -o -4 route show to default | sort -k11 -n | awk '{print $5; exit}')"
IP="$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}' | cut -d/ -f1)"
CIDR="$(ip -4 -o addr show dev "$IFACE" | awk '{print $4}')"
SUBNET="$(ip -o -4 route show dev "$IFACE" proto kernel scope link | awk '{print $1; exit}')"
GW="$(ip -o -4 route show to default dev "$IFACE" | awk '{print $3; exit}')"
TABLE=100

echo "iface   : $IFACE"
echo "primary : $IP  (cidr $CIDR, subnet $SUBNET, gw $GW)"

have_rule() { ip rule show | grep -q "from ${IP} lookup ${TABLE}"; }

if (( CHECK_ONLY )); then
    # Count how many same-subnet routes outrank the primary's. More than zero and
    # the rule below is load-bearing, not cosmetic.
    n_secondary=$(ip -o -4 route show "$SUBNET" proto kernel scope link | grep -cv " dev ${IFACE} ")
    echo "same-subnet kernel routes on OTHER interfaces: $n_secondary"
    if have_rule; then
        echo "OK: 'from $IP lookup $TABLE' is present."
        exit 0
    fi
    if (( n_secondary == 0 )); then
        echo "OK: single-NIC node, no rule needed."
        exit 0
    fi
    echo "BROKEN: $n_secondary secondary same-subnet routes and NO source rule for $IP." >&2
    echo "        Cross-node TCP and ICMP will fail in one direction. Run this" >&2
    echo "        script without --check." >&2
    exit 1
fi

sudo ip route replace "$SUBNET" dev "$IFACE" proto static scope link table "$TABLE"
sudo ip route replace default via "$GW" dev "$IFACE" proto static onlink table "$TABLE"
sudo ip rule del from "$IP" lookup "$TABLE" 2>/dev/null || true
sudo ip rule add from "$IP" lookup "$TABLE" pref 32749

echo "added   : ip rule from $IP lookup $TABLE"
ip route get "$GW" from "$IP" 2>/dev/null | head -1

# Prove it end to end where we can: any peer given on the command line, or the
# configured PD peers. A rule that is present but ineffective is worse than none.
PEERS="${*:-}"
[[ "$PEERS" == "--check" ]] && PEERS=""
if [[ -z "$PEERS" && -f "$(dirname "$0")/env_common.sh" ]]; then
    PEERS="$(cd "$(dirname "$0")" && set +eu && source ./env_common.sh >/dev/null 2>&1; echo "${PREFILL_IPS:-} ${DECODE_IPS:-}")"
fi
for p in $PEERS; do
    [[ "$p" == "$IP" ]] && continue
    if ping -c1 -W2 -I "$IFACE" "$p" >/dev/null 2>&1; then
        echo "ping OK : $p"
    else
        echo "ping FAIL: $p  (run this script on THAT node too -- the drop is on" >&2
        echo "           the REPLY path, so both ends need the rule)" >&2
    fi
done
