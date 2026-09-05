#!/usr/bin/env python3
"""Generate the 1P1D-vs-two-independent-instances table from the bench JSONs.

    python3 gen_pd_vs_solo_table.py [results_dir]

Both arms occupy the SAME 16 GPUs, so they are only comparable at matched *offered*
concurrency -- which is `max_concurrency` for the PD pair (one endpoint, the router)
but `2 x max_concurrency` for the two-instance arm (two endpoints, driven
simultaneously). That doubling is the whole reason this script exists: pairing the
two arms by the number in the filename would compare 128 offered requests against
256, and gen_bench_table.py deliberately does not know that two rows are one arm.

Throughput adds across the two instances. Latency does NOT: a client hitting the
slower of the two instances sees that instance's latency, never the average, so the
worse of the pair is printed and the spread is shown next to it.
"""
import json
import pathlib
import re
import sys

PD = "pd1p1d"
SOLO = "2xsolo"


def load(root):
    """-> {"pd"|"solo": {offered_total: [record, ...]}}"""
    arms = {"pd": {}, "solo": {}}
    for p in sorted(root.glob("*.json")):
        d = json.load(open(p))
        c = d.get("max_concurrency")
        if not c:
            continue
        if PD in p.name:
            arm, offered = "pd", int(c)
        elif SOLO in p.name:
            # "c64each" in the tag and max_concurrency=64 must agree, or the file
            # is mislabelled and the whole comparison silently shifts one rung.
            m = re.search(r"-c(\d+)each-", p.name)
            if not m or int(m.group(1)) != int(c):
                sys.exit(f"{p.name}: tag says c{m.group(1) if m else '?'}each but "
                         f"max_concurrency={c}")
            arm, offered = "solo", int(c) * 2
        else:
            continue
        d["_file"] = p.name
        arms[arm].setdefault(offered, []).append(d)
    return arms


def solo_agg(recs):
    if len(recs) != 2:
        return None, f"{len(recs)} instance log(s), need exactly 2"
    out = sum(r["output_throughput"] for r in recs)
    live = sum(r["concurrency"] for r in recs)
    req = sum(r["request_throughput"] for r in recs)
    # worst-instance latency + the spread across the two
    ttft = [r["median_ttft_ms"] for r in recs]
    tpot = [r["median_tpot_ms"] for r in recs]
    return {
        "out": out, "live": live, "req": req,
        "ttft": max(ttft), "ttft_lo": min(ttft),
        "tpot": max(tpot), "tpot_lo": min(tpot),
    }, None


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    arms = load(root)
    offered = sorted(set(arms["pd"]) | set(arms["solo"]))

    print("| offered conc (16 GPUs) | arm | out tok/s | vs PD | live conc | req/s | "
          "TTFT p50 (ms) | TPOT p50 (ms) |")
    print("|---|---|---|---|---|---|---|---|")
    notes = []
    for c in offered:
        pd = arms["pd"].get(c)
        pdr = pd[0] if pd and len(pd) == 1 else None
        if pd and len(pd) > 1:
            notes.append(f"offered {c}: {len(pd)} PD logs, using none")
        base = pdr["output_throughput"] if pdr else None
        if pdr:
            print("| {} | 1P1D (TP8+TP8) | {:.2f} | - | {:.1f} | {:.2f} | {:.0f} | "
                  "{:.2f} |".format(c, base, pdr["concurrency"],
                                    pdr["request_throughput"],
                                    pdr["median_ttft_ms"], pdr["median_tpot_ms"]))
        else:
            print(f"| {c} | 1P1D (TP8+TP8) | - | - | - | - | - | - |")
        s = arms["solo"].get(c)
        if not s:
            print(f"| {c} | 2x solo TP8 | - | - | - | - | - | - |")
            continue
        agg, err = solo_agg(s)
        if err:
            notes.append(f"offered {c} solo: {err}")
            continue
        delta = "-" if not base else "{:+.1f}%".format(100 * (agg["out"] / base - 1))
        print("| {} | 2x solo TP8 (c={} each) | {:.2f} | {} | {:.1f} | {:.2f} | "
              "{:.0f} ({:.0f}-{:.0f}) | {:.2f} ({:.2f}-{:.2f}) |".format(
                  c, c // 2, agg["out"], delta, agg["live"], agg["req"],
                  agg["ttft"], agg["ttft_lo"], agg["ttft"],
                  agg["tpot"], agg["tpot_lo"], agg["tpot"]))

    print("\n_`vs PD` is the two-instance arm's throughput relative to the 1P1D pair "
          "at the SAME offered concurrency and the same 16 GPUs. Two-instance "
          "throughput/req-rate/live-concurrency are SUMS over the two instances; "
          "its latency columns are the WORSE instance with the pair's range in "
          "parentheses, because no client experiences the average of two servers._")
    for n in notes:
        print(f"\n_note: {n}_")


if __name__ == "__main__":
    main()
