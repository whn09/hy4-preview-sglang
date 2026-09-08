#!/usr/bin/env python3
"""Publish the a2a-backend comparison table from the bench JSONs.

    python3 gen_a2a_table.py                      # all nightly TP8 rows
    python3 gen_a2a_table.py --glob 'results/mxfp8-nightly-tp8-*c64-n128.json'

Generated rather than transcribed, and the axes are read from each JSON's own
`server_info.server_args` rather than from its filename: the filename records what
was *asked for*, `server_args` records what the server actually resolved, and on
this kit they have disagreed (`env_common.sh` defaults `DISPATCH_DTYPE=bf16` for
deepep+MXFP8, so an arm launched with the variable unset still runs bf16).

The correctness column is not derivable from a benchmark at all -- it comes from
`96_logprob_parity.py` -- so it is a hand-maintained lookup keyed by the axes that
determine it, and any row whose axes are not in the table prints `?`. That is
deliberate: the pre-`atg` arm was the FASTEST row here while discarding seven
eighths of its expert work, so a throughput table for this model that cannot say
which rows are numerically valid is actively misleading.
"""
import argparse
import glob
import json
import os

# (moe_a2a_backend, disable_attn_tp_gather, dispatcher dtype) -> parity result.
# Measured 2026-09-08, B300-1, MXFP8 tp4 pairs, worst top-5 |dlogprob| against an
# a2a=none reference; the same-config noise floor of that harness is 0.593, so
# "at floor" is as good as this method can resolve. See 96_logprob_parity.py.
PARITY = {
    ("none", False, "auto"): "reference",
    ("none", True, "auto"): "reference",
    ("deepep", True, "bf16"): "0.410 (at floor)",
    ("deepep", False, "auto"): "7.48 WRONG",
}


def axes(blob):
    si = blob["server_info"]
    sa = si.get("server_args", si)
    return {
        "a2a": sa.get("moe_a2a_backend"),
        "mode": sa.get("deepep_mode"),
        "disp": sa.get("deepep_dispatcher_output_dtype"),
        "atg": bool(sa.get("disable_attn_tp_gather")),
        "cg_off": bool(sa.get("disable_cuda_graph")),
        "tp": sa.get("tp_size"),
        "ep": sa.get("ep_size"),
        "maxrun": sa.get("max_running_requests"),
        "memfrac": sa.get("mem_fraction_static"),
        "chunk": sa.get("chunked_prefill_size"),
        "spec": sa.get("speculative_algorithm"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/mxfp8-nightly-tp8-*c64-n128.json")
    a = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(a.glob)):
        b = json.load(open(p))
        rows.append((os.path.basename(p), b, axes(b)))
    if not rows:
        raise SystemExit(f"no JSONs match {a.glob!r}")

    # Every row must share the axes that are NOT under test, or the table is
    # comparing more than it claims to. Assert instead of printing a caveat.
    fixed = ("tp", "ep", "maxrun", "memfrac", "chunk", "spec")
    base = rows[0][2]
    for name, b, ax in rows[1:]:
        bad = [k for k in fixed if ax[k] != base[k]]
        if bad:
            raise SystemExit(
                f"FATAL: {name} differs from {rows[0][0]} in {bad} "
                f"({ {k: ax[k] for k in bad} } vs { {k: base[k] for k in bad} }); "
                "these rows are not comparable.")

    conc = rows[0][1].get("max_concurrency")
    print(f"# a2a backends on Hy4-preview MXFP8, tp{base['tp']} ep{base['ep']} "
          f"1 node, isl/osl 1024, c={conc}, max_running={base['maxrun']}, "
          f"spec={base['spec']}\n")
    hdr = ("| a2a | deepep_mode | dispatch | atg | cuda_graph | out tok/s | "
           "TPOT med ms | TTFT med ms | accept | parity |")
    print(hdr)
    print("|" + "---|" * (hdr.count("|") - 1))
    for name, b, ax in sorted(rows, key=lambda r: -r[1]["output_throughput"]):
        key = (ax["a2a"], ax["atg"], ax["disp"])
        print(f"| {ax['a2a']} | {ax['mode']} | {ax['disp']} | "
              f"{int(ax['atg'])} | {'off' if ax['cg_off'] else 'on'} | "
              f"**{b['output_throughput']:.2f}** | {b['median_tpot_ms']:.2f} | "
              f"{b['median_ttft_ms']:.0f} | {b['accept_length']:.2f} | "
              f"{PARITY.get(key, '?')} |")


if __name__ == "__main__":
    main()
