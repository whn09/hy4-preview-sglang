#!/usr/bin/env python3
"""Generate the Hy4-preview benchmark table from raw bench_serving logs.

Never hand-copy these numbers -- publish from this generator:

    python3 gen_bench_table.py [results_dir]

Every row is one 91_bench.sh log. The axes come from the log's own `###` header
(written from the RUNNING container's env, not from the launching shell), so a row
can never be labelled with a config it did not run. Rows whose header records
rc != 0 are dropped with a note rather than silently averaged in.

Both a time column and a rate column are printed on purpose: tok/s alone has
inverted a conclusion before, because its denominator (GPU count) changes between
a TP4 arm, a TP8 arm and a 1P1D pair (which occupies 2x TP).
"""
import pathlib
import re
import sys

HDR = [
    ("quant",   r"^### .*\bquant=(\S+)"),
    ("tp",      r"^### .*\btp=(\S+)"),
    ("nnodes",  r"^### .*\bnnodes=(\S+)"),
    ("topo",    r"^### .*\btopo=(\S+)"),
    ("gpus",    r"^### .*\bgpus=(\S+)"),
    ("backend", r"^### .*\bbackend=(\S+)"),
    # MoE geometry. Absent from any log written before the EP/a2a knobs existed;
    # such a row WAS pure TP (the flags could not be passed), but it says so by
    # having no header line, not by asserting it, so it prints as "-".
    ("a2a",     r"^### .*\bmoe_a2a=(\S+)"),
    ("ep",      r"^### .*\bmoe_a2a=\S+ ep=(\S+)"),
    ("profile", r"^### .*\bprofile=(\S+)"),
    ("spec",    r"^### .*\bspec=(\S+)"),
    ("isl",     r"^### .*\bisl=(\S+)"),
    ("osl",     r"^### .*\bosl=(\S+)"),
    ("n",       r"^### .*\bnum_prompts=(\S+)"),
    ("conc",    r"^### .*\bconcurrency=(\S+)"),
    ("rc",      r"^### .*\brc=(\S+)"),
]

FIELDS = [
    ("ok",       r"Successful requests:\s+(\d+)"),
    ("dur",      r"Benchmark duration \(s\):\s+([\d.]+)"),
    ("in_tps",   r"Input token throughput \(tok/s\):\s+([\d.]+)"),
    ("out_tps",  r"Output token throughput \(tok/s\):\s+([\d.]+)"),
    ("live",     r"Concurrency:\s+([\d.]+)"),
    ("ttft_p50", r"Median TTFT \(ms\):\s+([\d.]+)"),
    ("tpot_p50", r"Median TPOT \(ms\):\s+([\d.]+)"),
    # ITL is the cleaner decode-step number: TPOT averages over the whole request
    # and picks up prefill interleaving. Under MTP the two diverge -- quote both.
    ("itl_p50",  r"Median ITL \(ms\):\s+([\d.]+)"),
    ("e2e_p50",  r"Median E2E Latency \(ms\):\s+([\d.]+)"),
]


def parse(path):
    txt = path.read_text(errors="replace")
    row = {}
    for name, pat in HDR:
        m = re.search(pat, txt, re.M)
        row[name] = m.group(1) if m else "?"
    for name, pat in FIELDS:
        m = re.search(pat, txt)
        row[name] = float(m.group(1)) if m else None
    return row


def cell(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    logs = sorted(root.glob("*.log"))
    if not logs:
        print(f"no logs in {root}/")
        return

    rows, failed, unfinished = [], [], []
    for p in logs:
        r = parse(p)
        r["tag"] = p.stem
        if r["rc"] == "?":
            unfinished.append(p.name)
            continue
        if r["rc"] != "0" or r["ok"] in (None, 0):
            failed.append(f"{p.name} (rc={r['rc']})")
            continue
        rows.append(r)

    # Sort by the axes, not by filename, so an arm's ladder reads in order.
    def key(r):
        def num(x):
            try:
                return float(x)
            except (TypeError, ValueError):
                return -1.0
        return (r["quant"], r["topo"], num(r["tp"]), r["profile"],
                num(r["isl"]), num(r["osl"]), num(r["conc"]))

    print("| quant | topo | moe | profile | spec | ISL/OSL | conc | reqs | dur (s) | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | out tok/s | GPUs | out tok/s/GPU |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    guessed = False
    for r in sorted(rows, key=key):
        # gpus comes from the RUNNING container (BENCH_GPUS). Falling back to TP is
        # correct only for single-node rows -- a 1P1D pair occupies TP GPUs on each
        # of two hosts -- so an older log without the header is flagged, not fixed.
        try:
            gpus = int(float(r["gpus"]))
        except (TypeError, ValueError):
            try:
                gpus = int(float(r["tp"]))
            except (TypeError, ValueError):
                gpus = 0
            guessed = True
        per_gpu = r["out_tps"] / gpus if (r["out_tps"] and gpus) else None
        topo = r["topo"] if r["topo"] != "?" else f"tp{r['tp']}x{r['nnodes']}node*"
        # "a2a/ep" in one column: EP alone is ambiguous because an a2a-spanning
        # backend forces EP = TP, so the pair is the only honest label.
        moe = "-" if r["a2a"] == "?" else "{}/ep{}".format(r["a2a"], r["ep"])
        print("| {} | {} | {} | {} | {} | {}/{} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["quant"], topo, moe, r["profile"], r["spec"],
            r["isl"], r["osl"], r["conc"], cell(r["ok"], "{:.0f}"), cell(r["dur"]),
            cell(r["ttft_p50"]), cell(r["tpot_p50"]), cell(r["itl_p50"]),
            cell(r["e2e_p50"]), cell(r["out_tps"]), gpus or "-", cell(per_gpu)))

    print("\n_out tok/s/GPU denominator = the GPUs the whole serving instance "
          "occupies, read from the container's BENCH_GPUS: TP size for a "
          "single-node row, 2x TP for a 1P1D pair. A TP4 MXFP8 row, a TP8 BF16 "
          "row and a PD row do not share it. `moe` = a2a backend / EP degree; "
          "`-` means the log predates the knob, i.e. pure TP MoE._")
    if guessed:
        print("\n_Rows marked `*` predate the topo/gpus header; their GPU count was "
              "assumed = TP, which is WRONG for any disaggregated row. Re-run or "
              "treat their per-GPU column as unlabelled._")
    if failed:
        print("\nFAILED runs (row omitted, not zero): " + ", ".join(failed))
    if unfinished:
        print("\nNO rc IN HEADER (still running or killed): " + ", ".join(unfinished))


if __name__ == "__main__":
    main()
