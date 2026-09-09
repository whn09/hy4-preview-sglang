#!/usr/bin/env python3
"""Compare two running Hy4 servers token-for-token, to decide whether an arm is
numerically trustworthy before any of its throughput numbers get quoted.

    python3 96_logprob_parity.py --ref  B300-4:30000 \
                                 --test B300-3:30000

Why this exists: the deepep_v2 arm needs a source patch that sets
`mxfp8_act_gran_k` on the v2 pre-permute (see patches/README.md). Get that wrong
and the masked path *asserts*, but the contiguous path silently produces wrong
numbers -- a server that boots, answers, and benchmarks fine while computing the
wrong thing. A wrong-but-fast arm is the worst possible result, so parity is a
gate on quoting v2 numbers, not a nice-to-have.

Two comparisons, because they fail differently:

  greedy continuation   temperature 0, N tokens, ids compared position by
                        position. Catches gross breakage immediately and is
                        readable when it fails (you see where they diverge).
                        Insensitive to small numeric drift -- argmax absorbs it.

  first-token top-5     the logprob VALUES for the top 5 candidates of the first
                        generated token. Sensitive to drift that argmax hides, so
                        this is the one that distinguishes "different kernel,
                        same math" from "different math". Reported as a max
                        absolute delta rather than pass/fail on equality: two
                        different reduction orders are not bit-identical and
                        should not be, so judge the magnitude.

Both arms must differ in exactly one thing. `chunked_prefill_size` in particular
must match: the v2 arm cannot run at the default 16384, so the v1 reference has
to be launched with the same CHUNKED_PREFILL or this script is comparing chunk
sizes too. It prints both servers' resolved args for the axes that matter and
refuses to run when one of them disagrees.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

# Deliberately mixed: a short factual prompt, a long-ish one that crosses more
# than one MoE token block, and one that pushes the router toward experts a
# short prompt never reaches.
PROMPTS = [
    "The capital of France is",
    "Explain in one paragraph why the sky appears blue during the day but red at sunset.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n",
    "Translate to Chinese: The quick brown fox jumps over the lazy dog.",
]

# The axes that must match for the comparison to mean anything. moe_a2a_backend
# is the one that is SUPPOSED to differ, so it is reported and not enforced.
MUST_MATCH = [
    "chunked_prefill_size",
    "tp_size",
    "quantization",
    "speculative_algorithm",
    "speculative_num_draft_tokens",
    "max_running_requests",
    "page_size",
    "kv_cache_dtype",
    "enable_dp_attention",
]
# disable_attn_tp_gather is reported, not enforced, and it is the one axis where
# "differs" is correct rather than sloppy: require_attn_tp_gather() is False for
# an a2a=none arm anyway (utils/common.py:3887 tests the a2a backend), so setting
# it only on the DeepEP arm makes the two arms agree on the layout instead of
# adding an axis. See the DISABLE_ATTN_TP_GATHER block in start_server.sh.
REPORT_ONLY = ["moe_a2a_backend", "deepep_mode", "deepep_dispatcher_output_dtype",
               "ep_size", "deepep_v2_mode", "moe_runner_backend",
               "speculative_moe_a2a_backend", "disable_attn_tp_gather"]


def post(host, path, payload, timeout=600):
    req = urllib.request.Request(
        f"http://{host}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(host, path, timeout=30):
    with urllib.request.urlopen(f"http://{host}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def generate(host, prompt, n_tokens):
    """Greedy, with the first token's top-5 logprobs.

    top_logprobs_num is requested on the OUTPUT side only: input logprobs would
    add a full prefill's worth of payload and the first generated token is where
    a dispatch-path numerics bug shows up.
    """
    return post(host, "/generate", {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": n_tokens,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "top_logprobs_num": 5,
        "logprob_start_len": -1,
    })


def out_token_ids(res):
    lp = res["meta_info"].get("output_token_logprobs") or []
    return [t[1] for t in lp]


def first_top5(res):
    """[(token_id, logprob), ...] for the first generated token, or []."""
    top = res["meta_info"].get("output_top_logprobs") or []
    if not top:
        return []
    return [(t[1], t[0]) for t in top[0]]


def server_args(host):
    """The resolved args, from either /get_server_info shape.

    The EFA-base image (sglang 0.5.18) nests them under "server_args"; the
    2026-09 nightly flattens them to the top level. Detect by a key that only
    ever lives in the args, so a future third shape fails loudly here instead of
    silently comparing two dicts of Nones -- every MUST_MATCH axis would then
    read None on both sides and the gate would pass by vacuity.
    """
    info = get(host, "/get_server_info")
    if isinstance(info.get("server_args"), dict):
        return info["server_args"]
    if "chunked_prefill_size" in info:
        return info
    raise SystemExit(f"FATAL: {host}/get_server_info has neither a 'server_args' "
                     f"dict nor a top-level 'chunked_prefill_size'; keys: "
                     f"{sorted(info)[:12]}...")


def collect(host, tokens):
    """Everything one arm contributes to the comparison, as plain JSON.

    Split out so the two arms need not be alive at the same time: on a single
    node they cannot be -- one server owns all 8 GPUs -- and a sequential
    capture is exactly as valid as a simultaneous one here, because both
    comparisons are deterministic (temperature 0) and neither reads a clock.
    """
    blob = {"host": host, "tokens": tokens,
            "server_args": server_args(host),
            "results": [generate(host, p, tokens) for p in PROMPTS]}
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", help="host:port of the TRUSTED arm (e.g. a2a=none)")
    ap.add_argument("--test", help="host:port of the arm under test (e.g. deepep_v2)")
    ap.add_argument("--capture", metavar="HOST:PORT",
                    help="single-arm mode: record this server to --out and exit. "
                         "Use it twice (relaunching in between) when one node has "
                         "to host both arms, then --compare the two files.")
    ap.add_argument("--out", metavar="PATH", help="where --capture writes")
    ap.add_argument("--compare", nargs=2, metavar=("REF.json", "TEST.json"),
                    help="compare two --capture files instead of two live servers")
    ap.add_argument("--noise-floor", action="store_true",
                    help="allow both arms to run the SAME moe_a2a_backend. For "
                         "measuring what this pair scores with nothing under test: "
                         "a DeepEP arm cannot avoid also differing in CUDA graphs "
                         "(DEEPEP_MODE=normal disables capture) and in which GPUs "
                         "it runs on, and --tol is a tolerance someone chose, not a "
                         "measurement of those. Without this number a nonzero "
                         "|dlogprob| cannot be attributed to the backend.")
    ap.add_argument("--tokens", type=int, default=64, help="greedy continuation length")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max |delta| in a top-5 logprob to still call it agreement")
    args = ap.parse_args()

    # ---- 0. resolve the two blobs, live or from disk ----
    if args.capture:
        if not args.out:
            print("FATAL: --capture needs --out PATH", file=sys.stderr)
            return 2
        try:
            blob = collect(args.capture, args.tokens)
        except (urllib.error.URLError, OSError) as e:
            print(f"FATAL: cannot reach {args.capture}: {e}", file=sys.stderr)
            return 2
        with open(args.out, "w") as f:
            json.dump(blob, f)
        sa = blob["server_args"]
        print(f"captured {args.capture} -> {args.out}  "
              f"(a2a={sa.get('moe_a2a_backend')} ep={sa.get('ep_size')} "
              f"tokens={args.tokens}, {len(blob['results'])} prompts)")
        return 0

    if args.compare:
        try:
            ref_blob = json.load(open(args.compare[0]))
            test_blob = json.load(open(args.compare[1]))
        except OSError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 2
        if ref_blob["tokens"] != test_blob["tokens"]:
            print(f"FATAL: the two captures used different --tokens "
                  f"({ref_blob['tokens']} vs {test_blob['tokens']}); recapture.",
                  file=sys.stderr)
            return 2
    elif args.ref and args.test:
        try:
            ref_blob = collect(args.ref, args.tokens)
            test_blob = collect(args.test, args.tokens)
        except (urllib.error.URLError, OSError) as e:
            print(f"FATAL: cannot reach a server: {e}", file=sys.stderr)
            return 2
    else:
        print("FATAL: give either --ref and --test, or --capture/--out, "
              "or --compare REF.json TEST.json", file=sys.stderr)
        return 2

    return compare(ref_blob, test_blob, args.tol, args.noise_floor)


def compare(ref_blob, test_blob, tol, noise_floor=False):
    # ---- 1. the two servers must differ in exactly one axis ----
    ref_args = ref_blob["server_args"]
    test_args = test_blob["server_args"]

    print("axis                          ref                 test")
    print("-" * 68)
    for k in REPORT_ONLY:
        print(f"  {k:<27} {str(ref_args.get(k)):<19} {str(test_args.get(k))}")
    bad = []
    for k in MUST_MATCH:
        r, t = ref_args.get(k), test_args.get(k)
        flag = "" if r == t else "   <-- MISMATCH"
        if r != t:
            bad.append(k)
        print(f"  {k:<27} {str(r):<19} {str(t)}{flag}")
    if ref_args.get("moe_a2a_backend") == test_args.get("moe_a2a_backend"):
        if not noise_floor:
            print("\nFATAL: both servers run the same moe_a2a_backend "
                  f"({ref_args.get('moe_a2a_backend')!r}); there is nothing to compare.",
                  file=sys.stderr)
            print("       Pass --noise-floor if that is deliberate (see its help).",
                  file=sys.stderr)
            return 2
        print("\nNOISE FLOOR: same backend on both arms. The numbers below are what "
              "this pair scores when NOTHING under test differs, so they are the "
              "floor a real arm has to be judged against -- not a pass/fail.")
    if bad:
        print(f"\nFATAL: {len(bad)} axis/axes differ besides the backend: {', '.join(bad)}.",
              file=sys.stderr)
        print("       Relaunch so the pair differs only in the thing under test.", file=sys.stderr)
        return 2
    print("\nonly moe_a2a_backend (and what it forces) differs -- comparison is valid\n")

    # ---- 2. greedy continuation + first-token top-5 ----
    n_ok = 0
    worst = 0.0
    for i, prompt in enumerate(PROMPTS):
        r = ref_blob["results"][i]
        t = test_blob["results"][i]
        ref_ids, test_ids = out_token_ids(r), out_token_ids(t)

        if ref_ids == test_ids:
            verdict = f"IDENTICAL ({len(ref_ids)} tokens)"
        else:
            div = next((j for j, (a, b) in enumerate(zip(ref_ids, test_ids)) if a != b),
                       min(len(ref_ids), len(test_ids)))
            verdict = f"DIVERGES at token {div}/{min(len(ref_ids), len(test_ids))}"

        # top-5: compare the candidate SET/order and the values
        r5, t5 = first_top5(r), first_top5(t)
        if r5 and t5:
            same_ids = [x[0] for x in r5] == [x[0] for x in t5]
            dmap = dict(t5)
            deltas = [abs(lp - dmap[tid]) for tid, lp in r5 if tid in dmap]
            maxd = max(deltas) if deltas else float("nan")
            worst = max(worst, maxd if maxd == maxd else 0.0)
            top5 = f"ids {'same' if same_ids else 'DIFFER'}, max|dlogprob| {maxd:.6f}"
        else:
            top5 = "no top_logprobs returned"

        ok = (ref_ids == test_ids)
        n_ok += ok
        print(f"[{i + 1}/{len(PROMPTS)}] {'PASS' if ok else 'FAIL'}  {verdict}")
        print(f"         top-5: {top5}")
        print(f"         prompt: {prompt[:56]!r}")
        if not ok:
            print(f"         ref  text: {r['text'][:120]!r}")
            print(f"         test text: {t['text'][:120]!r}")

    print(f"\n{n_ok}/{len(PROMPTS)} prompts produced an identical greedy continuation; "
          f"worst top-5 |dlogprob| {worst:.6f} (tol {tol})")
    if noise_floor:
        # Not a verdict: this run had nothing under test, so the numbers ARE the
        # floor. Measured 2026-09-08, Hy4-preview MXFP8 tp4 pair (a2a=none both
        # sides, GPUs 0-3 vs 4-7, graphs on vs off): 2/4 identical, worst 0.593124
        # -- an order of magnitude above the 0.05 tol. On a model this
        # near-degenerate the tol is unreachable by ANY backend, so quote an arm
        # against THIS number, not against the tol.
        print(f"FLOOR: {n_ok}/{len(PROMPTS)} identical, worst {worst:.6f}. Judge a "
              f"real arm against this, not against tol={tol}: an arm at or under "
              "the floor is indistinguishable from the reference here.")
        return 0
    if n_ok == len(PROMPTS) and worst <= tol:
        print("VERDICT: PARITY -- this arm's numbers are quotable.")
        return 0
    print("VERDICT: NO PARITY vs tol -- but compare against the measured floor before\n"
          "         concluding the arm is wrong (--noise-floor). A boot-and-answer\n"
          "         server proves nothing either way; the contiguous MXFP8 path\n"
          "         fails silently.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
