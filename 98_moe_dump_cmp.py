#!/usr/bin/env python3
"""Diff two arms' MoE boundary dumps (98_moe_dump.sh) and say which stage diverged.

    python3 98_moe_dump_cmp.py results/dumps            # both arms, all ranks
    python3 98_moe_dump_cmp.py results/dumps --tag hy4 --layer 1

Printed rather than eyeballed, because the interesting comparison is not "are
these equal" -- they cannot be, the two arms use different grouped-GEMM
permutations and FP8 rounding -- but WHICH tensor stops being close. The MoE
input must match to attention noise; the routing must match exactly (topk is
integer); a routed output that is off by ~1 relative is a broken expert path.

TWO THINGS MAKE A NAIVE PER-RANK DIFF WRONG HERE, both measured 2026-09-08:

  * ROW COUNT. The a2a=none arm dumped 5 rows for a 5-token prompt; the DeepEP
    arm dumped 8. forward_deepep runs on a token count padded up for the
    dispatcher and passes num_token_non_padded separately, so the tail rows are
    padding. Compare the leading min(rows) only -- a shape mismatch is padding,
    not a different batch, and bailing out on it hides the real diff.

  * REDUCTION. In pure-EP a2a=none, self.experts() returns only THIS rank's
    local experts' contribution and dsv2 all-reduces afterwards, so every rank
    holds a different partial. DeepEP's combine already returns the full sum, so
    every rank holds the same total. Comparing rank0 to rank0 across those two
    conventions compares 1/8 of a sum against the whole thing. This script
    infers the convention from the data (ranks agree => already combined, ranks
    differ => partial, sum them) and says which it picked.
"""
import argparse
import glob
import os
import sys

import torch


def rel(a, b):
    d = (a.float() - b.float()).norm().item()
    n = max(a.float().norm().item(), 1e-30)
    return d / n


def load_arm(dumpdir, tag, arm, layer):
    pat = os.path.join(dumpdir, f"moe_dump-{tag}-{arm}-layer{layer}-rank*.pt")
    out = {}
    for p in sorted(glob.glob(pat)):
        d = torch.load(p, map_location="cpu", weights_only=False)
        out[d["rank"]] = d
    return out


def describe_arm(name, ranks):
    """Report cross-rank structure and return (reduction_kind, routed_out)."""
    rs = sorted(ranks)
    r0 = ranks[rs[0]]["routed_out"]
    norms = {r: ranks[r]["routed_out"].norm().item() for r in rs}
    zeros = [r for r in rs if norms[r] == 0.0]
    agree = [r for r in rs if torch.equal(ranks[r]["routed_out"], r0)]

    print(f"[{name}] ranks={rs} rows={r0.shape[0]}")
    # Per-ROW norms, not just the total: padding rows and dropped tokens both
    # show up here and nowhere else. A row that is 0 in the input but nonzero in
    # topk is padding the gate still scored; a row that is nonzero in the input
    # and 0 in the output is a token the MoE dropped.
    h0 = ranks[rs[0]]["hidden_states"]
    print("  input  row norms: " + " ".join(f"{v:.4f}" for v in h0.norm(dim=-1).tolist()))
    tot = torch.stack([ranks[r]["routed_out"] for r in rs]).sum(0)
    print("  out(sum) row norms: " + " ".join(f"{v:.4f}" for v in tot.norm(dim=-1).tolist()))
    print("  topk_ids row0: " + str(ranks[rs[0]]["topk_ids"][0].tolist()))
    print("  routed_out norm per rank: "
          + "  ".join(f"r{r}={norms[r]:.4f}" for r in rs))
    if zeros:
        print(f"  !! EXACTLY ZERO on ranks {zeros} -- those ranks contributed "
              "nothing at all")
    if len(agree) == len(rs):
        print("  all ranks bit-identical => already combined (DeepEP convention)")
        return "combined", r0
    print(f"  ranks differ ({len(agree)}/{len(rs)} match rank{rs[0]}) => per-rank "
          "partial, summing for the comparison (pure-EP convention)")
    return "partial", torch.stack([ranks[r]["routed_out"] for r in rs]).sum(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumpdir")
    ap.add_argument("--tag", default="hy4")
    ap.add_argument("--layer", type=int, default=1)
    # `arm[:tag]`, because a fix is a new axis: the arm before and after a
    # correctness flag must land in differently tagged files, and the comparison
    # is then across tags (feedback_stamp_every_variable_in_filenames).
    ap.add_argument("--arms", nargs=2, default=["normal", "deepep"])
    a = ap.parse_args()

    names, tags = [], []
    for spec in a.arms:
        arm, _, tag = spec.partition(":")
        names.append(spec)
        tags.append((arm, tag or a.tag))
    a.arms = names

    A = load_arm(a.dumpdir, tags[0][1], tags[0][0], a.layer)
    B = load_arm(a.dumpdir, tags[1][1], tags[1][0], a.layer)
    for nm, d in zip(a.arms, (A, B)):
        if not d:
            raise SystemExit(
                f"FATAL: no dumps for arm '{nm}' tag={a.tag} layer={a.layer} in "
                f"{a.dumpdir}")
    print(f"tag={a.tag} layer={a.layer}   arms: {a.arms[0]} vs {a.arms[1]}\n")

    kA, rA = describe_arm(a.arms[0], A)
    print()
    kB, rB = describe_arm(a.arms[1], B)
    print()

    # Leading rows only; the tail of the longer arm is dispatcher padding.
    ra, rb = (min(A), min(B))
    ha, hb = A[ra]["hidden_states"], B[rb]["hidden_states"]
    n = min(ha.shape[0], hb.shape[0])
    if ha.shape[0] != hb.shape[0]:
        print(f"rows      : {ha.shape[0]} vs {hb.shape[0]} -- comparing the "
              f"leading {n} (the rest is padding)")
    in_rel = rel(ha[:n], hb[:n])
    print(f"input     : rel_l2={in_rel:.3e}  max_abs={(ha[:n] - hb[:n]).abs().max():.3e}")

    ia, ib = A[ra]["topk_ids"][:n], B[rb]["topk_ids"][:n]
    same_ids = torch.equal(ia, ib)
    print(f"topk_ids  : identical={same_ids}", end="")
    if not same_ids:
        # Order can differ without the SET differing; only the set matters to the
        # MoE result, so report both before blaming the gate.
        sa = torch.sort(ia.long(), dim=-1).values
        sb = torch.sort(ib.long(), dim=-1).values
        print(f"  same_set={torch.equal(sa, sb)}  "
              f"differing_slots={(sa != sb).float().mean().item():.1%}", end="")
    print()

    wa, wb = A[ra]["topk_weights"][:n], B[rb]["topk_weights"][:n]
    print(f"topk_wts  : rel_l2={rel(wa, wb):.3e}  sum={wa.sum():.6f} vs {wb.sum():.6f}")

    xa, xb = rA[:n], rB[:n]
    r = rel(xa, xb)
    print(f"routed_out: rel_l2={r:.3e}  ({kA} vs {kB})  "
          f"norm={xa.norm():.6f} vs {xb.norm():.6f}")
    per_tok = (xa - xb).norm(dim=-1) / xa.norm(dim=-1).clamp_min(1e-30)
    print(f"            per-token rel: min={per_tok.min():.3e} "
          f"median={per_tok.median():.3e} max={per_tok.max():.3e}")
    # A CONSTANT ratio is a scale factor, not wrong maths. The one to expect here
    # is routed_scaling_factor (2.827 for Hy4-preview): forward_normal's runner
    # fuses it into the expert epilogue, so it is already in the dumped tensor,
    # while forward_deepep applies it AFTER self.experts() returns -- downstream
    # of the dump point. Measured 2026-09-08: ratio 0.3536-0.3538 vs 1/2.827 =
    # 0.35373, i.e. the two arms agree and only the scale's position differs.
    ratio = (xb.norm(dim=-1) / xa.norm(dim=-1).clamp_min(1e-30))
    print(f"            norm ratio B/A per token: "
          + " ".join(f"{v:.4f}" for v in ratio.tolist()))
    spread = (ratio.max() / ratio.min().clamp_min(1e-30) - 1).item()
    const_scale = spread < 1e-2 and abs(ratio.median().item() - 1.0) > 1e-3
    if const_scale:
        m = ratio.median().item()
        print(f"            ratio is CONSTANT to {spread:.1e} => a pure scale of "
              f"{m:.5f} (1/{1 / m:.4f}), not a numerics difference")

    print()
    if in_rel > 1e-2:
        print("VERDICT: the MoE inputs already differ -- fix the harness first "
              "(different prompt, batch, or a stolen dump) before reading further.")
    elif not same_ids:
        print("VERDICT: routing differs. The gate/topk glue, not the expert path: "
              "forward_deepep passes num_token_non_padded and an "
              "ExpertLocationDispatchInfo that forward_normal does not.")
    elif const_scale:
        m = ratio.median().item()
        print(f"VERDICT: same input, same routing, and routed output differing by a "
              f"CONSTANT {m:.5f} (1/{1 / m:.4f}). Check that against the config's "
              "routed_scaling_factor -- if it matches, the expert path agrees and "
              "the two arms merely apply that scale on opposite sides of this dump "
              f"point. Residual after dividing it out: "
              f"rel_l2={rel(xa * m, xb):.3e}.")
    elif r > 1e-1:
        print("VERDICT: same input, same routing, grossly different routed output "
              "-> the expert compute/dispatch path.")
    else:
        print("VERDICT: input, routing and routed output all agree. The divergence "
              "is downstream of self.experts() -- routed_scaling_factor or the "
              "shared-expert add, which the two forwards implement separately.")


if __name__ == "__main__":
    main()
