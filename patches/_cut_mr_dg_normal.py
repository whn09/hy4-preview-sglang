#!/usr/bin/env python3
"""Regenerate patches/mr_dg_normal.diff from a base image's deep_gemm.py.

Cut this way rather than hand-edited, because the patch touches FOUR places in
one function and every one of them is a `128` that must become the same
variable -- a hand-cut diff that updates three of four produces a scale buffer
whose lane count disagrees with ep_scatter's group, and ep_scatter's own assert
is the only thing that would catch it.

Usage, on a host that has the base image:

    python3 patches/_cut_mr_dg_normal.py \
      --image lmsysorg/sglang:nightly-dev-cu13-20260908-20ca564b \
      --after patches/mr_deep_gemm.diff \
      --out patches/mr_dg_normal.diff

`--after` is applied to the extracted file first, because mr_dg_normal.diff is
cut against a tree mr_deep_gemm.diff has already patched (the Dockerfile applies
them in that order) and their hunks are close enough to shift each other's line
numbers.
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

TARGET = "srt/layers/moe/moe_runner/deep_gemm.py"
FUNC = "def pre_permute_deepep_normal_to_deep_gemm("

# ---- the four edits, as (find, replace) pairs applied inside FUNC only ----

QUANT_BLOCK = '''    hidden_states_dtype = hidden_states.dtype
'''

QUANT_BLOCK_NEW = '''    hidden_states_dtype = hidden_states.dtype

    # LOCAL EXPERIMENT (HYV4 MXFP8 on deepep v1 normal).
    #
    # This function fixes the activation quantisation group at 128 in three
    # places -- the scale buffer's lane count below, ep_scatter's default
    # quant_block_size, and the mxfp8_act_gran_k the contiguous GEMM reads. On a
    # DeepSeek-style FP8 checkpoint (block 128) all three agree with the weights.
    # On HYV4's MXFP8 weights, weight_block_size is [1, 32] and they do not, so
    # deep_gemm's layout.hpp:108 assert fires -- and it is RIGHT to.
    #
    # Do NOT silence that assert by declaring the activation 128-grouped. That
    # was tried: measured 2026-09-08 on a B300 at TP8/EP8 with MTP on, the arm
    # serves 2711 tok/s and is numerically broken -- 0/4 prompts matched an
    # a2a=none reference on the same image, worst top-5 |dlogprob| 7.37, output
    # degenerate ("the capital of the the Franks and the capital of the Franks
    # and ..."). The FP8 dispatch really did quantise at 128 and nothing
    # downstream can undo it. Upstream says as much in the sibling file:
    # token_dispatcher/deepep.py raises "MXFP8 DeepEP dispatch is supported only
    # on Ascend A5 in low-latency mode", i.e. there is no MXFP8 dispatch on CUDA.
    #
    # So take the other route -- dispatch BF16
    # (--deepep-dispatcher-output-dtype bf16) and quantise here at the
    # checkpoint's own group, which is exactly what
    # pre_permute_standard_to_deep_gemm already does: same kernel, same flags,
    # same ep_scatter argument, same mxfp8_act_gran_k. Twice the dispatch bytes,
    # and the only granularity that matches these weights.
    # Gate on "the payload arrived unquantised and the weights are not BF16",
    # not on use_mxfp8, even though MXFP8 is the only checkpoint that REQUIRES
    # this branch. Two reasons, one of them the whole reason this generalisation
    # exists:
    #   * an FP8 block-128 checkpoint launched with
    #     --deepep-dispatcher-output-dtype bf16 otherwise reaches the contiguous
    #     GEMM with BF16 activations against FP8 weights, which is not a
    #     supported combination -- upstream only writes BF16 into input_tensor
    #     when the WEIGHTS are BF16 too;
    #   * it makes a known-good model a control for this route. The MXFP8 arm is
    #     wrong (0/4 parity) and the same image's FP8 block-128 DeepEP arm is
    #     right, but that arm ran the default FP8 dispatch -- so as of 2026-09-08
    #     the BF16 dispatch had never been exercised on a checkpoint whose
    #     correct answer we know. With this condition it can be:
    #     `DISPATCH_DTYPE=bf16 bash 97_a2a_control.sh deepep`.
    act_gran_k = (
        quant_info.block_shape[1]
        if (quant_info.use_mxfp8 and quant_info.block_shape)
        else 128
    )
    quantised_weights = quant_info.w13_weight.dtype != torch.bfloat16
    if quantised_weights and hidden_states.dtype == torch.bfloat16:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_group_quant_fp8,
        )

        assert hidden_states_scale is None, (
            "a BF16 DeepEP dispatch must not carry an activation scale; got one, "
            "so this is not the payload this branch was written for"
        )
        bf16_source = hidden_states
        hidden_states, hidden_states_scale = sglang_per_token_group_quant_fp8(
            hidden_states,
            act_gran_k,
            column_major_scales=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_tma_aligned=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
            scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        )
        dispose_tensor(bf16_source)
    elif quant_info.use_mxfp8:
        raise RuntimeError(
            f"DeepEP normal dispatch delivered {hidden_states.dtype} for an "
            f"MXFP8 checkpoint (weight_block_size [1, {act_gran_k}]). The FP8 "
            "dispatch quantises at a fixed 128 group, which does not match these "
            "weights: the arm then either trips deep_gemm's layout assert or, if "
            "that is silenced, serves fluent nonsense. Launch with "
            "--deepep-dispatcher-output-dtype bf16."
        )
'''

UE8M0_LANES = '''            (ceil_div(K // 128, 4), all_tokens),'''
UE8M0_LANES_NEW = '''            (ceil_div(K // act_gran_k, 4), all_tokens),'''

FP32_LANES = '''            (all_tokens, K // 128),'''
FP32_LANES_NEW = '''            (all_tokens, K // act_gran_k),'''

SCATTER = '''        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )'''
SCATTER_NEW = '''        scale_ue8m0=deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
        quant_block_size=act_gran_k,
    )'''

SETTER = '''    running_state["output_index"] = output_index'''
SETTER_NEW = '''    # Whatever group the activation was actually quantised at -- read by
    # _run_contiguous_gemm's recipe_a, whose default is block_shape[1].
    running_state["mxfp8_act_gran_k"] = act_gran_k

    running_state["output_index"] = output_index'''

EDITS = [
    (QUANT_BLOCK, QUANT_BLOCK_NEW),
    (UE8M0_LANES, UE8M0_LANES_NEW),
    (FP32_LANES, FP32_LANES_NEW),
    (SCATTER, SCATTER_NEW),
    (SETTER, SETTER_NEW),
]


def extract(image, dest):
    script = (
        'SGL=$(cd / && python3 -c "import sglang, os; '
        'print(os.path.dirname(sglang.__file__))"); '
        f'cat "$SGL/{TARGET}"'
    )
    out = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "bash", image, "-c", script],
        check=True, capture_output=True, text=True,
    ).stdout
    dest.write_text(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--after", action="append", default=[],
                    help="diff(s) to apply to the extracted file before cutting")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        base = td / "base.py"
        extract(args.image, base)
        for d in args.after:
            subprocess.run(["patch", "--forward", str(base), str(Path(d).resolve())],
                           check=True, capture_output=True)

        text = base.read_text()
        # Confine every edit to the one function: several of these strings
        # (the scatter tail, the dtype capture) also occur in its siblings, and
        # editing those would change paths this patch has no business touching.
        start = text.index(FUNC)
        end = text.index("@register_post_permute", start)
        body, new_body = text[start:end], text[start:end]
        for find, repl in EDITS:
            n = new_body.count(find)
            if n != 1:
                raise SystemExit(
                    f"FATAL: expected exactly 1 occurrence in "
                    f"pre_permute_deepep_normal_to_deep_gemm, found {n}:\n"
                    f"---\n{find}\n---\n"
                    "The nightly moved. Re-read the function and update this script."
                )
            new_body = new_body.replace(find, repl)

        new = td / "new.py"
        new.write_text(text[:start] + new_body + text[end:])
        compile(new.read_text(), "new.py", "exec")

        diff = subprocess.run(
            ["diff", "-u",
             "--label", f"a/python/sglang/{TARGET}",
             "--label", f"b/python/sglang/{TARGET}",
             str(base), str(new)],
            capture_output=True, text=True,
        )
        if diff.returncode != 1:
            raise SystemExit(f"FATAL: diff produced no change (rc={diff.returncode})")
        Path(args.out).write_text(diff.stdout)
        print(f"wrote {args.out} ({len(diff.stdout.splitlines())} lines, "
              f"{len(EDITS)} edits, base={args.image})")
        assert body != new_body


if __name__ == "__main__":
    main()
