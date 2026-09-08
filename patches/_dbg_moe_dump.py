#!/usr/bin/env python3
"""Dump one sparse layer's MoE boundary tensors, so two arms can be diffed offline.

WHY. As of 2026-09-08 every explanation for the DeepEP arm's wrongness on the
MXFP8 Hy4 checkpoint has been eliminated by an arm that WORKS:

  * the compact ep_scatter + contiguous grouped-GEMM chain at group 32 -- an
    a2a=none arm pinned to SGLANG_DEEPGEMM_STANDARD_LAYOUT=compact is coherent;
  * eager execution (deepep normal disables CUDA graphs) -- DISABLE_CUDA_GRAPH=1
    with a2a=none is coherent;
  * dropped routed tokens -- patches/_dbg_scatter_stats.py measured every recv
    token placed exactly once, local ids, non-local marked -1;
  * DeepEP-normal dispatch/ep_gather/combine themselves, at BOTH dispatch
    dtypes -- Qwen3-30B-A3B-FP8 (block 128) through the same image and the same
    ep=8 shape is coherent with fp8 AND with bf16 dispatch + local quantisation,
    i.e. the exact code path this kit's patch adds.

What Qwen does NOT exercise is the model glue: Qwen3MoE has its own sparse
block, while Hy4 goes through DeepseekV2MoE.forward_deepep. So stop reasoning
about the code and compare the tensors. Dump, for the first sparse layer of the
first real forward:

    hidden_states  the MoE input        -- must match across arms, else the two
                                          runs are not comparable at all
    topk_ids       the routing decision -- forward_deepep calls self.topk() with
                                          num_token_non_padded and an
                                          ExpertLocationDispatchInfo that
                                          forward_normal does not pass
    topk_weights   the gate values
    routed_out     self.experts(...) output, BEFORE routed_scaling_factor and
                   before the shared expert is added

Reading the diff: input differs -> harness error, not a bug; topk differs ->
gate/topk glue; only routed_out differs -> the expert path; all match -> the
scaling/shared-add tail (forward_deepep's own add_ vs forward_normal's
maybe_fuse_routed_scale_and_shared_add).

Usage (inside an image, edits site-packages in place):

    python3 patches/_dbg_moe_dump.py

Then run with SGLANG_HY4_DBG_MOE_DUMP=<tag>. Dumps are keyed by arm tag, layer
and rank, and land in /tmp inside the container.

The trigger is a FILE, /tmp/hy4_dbg_go, not the first forward: the first forwards
of any server are DeepGEMM warmup and CUDA-graph capture on dummy tokens, and
dumping those would compare two batches of garbage and look like a clean match.
Touch the file only after the server is healthy and immediately before the one
request being compared -- anything that generates in between (including
/health_generate) steals the dump.
"""
import os
import subprocess
import sys

TARGET = "srt/models/deepseek_v2.py"

HELPER = '''

# LOCAL DEBUG (SGLANG_HY4_DBG_MOE_DUMP=<tag>); see patches/_dbg_moe_dump.py.
_HY4_DBG_DUMPED = set()


def _hy4_dbg_armed():
    import os

    if not os.environ.get("SGLANG_HY4_DBG_MOE_DUMP"):
        return False
    # One layer only. Every later layer's input already depends on this layer's
    # output, so a second dump cannot distinguish cause from consequence -- and
    # 78 layers x 8 ranks of tensors fills /tmp.
    if _HY4_DBG_DUMPED:
        return False
    return os.path.exists("/tmp/hy4_dbg_go")


def _hy4_dbg_snapshot(x):
    # The MoE runner disposes its input IN PLACE (dispose_tensor resizes the
    # storage to 0), so reading hidden_states after self.experts() returns a
    # 0-row tensor: measured 2026-09-08, the dump logged "tokens=0
    # in_norm=0.000000" beside a (5, 8) topk. Clone before the call or there is
    # no input left to compare the two arms on.
    return x.detach().clone() if _hy4_dbg_armed() else None


def _hy4_dbg_dump(arm, layer_module, hidden_states, topk_output, routed_out):
    import os

    tag = os.environ.get("SGLANG_HY4_DBG_MOE_DUMP")
    if not tag or hidden_states is None:
        return
    layer_id = getattr(layer_module, "layer_id", -1)
    if _HY4_DBG_DUMPED:
        return

    import torch
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else -1
    _HY4_DBG_DUMPED.add((layer_id, rank))
    topk_weights, topk_ids = topk_output[0], topk_output[1]
    path = f"/tmp/moe_dump-{tag}-{arm}-layer{layer_id}-rank{rank}.pt"
    torch.save(
        {
            "arm": arm,
            "tag": tag,
            "layer_id": layer_id,
            "rank": rank,
            "hidden_states": hidden_states.detach().float().cpu(),
            "topk_ids": topk_ids.detach().cpu(),
            "topk_weights": topk_weights.detach().float().cpu(),
            "routed_out": routed_out.detach().float().cpu(),
        },
        path,
    )
    logger.warning(
        "HY4_DBG_MOE_DUMP wrote %s (tokens=%d topk=%s routed_norm=%.6f "
        "in_norm=%.6f)",
        path,
        hidden_states.shape[0],
        tuple(topk_ids.shape),
        float(routed_out.detach().float().norm()),
        float(hidden_states.detach().float().norm()),
    )
'''

# Anchored on the `self.experts(...)` call in each forward, because that is the
# one point where input, routing and routed output are all still in scope and
# none of them has been fused into anything yet.
NORMAL_ANCHOR = """        else:
            final_hidden_states = self.experts(
                hidden_states,
                topk_output,
            )
"""
NORMAL_NEW = """        else:
            _hy4_dbg_in = _hy4_dbg_snapshot(hidden_states)
            final_hidden_states = self.experts(
                hidden_states,
                topk_output,
            )
            _hy4_dbg_dump(
                "normal", self, _hy4_dbg_in, topk_output, final_hidden_states
            )
"""

DEEPEP_ANCHOR = """        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
"""
DEEPEP_NEW = """        _hy4_dbg_in = _hy4_dbg_snapshot(hidden_states)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        _hy4_dbg_dump("deepep", self, _hy4_dbg_in, topk_output, final_hidden_states)
"""


def main():
    sgl = subprocess.run(
        [sys.executable, "-c",
         "import sglang, os; print(os.path.dirname(sglang.__file__))"],
        cwd="/", check=True, capture_output=True, text=True,
    ).stdout.strip()
    path = os.path.join(sgl, TARGET)
    text = open(path).read()

    if "_hy4_dbg_dump" in text:
        print("already instrumented")
        return

    for name, anchor, new in (
        ("forward_normal", NORMAL_ANCHOR, NORMAL_NEW),
        ("forward_deepep", DEEPEP_ANCHOR, DEEPEP_NEW),
    ):
        n = text.count(anchor)
        if n != 1:
            raise SystemExit(
                f"FATAL: {name} anchor occurs {n} times, expected 1; the file moved"
            )
        text = text.replace(anchor, new)

    # The helper needs `logger`, which is defined right after the imports.
    marker = "logger = logging.getLogger(__name__)\n"
    if text.count(marker) != 1:
        raise SystemExit("FATAL: no unique logger definition to anchor the helper on")
    text = text.replace(marker, marker + HELPER, 1)

    compile(text, path, "exec")
    open(path, "w").write(text)
    print(f"instrumented {path}")


if __name__ == "__main__":
    main()
