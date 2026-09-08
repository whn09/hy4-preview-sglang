#!/usr/bin/env python3
"""Insert env-gated scatter statistics into the DeepEP-normal pre-permute.

WHY a separate debug edit instead of another entry in the patch set: this answers
one question and then should go away. Two DeepEP arms on the MXFP8 checkpoint --
fp8 dispatch (activation quantised at 128 against [1,32] weights) and bf16
dispatch (quantised locally at 32) -- produce the SAME kind of wrongness, worst
top-5 |dlogprob| 7.37 vs 7.48 against an a2a=none control pinned to the same
compact layout. A granularity error would not be indifferent to the granularity.
The remaining explanation with that signature is that the routed-expert
contribution is largely DROPPED, which would leave attention plus the shared
expert running and produce exactly the fluent prompt-echo we see.

ep_scatter decides what is dropped, in one line:

    valid = (expert_id >= 0) & (expert_id < num_experts)   # expert_id = id - expert_start

and the DeepEP-normal caller passes no `expert_start` (so 0) with
num_experts = len(num_recv_tokens_per_expert) = num_local_experts, i.e. 32 here.
That is only correct if deep_ep re-indexes recv_topk_idx to LOCAL ids. If it
returns global ids instead, rank 0's own experts (0..31) still validate and every
other rank drops everything -- 7/8 of the MoE lost, uniformly across dispatch
dtypes.

So print, for one layer, the id range that arrived and the fraction of
assignments ep_scatter actually placed. Two numbers settle it.

Usage (inside an image, edits site-packages in place):

    python3 patches/_dbg_scatter_stats.py

Then run the server with SGLANG_HY4_DBG_SCATTER=1. Output goes to the scheduler
log, once per rank, for the first sparse layer only -- printing every layer of 78
on 8 ranks buries the run and perturbs the timing of the very path being measured
(feedback_ep_buffer_debug_timing_trap).
"""
import os
import subprocess
import sys

TARGET = "srt/layers/moe/moe_runner/deep_gemm.py"
ANCHOR = """    dispose_tensor(hidden_states)
    if hidden_states_scale is not None:
        dispose_tensor(hidden_states_scale)
"""

DEBUG = '''
    # LOCAL DEBUG (SGLANG_HY4_DBG_SCATTER=1). Removed once the drop question is
    # answered; see patches/_dbg_scatter_stats.py for what it is asking.
    if os.environ.get("SGLANG_HY4_DBG_SCATTER") == "1":
        global _HY4_DBG_SCATTER_DONE
        try:
            _HY4_DBG_SCATTER_DONE
        except NameError:
            _HY4_DBG_SCATTER_DONE = False
        if not _HY4_DBG_SCATTER_DONE:
            _HY4_DBG_SCATTER_DONE = True
            import torch.distributed as _dist

            _rank = _dist.get_rank() if _dist.is_initialized() else -1
            _assign = topk_ids.numel()
            _placed = int((output_index >= 0).sum().item())
            # A rank can legitimately receive zero tokens (measured: rank 4 of 8
            # during warmup), and .min() on an empty tensor RAISES -- the debug
            # print then kills the scheduler it was added to observe.
            _lo = int(topk_ids.min().item()) if _assign else 0
            _hi = int(topk_ids.max().item()) if _assign else 0
            _neg = int((topk_ids < 0).sum().item())
            logger.warning(
                "HY4_DBG_SCATTER rank=%d n_local_experts=%d recv_tokens=%d "
                "all_tokens=%d assignments=%d placed=%d (%.1f%%) "
                "topk_id_range=[%d,%d] negatives=%d act_gran_k=%d",
                _rank,
                len(num_recv_tokens_per_expert),
                hidden_states_shape[0],
                all_tokens,
                _assign,
                _placed,
                100.0 * _placed / max(_assign, 1),
                _lo,
                _hi,
                _neg,
                act_gran_k,
            )
'''


def main():
    sgl = subprocess.run(
        [sys.executable, "-c",
         "import sglang, os; print(os.path.dirname(sglang.__file__))"],
        cwd="/", check=True, capture_output=True, text=True,
    ).stdout.strip()
    path = os.path.join(sgl, TARGET)
    text = open(path).read()

    if "HY4_DBG_SCATTER" in text:
        print("already instrumented")
        return
    # Anchor on the disposal pair that follows ep_scatter: output_index is written
    # by then, and hidden_states has not yet been freed out from under the read.
    n = text.count(ANCHOR)
    if n != 1:
        raise SystemExit(f"FATAL: anchor occurs {n} times, expected 1; the file moved")
    text = text.replace(ANCHOR, ANCHOR + DEBUG)
    if "\nimport os\n" not in text:
        text = text.replace("\nimport logging\n", "\nimport logging\nimport os\n", 1)
    compile(text, path, "exec")
    open(path, "w").write(text)
    print(f"instrumented {path}")


if __name__ == "__main__":
    main()
