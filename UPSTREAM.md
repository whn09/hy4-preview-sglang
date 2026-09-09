# What of this kit belongs upstream, and what does not

**Submitted 2026-09-09**, all on `sgl-project/sglang` `main` @ **`76eea36e38`**, from the
`whn09/sglang` fork:

| upstream | what | branch |
|---|---|---|
| [issue #38606](https://github.com/sgl-project/sglang/issues/38606) | D — HYV4 + a2a computes 1 token in 8 | — |
| [PR #38607](https://github.com/sgl-project/sglang/pull/38607) | the `model_overrides` declaration that fixes D | `fix/hyv4-attn-tp-gather` |
| [PR #38608](https://github.com/sgl-project/sglang/pull/38608) | A — `get_model_config_for_expert_location` | `fix/hyv4-expert-location` |
| [PR #38609](https://github.com/sgl-project/sglang/pull/38609) | A′ — the assert before the early return | `fix/eplb-assert-order` |

C and B1 below are **not** submitted; the rest of this file is the triage they came from.

Re-verified 2026-09-09 against `sgl-project/sglang` `origin/main` @ **`8ab9982851`**
(2026-09-09 09:44 +0800), i.e. four days after HYV4 merged (PR #36805, `55bf338`).
**Every finding below is still present at that tip**, and `gh search issues --repo
sgl-project/sglang` finds no existing issue or PR for any of them.

The honest frame for all of it is **enablement and correctness, not speed**. We have no
throughput argument to make: the one numerically-correct Hy4 DeepEP arm we ever got is
**−6.6%** against a matched-eager `a2a=none` control and 5.5x slower than the graph-on
one (`results/a2a_backends_tp8_c64.md`), and `deepep_v2` has never served a token on this
model. Any PR that implies "DeepEP is now good on Hy4" would be unsupported.

## Triage

| # | finding | upstream site @ `8ab9982851` | our artifact | verdict |
|---|---|---|---|---|
| D | HYV4 + DeepEP silently computes 1 token in 8 | `utils/common.py:3879`, `models/hunyuan_v4.py:606` | flag `--disable-attn-tp-gather` | **sent: #38606 + #38607** |
| A | HYV4 has no expert-location hook ⇒ bare `AssertionError` | `models/hunyuan_v4.py:667`, `eplb/expert_location.py:758` | `patches/hunyuan_v4.diff` | **sent: #38608** |
| A′ | the assert fires on a value the next line discards | `eplb/expert_location_dispatch.py:45` vs `:47` | — | **sent: #38609** |
| C | DSV4 silu post-quant kernel only exists at group 128 | `kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh:117,263,403,486` | `patches/silu_group32.diff` | **PR, needs a test** |
| B1 | DeepEP-**normal** pre-permute hard-codes group 128 | `moe_runner/deep_gemm.py:1333` (`:1381`, `:1387`) | `patches/mr_dg_normal.diff` | **PR, needs discussion** |
| B2 | DeepEP-**v2** pre-permute sets no `mxfp8_act_gran_k` | `moe_runner/deep_gemm.py:1672` | `patches/mr_deep_gemm.diff` hunk A | hold — untested |
| F | bf16 dispatch + masked runner crashes on `None` | `moe_runner/deep_gemm.py` masked path | the guard in `mr_dg_normal.diff` | fold into B1 |
| E | v2 arch whitelist + v2 MXFP8 quant gate | `arg_groups/moe_hook.py:456`, `fused_moe_triton/layer.py` | `moe_hook.diff`, `fmt_layer.diff` | **hold** — no working arm |

D and A are worth submitting **independently of everything else we are doing**: they
break, and silently corrupt, every DeepEP arm on Hy4 at any quantisation. C and B1 are
what the MXFP8 checkpoint additionally needs. E is ours to keep local until a v2 arm
exists.

## D — the one that should be reported whether or not we send code

Severity first: with `--moe-a2a-backend deepep*` and DP attention off,
`require_attn_tp_gather()` (`utils/common.py:3879`) returns True on "the a2a backend is
not none" alone, so the scheduler hands the MoE an attn-TP **sequence-shard**
`num_token_non_padded` (`forward_batch_info.py:259`). That contract holds for models
whose `LayerCommunicator` gives the MoE the scattered shard. `hunyuan_v4.py` has no
`LayerCommunicator` at all — it uses `hc_attn_layer`/`hc_mlp_layer` and calls
`self.mlp(hidden_states, forward_batch)` on the **full padded width** on every rank
(`:606-607`). Measured on B300-1, tp8/ep8, a 5-token prompt padded to 8:

| | `a2a=none` | `a2a=deepep` |
|---|---|---|
| `topk_ids` | 5 real rows | row 0 real, **rows 1-7 all `-1`** |
| `routed_out` | 8 distinct EP partials | r0-r4 bit-identical, **r5-r7 exactly `0.0`** |
| first-token top-5 \|dlogprob\| vs reference | — | **7.48** (0.410 with the flag; floor 0.593) |
| out tok/s @ c=64 | 2324.38 | **2711.81 (+16.7%)** |

The last row is why this is worth an issue on its own: the broken arm is **faster**, the
server is healthy, and the completions are fluent. Nothing short of a logprob comparison
catches it, and `docs/src/snippets/configs/tencent/hy4-preview.jsx` ships
`--moe-a2a-backend deepep` as a Playground toggle, so this is reachable from the
documented UI.

The fix is not the flag — the flag is our workaround. Upstream has the idiomatic channel
already: `arg_groups/model_overrides/` (30 files, keyed on `hf_config.architectures[0]`,
`@_register_for(...) -> dict`). Two things about it that were not obvious and that
**PR #38607** had to get right:

* HYV4 is **already claimed** by `model_overrides/deepseek_v2.py` (it is in that module's
  `@_register_for` list, and `__init__.py` forbids two modules declaring one field for one
  arch), so the declaration goes in that file's existing HYV4 branch — *not* in a new
  `model_overrides/hunyuan_v4.py`, which is what an earlier draft of this file said.
* `disable_attn_tp_gather` was not tagged `resolvable=True`, so a declaration would have
  been rejected; tagging it obliges extending the pinned resolvable-field set in
  `test/registered/unit/test_model_overrides.py:57` in the same commit.

The declaration is also **unconditional**, not gated on `moe_a2a_backend != "none"`:
`overrides.py:1668`'s post-process turns `--enable-waterfill` into
`moe_a2a_backend=deepep` *after* the model overrides run, so a gate there would let the
wrong-output path back in. It is a no-op whenever `require_attn_tp_gather()` would have
returned False anyway. `require_attn_tp_gather`'s own comment already describes HYV4's case:
*"Opt-out for models that manage SP scatter/gather at the model level and do not consume
the upstream gathered_buffer."* HYV4 is such a model and simply is not opted out.

The alternative — give `hunyuan_v4.py` a real `LayerCommunicator` — is upstream's design
call and much larger; the issue should offer both and let them pick.

## A — the missing hook, and A′ the assert that hides it

`HYV4ForCausalLM` is `nn.Module, DeepseekV2WeightLoaderMixin` (`:667`), so it does not
inherit `DeepseekV2ForCausalLM.get_model_config_for_expert_location`, while its sparse
layers *are* `DeepseekV2MoE` (`:556`) and do take `forward_deepep`. Absent hook ⇒
`expert_location.py:758`'s `hasattr` is False ⇒ metadata stays `None` ⇒
`expert_location_dispatch.py:45` `assert expert_location_metadata is not None`.

Two things make this a good PR: the body is DeepSeek's unchanged (the checkpoint uses
`num_hidden_layers=78`, `n_routed_experts=256`, `n_group=1` verbatim), and the symptom is
maximally misleading — a **bare** `AssertionError`, no message, ~3 min after weight load,
during decode graph capture, so it reads as a graph/memory failure.

A′ is separable and helps every model, not just Hy4: at `:45` the assert runs *before*
`:47`'s `if ep_dispatch_algorithm is None: return None`. When no EPLB dispatch algorithm
is configured — the default — the assert fires on a value the function was about to
discard. Swapping the two lines turns "crash for any model lacking the hook" into "works,
as it would have anyway". Send it as its own 2-line PR; do not use it as a reason to skip
A, because EPLB does need the hook.

## C — generalise the kernel instead of substituting a different activation

`silu_and_mul_masked_post_quant.cuh` already takes `kGroupSize` as a template parameter
and then ignores it: both device kernels shadow it with `constexpr uint32_t kGroupSize =
128u` (`:117`, `:403`) and both host wrappers `static_assert(kGroupSize == 128)` (`:263`,
`:486`). `patches/silu_group32.diff` deletes the shadowing constants, derives
`kWorkThreads = kGroupSize / 8u`, and forwards the parameter — 8 edits, no launch-config
change (`num_threads = hidden_dim / 8` either way; only the partition within the block
moves, and `warp::reduce_max<kWorkThreads>` is already width-generic for any power of two
≤ 32).

Why bother upstream rather than keep it local: the only other way to serve an MXFP8
`[1,32]` checkpoint through DeepEP is the Triton kernel, whose clamp lives inside
`if GEMM1_ALPHA > 0` (a *different* activation), and `assert swiglu_limit is None` blocks
the combination anyway. So the alternative is silently wrong, which is the exact hazard
B1 documents.

What a reviewer will ask, and where we stand:

* **numerics.** The *contiguous* instantiation at 32 is validated end-to-end: it is what
  the corrected v1 arm runs, and that arm agrees with the `a2a=none` reference on the MoE
  boundary dump (`rel_l2=0` input, identical `topk_ids`, `routed_out` ratio constant at
  `1/routed_scaling_factor`) and lands at the parity noise floor. The **varlen/masked**
  instantiation at 32 is **compile-validated only** — it is on the v2 masked path we have
  never run. Say so in the PR.
* **a test.** There is none to extend: `git ls-tree` finds no unit test for this kernel
  (`test_inkling_silu_and_mul.py` is a different kernel). A `group_size ∈ {32,128}`
  reference-vs-kernel test would have to come with the PR. That is the main cost.
* keep `_masked_activation_unsupported_reason`'s `group_size != 128` guard
  (`deep_gemm.py:~185`) **out** of this PR. Relaxing it changes the *standard* path's
  masked-vs-compact choice for every group-32 MXFP8 model, which is a behaviour change we
  have not measured. Kernel capability first.

## B1 — the DeepEP-normal path, and the argument it has to win

`pre_permute_deepep_normal_to_deep_gemm` (`:1333`) fixes the activation group at 128 in
three places — the scale buffer's lane count (`:1381`, `:1387`), `ep_scatter`'s default
`quant_block_size`, and the `mxfp8_act_gran_k` it never writes — while
`pre_permute_deepep_ll_to_deep_gemm` sets the key (`:1305`) and
`pre_permute_standard_to_deep_gemm` both sets it and passes
`quant_block_size=block_shape[1]` (`:1139`). Upstream's own comment at `:700` names the
rule ("gran_k is set by the dispatch path … not inferable from K; inferring it silently
mis-reads the activation scale") and lists only standard and DeepEP-LL. So this is a gap
in a rule upstream already wrote down.

The part that needs discussion is that **setting the key to 128 is the wrong fix here**.
There is no MXFP8 DeepEP dispatch on CUDA at all (`token_dispatcher/deepep.py:486`,
`:494` — Ascend A5, low-latency only), so an MXFP8 checkpoint must dispatch **bf16** and
quantise at the checkpoint's own group in the pre-permute, which is exactly what the
standard path does. Declaring the 128-grouped fp8 dispatch acceptable instead is measured
to serve fluent nonsense (0/4 prompts, worst top-5 `|dlogprob|` 7.37, degenerate
repetition) at 2711 tok/s. `mr_dg_normal.diff` therefore does the real thing: quantise
bf16 dispatch output here at `block_shape[1]`, plumb `act_gran_k` into the scale buffer,
`ep_scatter` and `running_state`, and raise a clear error on fp8+MXFP8 instead of a
`layout.hpp:108` assert. ~60 lines, and the F row folds in for free: a bf16 dispatch into
the *masked* runner currently dies on `hidden_states_scale` being `None` rather than
saying so.

This PR depends on C (without the group-32 kernel the arm cannot reach the GEMM), so
either send C first or say plainly that B1 is only exercisable on top of it.

## Hold: E, the two v2 gates

`moe_hook.diff` (add `HYV4ForCausalLM` to `validated_architectures`, `:456`) and
`fmt_layer.diff` (let MXFP8 `[1,32]` through `_validate_deepep_v2_quant_method`) are
**validation gates, and they are doing their job**. Proposing to loosen a gate whose
error message says "is not validated" requires a validated arm, and we do not have one:
`deepep_v2` has never served a request on Hy4 — the nightly's NCCL 2.30.7 blocks GIN
(which deep_ep v2 asserts even single-node), and on the EFA image the arm stops at C's
kernel. Its masked runner also needs an MXFP8 dispatch that does not exist on CUDA, so
even with the gates open there is no complete path today. B2 is held for the same reason:
one line, zero runtime evidence.

If the v2 arm ever comes up and passes `96_logprob_parity.py`, these two become the
natural follow-up PR — with the measurement attached, which is the only thing that would
make them reviewable.

## Order to send

1. ~~**D as an issue**, with the override PR attached as the cheap fix.~~ **Sent**:
   [#38606](https://github.com/sgl-project/sglang/issues/38606) +
   [#38607](https://github.com/sgl-project/sglang/pull/38607).
2. ~~**A** and **A′** as two small PRs.~~ **Sent**:
   [#38608](https://github.com/sgl-project/sglang/pull/38608) (12 lines) and
   [#38609](https://github.com/sgl-project/sglang/pull/38609) (6/−3). #38609 does not
   depend on #38608.
3. **C**, once the group-32 unit test exists.
4. **B1**, on top of C, with the "no MXFP8 dispatch on CUDA" reasoning stated up front.

What the three sent PRs say about validation, so a reviewer's reply is not a surprise:
each one states that its unit tests were **not run locally** (no torch-capable
environment on the Mac they were written on), #38607 states plainly that it makes the arm
**slower** (2711.81 → 2324.38 out tok/s at c=64, which is the cost of routing 8× the
tokens), and #38608/#38609 carry no perf claim at all.

Nothing here is blocked on hardware. A, A′ and D are pure source changes reviewable
without a B300; C and B1 want one B300-hour each to re-confirm on a fresh `main` build
before submission, because our evidence was taken on `nightly-dev-cu13-20260908-20ca564b`
plus six patches, not on `8ab9982851`.
