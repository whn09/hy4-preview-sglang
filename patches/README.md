# Hy4-preview (HYV4) on `--moe-a2a-backend deepep_v2` — three patches

Base image: **`lmsysorg/sglang:hy4-preview`** (sglang `0.5.18`, python 3.12).
The image already ships `deep_ep` and `deep_gemm` in
`/usr/local/lib/python3.12/dist-packages`, so nothing is installed — this is a
three-file source patch and nothing else.

Build and run:

```bash
docker build -t hy4-preview-v2:latest -f Dockerfile.deepep_v2 .
IMAGE=hy4-preview-v2:latest A2A_BACKEND=deepep_v2 CHUNKED_PREFILL=2048 \
  bash 10_launch_standalone.sh
```

`CHUNKED_PREFILL` is not optional on this backend and the default would refuse to
boot — see "Capacity" below.

The diffs are marked `LOCAL EXPERIMENT ONLY`. **Passing a gate is not a
correctness proof** — and here that warning has teeth, because blocker 3 is a
numerics bug, not a gate. Nothing in this directory has been run yet.

| # | file | what | kind |
|---|---|---|---|
| 1 | `moe_hook.diff` | architecture whitelist | validation |
| 2 | `fmt_layer.diff` | the quant gate rejects MXFP8 outright | validation |
| 3 | `mr_deep_gemm.diff` | MXFP8 activation granularity never set on the v2 path | **real bug** |

## Why three, when Kimi-K3 needed five

`../../kimi-k3-sglang/patches/README.md` lists five blockers for K3 on the same
backend. Three of those five **do not apply to Hy4**, and each was checked against
the extracted source of this exact image rather than assumed:

| K3 blocker | Hy4 |
|---|---|
| model file's hand-rolled EP-a2a list omits v2 | **N/A.** `srt/models/hunyuan_v4.py` keeps no a2a bookkeeping at all: `:533` builds `DeepseekV2MoE` and `:583` calls `self.mlp(hidden_states, forward_batch)`. All the DP-gather / TP-reduce logic lives in `deepseek_v2.py`, which handles `is_deepep_v2()` at `:757`, `:847`, `:870`, `:2776` and `:3078`. Hy4's MoE region **is** the whitelisted DeepseekV3 region. |
| `assert activation == "silu"` | **N/A.** `config.json` has `hidden_act: "silu"`. (K3's is `situ`.) |
| `ep_scatter_from_psum` missing kernel args (sglang PR #37211) | **N/A in this image.** `kernels/ops/moe/ep_moe_kernels.py:1252` and the `_fwd_kernel_ep_scatter_2` definition at `:1056` agree exactly — neither takes `expert_start`/`num_experts`. Applying K3's diff here would *introduce* the bug. |

That last row is the reason these patches are re-cut rather than symlinked: the
K3 set is correct for a 2026-09-01 nightly and wrong for this image.

## Blocker 1 — architecture whitelist (`moe_hook.diff`)

`srt/arg_groups/moe_hook.py:427` `validate_deepep_v2_model_architecture`:

```
ValueError: DeepEP v2 MoE is not validated for 'HYV4ForCausalLM'; supported
architectures are ['DeepseekV3ForCausalLM', 'DeepseekV4ForCausalLM',
'Qwen3MoeForCausalLM']. Other model workflows may require an all-reduce after
A2A combine. Use --moe-a2a-backend deepep.
```

Fires during config resolution, before any weight load.

The error's own rationale — "may require an all-reduce after A2A combine" — is
what makes this one safe **here** and not on K3: the hazard is a model file that
does its own reduction while the v2 dispatcher has already combined. Hy4 has no
such code (see the table above), so adding `HYV4ForCausalLM` is adding a model
whose MoE region is already on the list under a different name.

## Blocker 2 — the quant gate rejects MXFP8 (`fmt_layer.diff`)

`srt/layers/moe/fused_moe_triton/layer.py:239` `_validate_deepep_v2_quant_method`,
called per MoE layer from `FusedMoE.__init__`:

```
ValueError: --moe-a2a-backend deepep_v2 requires 128x128 blockwise FP8 experts
with dynamic activation scaling, but this layer selected MXFP8 weights.
Use a compatible checkpoint or --moe-a2a-backend deepep.
```

Hy4 hits it on the **second** branch (`quant_method.use_mxfp8`), and would then
hit the fourth (`weight_block_size != [128,128]`; Hy4's is `[1,32]`), so the patch
returns early for the MXFP8 case rather than deleting one condition.

It is stricter than the kernels behind it — **both** deep_gemm v2 paths already
carry a full `use_mxfp8` branch:

| | evidence |
|---|---|
| contiguous runner | `moe_runner/deep_gemm.py:362` → `recipe_a = recipe_b = tuple(block_shape)` |
| masked runner | `:637-651` → `recipe_b = block_shape`, `recipe_a = (block_shape[0], gran_k_act)` |
| the runner constructor validates MXFP8 explicitly | `:284-291` — "MXFP8 requires block_shape [1, 32]", "MXFP8 requires DEEPGEMM_SCALE_UE8M0=True" |
| the v2 dispatcher's activation quant is method-agnostic | `token_dispatcher/deepep_v2.py:27` `_SCALE_BLOCK_SIZE = 128`, `:124` `sglang_per_token_group_quant_fp8` |

The gate also requires `--moe-runner-backend deep_gemm`, which this kit already
passes for MXFP8 (`start_server.sh:59`) and which the runtime defaults to for HYV4
anyway.

## Blocker 3 — `mxfp8_act_gran_k` is never set on the v2 path (`mr_deep_gemm.diff`)

**This is the one that is not a loosened gate.** Two hunks.

The MXFP8 gateup GEMM needs to know at what granularity the *activations* were
quantised, and that is a property of the **dispatch** path, not of the checkpoint.
Upstream says so itself, at `moe_runner/deep_gemm.py:640`:

```python
# gran_k is set by the dispatch path (standard=block_shape[1], DeepEP-LL=128),
# not inferable from K; inferring it silently mis-reads the activation scale.
gran_k_act = running_state.get("mxfp8_act_gran_k", quant_info.block_shape[1])
```

Three dispatch paths set the key — `:939` (standard), `:1071` (deepep_normal),
`:1144` (deepep_ll, explicitly `= 128`). **`pre_permute_deepep_v2_to_deep_gemm`
sets it nowhere**, and the comment above does not mention v2. So on an MXFP8
checkpoint the masked v2 path falls back to `block_shape[1]` = **32** while the v2
dispatcher emitted **128**-group scales.

* **Hunk A** sets `running_state["mxfp8_act_gran_k"] = 128` in
  `pre_permute_deepep_v2_to_deep_gemm` (`:1510`), mirroring the DeepEP-LL path,
  because `_SCALE_BLOCK_SIZE` in the v2 dispatcher is a module constant and is not
  conditioned on the quant method.
* **Hunk B** makes `_run_contiguous_gemm` (`:362`) read the same key instead of
  inferring `recipe_a` from `block_shape`. For every non-v2 dispatch the `.get()`
  returns `block_shape[1]`, i.e. byte-identical to current behaviour — the masked
  runner has always done it this way, and only the contiguous one hard-coded it.

How it fails without hunk A, and this is the useful part: the masked path
*asserts*, so the symptom is loud rather than silent. With `K = 6144`
(`hidden_size`) it computes

```
assert ceil_div(K, gran_k_act * 4) == act_sf_last
      gran_k_act = 32  -> ceil_div(6144, 128) = 48
      dispatch carries ceil_div(6144 / 128, 4) = 12       -> AssertionError
```

The **contiguous** path (hunk B) has no such assert: it would hand deep_gemm a
`recipe_a` of `(1,32)` for a 128-grouped scale tensor and produce wrong numbers.
That is why this is filed as a bug rather than as a gate, and why
`require_deepep_v2_image()` verifies a marker instead of trusting `IMAGE`.

## Not blockers, but they will bite

* **Capacity, and this one blocks the default launch.**
  `validate_deepep_v2_dispatch_token_budget` (`moe_hook.py:361`) checks the env var
  `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK` against two budgets:

  | budget | value here | vs `V2_CAP=2048` |
  |---|---|---|
  | prefill: `max_prefill_buffer_tokens()` = `chunked_prefill_size` | **16384** — `memory_hook.py:124` defaults it that way for any GPU >160 GiB, and B300 has 279 | **fails**: `required=16384, capacity=2048` |
  | decode: `graph_bs * tokens_per_request` | `graph_bs` = min(cuda-graph max_bs **512**, `max_running_requests`/dp), tokens/req **4** with MTP | fits only at the 512 default; a lower `V2_CAP` needs `MAX_RUNNING` |

  So a v2 arm needs `CHUNKED_PREFILL` set. `build_moe_args()` refuses without it
  rather than lowering it silently — a chunk size chosen for v2 alone would make
  every prefill comparison a chunk-size comparison too, so **the v1 baseline must
  get the same `CHUNKED_PREFILL`**. It is in the results tag (`-chunk<N>`) for the
  same reason. Note the direction of the trade: on K3, prefill throughput wanted
  `CAP >= ISL` (2048→8192 was +158% tok/s), and here the cap is what forces the
  chunk down — so expect the v2 arm to lose prefill throughput to the chunk before
  any a2a effect is visible, and read the two phases separately.

  Capacity costs GPU memory: on K3 the ElasticBuffer ran ~10.5 GiB per 1024 and
  4096 OOM'd. **Hy4's ceiling is unmeasured**, and at TP8 there is more headroom
  than K3 had, so do not carry the K3 number over as if it were measured here.
* **BF16 cannot use v2 at all.** The quant gate rejects non-`Fp8MoEMethod` layers
  on its first branch, before anything this patch touches. `build_moe_args()`
  refuses the combination up front.
* **`hidden_size % 128`.** `deepep_v2.py:267` requires it; Hy4's 6144 is fine.

## Validating, before quoting any number

The v1 baseline runs unpatched on the stock image and is the reference:

```bash
A2A_BACKEND=deepep bash 10_launch_standalone.sh      # v1, stock image
```

Then compare **first-token top-5 logprobs**, not prose. A coherent-looking
completion is not evidence: blocker 3's contiguous half produces a *plausible*
answer from a mis-scaled GEMM. `95_features_test.py` has the probe prompts.

## Re-cutting a diff against a new base image

```bash
P=/opt/dlami/nvme/patch; mkdir -p $P/orig
cid=$(docker create --entrypoint true lmsysorg/sglang:hy4-preview)
B=/sgl-workspace/sglang/python/sglang
docker cp $cid:$B/srt/arg_groups/moe_hook.py               $P/orig/moe_hook.py
docker cp $cid:$B/srt/layers/moe/fused_moe_triton/layer.py $P/orig/fmt_layer.py
docker cp $cid:$B/srt/layers/moe/moe_runner/deep_gemm.py   $P/orig/mr_deep_gemm.py
docker rm -f $cid
for f in moe_hook fmt_layer mr_deep_gemm; do
  cp $P/orig/$f.py $P/$f.py
  patch $P/$f.py < patches/$f.diff        # edit, then:
  diff -u $P/orig/$f.py $P/$f.py > patches/$f.diff
done
```

The diffs are applied **by target path**, not with `-p<n>`: they were captured
against a `docker cp`'d copy, so no strip level reaches the real tree. The
Dockerfile's apply loop is idempotent and fail-closed — forward, else
already-applied (proved by a reverse dry-run), else **fail the build**. A build
break here is the intended signal that the base image moved.
