# Hy4-preview (HYV4) on DeepEP — four patches **and a library swap**

Base image: **`hy4-preview-efa:latest`**, not the stock `lmsysorg/sglang:hy4-preview`.
That is a requirement, not a preference — see "Blocker 7" below. sglang `0.5.18`,
python 3.12.

Two independent things are wrong on the stock image, and only one of them is a
patch:

* **four source patches**, listed below. Patch 4 is needed by v1 `deepep` as well
  as by v2, so on the stock image *no* DeepEP arm on Hy4 reaches a served token.
  The other three do nothing unless `moe_a2a_backend == deepep_v2`.
* **the DeepEP library itself is v1-only** (`sgl-deep-ep 0.1.0`, no
  `ElasticBuffer`). No patch can fix that; `Dockerfile.deepep_v2` replaces the
  package with `deep_ep 2.1.0+97d8f9b` out of `kimi-k3-efa-v2:nccl2312`. That
  library needs NCCL ≥ 2.31, which is why the base must be the EFA image.

Build and run:

```bash
docker build -t hy4-preview-v2:latest -f Dockerfile.deepep_v2 .
IMAGE=hy4-preview-v2:latest A2A_BACKEND=deepep_v2 CHUNKED_PREFILL=2048 \
  MEM_FRACTION=0.92 bash 10_launch_standalone.sh
```

`MEM_FRACTION` is there because the ElasticBuffer is memory the `none` arm never
pays for — see "Not blockers" below.

`CHUNKED_PREFILL` is not optional on this backend and the default would refuse to
boot — see "Capacity" below.

The diffs are marked `LOCAL EXPERIMENT ONLY`. **Passing a gate is not a
correctness proof** — and here that warning has teeth, because blockers 3 and 4
are bugs, not gates.

| # | file | what | kind | affects |
|---|---|---|---|---|
| 1 | `moe_hook.diff` | architecture whitelist | validation | v2 |
| 2 | `fmt_layer.diff` | the quant gate rejects MXFP8 outright | validation | v2 |
| 3 | `mr_deep_gemm.diff` | MXFP8 activation granularity never set on the v2 path | **real bug** | v2 |
| 4 | `hunyuan_v4.diff` | `get_model_config_for_expert_location` missing on HYV4 | **real bug** | v2 **and v1** |

Three more blockers are *not* patches:

* **5** is an env flag — `--speculative-moe-a2a-backend`, handled by `env_common.sh`.
* **6+7** are one missing kernel, and they are where both DeepEP arms currently
  stop. See **"The wall"** below. Nothing in this directory fixes that, and
  nothing can: clamped-swiglu post-quant is unimplemented at `group_size 32`,
  which is Hy4's MXFP8 block shape.

So as of 2026-09-05 these patches are necessary but **not sufficient** on the
MXFP8 checkpoint. They are still worth keeping: they close every blocker up to the
kernel, so a fixed kernel is the only thing standing between this kit and a
working v2 arm.

## Why four, when Kimi-K3 needed five

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

## Blocker 4 — HYV4 has no expert-location hook (`hunyuan_v4.diff`)

**This one is not a v2 problem at all — it breaks v1 `deepep` identically**, and
it is the reason the stock image has no working DeepEP arm on Hy4.

The symptom, measured 2026-09-05 on B300-3 (TP8, MXFP8, MTP on):

```
File ".../srt/layers/moe/token_dispatcher/deepep.py", ... ExpertLocationDispatchInfo.init_new(...)
File ".../srt/eplb/expert_location_dispatch.py", line 45, in init_new
AssertionError
```

A bare `AssertionError` with no message, ~3 min after the weight load finishes,
during decode CUDA-graph capture. Everything before it looks healthy — including
the line that says DeepEP v2 is enabled — so it reads like a graph-capture or
memory problem rather than a missing model method.

The chain:

| step | file | what |
|---|---|---|
| the hook is looked up by name | `eplb/expert_location.py:762` `ModelConfigForExpertLocation.from_model_config` | `getattr(model_class, "get_model_config_for_expert_location", None)`; **returns `None` when absent** |
| `None` propagates | `init_trivial` → `_init_common` | global metadata stays `None` |
| the assert | `eplb/expert_location_dispatch.py:45` `init_new` | asserts the metadata is not `None` — *before* its own `if ep_dispatch_algorithm is None: return None` on the next line, so it fires on a value it was about to discard |

`DeepseekV2ForCausalLM` defines the hook (`deepseek_v2.py:3206`).
`HYV4ForCausalLM` is `nn.Module, DeepseekV2WeightLoaderMixin` — it reuses
DeepSeek's *layers*, not its *model class* — so it never inherits it, while its
sparse layers are `DeepseekV2MoE` and do go through `forward_deepep`. Hence: the
model is DeepSeek enough to take the DeepEP path and not DeepSeek enough to
answer the question that path asks.

The patch is DeepSeek's body unchanged, because the Hy4 checkpoint uses the same
three config keys verbatim:

| field | source key | Hy4 value |
|---|---|---|
| `num_layers` | `num_hidden_layers` | 78 |
| `num_logical_experts` | `n_routed_experts` | 256 |
| `num_groups` | `n_group` | 1 |

So this is a genuinely missing 6-line hook, not a workaround — worth reporting
upstream against PR #36805.

## The other two, which are not patches

**Blocker 5 — the speculative draft inherits the backend.** Config-time, and it
kills the launch ~45 s in, *after* the log says DeepEP v2 is on:

```
ValueError: DeepEP v2 MoE is not validated as a speculative draft backend
```

`validate_deepep_v2_speculative_draft` (`moe_hook.py:343`). With MTP on, the
draft model inherits `moe_a2a_backend`, and upstream refuses v2 there. Upstream
also names the escape hatch, so no patch is needed:
`--speculative-moe-a2a-backend`. `env_common.sh` passes it as
`SPEC_A2A_BACKEND` (default `none`) whenever `A2A_BACKEND=deepep_v2` and
`SPEC=on`, and puts `draft<backend>` in the results tag when it is not `none`.
This keeps the v2 code under test on the target model only — which is also what
makes the arm honest: the draft is *not* running v2.

**Blockers 6 and 7 — one missing kernel, and it is where both DeepEP arms stop.**
See "The wall" below. They looked like two bugs and are one: `deepep` (v1) hits it
at *run* time and `deepep_v2` at *compile* time.

## Three environment blockers between the library and the kernel

All three were hit on the way to "The wall", all three look like something else
entirely, and all three are handled by the kit now — recorded so they are not
re-diagnosed.

**`cuobjdump` is missing from this image's `CUDA_HOME`.** deep_ep's C++ JIT compiles
a kernel and then shells out to `$cuda_home/bin/cuobjdump -symbols kernel.cubin` to
find the symbol to launch (`csrc/jit/kernel_runtime.hpp:24`), asserting the exit
code at `:33`. This image's `CUDA_HOME` is the pip nvcc wheel
(`.../nvidia/cu13`), which ships `nvcc` and `ptxas` but not `cuobjdump` — that is a
separate wheel. The symptom is maximally misleading: every rank compiles
*successfully* (ptxas even reports `Used 42 registers`) and then dies with
`Assertion exception ... exit_code == 0` and **no other output**, surfaced as
`Capture cuda graph failed`. The K3 image is unaffected only because its
`CUDA_HOME` is `/usr/local/cuda`. `Dockerfile.deepep_v2` symlinks the one binary;
the marker asserts it. Note the assert is in `kernel_runtime.hpp`, not
`compiler.hpp` — reading it as a *compiler* failure sends you to look at nvcc
flags, which are fine.

**GIN type 5 cannot come up on a single-node instance.** `--device=/dev/infiniband`
is required even at `NNODES=1`, because deep_ep asserts a GIN backend exists
(`csrc/kernels/backend/nccl.cu:87`) whether or not anything crosses the wire. But
`NCCL_GIN_TYPE=5` (EFA_GDA) then resolves to `NCCL_GIN_TYPE_NONE` and fails the
assert at `nccl.cu:101`; the same box needs **type 3** (GDAKI) for a single-node
`direct` instance, and type 5 for a cross-node `hybrid` one. K3 measured the same
rule from the other side (`nccl.cu:188`, "DevComm setup failed on all available
backends"). This costs nothing — a single-node instance keeps its a2a on NVLink and
never touches a NIC; GIN merely has to exist. `export_gin_envs()` in
`env_common.sh` resolves it from `NNODES` plus the device list, so it is one
function rather than a flag to remember, and echoes what it chose:

```
GIN: type=3 sym_kernels=default hca=ibp198s0f0 (nnodes=1, devices: 16 rdmap + 2 other)
```

On the cross-node path it also sets the two settings that always travel with type
5 — `NCCL_SYM_GIN_KERNELS_ENABLE=0` and `NCCL_IB_HCA=rdmap` (a *prefix*, covering
all 16 rails, mandatory because this box's device list is mixed: 16 `rdmap*` EFA
plus 2 `ibp*` ConnectX-7).

**GIN type 5 also needs `--privileged` and `/dev/gdrdrv`, and says "network" when it
does not have them.** This one is invisible at `NNODES=1` and cost the most time, so
it is worth stating precisely. On the 2-node arm, with the two settings above in
place and echoed, with `/dev/gdrdrv` present *on the host*, and with the 16-rank TP
communicator already running healthily over EFA (`NET/Libfabric/0..7/GDRDMA`), the
run still died at `nccl.cu:101` — this time on `props.railedGinType`, which is the
field `hybrid` mode reads where `direct` reads `props.ginType`. At
`NCCL_DEBUG_SUBSYS=GIN` the real sequence is visible:

```
NCCL WARN NET/OFI Failed to initialize GDRCopy: Failed to open gdr handle
NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)
NCCL INFO NET/OFI gin: Initializing
NCCL INFO GIN/Plugin: Failed to initialize any GIN plugin
```

The GIN plugin loads, tries to initialize, cannot get a GDRCopy handle, and gives
up — and NCCL then reports the rail team's GIN type as `NONE`, which DeepEP
correctly but unhelpfully reports as "usually due to a network configuration issue".
Type 5 works by having the GPU write a WQE and ring the NIC doorbell in MMIO itself,
so this is a mapping-permission failure, not a fabric one. Adding
`--privileged --device=/dev/gdrdrv` (now done by `10_launch_standalone.sh` for
`deepep*` arms only) takes `grep -c nccl.cu:101` from 14 to **0** with no other
change, and the run advances to "The wall" below. The K3 kit reached the same
conclusion from the other direction and made every arm privileged so that "a GIN
init failure could be a permissions artefact" stopped being a live hypothesis.

Two things follow. First, at the default `NCCL_DEBUG=WARN` none of the four lines
above are printed, so the assert is the *only* output — always re-run a GIN failure
with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GIN,NET` before theorizing. Second, do not
try to reproduce this at 1 node: `direct` mode never reads `railedGinType`, so a
single-node arm comes up privileged or not.

## The wall: clamped-swiglu post-quant does not exist at group_size 32

Measured on B300-3, 2026-09-05, single node TP8 EP8. **Everything that
configuration or patching can fix now works, and then it stops here.** This is a
missing kernel, not a gate, and no env var reaches it.

The failure, during decode graph capture, from sglang's *own* JIT (not DeepEP's):

```
sgl_kernel_jit_dpsk_v4_silu_mul_quant_varlen_32_true_false_true_true
.../deepseek_v4/silu_and_mul_masked_post_quant.cuh(245): error: static assertion failed
    static_assert(kGroupSize == 128);
  ... with kGroupSize=32L, kScaleUE8M0=true, kSwizzle=false, kUsePDL=true,
      kApplySwigluLimit=true
```

`kGroupSize=32` is Hy4's MXFP8 `weight_block_size` `[1, 32]`. `kApplySwigluLimit`
is Hy4's swiglu clamp. Together they name a combination that has no
implementation:

| candidate | why it cannot serve Hy4 |
|---|---|
| the DSV4 JIT kernel (`silu_and_mul_masked_post_quant`) | carries the clamp, but **both** device kernels hard-code `constexpr uint32_t kGroupSize = 128u` (`:116` and `:381`). The `kGroupSize` template parameter is only *asserted*, never plumbed through — so the assert is honest, not over-strict. |
| the Triton kernel (`silu_and_mul_masked_post_quant_fwd`) | takes `group_size` as a **runtime** argument, so 32 is fine — but its clamp sits inside `if GEMM1_ALPHA > 0` (`ep_moe_kernels.py:339-342`), where the activation is the gpt-oss form `gate*sigmoid(gate*α)*(up+1)`. At `α=0` you get plain silu and **no clamp at all**. Reaching it also requires `gemm1_alpha is not None`, which asserts `swiglu_limit is None`. |

So routing Hy4 to Triton would silently substitute a different activation. That is
the one thing blocker 3 teaches not to do.

**This is also blocker 6.** v1 `deepep` dies in the same file at `:361` — the
`kGroupSize = 128u` at `:381`, the other of the two hard-coded kernels — as
`RuntimeError: CUDA error: invalid argument` from `_run_contiguous_gemm ->
silu_and_mul_clamp`. v1 feeds the contiguous runner and gets a runtime failure; v2
feeds the masked runner and gets the compile-time assert. One defect, two faces.
Hence: **on this MXFP8 checkpoint no DeepEP arm can run, v1 or v2**, and
`DISABLE_CUDA_GRAPH=1` cannot help — the kernel is missing, not mis-launched.

Upstream already knows the constraint and even spells out this exact string:

```python
# moe_runner/deep_gemm.py:168  _masked_activation_unsupported_reason()
if group_size != 128:
    return f"masked activation group_size {group_size}, DSV4 JIT kernel requires 128"
```

But it guards only `_should_use_masked_standard_layout()`, i.e. the **standard**
(non-DeepEP) path's masked-vs-compact layout choice. That is precisely why
`A2A_BACKEND=none EP_SIZE=8` works: group 32 falls back to the compact layout. The
v2 masked runner never consults it, and "fall back to compact" is not available
there anyway — the masked layout is how v2 dispatch delivers tokens.

**What would actually unblock it** (either one, both real work):

1. instantiate the DSV4 CUDA kernel at `kGroupSize=32` — plumb the template
   parameter into the two device kernels instead of the `128u` constants; or
2. add a plain-silu-plus-clamp branch to the Triton kernel, independent of
   `GEMM1_ALPHA`.

Either way, gate the result on `96_logprob_parity.py` against
`A2A_BACKEND=none EP_SIZE=8` before quoting a number — the whole hazard here is an
activation that runs and is wrong.

**The one untested combination that might sidestep it:** BF16 weights with v1
`deepep`. No FP8 post-quant means this kernel is never reached. v2 is not an
option there (the quant gate rejects non-FP8 on its first branch, before anything
these patches touch).

## How far it does get, and what that proves

Worth recording, because each of these was its own blocker and all are now closed:

| stage | evidence |
|---|---|
| four patches applied | build prints `all four DeepEP patches accounted for` |
| v2-capable library present | `deep_ep=2.1.0+97d8f9b`, `nccl=2.31.2` in `/etc/hy4-deepep-v2-patched` |
| GIN initialises, 1 node | `GIN: type=3 sym_kernels=default hca=ibp198s0f0 (nnodes=1, devices: 16 rdmap + 2 other)` |
| the ElasticBuffer builds, on all 8 ranks | `Initialized DeepEP v2 ElasticBuffer: world_size=8 hidden_size=6144 num_topk=8 max_dispatch_tokens_per_rank=2048 use_fp8_dispatch=True allow_hybrid_mode=False num_bytes=203423744` |
| GIN initialises, 2 nodes, on **EFA** | `GIN: type=5 sym_kernels=0 hca=rdmap (nnodes=2, ...)`, then `devCommCreate: creating 11 contexts: 1 GIN connections`, and zero `nccl.cu:101` asserts |
| the ElasticBuffer builds cross-node, all 16 ranks | `Initialized DeepEP v2 ElasticBuffer: world_size=16 ... allow_hybrid_mode=True num_bytes=570425344` |
| DeepEP's own dispatch kernel compiles **and loads** | ptxas `Used 42 registers`, and zero `kernel_runtime.hpp:33` asserts after the cuobjdump fix |
| then, identically at 1 and 2 nodes | the static_assert above |

So the v2 *plumbing* on Hy4 is proven end-to-end, and — after the privileged fix —
proven **cross-node on the EFA rails** as well, not merely on NVLink. What is
missing is one activation kernel downstream of dispatch. Because that kernel sits
after dispatch in the MoE forward, the node count is irrelevant to it: the 2-node
arm reaches exactly the same `static_assert` at exactly the same point in decode
graph capture. Full run logs and the reproduce command are in
`../results/deepep_v2_enablement/`.

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
  4096 OOM'd. On **Hy4** the buffer itself is small — `num_bytes=203423744`,
  i.e. **194 MiB per rank** at `V2_CAP=2048`, TP8, `hidden_size=6144`,
  `num_topk=8`. Do not carry the K3 number over; it was a different world size and
  a different hidden size.
* **The v2 arm needs `MEM_FRACTION`, and this is a launch blocker.** The buffer is
  194 MiB but it is not the cost that matters: sglang's default
  `mem_fraction_static` for this config is **0.963**, tuned with no DeepEP present,
  and the DSA decode graph then asks for 8 GiB workspaces it cannot get. Measured
  on B300-3 at `V2_CAP=2048`:

  | | outcome |
  |---|---|
  | defaults (`0.963`, graph max_bs 512) | OOM in `dsa_backend.py:1222 init_cuda_graph_state`, "Tried to allocate 8.00 GiB ... 7.46 GiB is free" |
  | `MEM_FRACTION=0.92`, max_bs 512 | gets one step further, then OOM inside the capture forward at `deepgemm_paged_mqa_logits`, 1.66 GiB free |
  | `MEM_FRACTION=0.85` + `MAX_RUNNING=128` | no memory error; reaches the MoE forward |

  `MAX_RUNNING` is the stronger lever, because it sets the decode graph's `max_bs`
  and every one of those workspaces scales with it. Note both knobs are comparison
  axes, so **the reference arm must be given the same two values** — otherwise the
  pair differs in graph size and KV pool as well as in backend.
* **BF16 cannot use v2 at all.** The quant gate rejects non-`Fp8MoEMethod` layers
  on its first branch, before anything this patch touches. `build_moe_args()`
  refuses the combination up front.
* **`hidden_size % 128`.** `deepep_v2.py:267` requires it; Hy4's 6144 is fine.

## Validating, before quoting any number

The reference is `A2A_BACKEND=none EP_SIZE=8` **on this same image** — not v1
`deepep` (blocker 6) and not the stock image (blocker 4):

```bash
IMAGE=hy4-preview-v2:latest A2A_BACKEND=none EP_SIZE=8 CHUNKED_PREFILL=2048 \
  bash 10_launch_standalone.sh
```

Then compare **first-token top-5 logprobs**, not prose. A coherent-looking
completion is not evidence: blocker 3's contiguous half produces a *plausible*
answer from a mis-scaled GEMM. `96_logprob_parity.py` automates exactly this and
refuses to run unless the two servers' resolved args differ only in the backend:

```bash
python3 96_logprob_parity.py --ref B300-4:30000 --test B300-3:30000
```

## Re-cutting a diff against a new base image

```bash
P=/opt/dlami/nvme/patch; mkdir -p $P/orig
cid=$(docker create --entrypoint true lmsysorg/sglang:hy4-preview)
B=/sgl-workspace/sglang/python/sglang
docker cp $cid:$B/srt/arg_groups/moe_hook.py               $P/orig/moe_hook.py
docker cp $cid:$B/srt/layers/moe/fused_moe_triton/layer.py $P/orig/fmt_layer.py
docker cp $cid:$B/srt/layers/moe/moe_runner/deep_gemm.py   $P/orig/mr_deep_gemm.py
docker cp $cid:$B/srt/models/hunyuan_v4.py                 $P/orig/hunyuan_v4.py
docker rm -f $cid
for f in moe_hook fmt_layer mr_deep_gemm hunyuan_v4; do
  cp $P/orig/$f.py $P/$f.py
  patch $P/$f.py < patches/$f.diff        # edit, then:
  diff -u $P/orig/$f.py $P/$f.py > patches/$f.diff
done
```

`$B` above is hard-coded on purpose. Do **not** derive it with
`os.path.dirname(sglang.__file__)` from a shell whose cwd is `/sgl-workspace`:
that directory contains the repo checkout `sglang/` with no `__init__.py`, so
`import sglang` binds a namespace package and `__file__` is `None`. That is
exactly how the `hy4-preview-efa`-based build failed on 2026-09-04 — the probe
raised before a single patch was tried, and the only thing buildkit showed was
the command echo, which looked like "the diffs no longer apply". The stock image
has `WORKDIR /` and hides this; the EFA image has `WORKDIR /sgl-workspace` and
does not. The Dockerfile now does `cd /` first and asserts `$SGL/srt` exists.

The diffs are applied **by target path**, not with `-p<n>`: they were captured
against a `docker cp`'d copy, so no strip level reaches the real tree. The
Dockerfile's apply loop is idempotent and fail-closed — forward, else
already-applied (proved by a reverse dry-run), else **fail the build**. A build
break here is the intended signal that the base image moved.
