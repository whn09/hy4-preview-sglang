# Hy4-preview on AWS p6-b300 (SGLang)

Test kit for **Tencent Hy4-preview** (HYV4) on 8x B300 nodes, modelled on the
`kimi-k3-sglang` kit. Two arms:

| arm | hosts | image | status |
|---|---|---|---|
| single node, MXFP8 **TP8** (default) | any one B300 | stock `lmsysorg/sglang:hy4-preview` | **not yet measured** |
| single node, MXFP8 TP4 (`TP_SIZE=4`) | B300-3 | stock `lmsysorg/sglang:hy4-preview` | **measured** (7 points), see below |
| 1P1D PD disaggregation | B300-1 prefill / B300-2 decode | `hy4-preview-efa:latest` (ECR, see below) | **boots healthy; not yet benchmarked** |
| EP / DeepEP arms (`EP_SIZE`, `A2A_BACKEND`) | any one B300 | either | **wired and gated; not yet measured** |

**The default TP changed from 4 to 8** for both quantizations. See "TP8, and why it
is not a guess" below; the seven measured rows further down are the **TP4**
campaign and are labelled as such -- TP is in every filename, so the two arms
cannot collide.

The 1P1D pair reached `HTTP 200 /health` on both sides simultaneously on
2026-09-04 at 18:59 KST -- so the EFA image, the Mooncake swap, the bootstrap port
and the MXFP8 TP4 geometry are all confirmed to come up on real hardware. The
cluster was shut down about a minute later, before the router and the first bench
point ran, so there are **no PD numbers yet and none are claimed here**.

Everything in the single-node arm comes from a **verified cookbook cell**
(`docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` and
`docs/src/snippets/configs/tencent/hy4-preview.jsx` in the sglang tree). PD is
**not** a cookbook cell -- the cookbook publishes only single-node TP recipes --
so it is our own arm and is labelled as such everywhere.

---

## The model, as the runtime actually configures it

770B total / 49B active MoE. 78 layers (layer 0 dense, 77 sparse), 256 routed +
1 shared expert, top-8 sigmoid routing (routed scaling 2.827), expert
intermediate 2048, bounded SwiGLU clamp 10.0.

* **MLA + DeepSeek Sparse Attention (DSA) on every layer.** q_lora_rank 2048,
  kv_lora_rank 512, 192 nope + 64 rope, v_head_dim 256, indexer top-k 2048, 32
  index heads with an FP8 index cache. SGLang auto-selects
  `attention_backend=dsa` with `flashmla_sparse` for both prefill and decode,
  `page_size 64`, and forces the KV dtype to bfloat16 on SM10x.
* **iHC residual gating** (`enable_ihc`, `hc_mult 4`) -- distinct from
  DeepSeek-V4's mHC. Logs `HYV4 MLA output gate backend: hpc` and
  `HY4 iHC: using hpc.fuse_ihc_pre/head`.
* **MTP (NextN)** draft layer `model.mtp_layers.0` (~10B params, ~0.7B active)
  ships **inside both checkpoints**, so there is no separate draft path to point
  at. SGLang records it internally as `speculative_algorithm='EAGLE'` with
  `speculative_draft_model_path` = the same checkpoint.
* **MXFP8** (ModelOpt). Straight out of `tencent/Hy4-preview-FP8/config.json`:
  `"quantization_config": {"quant_method": "modelopt", "quantization":
  {"quant_algo": "MXFP8", ...}}`. **Never pass `--quantization`** -- it overrides
  the checkpoint's own recipe. The repo has **no `hf_quant_config.json`** (that is
  ModelOpt's schema name; here it is inlined into `config.json`).

  sglang maps `quant_algo MXFP8` onto `Fp8Config(weight_block_size=[1,32],
  use_mxfp8=True)`, so the server logs plain `quant_method=Fp8MoEMethod` even on
  an MXFP8 checkpoint -- **that log line is not evidence of DeepSeek-style
  blockwise FP8**, and reading it that way is the obvious wrong turn here.

  **SM100+ only**, from `fp8.py get_min_capability`: `return 100 if self.use_mxfp8
  else 80`. There is one MXFP8 -> block-fp8[128,128]-at-load path that would run on
  older parts, but it is **AMD-only** (`mxfp8_block_convert_required()` opens with
  `if not torch.version.hip: return False`) and it reports capability 94, a gfx942
  code that SM90's 90 fails regardless. So **H200 cannot serve this FP8 checkpoint**
  -- source-level, not measured. `tencent/Hy4-preview` (BF16, ~1.5 TB) is the only
  H200 path, and at 1.5 TB that is 2 nodes / TP16.
* Text-only: image input is rejected with HTTP 400 **by design**.
* The model rejects pipeline parallelism and `--enable-prefill-cp` before
  allocation.

Structural tokens are suffixed (`<think:opensource>`, `<tool_calls:opensource>`,
`<arg_key:opensource>`, `<arg_value:opensource>`); `--reasoning-parser auto
--tool-call-parser auto` detect them from the chat template.
`reasoning_effort` defaults to **high**, and `no_think` is *not* a standard
OpenAI tier -- it has to go through `chat_template_kwargs` / `extra_body`.

### Cookbook cells that apply to this hardware

Verified for B300 (288 GB): **MXFP8 TP4 single node** and **BF16 TP8 single
node**. H200/B200/GB300 BF16 are 2-node TP16/TP8 cells and still marked
in-progress upstream.

---

## TP8, and why it is not a guess

Both arms now default to `TP_SIZE=8`, the whole node. The b300 cell the cookbook
marks verified for MXFP8 is TP**4**, so this needs saying plainly: TP8 MXFP8 on
b300 is **not a verified cell**, it is the intersection of two verified cells.

* The **b200 MXFP8** verified cell is `--tp 8 --moe-runner-backend deep_gemm
  --fp8-gemm-backend deep_gemm` -- the exact flag set this kit emits, on a
  smaller GPU (192 GB). So MXFP8-at-TP8 is exercised upstream.
* The **b300 BF16** verified cell is `--tp 8`. So TP8-on-b300 is exercised
  upstream.
* The cookbook's TP knob **does not disable** TP8 for b300; the only disables on
  the value 8 are `h200/b200 + bf16` (~190 GB/rank does not fit) and
  `gb300 + single` (4-GPU hosts).
* 64 attention heads / 8 = 8 per rank, 32 DSA index heads / 8 = 4 per rank,
  `moe_intermediate_size` 2048 / 8 = 256 -- nothing is indivisible at 8.

What it buys: per-rank weights halve to **~95 GB**, so roughly twice as much of
the 288 GB is left for the KV pool and the MoE gets 8 ranks of compute instead of
4. What it costs: one more all-reduce hop per layer, and one instance now occupies
the whole node -- the TP4 arm deliberately left GPUs 4-7 free. `TP_SIZE=4` still
reproduces the old arm exactly.

Note that TP4 and TP8 are **not** a like-for-like per-GPU comparison of the same
service: they have different KV pools and therefore different admission
behaviour. Compare them on `out tok/s/GPU` *and* on latency at matched
concurrency, and expect the TP8 arm to win where the TP4 arm was pool-limited.

---

## MoE parallelism: EP and the a2a backends

Short answers: **EP=8 works and is one env var**, `deepep` is the only all-to-all
backend the cookbook offers for this model, and `deepep_v2` is blocked on the
stock image but **runs on a patched one** -- three source patches, one of which is
a numerics fix rather than a loosened gate (`patches/README.md`).

```bash
EP_SIZE=8 bash 10_launch_standalone.sh                 # EP, no a2a library
A2A_BACKEND=deepep bash 10_launch_standalone.sh        # DeepEP (forces EP = TP)
DEEPEP_MODE=low_latency A2A_BACKEND=deepep bash 10_launch_standalone.sh
DP_ATTN=8 EP_SIZE=8 bash 10_launch_standalone.sh       # + attention DP

docker build -t hy4-preview-v2:latest -f Dockerfile.deepep_v2 .   # ~1 min
IMAGE=hy4-preview-v2:latest A2A_BACKEND=deepep_v2 CHUNKED_PREFILL=2048 \
  bash 10_launch_standalone.sh                         # DeepEP v2, patched image
```

### EP=8 is arithmetically free on this model

256 routed experts + 1 shared, top-8 sigmoid routing, **`n_group` 1 /
`topk_group` 1** (so there is no expert-group constraint to satisfy),
`moe_intermediate_size` 2048. 256/8 = 32 experts per rank; `moe_tp_size` =
TP/EP/moe_dp = 1, so each rank holds its 32 experts whole. `build_moe_args()`
re-checks all three divisibilities and names the constant that failed.

### Two different things are both called "EP"

* **`EP_SIZE=8` with `A2A_BACKEND=none`** is still real expert parallelism. It
  runs through `StandardDispatcher`, which builds a `local_expert_mapping` and
  masks: every rank sees every token, computes only its own 32 experts, and the
  existing post-MoE all-reduce does the combine. **No dispatch/combine
  collective, no DeepEP, nothing extra in the image** -- it works on any
  interconnect. On one node over NVLink that is cheap. What it never does is
  shrink activation traffic, which is why it is not the cross-node answer.
* **`A2A_BACKEND=deepep`** replaces that with a real all-to-all: each rank sends
  only the tokens its experts were selected for. This is the cookbook's own
  option, labelled there `"DeepEP (EP = TP)"`.

### EP is silently rewritten for every a2a-spanning backend

From `arg_groups/overrides.py`:

```python
@register_post_process
def _a2a_ep_size(view):
    if view.moe_a2a_backend in _A2A_EP_SPANNING_BACKENDS:   # deepep, deepep_v2,
        ...                                                 # mooncake, nixl,
        return {"ep_size": view.tp_size}                    # flashinfer, mori,
                                                            # pplx, megamoe, ...
```

So `--ep-size 4 --moe-a2a-backend deepep` at TP8 **runs EP=8** and only says so
in an `logger.info`. This kit therefore resolves EP itself and passes the
resolved number, so the flag, the startup line and the results filename can
never disagree -- and it *refuses* an `EP_SIZE` that a2a would overwrite instead
of quietly honouring the other value.

### Every a2a backend, and what happens if you ask for it

`MoeA2ABackend` (`layers/moe/utils.py`) has twelve members. What each one means
for **this** model:

| `A2A_BACKEND` | this kit | why |
|---|---|---|
| `none` (default) | runs | the verified cell: pure TP MoE, or masked EP when `EP_SIZE>1` |
| `deepep` | runs | the cookbook's own experimentation override, `EP = TP` |
| `deepep_v2` | patched image only | three upstream blockers, all in `patches/`: the architecture whitelist (`DeepseekV3/V4`, `Qwen3Moe` -- Hy4 is `HYV4ForCausalLM`), the quant gate rejecting MXFP8, and `mxfp8_act_gran_k` never being set on the v2 pre-permute. `require_deepep_v2_image()` refuses the stock image up front |
| `megamoe` | **refused** | the cookbook omits it: its fused path is not wired for Hy4's sigmoid-scored, bounded-SwiGLU experts |
| `mori` | **refused** | ROCm |
| `ascend_fuseep`, `ascend_tp` | **refused** | Ascend NPU |
| `mooncake`, `nixl`, `flashinfer`, `pplx`, `customized` | `ALLOW_UNVALIDATED_A2A=1` | real code paths, EP-spanning, but not exercised on Hy4 |

**`mooncake`/`nixl` in that list are the MoE expert all-to-all transport, not the
PD KV transport.** The PD one is `TRANSFER_BACKEND`, a completely unrelated flag
that happens to take the same two words. The gate says so in its error message,
because a typo between them would otherwise "work".

### Things to know before reading an EP number

* **DeepEP is in the image** -- checked: the stock `lmsysorg/sglang:hy4-preview`
  ships both `deep_ep` and `deep_gemm` in
  `/usr/local/lib/python3.12/dist-packages`, so nothing needs installing for either
  DeepEP arm. `require_deepep_image()` stays, because it checks
  `find_spec("deep_ep")` on the host *before* the ~10 min weight load and the next
  base image is not obliged to keep the module.
* **`DEEPEP_MODE=normal` disables CUDA graphs for both phases**
  (`moe_hook.py`: "Cuda graph is disabled because deepep_mode=`normal`"). Any
  latency comparison against another arm then also carries graph-on/graph-off,
  which was worth 3.27x on K3. `auto` (the default) is normal-for-prefill,
  low-latency-for-decode.
* **MXFP8 + DeepEP-LL is handled but mixes granularities.** DeepEP's low-latency
  dispatch quantises activations at a fixed 128 block while the checkpoint's
  weight block is `[1, 32]`; the deep_gemm runner has an explicit
  `running_state["mxfp8_act_gran_k"] = 128` for exactly this combination, so it
  is anticipated upstream rather than accidental. It is still not a cell anybody
  has published numbers for.
* **Hy4's bounded SwiGLU already forces the compact DeepGEMM layout.**
  `_masked_activation_unsupported_reason()` requires group 128, MXFP8 is group 32,
  so the masked standard layout is off and the log says so. That is pre-existing
  at TP4/EP1, not something EP introduces -- do not read it as an EP failure.
* **The `deepep_v2` arm needs `CHUNKED_PREFILL`, and needs it on the baseline
  too.** Its per-rank dispatch capacity is an env var
  (`SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`, exposed here as `V2_CAP`,
  default 2048) and the budget check compares it against `chunked_prefill_size`,
  which the runtime defaults to **16384** on a 279 GiB B300. So the v2 arm cannot
  boot at the default chunk; `build_moe_args()` refuses without `CHUNKED_PREFILL`
  rather than lowering it silently, because a chunk chosen for v2 alone turns every
  prefill comparison into a chunk-size comparison. Both `-chunk<N>` and `-mr<N>`
  are in the filename now.
* **A v2 number is not quotable until its logits match v1's.** Blocker 3 is a real
  numerics bug on the contiguous path, and a mis-scaled GEMM there still produces
  fluent text -- compare first-token top-5 logprobs against the `deepep` arm on the
  stock image before reading any throughput row. `patches/README.md` has the
  arithmetic.
* **`DP_ATTN` is offered by the cookbook but is not a verified cell**, and
  `server_args`' own help for `--enable-dp-attention` says the DP size "should be
  equal to the tp size" while the cookbook offers 4 with TP 8. Treat any DP-attn
  number as exploratory, and note that on an MLA model it changes the KV pool
  shape, i.e. it is an axis, not a tweak.
* Every one of these axes is in the **filename** (`TOPO` gains
  `-a2a<backend>[cap<n>]-chunk<n>-mr<n>-ep<n>-dp<n>`) and in the log header
  (`### moe_a2a=... ep=... v2_cap=... chunk=...`),
  and `gen_bench_table.py` prints an `moe` column. A row from before these knobs
  existed shows `-`, which means pure TP -- it does not mean unknown.

---

## Files

```
00_download_models.sh     hf download -> /opt/dlami/nvme/models/  (758 G MXFP8)
env_common.sh             all config + the traps, sourced by host AND container
04_fix_multinic_routing.sh HOST, EVERY NODE, EVERY BOOT: the 17-ENI routing fix
05_pull_pd_image.sh       pull hy4-preview-efa from ECR instead of rebuilding
10_launch_standalone.sh   single-node container (stock image)
20_launch_prefill.sh      1P1D prefill side  -> _pd_launch.sh
21_launch_decode.sh       1P1D decode side   -> _pd_launch.sh
_pd_launch.sh             shared PD host launcher (not an entry point)
22_launch_router.sh       sglang_router in front of the pair
start_server.sh           IN-CONTAINER: serves all three arms (PD_ROLE switches)
Dockerfile                hy4-preview-efa:latest -- EFA + Mooncake, PD only
Dockerfile.deepep_v2      hy4-preview-v2:latest  -- 3 source patches, nothing else
patches/                  the deepep_v2 diffs + per-blocker root cause (README.md)
90_smoke_test.sh          health / reasoning / no_think / streaming
95_features_test.py       asserting harness: 12 functional checks
91_bench.sh               one bench_serving point -> results/<TAG>.log
92_sweep.sh               concurrency ladder + table
gen_bench_table.py        results/*.log -> markdown (never transcribe by hand)
sync.sh                   push scripts / pull results (NEVER --delete)
```

`start_server.sh` is one file rather than the K3 kit's
`start_standalone`/`start_prefill`/`start_decode` trio: the three arms differ by
three flags, and the K3 pair has already drifted.

---

## Single node (B300-3)

```bash
bash 00_download_models.sh mxfp8          # 758 G, ~4 min at ~3 GB/s
bash 10_launch_standalone.sh              # MXFP8 TP8, MTP on
PROFILE=high-throughput bash 10_launch_standalone.sh   # MTP off
TP_SIZE=4 bash 10_launch_standalone.sh    # the TP4 arm the table below measured
docker logs -f hy4-preview
bash 90_smoke_test.sh
CONCS="1 16 64 256" bash 92_sweep.sh
```

Cold start is **~10.5 min**: 61 s of that is the weight load; the rest is
deep_gemm JIT plus CUDA-graph capture. The JIT stall reads exactly like a hang --
it is not. The caches in `CACHE_MOUNTS` are what stops the next launch paying it
again.

KV pool, **TP4** MXFP8 on B300: **520,512 tokens / 44.91 GB + 0.62 GB indexer per
rank** with MTP on, **566,464 tokens / 48.87 GB** with MTP off. At TP8 the weights
drop to ~95 GB/rank so there is far more room, but the pool is sized from
`mem_fraction_static` against what is left after the weights and the graphs --
read the number out of the startup log rather than scaling these two by hand.

`--tp-size`, not the cookbook's `--tp`: this build has **no `--tp` option at
all**. The cookbook's command works only through argparse prefix matching, which
breaks silently the day another `--tp*` flag is added.

`--cap-add SYS_NICE` is required (the K3 kit gets it free from `--privileged`);
without it every rank logs "User lacks permission to set NUMA affinity" and runs
with whatever NUMA placement it inherited.

### Functional checks

All 12 pass: reasoning/content separation, `no_think` via `chat_template_kwargs`,
`reasoning_effort=high`, tool calls non-streaming **and** streaming (including
`arg_key`/`arg_value` fragment reassembly and JSON type coercion), and the
text-only 400 contract.

```bash
docker run --rm --net=host -v $PWD:/w -w /w \
  --entrypoint python3 lmsysorg/sglang:hy4-preview 95_features_test.py
```

### Measured: MXFP8 **TP4**, 1024 in / 1024 out, B300

Generated by `gen_bench_table.py`, not transcribed. This is the **TP4** campaign
of 2026-09-04 -- the kit's default is now TP8, so reproduce it with
`TP_SIZE=4`. These logs also predate the `topo`/`gpus`/`moe` headers, so the
generator marks them `*` and prints `moe` as `-` (they were pure TP: the EP knobs
did not exist yet).

| profile | spec (MTP) | conc | reqs | dur (s) | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | out tok/s | out tok/s/GPU |
|---|---|---|---|---|---|---|---|---|---|
| low-latency | **on** | 1 | 8 | 49.68 | 131.19 | **5.48** | 5.39 | 164.91 | 41.23 |
| high-throughput | off | 1 | 8 | 135.11 | 123.74 | 16.39 | 16.37 | 60.63 | 15.16 |
| low-latency | **on** | 16 | 32 | 40.39 | 2954.37 | **13.73** | 12.64 | **811.20** | 202.80 |
| high-throughput | off | 16 | 32 | 64.52 | 1136.56 | 30.42 | 30.37 | 507.85 | 126.96 |
| low-latency | **on** | 64 | 128 | 76.43 | 3153.71 | **22.18** | 17.15 | **1714.84** | 428.71 |
| high-throughput | off | 64 | 128 | 108.06 | 3243.44 | 49.64 | 48.81 | 1212.99 | 303.25 |
| high-throughput | off | 256 | 512 | 184.28 | 10047.90 | 80.18 | 72.60 | 2845.11 | 711.28 |

**MTP-on wins on both axes at every measured point** -- 2.72x output throughput
at c=1, 1.60x at c=16, 1.41x at c=64 -- which contradicts the cookbook's
rationale for the high-throughput profile ("at saturation the draft+verify
overhead outweighs the speedup"). Two things to hold onto before repeating that
as a conclusion:

1. **The MTP-on arm is admission-capped at 48.** With speculation on, SGLang
   resets `max_running_requests` to 48 unless it is set explicitly (the same trap
   as K3/DSPARK). Live concurrency at c=64 was 53.2 for MTP-on vs 64.0 for
   MTP-off, so the spec-on arm won c=64 *while serving fewer requests at a time*.
   That strengthens the per-request latency claim and weakens the throughput
   claim -- the two arms do not have the same queue.
2. **The crossover, if it exists, is above c=64.** The MTP-on c=256 row is
   missing: B300-3 was handed back before it ran. Do not infer it from the
   trend -- 48-slot admission is exactly the kind of ceiling that flips a curve.

`out tok/s/GPU` denominator is the GPUs the instance occupies (4 here). A TP8
BF16 row would not share it, and a 1P1D row does not either -- it is 2x TP.

---

## 1P1D PD disaggregation (B300-1 prefill, B300-2 decode)

```bash
# EVERY node, EVERY boot -- nothing cross-node works without it (see below):
bash 04_fix_multinic_routing.sh

# ONCE, on one host (~10-15 min, no GPU needed), then push:
docker build -t hy4-preview-efa:latest -f Dockerfile .

# every other host pulls it -- do not rebuild per node (see below):
bash 05_pull_pd_image.sh

# B300-1:
bash 20_launch_prefill.sh
# B300-2:
bash 21_launch_decode.sh
# then, anywhere that can reach both:
bash 22_launch_router.sh
ENDPOINT=localhost:8000 SRV=hy4-prefill bash 91_bench.sh
```

**TP8 per side** now, MXFP8, GPUs 0-7 on each host -- so the pair occupies two
whole nodes and `BENCH_GPUS` is 16, which is the `out tok/s/GPU` denominator.
`TP_SIZE=4` on both sides reproduces the half-node pair the 18:59 KST boot used.

`TP_SIZE`, `TRANSFER_BACKEND`, `SPEC` and the MoE geometry (`A2A_BACKEND` /
`EP_SIZE` / `DP_ATTN`) must **match on both sides**: none of them is negotiated at
handshake time, and a mismatch shows up as a stalled request or a blacklisted
Mooncake session rather than as a startup error.

### The 17-ENI routing trap: run `04_fix_multinic_routing.sh` after every boot

These instances come up with **17 ENA "interface" ENIs in one subnet** on top of
the 16 efa-only ones. `ec2-net-utils` writes an `ip rule` for each *secondary* IP
(tables 101-116) but not for the primary, and in the main table the 16 secondary
`172.31.16.0/20 ... proto kernel` routes carry metric 0 against the primary's
100. So a packet sourced from the primary IP egresses a *secondary* ENI, that
ENI's source/destination check rejects a source address it does not own, and the
frame is dropped before the wire. Captured on B300-2 while B300-1 pinged it:

```
enp71s0  In  IP 172.31.17.128 > 172.31.29.80: ICMP echo request     <- arrives
enp170s0 Out IP 172.31.29.80 > 172.31.17.128: ICMP echo reply       <- wrong NIC
```

Requests arrive, replies vanish: 100% loss between two instances in one subnet
whose security group already allows all traffic from itself. Every AWS-side check
passes (SG, subnet, NACL, source/dest flag, `hostname -I`), ssh works, and
`/health` returns 200 on each node *locally* -- so it reads as a security group
that was not reattached after the relaunch, and `22_launch_router.sh`'s
"not accepting connections" reads as a server still loading, which a cold Hy4
genuinely does for ~10 min. **Ping the peer.** If ICMP fails too, no server is
involved. The router now makes that distinction for you and names the fix.

`bash 04_fix_multinic_routing.sh` adds one source rule per node
(`from <primary> lookup 100`) and pings the configured peers to prove it took.
`--check` reports without changing anything and exits 1 when broken. It must run
on **both** ends -- the drop is on the reply path -- and it is **not persistent**
by design, so it is lost on every reboot. Verified 2026-09-05: all 12 ordered
pairs across B300-1/2/3/4 ping after it, none of the cross pairs did before.

### Why PD needs its own image

The KV cache crosses the wire, and on p6-b300 that wire is EFA. The stock image
cannot do it, measured rather than assumed:

```
$ ldd  mooncake/engine.so | grep -i fabric        -> nothing
$ strings mooncake/engine.so | grep -oE '[A-Za-z]*Transport' | sort -u
  RdmaTransport  TcpTransport  IbgdaDeviceTransport          # no EFA
```

The base image's `mooncake-transfer-engine-cuda13 0.3.12.post1` is built
**without** EFA. Its only RDMA path is libibverbs, which segfaults on EFA
hardware; what is left is `TcpTransport` over ENA. **A "Mooncake PD run" on the
stock image is a TCP run that never says so.** Note that
`strings engine.so | grep -ci efa` returns 389 on that wheel -- "efa" is a
substring of "d-efa-ult" -- so that grep passes on exactly the build it should
reject. The discriminator is linking libfabric and carrying `fi_getinfo` /
`fi_mr_reg`; `require_efa_image()` in `env_common.sh` checks those, plus the
GPU-MR CUDA-context fix and `MC_MAX_CONCURRENT_REG_MR`.

Nor can it be fixed by mounting the host's EFA stack in. Verified on B300-1,
2026-09-04: the base ships rdma-core 50.0 (IBVERBS_1.14) while the host has
64.0amzn0 (IBVERBS_1.18), so host libfabric loaded in fails with
`libefa.so.1: version 'EFA_1.7' not found (required by libfabric.so.1)`. Hence
the Dockerfile replaces **rdma-core and the wheel**.

`./Dockerfile` is the K3 image with the base swapped and the K3-specific parts
(DeepEP + 5 source patches) dropped: GDRCopy 2.5.2, aws-efa-installer 1.50.0,
pip NCCL 2.31.2, and `mooncake-transfer-engine-efa-cuda13`.

**One stated confound:** that NCCL upgrade makes the PD image's intra-node TP
all-reduce a *different NCCL* from the stock image the single-node arm runs, so a
PD-vs-single-node comparison carries an NCCL version as a second axis. Nothing in
1P1D needs >= 2.31 (each side is a single-node TP server, so NCCL never leaves the
node); it is carried over so the image stays usable for a future cross-node arm. To
remove the axis: `docker build --build-arg NCCL_PIP_VER= ...` and say so in the
results header.

### Build once, pull everywhere (ECR)

```
579019700964.dkr.ecr.ap-northeast-2.amazonaws.com/hy4-preview-sglang-b300
  :efa1.50.0-nccl2.31.2-mc0.3.13.post1     <- use this
  :latest                                  <- convenience alias, moves
```

`bash 05_pull_pd_image.sh` pulls it, aliases it to `hy4-preview-efa:latest`, and
then **verifies the three EFA discriminators inside the pulled image** before
declaring success -- a pull that resolved to a non-EFA build fails there rather
than hours later as `mooncake session ... is not alive`.

Two reasons this is not just a time saving:

* **A per-node rebuild is not the same image.** `lmsysorg/sglang:hy4-preview` is a
  moving tag, so two builds a day apart can carry different sglang source, and the
  prefill/decode pair would then be running different code with nothing in the
  logs saying so.
* **A build is a noisy neighbour.** The apt/EFA/GDRCopy stages are CPU- and
  network-heavy. Building on a node that is serving somebody else's benchmark
  perturbs their numbers, and `docker build` cannot be un-run.

The pinned tag names the three versions a PD result actually depends on, so a
`results/` log referring to it is reproducible. A log referring to `:latest` is
not -- state the pinned tag in the results header.

### The alternative backend, for the record

NIXL is a real option here and was verified on B300-1 the same day:
`libplugin_LIBFABRIC.so` **is** in the stock image, but its bundled libfabric is
too old for it (`version 'FABRIC_1.7' not found`). With the EFA image's
libfabric it loads, `create_backend("LIBFABRIC")` succeeds, GPU memory registers,
and `fi_info -p efa` enumerates 48 EFA domains (`efa-direct`, `FI_PROTO_EFA`)
inside the container. Select it with `TRANSFER_BACKEND=nixl` on **both** sides;
`env_common.sh` then sets `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`, whose
default is `UCX` -- and UCX has no EFA support, so leaving it would quietly run
over TCP.

### Two PD traps that are not optional

* **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`.** With `:True` *every*
  KV block transfer fails (`efa_context.cpp: fi_read/fi_write failed: Invalid
  argument`) and the only user-visible symptom is `mooncake session ... is not
  alive`, because one failure blacklists the session. Registration is fine --
  zero `fi_mr_regattr` failures, all rails up. `expandable_segments` backs a
  tensor with a growable VA reservation whose physical mapping is remapped as it
  grows, so the dmabuf-derived MR registered at startup no longer describes the
  memory at transfer time and libfabric rejects the RMA with EINVAL.
* **`MC_MAX_CONCURRENT_REG_MR=8`.** Registration serializes on the EFA
  provider's per-domain lock, so **fewer threads is faster**: on K3's 1376
  buffers, 128 -> 130.1 s, 32 -> 51.0 s, 8 -> 20.7 s, 4 -> 24.8 s. Hy4 registers
  a different buffer count (MLA plus a separate FP8 indexer pool), so re-sweep
  before quoting 8 as tuned rather than inherited.

### Admission ceiling on the decode side

With MTP on, the decode server's `max_running_requests` is reset to 48, and on
the decode side that is the concurrency ceiling of **the whole pair**. A c=256
benchmark against this arm is measuring a 48-slot queue unless `MAX_RUNNING` is
set explicitly -- and if you set it, put it in the results tag.

### Measured: the first 1P1D point, and the proof it went over EFA

2026-09-05, B300-1 prefill / B300-2 decode, MXFP8, **TP8 per side**, MTP on,
`TRANSFER_BACKEND=mooncake`, EP=1, 1024 in / 1024 out, c=16, n=64:

| out tok/s | total tok/s | req/s | live conc | TTFT med / mean / p99 | TPOT med | ITL med |
|---|---|---|---|---|---|---|
| 1128.79 | 2257.58 | 1.10 | 14.69 | 1065 / 2690 / 6089 ms | 10.00 ms | 38.49 ms |

Read it as a functional result, not a competitive one. `ITL / TPOT = 3.85`, which
is MTP accepting close to its full 4 tokens per step. And the pair spans **16
GPUs** holding two complete copies of the weights, so 70.5 out tok/s/GPU against
the single-node TP4 arm's 202.8 at the same c=16 -- c=16 cannot load a 16-GPU
pair, and the decode side's 48-slot ceiling is the thing to push against before
any PD-vs-single-node claim is worth making.

**Mooncake really carried the KV over EFA** -- image-capable is not
went-over-EFA, so this was checked three ways rather than grepped for once:

* both sides log `EfaTransport` installed with `provider: efa`, and across the
  whole run there is not one `fi_write failed`, `session ... is not alive`,
  `blacklist`, `KVTransferError` or `TcpTransport`;
* the EFA **hardware** counters agree exactly. Over 5 requests, prefill's
  `rdma_write_bytes` rose by 260,229,120 B in 6,720 `rdma_write_wrs` while
  decode's `rx_bytes` rose by the same 260,229,120 B -- a one-way RDMA-write push
  from prefill to decode, which is the shape the transfer is supposed to have;
* decode's `tx_bytes` did not move, confirming nothing came back over the KV
  path. Counters live in
  `/sys/class/infiniband/rdmap*/ports/1/hw_counters/`; read them on the **host**,
  before and after, and diff -- they are the only source here that a
  configuration mistake cannot fake.

---

## Benchmarking rules this kit enforces

* **Every axis is in the filename**, and it is read from the *running container*
  (`read_cenv`), not from the invoking shell. `bash 91_bench.sh` without the same
  `PROFILE=` that launched the server would otherwise stamp a spec-off run
  "low-latency" and overwrite the spec-on log. Topology is in there too, so a
  1P1D row and a single-node row cannot collide.
* **`out tok/s/GPU` uses `BENCH_GPUS` from the container**, which is `TP` for a
  single node and `2 x TP` for a 1P1D pair. Computing it from TP would credit a
  PD arm with double its real per-GPU throughput.
* **Tables are generated, never transcribed** (`gen_bench_table.py`). Rows whose
  header records `rc != 0` are dropped with a note rather than averaged in.
* **Both a time column and a rate column** are always printed: tok/s alone has
  inverted a conclusion before, because its denominator changes between arms.
* `--flush-cache` per run, since the random dataset is seeded and a second run
  would otherwise hit the first run's radix cache. `DISABLE_RADIX=1` for a fully
  cache-free measurement.
* `--tokenizer` must be the **local** path: the server reports its model path as
  `/models/Hy4-preview-FP8`, which bench_serving would otherwise try to resolve
  as an HF repo id.
* `sync.sh` never uses a bare `rsync --delete` -- `results/` exists only on the
  hosts, and a `--delete` push would wipe every benchmark log.

## Known gaps

* **No TP8 numbers at all.** The default changed to TP8 on 2026-09-05 on the
  source-level grounds above; every measured row in this file is TP4. The first
  thing to run on the next node is the same 1/16/64/256 ladder at TP8, both
  profiles, so the TP4 rows get a partner.
* **No EP or DeepEP numbers.** The knobs are wired and gated but nothing has run.
  Two unknowns are worth settling in the first ten minutes: whether the stock
  image even contains `deep_ep` (`require_deepep_image()` answers that in one
  second), and whether `EP_SIZE=8 A2A_BACKEND=none` at TP8 is faster or slower
  than pure TP -- on one node with NVLink it is not obvious in either direction,
  because masked EP trades MoE FLOPs per rank for zero extra communication.
* **MTP-on c=256** single-node row. Started at 18:58 KST and was killed ~2 min in
  by the cluster shutdown; `results/` therefore contains a header-only log for it,
  which `gen_bench_table.py` reports under "NO rc IN HEADER" rather than treating
  as a row. This is the point that would locate the MTP crossover, if there is one.
* **1P1D is one point wide.** c=16 at 1k/1k exists and Mooncake/EFA is proven
  (above); the ladder does not. c=16 leaves a 16-GPU pair idle, so the interesting
  rows are the ones that push the decode side's 48 slots -- and the pair has to be
  compared against a **TP8 single node**, which also does not exist yet, not
  against the TP4 rows. Resuming a cold pair:

  ```bash
  bash 04_fix_multinic_routing.sh       # every node, or nothing talks
  # the instance store is WIPED by stop/start -- re-download the weights first:
  bash 00_download_models.sh mxfp8      # ~4 min at ~3 GB/s, per host
  bash 05_pull_pd_image.sh              # ~75 s, the image itself is in ECR
  # B300-1: bash 20_launch_prefill.sh   # B300-2: bash 21_launch_decode.sh
  bash 22_launch_router.sh              # then 91_bench.sh through :8000
  ```

  `results/` lives on the EBS root volume, so it survives a stop/start; the weights
  under `/opt/dlami/nvme` do not. Budget ~15 min from a cold stopped instance to the
  first PD request.
* **BF16 TP8** arm never run (`QUANT=bf16` is wired and is a verified cell).
* The base image tag `lmsysorg/sglang:hy4-preview` is a **moving tag** and is not
  pinned by digest. Pin it before a real measurement campaign, or a re-pull
  silently changes what every `results/` log refers to.
