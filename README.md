# Hy4-preview on AWS p6-b300 (SGLang)

Test kit for **Tencent Hy4-preview** (HYV4) on 8x B300 nodes, modelled on the
`kimi-k3-sglang` kit. Two arms:

| arm | hosts | image | status |
|---|---|---|---|
| single node, MXFP8 TP4 | B300-3 | stock `lmsysorg/sglang:hy4-preview` | **measured** (7 points), see below |
| 1P1D PD disaggregation | B300-1 prefill / B300-2 decode | `hy4-preview-efa:latest` (ECR, see below) | **boots healthy; not yet benchmarked** |

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
* **MXFP8** (ModelOpt, UE8M0 group-32 weight scales, dynamic activations)
  self-describes through the checkpoint's `hf_quant_config`. **Never pass
  `--quantization`** -- it overrides the checkpoint's own recipe. Requires
  SM100+; H200/SM90 cannot serve the FP8 checkpoint at all.
* Text-only: image input is rejected with HTTP 400 **by design**.
* The model rejects pipeline parallelism and `--enable-prefill-cp` before
  allocation.

Structural tokens are suffixed (`<think:opensource>`, `<tool_calls:opensource>`,
`<arg_key:opensource>`, `<arg_value:opensource>`); `--reasoning-parser auto
--tool-call-parser auto` detect them from the chat template.
`reasoning_effort` defaults to **high**, and `no_think` is *not* a standard
OpenAI tier -- it has to go through `chat_template_kwargs` / `extra_body`.

### Cookbook cells that apply to this hardware

Verified: **MXFP8 TP4 single node** and **BF16 TP8 single node** (B300, 288 GB).
H200/B200/GB300 BF16 are 2-node TP16/TP8 cells and still marked in-progress
upstream. `QUANT=mxfp8|bf16` in `env_common.sh` picks the weights *and* the TP
size together, because on this hardware they are one decision.

---

## Files

```
00_download_models.sh     hf download -> /opt/dlami/nvme/models/  (758 G MXFP8)
env_common.sh             all config + the traps, sourced by host AND container
05_pull_pd_image.sh       pull hy4-preview-efa from ECR instead of rebuilding
10_launch_standalone.sh   single-node container (stock image)
20_launch_prefill.sh      1P1D prefill side  -> _pd_launch.sh
21_launch_decode.sh       1P1D decode side   -> _pd_launch.sh
_pd_launch.sh             shared PD host launcher (not an entry point)
22_launch_router.sh       sglang_router in front of the pair
start_server.sh           IN-CONTAINER: serves all three arms (PD_ROLE switches)
Dockerfile                hy4-preview-efa:latest -- EFA + Mooncake, PD only
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
bash 10_launch_standalone.sh              # MXFP8 TP4, MTP on
PROFILE=high-throughput bash 10_launch_standalone.sh   # MTP off
docker logs -f hy4-preview
bash 90_smoke_test.sh
CONCS="1 16 64 256" bash 92_sweep.sh
```

Cold start is **~10.5 min**: 61 s of that is the weight load; the rest is
deep_gemm JIT plus CUDA-graph capture. The JIT stall reads exactly like a hang --
it is not. The caches in `CACHE_MOUNTS` are what stops the next launch paying it
again.

KV pool, TP4 MXFP8 on B300: **520,512 tokens / 44.91 GB + 0.62 GB indexer per
rank** with MTP on, **566,464 tokens / 48.87 GB** with MTP off.

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

### Measured: MXFP8 TP4, 1024 in / 1024 out, B300

Generated by `gen_bench_table.py`, not transcribed.

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

TP4 per side, MXFP8, GPUs 0-3 on each host. `TRANSFER_BACKEND` and `SPEC` must
**match on both sides**: they are not negotiated at handshake time, and a
mismatch shows up as a stalled request or a blacklisted Mooncake session rather
than as a startup error.

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
1P1D needs >= 2.31 (each side is single-node TP4, so NCCL never leaves the node);
it is carried over so the image stays usable for a future cross-node arm. To
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

* **MTP-on c=256** single-node row. Started at 18:58 KST and was killed ~2 min in
  by the cluster shutdown; `results/` therefore contains a header-only log for it,
  which `gen_bench_table.py` reports under "NO rc IN HEADER" rather than treating
  as a row. This is the point that would locate the MTP crossover, if there is one.
* **1P1D numbers.** Nothing was measured. Resuming needs only:

  ```bash
  # the instance store is WIPED by stop/start -- re-download the weights first:
  bash 00_download_models.sh mxfp8      # ~4 min at ~3 GB/s, per host
  bash 05_pull_pd_image.sh              # ~75 s, the image itself is in ECR
  # B300-1: bash 20_launch_prefill.sh   # B300-2: bash 21_launch_decode.sh
  bash 22_launch_router.sh              # then 91_bench.sh through :8000
  ```

  `results/` lives on the EBS root volume, so it survives a stop/start; the weights
  under `/opt/dlami/nvme` do not. Budget ~15 min from a cold stopped instance to the
  first PD request.
* **The Mooncake KV path was never observed carrying traffic.** Both sides came up
  with `MOONCAKE_PROTOCOL=efa` exported and the EFA-verified image, but "the image
  can do EFA" is not "the transfer went over EFA". Before quoting any PD number,
  grep the prefill log for the mooncake protocol line and for `fi_mr_reg` failures,
  and confirm a non-zero KV-transfer count -- a silent fallback is exactly the
  failure this whole image exists to prevent.
* **BF16 TP8** arm never run (`QUANT=bf16` is wired and is a verified cell).
* The base image tag `lmsysorg/sglang:hy4-preview` is a **moving tag** and is not
  pinned by digest. Pin it before a real measurement campaign, or a re-pull
  silently changes what every `results/` log refers to.
