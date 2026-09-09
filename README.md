# Hy4-preview on AWS p6-b300 and p5en (SGLang)

Test kit for **Tencent Hy4-preview** (HYV4) on AWS, modelled on the
`kimi-k3-sglang` kit: host and container launchers, a bench harness that stamps
every axis into the filename, and the trap each arm hides. It reads in order --
**§1** what the model is, **§2** how to start one, **§3** what has actually been
measured -- with **§4** as the reference to reach for when something behaves
strangely, and **§5** the list of what is still missing.

| arm | hosts | image | status |
|---|---|---|---|
| single node, MXFP8 **TP8** (default) | any one B300 | stock `lmsysorg/sglang:hy4-preview` | **measured**, one point (c=16) |
| single node, MXFP8 TP4 (`TP_SIZE=4`) | B300-3 | stock | **measured**, 7 points, both profiles |
| TP8 EP / DeepEP `a2a` arms | any one B300 | stock (v1) / patched (v2) | **measured**, 4 rows at c=64 |
| 1P1D PD disaggregation | B300-1 prefill / B300-2 decode | `hy4-preview-efa:latest` (ECR) | **measured**, 4-point ladder + a 2x-solo control |
| 2-node BF16 TP16, `a2a=none` | P5EN-3 + P5EN-4 | stock + host EFA stack | **boots and serves**; only TCP-fallback numbers so far |
| 2-node MXFP8 TP16, `deepep_v2` | B300-3 + B300-4 | patched | **cannot serve**: one CUDA kernel does not exist |

Three results that change what you would otherwise do:

* **DeepEP costs almost nothing on this model; CUDA graphs are worth 5x.**
  Against a matched-eager control the correct DeepEP arm is **-6.6%**, and the row
  that looked 5.5x faster differed from it only by replaying CUDA graphs.
* **1P1D loses to two independent TP8 servers on the same 16 GPUs** -- by 5.8% at
  offered concurrency 128 and by **54.9%** at 256. Its one real win is TPOT.
* **Every Hy4 a2a arm computed one token in `attn_tp_size`** until
  `--disable-attn-tp-gather` was passed, and the broken arm benchmarked *faster*
  than the correct one. Upstream issue #38606 / PR #38607; `UPSTREAM.md` triages
  all eight findings and says which four are still held.

Provenance: the single-node arm comes from a **verified cookbook cell**
(`docs/cookbook/autoregressive/Tencent/Hy4-Preview.mdx` and
`docs/src/snippets/configs/tencent/hy4-preview.jsx` in the sglang tree). PD, EP
and every a2a arm are **not** cookbook cells -- the cookbook publishes only
single-node TP recipes -- so they are our own arms and are labelled as such
everywhere.

---

## 1. The model, as the runtime actually configures it

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

## 2. Quick start

### Which arm can this hardware run?

| you have | run | why |
|---|---|---|
| one **p6-b300** (8x B300, 288 GB) | MXFP8 TP8, single node | the default, and the only single-node arm there is |
| two p6-b300 | two independent TP8 servers; 1P1D only if you are TPOT-bound | the pair loses on throughput at every measured concurrency (§3) |
| one **p5en** (8x H200, 143 GB each) | nothing serves | BF16 misses one node by 330 GiB, and MXFP8 needs compute capability 100 (§4) |
| two p5en | BF16 TP16 with `A2A_BACKEND=none`, and nothing else | three separate walls block every a2a backend at BF16 (§4) |
| four p5en | TP32 is legal arithmetic and untried | `TP_SIZE` must divide 64 attention heads |

Weights first, on every host that will hold a rank:

```bash
bash 00_download_models.sh mxfp8     # 758 G, ~4 min at ~3 GB/s   -- B300 only
bash 00_download_models.sh bf16      # ~1.5 TB                    -- p5en, TP16
```

### One B300: MXFP8 TP8, single node

```bash
bash 10_launch_standalone.sh              # MXFP8 TP8, MTP on
PROFILE=high-throughput bash 10_launch_standalone.sh   # MTP off
TP_SIZE=4 bash 10_launch_standalone.sh    # the TP4 arm measured in §3
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

### Two p5en: BF16 TP16 across two nodes

The only p5en geometry that exists, for the three reasons in §4. Both hosts run
the same line but `NODE_RANK`.

```bash
# both hosts, after every boot -- 16 EFA ENIs, same trap as the B300 hosts (§4)
bash 04_fix_multinic_routing.sh

# P5EN-3 (rank 0, the only host that binds :$PORT)
QUANT=bf16 NNODES=2 NODE_RANK=0 TP_SIZE=16 MEM_FRACTION=0.90 \
  DIST_INIT_ADDR=172.31.29.216 bash 10_launch_standalone.sh
# P5EN-4 (rank 1) -- same line, NODE_RANK=1, same DIST_INIT_ADDR
QUANT=bf16 NNODES=2 NODE_RANK=1 TP_SIZE=16 MEM_FRACTION=0.90 \
  DIST_INIT_ADDR=172.31.29.216 bash 10_launch_standalone.sh

# then, while it is generating, on either host:
bash 93_check_efa.sh          # reads the NIC counters; exits 1 on TCP fallback
```

`DIST_INIT_ADDR` is rank 0's **private (ENA)** IP and must be the same string on
both hosts; it changes on a stop/start, so read it rather than reuse the one above.
Only rank 0 binds `:$PORT` -- do not wait for "server is fired up" on rank 1.

`93_check_efa.sh` exists because no log answers this question: at
`NCCL_DEBUG=WARN` the transport is never printed, and at `INFO` you have to know
to look for `NET/OFI` vs `NET/Socket`. The counters cannot be faked by a
configuration mistake, which is the same argument §3 makes for
`rdma_write_bytes`.

### Two B300: 1P1D prefill/decode disaggregation

Read §3 before choosing this: on the same 16 GPUs two independent TP8 servers
beat the pair on throughput at every concurrency measured. It is here because it
wins on TPOT under load, and because the KV-over-EFA path is proven.

```bash
# EVERY node, EVERY boot -- nothing cross-node works without it (see §4):
bash 04_fix_multinic_routing.sh

# ONCE, on one host (~10-15 min, no GPU needed), then push:
docker build -t hy4-preview-efa:latest -f Dockerfile .

# every other host pulls it -- do not rebuild per node (see §4):
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
`TP_SIZE=4` on both sides reproduces the half-node pair of the first 2026-09-04
boot.

`TP_SIZE`, `TRANSFER_BACKEND`, `SPEC` and the MoE geometry (`A2A_BACKEND` /
`EP_SIZE` / `DP_ATTN`) must **match on both sides**: none of them is negotiated at
handshake time, and a mismatch shows up as a stalled request or a blacklisted
Mooncake session rather than as a startup error.

### MoE parallelism: the EP and a2a knobs

```bash
EP_SIZE=8 bash 10_launch_standalone.sh                 # EP, no a2a library
A2A_BACKEND=deepep bash 10_launch_standalone.sh        # DeepEP (forces EP = TP)
DEEPEP_MODE=low_latency A2A_BACKEND=deepep bash 10_launch_standalone.sh
DP_ATTN=8 EP_SIZE=8 bash 10_launch_standalone.sh       # + attention DP

docker build -t hy4-preview-v2:latest -f Dockerfile.deepep_v2 .   # ~1 min
IMAGE=hy4-preview-v2:latest A2A_BACKEND=deepep_v2 CHUNKED_PREFILL=2048 \
  bash 10_launch_standalone.sh                         # DeepEP v2, patched image
```

What each backend does, which ones are refused outright, and the six things to
know before reading an EP number: §4. The measured answer at TP8 is in §3.

### Checking that it works

```bash
bash 90_smoke_test.sh    # health, reasoning/content split, no_think, streaming
bash 93_check_efa.sh     # cross-node only, WHILE generating: is it really on EFA?
```

`95_features_test.py` asserts the model's own contracts rather than its speed, so
it needs no GPU of its own. All 12 pass: reasoning/content separation, `no_think` via `chat_template_kwargs`,
`reasoning_effort=high`, tool calls non-streaming **and** streaming (including
`arg_key`/`arg_value` fragment reassembly and JSON type coercion), and the
text-only 400 contract.

```bash
docker run --rm --net=host -v $PWD:/w -w /w \
  --entrypoint python3 lmsysorg/sglang:hy4-preview 95_features_test.py
```

**An arm with an a2a backend is not quotable until it passes a numerics gate**,
because a mis-routed MoE still produces fluent text -- and on this model the
broken arm was the *fast* one (§3). `bash 99_parity_pair.sh` holds both arms on one
node (TP4 each) and runs `96_logprob_parity.py` between them; when it fails,
`98_moe_dump.sh` + `98_moe_dump_cmp.py` name the tensor that stopped matching.

### Files

```
00_download_models.sh     hf download -> /opt/dlami/nvme/models/  (758 G MXFP8)
env_common.sh             all config + the traps, sourced by host AND container
04_fix_multinic_routing.sh HOST, EVERY NODE, EVERY BOOT: the 17-ENI routing fix
05_pull_pd_image.sh       pull hy4-preview-efa from ECR instead of rebuilding
06_build_v2_efa_image.sh  the v2 patches ON TOP of the EFA image (cross-node v2)
10_launch_standalone.sh   one container: single-node, or one rank-set of a TP16
20_launch_prefill.sh      1P1D prefill side  -> _pd_launch.sh
21_launch_decode.sh       1P1D decode side   -> _pd_launch.sh
_pd_launch.sh             shared PD host launcher (not an entry point)
22_launch_router.sh       sglang_router in front of the pair
start_server.sh           IN-CONTAINER: serves all three arms (PD_ROLE switches)
Dockerfile                hy4-preview-efa:latest -- EFA + Mooncake, PD only
Dockerfile.deepep_v2      hy4-preview-v2:latest  -- 3 source patches, nothing else
Dockerfile.nightly_deepep sglang nightly + DeepEP, for the upstream-fix arms
patches/                  the deepep_v2 diffs + per-blocker root cause (README.md)
pr_validation/            evidence for the three submitted PRs (README.md)
UPSTREAM.md               the six findings: which are filed, which are held, why
90_smoke_test.sh          health / reasoning / no_think / streaming
95_features_test.py       asserting harness: 12 functional checks
91_bench.sh               one bench_serving point -> results/<TAG>.log
92_sweep.sh               concurrency ladder + table
93_check_efa.sh           HOST, while generating: is cross-node traffic on EFA?
96_logprob_parity.py      top-5 logprob parity between two running servers
97_a2a_control.sh         is DeepEP-normal == a2a=none on a NON-Hy4 model?
98_moe_dump.sh            one sparse layer's MoE boundary tensors, per rank
98_moe_dump_cmp.py        diffs two dumps and names the stage that diverged
99_parity_pair.sh         both parity arms on ONE node (TP4+TP4), correctness only
gen_bench_table.py        results/*.log -> markdown (never transcribe by hand)
gen_a2a_table.py          the a2a table, from each run's own resolved server_args
gen_pd_vs_solo_table.py   the 1P1D vs 2x-solo ladder
sync.sh                   push scripts / pull results (NEVER --delete)
```

`start_server.sh` is one file rather than the K3 kit's
`start_standalone`/`start_prefill`/`start_decode` trio: the three arms differ by
three flags, and the K3 pair has already drifted.

---

## 3. Results

Every number below comes from a `results/*.log` in this repo, and all three
tables are printed by a generator (`gen_bench_table.py`, `gen_a2a_table.py`,
`gen_pd_vs_solo_table.py`) -- for the two that also live under `results/`, that
copy is the authoritative one. The rules deciding whether two rows may be
compared at all come first, because most of the wrong conclusions in this
campaign came from comparing rows that differed on an axis nobody had written
down.

### How to read any number in this file

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

### Every measured row, in one table

Printed by `python3 gen_bench_table.py` from the `results/*.log` in this repo:
the whole campaign, every arm, in one place. The comparisons below read rows out
of it rather than re-tabulating them. (In the last line, the `dump_*` and
`parity_*` logs are correctness runs, not bench points, so they carry no `rc`
header by construction.)

| quant | topo | moe | profile | spec | ISL/OSL | conc | reqs | dur (s) | TTFT p50 (ms) | TPOT p50 (ms) | ITL p50 (ms) | E2E p50 (ms) | out tok/s | GPUs | out tok/s/GPU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mxfp8 | tp4x1node* | - | high-throughput | off | 1024/1024 | 1 | 8 | 135.11 | 123.74 | 16.39 | 16.37 | 16886.84 | 60.63 | 4 | 15.16 |
| mxfp8 | tp4x1node* | - | high-throughput | off | 1024/1024 | 16 | 32 | 64.52 | 1136.56 | 30.42 | 30.37 | 32242.91 | 507.85 | 4 | 126.96 |
| mxfp8 | tp4x1node* | - | high-throughput | off | 1024/1024 | 64 | 128 | 108.06 | 3243.44 | 49.64 | 48.81 | 54005.60 | 1212.99 | 4 | 303.25 |
| mxfp8 | tp4x1node* | - | high-throughput | off | 1024/1024 | 256 | 512 | 184.28 | 10047.90 | 80.18 | 72.60 | 92069.56 | 2845.11 | 4 | 711.28 |
| mxfp8 | tp4x1node* | - | low-latency | on | 1024/1024 | 1 | 8 | 49.68 | 131.19 | 5.48 | 5.39 | 5740.06 | 164.91 | 4 | 41.23 |
| mxfp8 | tp4x1node* | - | low-latency | on | 1024/1024 | 16 | 32 | 40.39 | 2954.37 | 13.73 | 12.64 | 18881.29 | 811.20 | 4 | 202.80 |
| mxfp8 | tp4x1node* | - | low-latency | on | 1024/1024 | 64 | 128 | 76.43 | 3153.71 | 22.18 | 17.15 | 29978.26 | 1714.84 | 4 | 428.71 |
| mxfp8 | nightly-tp8-deepep-mxfp8-g32 | deepep/ep8 | low-latency | on | 1024/1024 | 1 | 32 | 184.84 | 181.56 | 5.42 | 5.36 | 5730.20 | 177.28 | 8 | 22.16 |
| mxfp8 | nightly-tp8-deepep-mxfp8-g32 | deepep/ep8 | low-latency | on | 1024/1024 | 64 | 128 | 48.33 | 4926.15 | 16.46 | 12.45 | 22059.11 | 2711.81 | 8 | 338.98 |
| mxfp8 | nightly-tp8-deepep-mxfp8-g32-atg1-normal-dispbf16 | deepep/ep8 | low-latency | on | 1024/1024 | 64 | 128 | 307.64 | 4286.77 | 104.29 | 95.74 | 111113.89 | 426.06 | 8 | 53.26 |
| mxfp8 | nightly-tp8-none-mxfp8-g32 | none/ep8 | low-latency | on | 1024/1024 | 1 | 32 | 205.47 | 131.58 | 5.73 | 5.63 | 6006.94 | 159.48 | 8 | 19.93 |
| mxfp8 | nightly-tp8-none-mxfp8-g32 | none/ep8 | low-latency | on | 1024/1024 | 64 | 128 | 56.39 | 2985.41 | 19.57 | 15.27 | 22990.08 | 2324.38 | 8 | 290.55 |
| mxfp8 | nightly-tp8-none-mxfp8-g32-cgoff1-compact | none/ep8 | low-latency | on | 1024/1024 | 64 | 128 | 287.23 | 2583.86 | 95.98 | 87.83 | 101858.12 | 456.33 | 8 | 57.04 |
| mxfp8 | pd1p1d-tp8-mooncake | none/epunknown | low-latency | on | 1024/1024 | 16 | 64 | 58.06 | 1065.44 | 10.00 | 38.49 | 11897.46 | 1128.79 | 16 | 70.55 |
| mxfp8 | pd1p1d-tp8-mooncake-mr128 | none/epunknown | low-latency | on | 1024/1024 | 64 | 128 | 40.51 | 2102.18 | 14.53 | 57.68 | 17295.53 | 3235.87 | 16 | 202.24 |
| mxfp8 | pd1p1d-tp8-mooncake-mr128 | none/epunknown | low-latency | on | 1024/1024 | 128 | 256 | 53.67 | 2776.69 | 18.86 | 76.20 | 22491.22 | 4884.05 | 16 | 305.25 |
| mxfp8 | pd1p1d-tp8-mooncake-mr128 | none/epunknown | low-latency | on | 1024/1024 | 256 | 512 | 103.33 | 24953.81 | 19.51 | 76.51 | 44343.81 | 5073.72 | 16 | 317.11 |
| mxfp8 | tp8x1node | none/epunknown | low-latency | on | 1024/1024 | 16 | 64 | 60.10 | 213.92 | 11.49 | 9.68 | 13495.50 | 1090.48 | 8 | 136.31 |
| mxfp8 | tp8x1node-solo | none/epunknown | low-latency | on | 1024/1024 | 32 | 128 | 65.26 | 300.07 | 13.64 | 11.20 | 14535.60 | 2008.37 | 8 | 251.05 |
| mxfp8 | tp8x1node-solo | none/epunknown | low-latency | on | 1024/1024 | 32 | 128 | 62.71 | 328.55 | 13.52 | 11.24 | 14277.38 | 2090.17 | 8 | 261.27 |
| mxfp8 | tp8x1node-solo | none/epunknown | low-latency | on | 1024/1024 | 64 | 256 | 97.36 | 439.96 | 19.19 | 14.56 | 20278.77 | 2692.42 | 8 | 336.55 |
| mxfp8 | tp8x1node-solo | none/epunknown | low-latency | on | 1024/1024 | 64 | 256 | 105.86 | 458.77 | 20.43 | 14.60 | 22838.62 | 2476.39 | 8 | 309.55 |
| mxfp8 | tp8x1node-solo | none/epunknown | low-latency | on | 1024/1024 | 128 | 512 | 131.45 | 741.01 | 27.40 | 19.39 | 29251.19 | 3988.61 | 8 | 498.58 |
| mxfp8 | tp8x1node-solo | none/epunknown | low-latency | on | 1024/1024 | 128 | 512 | 135.38 | 627.80 | 28.40 | 19.46 | 30476.66 | 3872.58 | 8 | 484.07 |

_out tok/s/GPU denominator = the GPUs the whole serving instance occupies, read from the container's BENCH_GPUS: TP size for a single-node row, 2x TP for a 1P1D pair. A TP4 MXFP8 row, a TP8 BF16 row and a PD row do not share it. `moe` = a2a backend / EP degree; `-` means the log predates the knob, i.e. pure TP MoE._

_Rows marked `*` predate the topo/gpus header; their GPU count was assumed = TP, which is WRONG for any disaggregated row. Re-run or treat their per-GPU column as unlabelled._

FAILED runs (row omitted, not zero): mxfp8-pd1p1d-tp8-mooncake-mr128-low-latency-specon-isl1024-osl1024-c8-n8.log (rc=1)

NO rc IN HEADER (still running or killed): download-mxfp8.log, dump_both.log, dump_deepep.log, dump_deepep_atg.log, dump_none.log, mxfp8-tp4x1node-low-latency-specon-isl1024-osl1024-c256-n512.log, parity_run.log, parity_run2.log, parity_run_floor.log

### MXFP8 TP8 vs TP4, at the one concurrency they share

At c=16 the TP8 row is **+34.4% absolute and -32.8% per GPU** against the TP4
low-latency row (1090.48 on 8 GPUs vs 811.20 on 4). That is the expected shape:
c=16 cannot load 8 GPUs, and TP8 pays one more all-reduce hop per layer. The
honest win is TTFT -- 213.92 vs 2954.37 ms -- because the TP4 arm was queueing at
c=16. Do not read it as "TP8 is worse per GPU": the ladder that would settle it
(1/16/64/256, both profiles) has not run, `num_prompts` differs (64 vs 32), and
neither arm passed `MAX_RUNNING`, so both are on the 48-slot MTP default.

### MTP (speculative decoding), measured at TP4

The seven `tp4x1node*` rows are the campaign of 2026-09-04, before the default
became TP8; reproduce them with `TP_SIZE=4`. Those logs predate the
`topo`/`gpus`/`moe` headers, which is what the generator's `*` and the `moe` `-`
mean -- they were pure TP, the EP knobs did not exist yet.

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

### What the a2a backends cost, at TP8

MXFP8, TP8/EP8, one B300, 1024/1024, c=64, `max_running=128`, `mem_fraction=0.85`,
`chunked_prefill=16384`, MTP on. Source of truth is
`results/a2a_backends_tp8_c64.md`, regenerated by `python3 gen_a2a_table.py`
from each run's own resolved `server_args`:

| a2a | deepep_mode | dispatch | atg | cuda_graph | out tok/s | TPOT med ms | TTFT med ms | accept | parity |
|---|---|---|---|---|---|---|---|---|---|
| deepep | auto | auto | 0 | on | 2711.81 | 16.46 | 4926 | 3.95 | **7.48 WRONG** |
| none | auto | auto | 0 | on | 2324.38 | 19.57 | 2985 | 3.73 | reference |
| none | auto | auto | 0 | off | 456.33 | 95.98 | 2584 | 3.72 | reference |
| deepep | normal | bf16 | 1 | on | 426.06 | 104.29 | 4287 | 3.73 | 0.410 (at floor) |

**The 5.5x collapse is not DeepEP.** Against the matched-eager control
(`a2a=none` + `--disable-cuda-graph`, layout pinned to `compact` so it exercises
the same grouped-GEMM path) the correct DeepEP arm is **-6.6%**: 426.06 vs 456.33
out tok/s, TPOT 104.29 vs 95.98 ms. The 2324.38 row is ~5.1x ahead of both purely
because decode is replaying CUDA graphs. Without that control the DeepEP arm
would have been written up as a 5.5x regression.

**The fast row is the wrong row.** Row 1 predates `--disable-attn-tp-gather` and
computed the MoE on one token in eight, returning exactly `0.0` from three of its
eight ranks -- which is *why* it is ahead of the reference
(`results/RETRACTED-nightly-tp8-deepep-pre-atg.md`). The parity column is the
worst top-5 `|dlogprob|` against the `a2a=none` reference. The corrected arm's
**0.410 is below the 0.593 noise floor** measured by comparing that reference
against *itself*, i.e. it is indistinguishable from run-to-run variation --
which is only knowable because the floor was measured rather than assumed to be
zero.

**The configuration is forced, not chosen.** MXFP8 has no CUDA DeepEP dispatch at
all, so a numerically correct arm must dispatch bf16; bf16 dispatch carries no
activation scale, so the masked (low-latency) runner dies; so `DEEPEP_MODE` must
be `normal`; and normal mode is where the eager-like decode appears. The open
anomaly: that row reports `disable_cuda_graph: False` and still performs exactly
like an eager arm. Mechanism unidentified, measured not inferred.

### 1P1D: the first point, and the proof it went over EFA

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

### 1P1D vs two independent TP8 servers, on the same 16 GPUs

Both arms ran **at the same wall-clock time** on four p6-b300 hosts, so neither
can be explained away by drift in the fabric or the hosts: arm A is B300-1
prefill + B300-2 decode behind the router, arm B is B300-3 and B300-4 as
self-contained TP8 servers, summed. Everything else matched, including
`MAX_RUNNING=128` verified from `/get_server_info` on all four servers rather
than from the launch env. Source of truth `results/pd_vs_2solo/README.md`,
regenerated by `python3 gen_pd_vs_solo_table.py results`:

| offered conc (16 GPUs) | arm | out tok/s | vs PD | live conc | req/s | TTFT p50 (ms) | TPOT p50 (ms) |
|---|---|---|---|---|---|---|---|
| 64 | 1P1D (TP8+TP8) | 3235.87 | - | 55.7 | 3.16 | 2102 | 14.53 |
| 64 | 2x solo TP8 (c=32 each) | 4098.54 | **+26.7%** | 59.3 | 4.00 | 329 | 13.64 |
| 128 | 1P1D (TP8+TP8) | 4884.05 | - | 111.0 | 4.77 | 2777 | 18.86 |
| 128 | 2x solo TP8 (c=64 each) | 5168.81 | **+5.8%** | 115.1 | 5.05 | 459 | 20.43 |
| 256 | 1P1D (TP8+TP8) | 5073.72 | - | 210.0 | 4.95 | 24954 | **19.51** |
| 256 | 2x solo TP8 (c=128 each) | 7861.19 | **+54.9%** | 239.8 | 7.68 | 741 | 28.40 |

**1P1D saturates at ~5.0k out tok/s; two instances do not.** From offered 128 to
256 the pair gains **+3.9%** while arm B gains **+52%**. That is the finding: the
pair is already at its ceiling by offered 128.

**The ceiling is the prefill side, and its TTFT says so.** PD's median TTFT goes
2102 -> 2777 -> **24954 ms**, a 9x jump for a 2x concurrency increase, while arm
B's stays sub-second. At 1024/1024 with MTP accepting ~3.8 tokens/step, one
node's worth of prefill cannot feed one node's worth of decode: work does not
split 1:1, so a 1:1 node split leaves the prefill node as the queue and the
decode node partly idle. Two independent servers are balanced by construction.
PD also admits *less* of the offered load -- live concurrency 210.0 of 256 vs
239.8 -- which rules out "PD trades concurrency for throughput".

**PD's one real win is TPOT under load**: 19.51 vs 28.40 ms at offered 256, and
it barely degrades across the ladder (14.53 -> 18.86 -> 19.51), because the decode
node never has prefill interleaved into its steps. If a deployment is TPOT-bound
and can accept a 25 s TTFT, that is the case for PD. Nothing here is a case for
PD on throughput at this shape.

Two caveats, both stamped in the filenames: arm A used `n = 2 x c` where arm B
used `4 x c` per instance (both >= 2 rounds of the queue, but arm A's runs are
shorter and noisier), and only the decode side ran `MEM_FRACTION=0.85` -- forced,
because at 128 slots it OOMs during graph capture at the default and the OOM
message's suggested `expandable_segments:True` must never be used with Mooncake.
Every resulting KV pool is >= 1.35M tokens against the ~262k this workload needs,
so that is not a throughput confound.

### p5en 2-node BF16 TP16: a TCP-fallback baseline only

This arm ran before the EFA fix existed, so it is a socket number, kept because
it is the only cross-node TP measurement in the file and because the size of the
effect is the point.

Measured 2026-09-09 on P5EN-3/4, `A2A_BACKEND=none NNODES=2 TP_SIZE=16
QUANT=bf16` on stock `lmsysorg/sglang:hy4-preview`:

| | |
|---|---|
| aggregate EFA `tx_bytes` delta over 5 s, all 16 NICs | **0** |
| ENA (`enp71s0`) tx / rx | **157-159 MB/s** each way |
| decode, `#running-req: 1`, `accept len 4.00` | **53.64 tok/s** = 74.6 ms/step |
| of which the 43.7B active parameters' weight read (5.5 GiB/GPU at ~4.8 TB/s) | ~1.1 ms |

So **>95% of the step was the TP all-reduce running on TCP**, and nothing failed:
the server was healthy and the output was correct. The cause is that
`lmsysorg/sglang:hy4-preview` contains **no `libnccl-net*.so`** -- NCCL cannot
speak EFA without aws-ofi-nccl, so it picks `NET/Socket`. `--device=/dev/infiniband`
does not help; it hands device nodes to a stack with no libfabric to drive them.

`a2a=none` needs the plugin **most**, not least: with no dispatch/combine
collective, 2 all-reduces x 78 layers *are* the entire cross-node traffic. The
pre-existing `require_gin_capable_image` guard was scoped to `deepep*`, which is
exactly why this got through.

The fix, and why mounting `/opt/amazon` alone is not enough, is in §4. No
post-fix p5en number exists yet -- see §5.

### Not measured: `deepep_v2` cannot serve this checkpoint at all

Every *configurable* blocker is closed. Four source patches apply, `deep_ep`
2.1.0+97d8f9b imports with `ElasticBuffer` present, NCCL GIN initializes (type 3
on one node, **type 5 / EFA_GDA** across two), the buffer is built on all 16 ranks
(`world_size=16 num_bytes=570425344 allow_hybrid_mode=True`), and DeepEP's own
dispatch kernel JITs. Both the 1-node and the 2-node arm then die in the same
place, at decode CUDA-graph capture, inside *sglang's* activation kernel:

```
silu_and_mul_masked_post_quant.cuh(245): error: static assertion failed
    static_assert(kGroupSize == 128);
```

Hy4's MXFP8 weight block is `[1, 32]`, so `kGroupSize=32`, and both device
implementations of the clamped SwiGLU post-quant hard-code 128. Full evidence,
including what the 2-node arm proved that the 1-node arm could not:
`results/deepep_v2_enablement/README.md`. This is finding C in `UPSTREAM.md`, and
the same kernel blocks v1 `deepep` on the masked path.

---

## 4. Reference: why the defaults are what they are, and what bites

None of this is needed to start a server. All of it is needed to explain one.

### TP8, and why it is not a guess

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

### MoE parallelism: EP and the a2a backends

Short answers: **EP=8 works and is one env var**, `deepep` is the only all-to-all
backend the cookbook offers for this model, and `deepep_v2` gets much further on a
patched image -- every configurable blocker closed, four patches applying cleanly,
one of them a numerics fix rather than a loosened gate -- and **still cannot serve
a token** (§3). `patches/README.md` has the per-blocker arithmetic.

#### EP=8 is arithmetically free on this model

256 routed experts + 1 shared, top-8 sigmoid routing, **`n_group` 1 /
`topk_group` 1** (so there is no expert-group constraint to satisfy),
`moe_intermediate_size` 2048. 256/8 = 32 experts per rank; `moe_tp_size` =
TP/EP/moe_dp = 1, so each rank holds its 32 experts whole. `build_moe_args()`
re-checks all three divisibilities and names the constant that failed.

#### Two different things are both called "EP"

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

#### EP is silently rewritten for every a2a-spanning backend

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

#### Every a2a backend, and what happens if you ask for it

`MoeA2ABackend` (`layers/moe/utils.py`) has twelve members. What each one means
for **this** model:

| `A2A_BACKEND` | this kit | why |
|---|---|---|
| `none` (default) | runs | the verified cell: pure TP MoE, or masked EP when `EP_SIZE>1` |
| `deepep` | runs | the cookbook's own experimentation override, `EP = TP` |
| `deepep_v2` | patched image, and even then **cannot serve** (§3) | three configurable blockers, all in `patches/`: the architecture whitelist (`DeepseekV3/V4`, `Qwen3Moe` -- Hy4 is `HYV4ForCausalLM`), the quant gate rejecting MXFP8, and `mxfp8_act_gran_k` never being set on the v2 pre-permute. `require_deepep_v2_image()` refuses the stock image up front. The fourth blocker is a missing CUDA kernel and is not patchable |
| `megamoe` | **refused** | the cookbook omits it: its fused path is not wired for Hy4's sigmoid-scored, bounded-SwiGLU experts |
| `mori` | **refused** | ROCm |
| `ascend_fuseep`, `ascend_tp` | **refused** | Ascend NPU |
| `mooncake`, `nixl`, `flashinfer`, `pplx`, `customized` | `ALLOW_UNVALIDATED_A2A=1` | real code paths, EP-spanning, but not exercised on Hy4 |

**`mooncake`/`nixl` in that list are the MoE expert all-to-all transport, not the
PD KV transport.** The PD one is `TRANSFER_BACKEND`, a completely unrelated flag
that happens to take the same two words. The gate says so in its error message,
because a typo between them would otherwise "work".

#### Things to know before reading an EP number

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
* **A v2 number is not quotable until its logits match v1's** -- moot so far, since
  the arm has never produced a token (§3), but it is the gate to apply the day it
  does. Blocker 3 is a real numerics bug on the contiguous path, and a mis-scaled GEMM there still produces
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

### Cross-node: NCCL silently uses TCP without the OFI plugin

What it costs, and how it presents, is in §3: a 2-node TP16 arm that came up
healthy, served correct output, and moved **0 bytes** over EFA.

Fixed in `env_common.sh`'s `build_efa_args`, called by `10_launch_standalone.sh`
for **every** `NNODES>1` arm. The host already has the whole stack (efa installer
3.3.0, libfabric 2.6.0, `/opt/amazon/ofi-nccl/lib/libnccl-net-ofi.so`), so it is
mounted rather than baked into a 49 GB image. Mounting `/opt/amazon` alone is
**not** enough -- the host's libfabric wants `EFA_1.7` / `IBVERBS_1.18` and the
image's distro rdma-core is older:

```
OSError: /lib/x86_64-linux-gnu/libefa.so.1: version `EFA_1.7' not found
```

so the two rdma-core sonames and the provider directory come along under
`/host-efa`. An image that ships its own aws-ofi-nccl (the
`deepep-v2-efa-official:sm90-*` ones pin a libfabric their DeepEP was built
against) is left alone. `EFA_INJECT=0` opts out and says loudly that the numbers
are socket numbers.

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

### PD: why it needs its own image

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

#### Build once, pull everywhere (ECR)

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

#### The alternative KV backend, for the record

NIXL is a real option here and was verified on B300-1 the same day:
`libplugin_LIBFABRIC.so` **is** in the stock image, but its bundled libfabric is
too old for it (`version 'FABRIC_1.7' not found`). With the EFA image's
libfabric it loads, `create_backend("LIBFABRIC")` succeeds, GPU memory registers,
and `fi_info -p efa` enumerates 48 EFA domains (`efa-direct`, `FI_PROTO_EFA`)
inside the container. Select it with `TRANSFER_BACKEND=nixl` on **both** sides;
`env_common.sh` then sets `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`, whose
default is `UCX` -- and UCX has no EFA support, so leaving it would quietly run
over TCP.

#### Two PD traps that are not optional

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

#### Admission ceiling on the decode side

With MTP on, the decode server's `max_running_requests` is reset to 48, and on
the decode side that is the concurrency ceiling of **the whole pair**. A c=256
benchmark against this arm is measuring a 48-slot queue unless `MAX_RUNNING` is
set explicitly -- and if you set it, put it in the results tag.

### p5en (H200): three constraints leave exactly one geometry

p5en.48xlarge is 8x H200 (143,771 MiB each) + 16 EFA NICs. Three independent
constraints leave exactly **one** runnable arm on it, and each one eliminates the
configuration you would otherwise reach for first.

#### MXFP8 is out: the checkpoint needs compute capability 100

`layers/quantization/fp8.py` `get_min_capability` returns **100** when
`use_mxfp8`, and H200 is 90. This is not a gate to loosen -- there is no sm_90
MXFP8 expert GEMM behind it. So on p5en, `QUANT=bf16` is the only option, which
means the 758 GiB `Hy4-preview-FP8` download is dead weight on these hosts.

#### One node is out: BF16 does not fit, by 330 GiB

| | bytes | GiB |
|---|---|---|
| BF16 weights (131 shards, apparent size) | 1,560,018,288,914 | **1453.3** |
| one node's HBM (8 x 143,771 MiB) | | **1123.2** |
| shortfall | | **330.1** |

That is before the KV pool, activations and the CUDA-graph pool, so `TP_SIZE=8
QUANT=bf16` cannot load at any `MEM_FRACTION`. Note the asymmetry with the FP8
checkpoint, which *would* have fit one node at 758 GiB (94.8 GiB/GPU) -- the
quantization that fits is the one the GPU cannot run.

`TP_SIZE` must divide `num_attention_heads=64`, so on 8-GPU nodes the legal
cross-node values are **TP16 (2 nodes)** and **TP32 (4 nodes)**. At TP16:

* weights **90.8 GiB/GPU**; `MEM_FRACTION=0.90` leaves roughly 35 GiB
* MLA latent KV is `kv_lora_rank + qk_rope_head_dim = 576` halves per layer =
  **87.75 KiB/token** over 78 layers, and it is replicated across TP ranks rather
  than sharded, so ~35 GiB is a **~400K token** pool. Arithmetic, not measured.

#### Only `A2A_BACKEND=none` is reachable at BF16

Three separate walls, none of them a gate this kit can open:

* `deepep_v2`: `_validate_deepep_v2_quant_method()` rejects anything that is not
  an `Fp8MoEMethod` *before* `patches/fmt_layer.diff` is ever consulted, and the
  v2 runner has no kernel behind `UnquantizedFusedMoEMethod`. `env_common.sh:449`
  refuses the combination rather than letting it fail 10 minutes in.
* `deepep` (v1) cross-node: v1 cannot reach an EFA NIC at all -- it dies on a late
  `NVSHMEM_QP_DEPTH` assert. v1's cross-node path is IB-only.
* `deepep` (v1) with BF16 experts at all: unquantized DeepEP MoE supports only
  `low_latency` mode, and that mode's unquantized masked runner then refuses with
  `forward_deepgemm_masked is deprecated`.

So the p5en arm measures **Hy4's TP all-reduce over EFA**, not MoE a2a, and it
cannot be used to unblock the `deepep_v2` gates (E in `UPSTREAM.md`).

p5en earns its keep another way: it is the CPU/unit-test box the three submitted
upstream PRs were validated on (`pr_validation/`), which needs no Hy4 checkpoint
and none of the above.

---

## 5. Known gaps

* **The TP8 ladder is one point wide.** c=16 exists (§3); 1, 64, 256 and the
  high-throughput profile do not, so the whole TP4-vs-TP8 question rests on one
  concurrency at which 8 GPUs are not loaded. Cheapest missing thing in the file.
* **MTP-on c=256 at TP4** was killed ~2 min in by a cluster shutdown, so
  `results/` holds a header-only log that `gen_bench_table.py` reports under
  "NO rc IN HEADER" rather than treating as a row. That is the point which would
  locate the MTP crossover, if there is one.
* **No p5en figure is a real number.** Every one of them is the TCP-fallback run,
  taken before `build_efa_args` existed. Re-run the same arm now that the host EFA
  stack is injected, confirm the transport
  (`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET`, then `grep -m1 'NET/'`), and run
  the 1/16/64/256 ladder at 1k/1k so it can be set against a B300 TP8 row.
* **`04_fix_multinic_routing.sh` has never been exercised on p5en's 16 EFA ENIs**,
  only on p6-b300's 18. It is written from the ENI inventory rather than a fixed
  count, but that is untested here.
* **TP32 on four p5en** is legal arithmetic and nothing more.
* **BF16 TP8 on B300** has never been run, though it is a verified cookbook cell.
* **`EP_SIZE=8 A2A_BACKEND=none` vs pure TP** at TP8 has never been measured. It
  is not obvious in either direction: masked EP trades MoE FLOPs per rank for zero
  extra communication, and on one node the communication is NVLink anyway.
* **The `deepep_v2` numerics gate has never been satisfied**, because the arm has
  never served a token (§3). The BF16 p5en arm cannot unblock it either: the quant
  gate rejects unquantized MoE before any patch is consulted.
* **Four of the eight triaged upstream findings are still held** -- C, B1, B2 and
  E, with F folding into B1 -- including C, the group-32 kernel that is the actual
  wall. `UPSTREAM.md` says what each one needs; C and B1 each need a B300 hour on
  a clean `main`.
* The base image tag `lmsysorg/sglang:hy4-preview` is a **moving tag** and is not
  pinned by digest. Pin it before a real measurement campaign, or a re-pull
  silently changes what every `results/` log refers to.
* Resuming a cold, *stopped* PD pair -- the instance store is **wiped** by
  stop/start while `results/` on the EBS root survives; budget ~15 min to the
  first request:

  ```bash
  bash 04_fix_multinic_routing.sh       # every node, or nothing talks
  bash 00_download_models.sh mxfp8      # ~4 min at ~3 GB/s, per host
  bash 05_pull_pd_image.sh              # ~75 s, the image itself is in ECR
  # B300-1: bash 20_launch_prefill.sh   # B300-2: bash 21_launch_decode.sh
  bash 22_launch_router.sh              # then 91_bench.sh through :8000
  ```
