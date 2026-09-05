# Can `--moe-a2a-backend deepep_v2` be enabled on Hy4-preview? 1 node and 2 nodes.

Measured 2026-09-05 on B300-3 / B300-4 (p6-b300, 8x B300 each, 16 EFA rails +
2 ConnectX-7 per host, kernel 7.0.0-1011-aws, gdrdrv 2.5.2).

Question asked: does `deepep_v2` come up at all -- **not** how fast it is. So the
deliverable is a list of blockers with the evidence for each, and the answer is
the same on 1 node and on 2: **every configurable blocker is now closed, and the
arm still cannot serve, because one CUDA kernel does not exist.**

## Answer

| | 1 node (TP8, ep=8, `direct`) | 2 nodes (TP16, ep=16, `hybrid`) |
|---|---|---|
| four source patches apply | yes | yes |
| `deep_ep` 2.1.0+97d8f9b imports, `ElasticBuffer` present | yes | yes |
| NCCL GIN initializes | yes, **type 3** | yes, **type 5** (EFA_GDA) |
| ElasticBuffer built on every rank | yes, `world_size=8 num_bytes=203423744` | yes, `world_size=16 num_bytes=570425344 allow_hybrid_mode=True` |
| DeepEP's own dispatch kernel JITs | yes, `Used 42 registers` | yes |
| **sglang's MoE activation kernel JITs** | **NO** | **NO** |
| serves a request | no | no |

Both arms die in the same place, at decode CUDA-graph capture:

```
sglang/kernels/jit/csrc/deepseek_v4/silu_and_mul_masked_post_quant.cuh(245): error: static assertion failed
    static_assert(kGroupSize == 128);
  detected during instantiation of class
    "sglang::SiluAndMulMaskedPostQuantKernel<kGroupSize=32L, kScaleUE8M0=true,
     kSwizzle=false, kUsePDL=true, kApplySwigluLimit=true>"
Failed to build JIT module sgl_kernel_jit_dpsk_v4_silu_mul_quant_varlen_32_true_false_true_true
```

Hy4's MXFP8 checkpoint has `weight_block_size [1,32]`, so `kGroupSize=32`, and
both device implementations of the clamped-swiglu post-quant hard-code `128u`.
See `../../patches/README.md` -> "The wall" for why the Triton path is not a
substitute (its clamp only exists inside `if GEMM1_ALPHA > 0`, which is a
different activation) and for why this blocks v1 `deepep` too.

## What 2 nodes proved that 1 node could not

1 node keeps its whole all-to-all on NVLink, so it never touches the fabric and
cannot exercise the cross-node transport at all. The 2-node run does, and it
found a blocker that is invisible at NNODES=1:

**`--privileged` + `/dev/gdrdrv` are required for GIN type 5, and their absence
does not look like a permissions problem.** Without them:

```
NCCL WARN NET/OFI Failed to initialize GDRCopy: Failed to open gdr handle
NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)
NCCL INFO NET/OFI gin: Initializing
NCCL INFO GIN/Plugin: Failed to initialize any GIN plugin
...
RuntimeError: Assertion exception (csrc/kernels/backend/nccl.cu:101):
  (allow_hybrid_mode ? props.railedGinType : props.ginType) != NCCL_GIN_TYPE_NONE
  and "NCCL GIN is unavailable. This is usually due to a network configuration
  issue, such as `allow_hybrid_mode=0` ... in multi-plane network."
```

Everything about that says *network*, and nothing about it is: the images were
identical, `NCCL_GIN_TYPE=5` / `NCCL_SYM_GIN_KERNELS_ENABLE=0` /
`NCCL_IB_HCA=rdmap` were all set and echoed, `/dev/gdrdrv` existed on the host,
the 16-rank TP NCCL communicator came up healthy over EFA
(`NET/Libfabric/0..7/GDRDMA`), and the ONLY thing missing was `railedGinType`.
Note also that `hybrid` mode reads `props.railedGinType` while `direct` reads
`props.ginType` -- so this cannot be reproduced at 1 node even with the same
container flags.

With them (`10_launch_standalone.sh` now adds both for `deepep*` arms), on the
same command:

```
NCCL INFO GIN/Plugin: Loaded gin plugin Libfabric_GDAKI (v14)
NCCL INFO NET/OFI gin: Initializing
NCCL INFO devCommCreate: creating 11 contexts: 1 GIN connections with 11 contexts each
Initialized DeepEP v2 ElasticBuffer: world_size=16 hidden_size=6144 num_topk=8
  max_dispatch_tokens_per_rank=2048 use_fp8_dispatch=True allow_hybrid_mode=True
  num_bytes=570425344
```

`grep -c nccl.cu:101` goes from 14 to **0**, and the run advances to the kernel
wall above. That is the whole delta.

## Files

`tp16x2node-a2adeepep_v2cap2048-hybrid-gin5-priv-rank{0,1}.log` -- the passing-GIN,
failing-kernel run. Both ranks are kept because the GIN lines interleave across
16 ranks and a single host's log is not a complete picture.

The non-privileged counterfactual was overwritten by the relaunch
(`docker rm -f` in the launcher); its distinguishing lines are quoted above.

## Reproduce

```bash
# rank 0 on B300-3, rank 1 on B300-4
IMAGE=hy4-preview-v2:latest A2A_BACKEND=deepep_v2 \
  NNODES=2 NODE_RANK=$R TP_SIZE=16 DIST_INIT_ADDR=<B300-3 primary IP> \
  CHUNKED_PREFILL=2048 V2_CAP=2048 MEM_FRACTION=0.85 MAX_RUNNING=128 \
  bash 10_launch_standalone.sh
```

`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=GIN,NET` is what makes the GIN decision
visible; at the default `WARN` a failed GIN plugin init prints nothing and you
see only the `nccl.cu:101` assert.
