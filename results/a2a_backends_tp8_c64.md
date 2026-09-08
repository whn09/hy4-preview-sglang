# a2a backends on Hy4-preview, TP8, c=64 — the only numerically-valid DeepEP row

Regenerate with `python3 gen_a2a_table.py` (it reads each JSON's own resolved
`server_info.server_args`, so the axes below are what the servers actually ran):

```
# a2a backends on Hy4-preview MXFP8, tp8 ep8 1 node, isl/osl 1024, c=64, max_running=128, spec=EAGLE

| a2a | deepep_mode | dispatch | atg | cuda_graph | out tok/s | TPOT med ms | TTFT med ms | accept | parity |
|---|---|---|---|---|---|---|---|---|---|
| deepep | auto | auto | 0 | on | **2711.81** | 16.46 | 4926 | 3.95 | 7.48 WRONG |
| none | auto | auto | 0 | on | **2324.38** | 19.57 | 2985 | 3.73 | reference |
| none | auto | auto | 0 | off | **456.33** | 95.98 | 2584 | 3.72 | reference |
| deepep | normal | bf16 | 1 | on | **426.06** | 104.29 | 4287 | 3.73 | 0.410 (at floor) |
```

## What this says

**The 5.5x collapse is not DeepEP.** Against the matched-eager control
(`a2a=none` + `--disable-cuda-graph`, layout pinned to `compact`), the correct
DeepEP arm is only **−6.6%** — 426.06 vs 456.33 out tok/s, TPOT 104.29 vs 95.98 ms.
The 2324.38 row is ahead of both by ~5.1x purely because decode is replaying CUDA
graphs. Without that control the DeepEP arm would have been written up as a 5.5x
regression, which is the same class of error as reading the pre-`atg` row as a win.

**The anomaly worth chasing next:** the corrected DeepEP row reports
`disable_cuda_graph: False` and still performs exactly like an eager arm. So graph
capture is being asked for and decode is not benefiting from it — measured, not
inferred, and the mechanism is unidentified. That is where a useful DeepEP arm on
this model would come from; the a2a work itself costs almost nothing here.

**The configuration is forced, not chosen.** Each step is a measured constraint:

1. MXFP8 checkpoint ⇒ there is no MXFP8 DeepEP dispatch on CUDA at all
   (`token_dispatcher/deepep.py:490` raises: Ascend A5, low-latency only), so a
   numerically correct arm must dispatch **bf16** and let the runner quantise at
   `block_shape[1] = 32` itself.
2. bf16 dispatch ⇒ no activation scale arrives with the tokens ⇒ the masked
   (low-latency) runner dies at `moe_runner/deep_gemm.py:718`,
   `act_sf_last = hidden_states_scale.shape[-1]` on `None`. So `DEEPEP_MODE` must be
   **normal**.
3. `DEEPEP_MODE=normal` ⇒ decode runs the normal-mode path, which is where the
   eager-like behaviour above appears.

So on this checkpoint every *fast* DeepEP configuration is numerically wrong and the
one correct configuration is eager. Fixing that means either MXFP8 dispatch support,
or a masked runner that accepts unquantised dispatch output.

Two axes are pinned across all four rows and asserted by the generator: `tp=8 ep=8`,
`max_running=128`, `mem_fraction=0.85`, `chunked_prefill=16384`, MTP on. Accept length
is 3.72-3.95 everywhere, so speculation is not a confound between rows.
