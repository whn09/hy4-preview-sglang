# The p5en TP16 c=1 point, twice: TCP fallback vs EFA

Two runs landed in **one filename** on 2026-09-09, because `91_bench.sh` appends to
the `.json` but overwrites the `.log`, and nothing in the tag records the
transport. Recovered here rather than deleted -- the pair is the measurement of
what the missing OFI plugin costs.

| | out tok/s | TTFT p50 (ms) | TPOT p50 (ms) | duration (s) | transport |
|---|---|---|---|---|---|
| first run | **46.31** | 555.52 | 19.78 | 707.5 | TCP over ENA |
| second run | **122.00** | 178.40 | 7.55 | 268.6 | EFA |
| | **2.63x** | 3.11x | **-61.8%** | 2.63x | |

Both are `bf16 tp16x2node low-latency spec=on isl/osl 1024/1024 c=1 n=32` on
P5EN-3 (rank 0) + P5EN-4 (rank 1), 16 GPUs.

## Which run is which, and how that was established

The `.log` that survives belongs to the **second** run only, so the assignment is
inferred rather than stamped. The chain:

* The container was relaunched at **07:27:41Z** and `docker inspect` shows the
  `build_efa_args` mounts on it: `/opt/amazon:/opt/amazon:ro`, the host
  `libefa.so.1` / `libibverbs.so.1` under `/host-efa`, and
  `LD_LIBRARY_PATH=/host-efa:/host-efa/libibverbs:/opt/amazon/ofi-nccl/lib:...`.
  So everything from 07:27:41 on had the OFI plugin available; nothing before it
  did.
* The surviving `.log` header says `started=2026-09-09T07:30:23Z` and its run
  lasted 268.6 s, ending 07:34:52. The rest of the ladder then follows with no
  gap: c=16 at 07:35:29 (37.5 s), c=64 at 07:36:45 (80.6 s), c=256 at 07:38:45
  (297.3 s). The 707.5 s run does not fit anywhere in that chain, so it ran
  before the 07:27:41 relaunch.
* The server log agrees on both sides of the restart: the pre-restart container
  printed `gen throughput (token/s): 53.64` at 07:14:55, the post-restart one
  printed `130.37` at 07:44:12 -- a 2.43x step at bs=1, against 2.63x measured at
  the bench level.

**So c=16 / c=64 / c=256 in this directory are all EFA runs.** Only the c=1 point
has a TCP twin, and only because it was the first thing run twice.

## What to fix so this cannot recur

The transport is not an axis in the tag, yet it moves the answer by 2.63x. Two
separate defects produced this file:

1. `TOPO` (`tp16x2node`) records the node count but nothing about the network. For
   any `NNODES>1` arm the tag should record whether the EFA stack was injected --
   that is a launch-time fact `build_efa_args` already knows.
2. `91_bench.sh` appends the `.json` and truncates the `.log`, so a re-run under
   the same tag leaves a `.json` with two rows and a `.log` describing one of
   them. `gen_bench_table.py` reads the `.log`, so the published table was not
   wrong -- but anything reading the `.json` gets both runs with no way to tell
   them apart.
