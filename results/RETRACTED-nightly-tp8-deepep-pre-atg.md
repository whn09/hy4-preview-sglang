# Retracted: the pre-`atg` nightly TP8 `deepep` throughput rows

These two files measured a DeepEP arm that was computing the MoE on **one token in
eight** and returning exactly `0.0` from three of its eight ranks
(`patches/README.md`, "Blocker 7"). They were taken before
`--disable-attn-tp-gather` existed, so their filenames carry no `atg` axis:

* `mxfp8-nightly-tp8-deepep-mxfp8-g32-low-latency-specon-isl1024-osl1024-c64-n128.{json,log}`
* `mxfp8-nightly-tp8-deepep-mxfp8-g32-low-latency-specon-isl1024-osl1024-c1-n32.{json,log}`

They are kept, not deleted, because they are the cleanest possible demonstration of
why a parity gate exists at all: at c=64 the broken arm reported **2711.81 out tok/s
against the a2a=none arm's 2324.38, i.e. +16.7%** — a wrong arm that is *faster*, for
the obvious reason that most of the expert work never happened. Nothing here was
detectably off from the outside: the server was healthy, the completions were fluent,
and the throughput moved in the direction a working DeepEP arm would have.

The corrected rows are the `...-atg1-...` files. The `-none-` rows are unaffected:
`require_attn_tp_gather()` tests the a2a backend, so the flag is a no-op on that arm
and it needs no axis in its name.
