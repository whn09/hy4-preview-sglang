"""Generate the PR #38609 result table from the arm JSONs. Never transcribe."""

import glob
import hashlib
import json
import os
import sys

RESULTS = sys.argv[1] if len(sys.argv) > 1 else "/opt/dlami/nvme/results"
PREFIX = sys.argv[2] if len(sys.argv) > 2 else "qwen3moe-fp8_tp2_ep2_a2a-deepep-auto"


def axes(tag):
    parts = tag.split("_")
    return {p.split("-", 1)[0]: p.split("-", 1)[1] for p in parts if "-" in p and p.split("-", 1)[0] in ("hook", "order", "epalg", "a2a")}


def digest(d):
    if d.get("outcome") != "served":
        return ""
    blob = json.dumps(d["completions"], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


rows = []
for path in sorted(glob.glob(os.path.join(RESULTS, PREFIX + "*.json"))):
    d = json.load(open(path))
    a = axes(os.path.basename(path)[:-5])
    rows.append(
        {
            "a2a": a.get("a2a", "none"),
            "hook": a.get("hook", "?"),
            "order": a.get("order", "?"),
            "epalg": a.get("epalg", "?"),
            "outcome": d["outcome"],
            "exception": d.get("exception", ""),
            "bare": d.get("exception_is_bare", False),
            "digest": digest(d),
        }
    )

control = next(
    (r["digest"] for r in rows if r["hook"] == "present" and r["order"] == "base" and r["epalg"] == "none"),
    "",
)

print(f"Model: {PREFIX}\n")
print("| a2a backend | hook | assert order | ep_dispatch_algorithm | outcome | exception | completions |")
print("|---|---|---|---|---|---|---|")
for r in rows:
    if r["outcome"] == "served":
        same = "same as control" if r["digest"] == control and control else r["digest"]
        exc = "—"
    else:
        same = "—"
        exc = f"`{r['exception']}`" + (" (**bare**)" if r["bare"] else "")
    print(
        f"| {r['a2a']} | {r['hook']} | {r['order']} | {r['epalg']} | **{r['outcome']}** | {exc} | {same} |"
    )
