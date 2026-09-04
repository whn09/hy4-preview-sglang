#!/bin/bash
# Download the Hy4-preview weights to local NVMe. Run on the HOST (uses the
# /opt/pytorch venv), not inside the container.
#
#   bash 00_download_models.sh          # MXFP8 (~760GB) -- the B300 recipe (TP8)
#   bash 00_download_models.sh bf16     # BF16  (~1.5TB) -- the B300 TP8 recipe
#   bash 00_download_models.sh both
#
# Both repos are public (Apache-2.0, gated=false), so HF_TOKEN is optional.
set -euo pipefail

NVME="${NVME:-/opt/dlami/nvme}"
MODEL_DIR="${MODEL_DIR:-$NVME/models}"
HF_HOME_DIR="${HF_HOME_DIR:-$NVME/hf}"

source /opt/pytorch/bin/activate
export HF_HOME="$HF_HOME_DIR"
# These repos route through Xet, which ignores HF_HUB_ENABLE_HF_TRANSFER (it
# warns the variable is deprecated). This is the replacement knob; measured
# ~3 GB/s sustained on the 169-shard MXFP8 repo from ap-northeast-2.
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DOWNLOAD_TIMEOUT=60

mkdir -p "$MODEL_DIR" "$HF_HOME_DIR"

# A freshly relaunched DLAMI has no `hf` CLI in /opt/pytorch, and the failure is
# a bare "hf: command not found" 1 second into a nohup'd download -- i.e. it looks
# like the download is running for as long as nobody reads the log. Install it
# here rather than documenting it as a prerequisite. hf_xet is what makes
# HF_XET_HIGH_PERFORMANCE above mean anything.
if ! command -v hf >/dev/null 2>&1; then
    echo "==> hf CLI missing (fresh instance); installing huggingface_hub[cli,hf_xet]"
    pip install -q -U "huggingface_hub[cli,hf_xet]"
    command -v hf >/dev/null 2>&1 || { echo "hf still not on PATH after install" >&2; exit 1; }
fi
hf version 2>/dev/null || true

dl() {
    local repo="$1" dest="$2"
    echo "==> $repo -> $dest"
    hf download "$repo" --local-dir "$dest" --max-workers 16
}

case "${1:-mxfp8}" in
    mxfp8|fp8) dl tencent/Hy4-preview-FP8 "$MODEL_DIR/Hy4-preview-FP8" ;;
    bf16)      dl tencent/Hy4-preview     "$MODEL_DIR/Hy4-preview" ;;
    both|all)
        dl tencent/Hy4-preview-FP8 "$MODEL_DIR/Hy4-preview-FP8"
        dl tencent/Hy4-preview     "$MODEL_DIR/Hy4-preview"
        ;;
    *) echo "usage: $0 [mxfp8|bf16|both]" >&2; exit 1 ;;
esac

echo
echo "==> done"
du -sh "$MODEL_DIR"/Hy4-preview* 2>/dev/null || true
