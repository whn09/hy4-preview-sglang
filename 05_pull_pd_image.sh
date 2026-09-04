#!/bin/bash
# HOST-side: get hy4-preview-efa:latest onto this node from ECR instead of
# rebuilding it.
#
#   bash 05_pull_pd_image.sh                     # the pinned tag below
#   IMAGE_TAG=latest bash 05_pull_pd_image.sh    # whatever was pushed last
#
# Build it once (10-15 min, and the apt/EFA stages are CPU- and network-heavy) and
# pull it everywhere else. That is not just a time saving: a node that is running
# somebody else's benchmark must not also be compiling GDRCopy, and two
# independently-built images are not the same image -- `lmsysorg/sglang:hy4-preview`
# is a MOVING tag, so a rebuild a day later can silently pick up different sglang
# source. Pulling makes every arm provably the same bits.
#
# Push side, on the node that built it:
#   aws ecr create-repository --region $ECR_REGION --repository-name $ECR_REPO
#   aws ecr get-login-password --region $ECR_REGION \
#     | docker login --username AWS --password-stdin $ECR_REGISTRY
#   docker tag hy4-preview-efa:latest $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG
#   docker push $ECR_REGISTRY/$ECR_REPO:$IMAGE_TAG
set -euo pipefail

ECR_ACCOUNT="${ECR_ACCOUNT:-579019700964}"
ECR_REGION="${ECR_REGION:-ap-northeast-2}"
ECR_REPO="${ECR_REPO:-hy4-preview-sglang-b300}"
# The default is the self-describing tag, not `latest`: the three versions in it
# (EFA installer / pip NCCL / mooncake wheel) are exactly the axes a PD result
# depends on, so a results/ log that names this tag is reproducible and one that
# names `latest` is not.
IMAGE_TAG="${IMAGE_TAG:-efa1.50.0-nccl2.31.2-mc0.3.13.post1}"
LOCAL_IMAGE="${PD_IMAGE:-hy4-preview-efa:latest}"

ECR_REGISTRY="${ECR_ACCOUNT}.dkr.ecr.${ECR_REGION}.amazonaws.com"
REMOTE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo "pull  : $REMOTE"
echo "as    : $LOCAL_IMAGE"

aws ecr get-login-password --region "$ECR_REGION" \
  | docker login --username AWS --password-stdin "$ECR_REGISTRY"

docker pull "$REMOTE"
# The launchers ask for the local name, so alias rather than editing env_common.sh
# on every host.
docker tag "$REMOTE" "$LOCAL_IMAGE"

# Verify what arrived rather than trusting the tag: a pull that resolved to a
# non-EFA build fails here instead of three hours later as "mooncake session is
# not alive". Same three discriminators require_efa_image() uses.
#
# This container gets NO --gpus, so there is no libcuda.so.1 injected: locate the
# package with find_spec (which does not execute it) instead of `import mooncake`,
# whose extension load would fail on a perfectly good image. Everything below is
# ldd/strings/metadata -- deliberately nothing that needs a driver.
docker run --rm --entrypoint bash "$LOCAL_IMAGE" -c '
set -e
MC=$(python3 -c "import importlib.util as u; s=u.find_spec(\"mooncake\"); print(list(s.submodule_search_locations)[0])")
test -f "$MC/engine.so" || { echo "FATAL: no engine.so under $MC"; ls -1 "$MC"; exit 1; }
ldd "$MC/engine.so" | grep -q libfabric \
  || { echo "FATAL: pulled image does not link libfabric -- NON-EFA mooncake."; exit 1; }
n=$(strings "$MC/engine.so" | grep -cE "^fi_(getinfo|mr_reg)" || true)
test "$n" -ge 2 || { echo "FATAL: only $n libfabric entry points."; exit 1; }
python3 -c "import importlib.metadata as m; print(\"mooncake:\", [d.metadata[\"Name\"]+\" \"+d.version for d in m.distributions() if d.metadata[\"Name\"].startswith(\"mooncake\")])"
python3 -c "import importlib.metadata as m; print(\"nccl    :\", m.version(\"nvidia-nccl-cu13\"))"
/opt/amazon/efa/bin/fi_info --version | head -1
echo "image OK: EFA mooncake present"'

echo "done. $LOCAL_IMAGE is ready on $(hostname)."
