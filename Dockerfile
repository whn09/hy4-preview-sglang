# Hy4-preview on AWS p6-b300 (8x B300, sm_103), with EFA-capable PD disaggregation.
#
# This is the kimi-k3 kit's image with the base swapped and the K3-specific parts
# removed. Layers on top of the cookbook's own hy4 image:
#   - GDRCopy:  GPU memory registration for EFA VRAM transfer
#   - EFA:      aws-efa-installer 1.50.0 (libfabric 2.6 efa-direct + aws-ofi-nccl
#               1.21.1) -- which also REPLACES the base image's older rdma-core
#   - NCCL:     pip nvidia-nccl-cu13 2.31.2
#   - Mooncake: the published EFA + CUDA 13 wheel, replacing the base image's
#               non-EFA one (KV transfer for PD disaggregation)
# No DeepEP and no source patches: Hy4's verified recipes are pure TP, the model
# needs no fixes in this base, and MoE a2a never leaves the node here.
#
# THE SINGLE-NODE ARMS DO NOT USE THIS IMAGE. 10_launch_standalone.sh runs the
# stock `lmsysorg/sglang:hy4-preview`, which is the cookbook's tested artefact.
# This file exists for one reason: PD disaggregation moves the KV cache between
# two hosts, and on this hardware that wire is EFA.
#
# Why the stock image cannot do PD over EFA -- measured, not assumed:
#   $ ldd mooncake/engine.so | grep -i fabric                     -> nothing
#   $ strings mooncake/engine.so | grep -oE '[A-Za-z]*Transport' | sort -u
#         -> RdmaTransport, TcpTransport, IbgdaDeviceTransport     (no EFA)
# The base's `mooncake-transfer-engine-cuda13 0.3.12.post1` is built WITHOUT EFA.
# Its only RDMA path is libibverbs, which segfaults on EFA hardware; what is left
# is TcpTransport over ENA. A "Mooncake PD run" on the stock image is a TCP run
# that never says so.
#
# And it cannot be fixed by mounting the host's EFA stack in: the base ships
# rdma-core 50.0 (IBVERBS_1.14) while the host has 64.0amzn0 (IBVERBS_1.18), so
# host libfabric loaded in fails with
#   "libefa.so.1: version `EFA_1.7' not found (required by libfabric.so.1)"
# (verified on B300-1, 2026-09-04). Hence: replace rdma-core AND the wheel here.
#
# Build (no GPU needed, ~10 min):
#   docker build -t hy4-preview-efa:latest -f Dockerfile .
#
# The base tag is a MOVING tag. Pin it by digest before a measurement campaign or
# a re-pull silently changes what every results/ log refers to.
ARG SGLANG_BASE=lmsysorg/sglang:hy4-preview
FROM ${SGLANG_BASE}

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        git cmake build-essential wget curl patch \
        libgflags-dev autoconf automake libtool ninja-build
# NOTE: do NOT purge /var/lib/apt/lists here -- the EFA installer below runs its
# own `apt-get install` (pciutils, tcl, environment-modules, ...) and needs the
# package index. It is purged at the end of the EFA step instead.

# ---- GDRCopy (GPU memreg for EFA) ----
# The kernel module is the HOST's; the launchers pass /dev/gdrdrv in. Only the
# user-space library is built here.
ARG GDRCOPY_VERSION=2.5.2
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz && \
    tar xf v${GDRCOPY_VERSION}.tar.gz && cd gdrcopy-${GDRCOPY_VERSION} && \
    make -j$(nproc) lib lib_install CUDA=/usr/local/cuda PREFIX=/usr/local && \
    rm -rf /tmp/gdrcopy* /tmp/v${GDRCOPY_VERSION}.tar.gz

# ---- AWS EFA installer (libfabric + EFA provider + aws-ofi-nccl) ----
# Bundled rdma-core + --skip-rdma-core so the build does not depend on whether
# the build host has EFA devices. This also replaces the base image's older
# rdma-core, which is the actual fix for the EFA_1.7 symbol mismatch above.
#
# PINNED on purpose. `latest` inside a cached RUN layer is not even honest
# freshness: it is re-resolved only when a line above it changes, so the image
# would claim the newest stack while shipping whatever was newest the day the
# layer was first built, with nothing recording which.
#
# The ncclGinPlugin_v14 gate is kept from the K3 image even though nothing in a
# Hy4 1P1D run needs EFA-GDA (each side is a single-node TP4 server, so NCCL
# never crosses the wire). It costs one `nm` and it is the only cheap way to
# catch a mislabelled pre-GA tarball: a 1.50.0 whose ChangeLog also opens
# `## [1.50.0]` can bundle aws-ofi-nccl 1.20.0-1, which exports only v11/v13.
# Keeping it means this image stays usable the day a cross-node Hy4 arm needs GDA.
ARG EFA_INSTALLER_VERSION=1.50.0
ENV HY4_EFA_INSTALLER=${EFA_INSTALLER_VERSION}
RUN set -eu; \
    cd /tmp; \
    curl -fSL --retry 3 -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz; \
    tar xzf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz; \
    grep -q "^## \[${EFA_INSTALLER_VERSION}\]" aws-efa-installer/ChangeLog.md \
      || { echo "FATAL: tarball is not EFA installer ${EFA_INSTALLER_VERSION}; its ChangeLog opens:"; \
           head -8 aws-efa-installer/ChangeLog.md; exit 1; }; \
    cd aws-efa-installer; \
    . /etc/os-release; \
    case "$VERSION_ID" in \
      24.04) DEBS_DIR=UBUNTU2404 ;; \
      22.04) DEBS_DIR=UBUNTU2204 ;; \
      *)     DEBS_DIR=UBUNTU2404 ;; \
    esac; \
    echo "Using EFA DEBS dir: $DEBS_DIR (Ubuntu $VERSION_ID)"; \
    (apt-get install -y ./DEBS/${DEBS_DIR}/x86_64/rdma-core/*.deb || true); \
    ./efa_installer.sh -y --skip-kmod -g --no-verify --skip-rdma-core; \
    ldconfig; \
    PLUGIN=$(find /opt/amazon -name 'libnccl-net*.so*' -type f 2>/dev/null | head -1); \
    test -n "$PLUGIN" || { echo "FATAL: the installer laid down no libnccl-net*.so under /opt/amazon:"; \
                           dpkg -l 'libnccl-ofi*' || true; exit 1; }; \
    echo "aws-ofi-nccl plugin: $PLUGIN"; \
    nm -D "$PLUGIN" | grep -qw ncclGinPlugin_v14 \
      || { echo "FATAL: $PLUGIN exports no ncclGinPlugin_v14 => no EFA-GDA. Harmless for"; \
           echo "       1P1D today, but it means the tarball is not the release it claims."; \
           dpkg -l 'libnccl-ofi*' || true; \
           echo "       exports it does have:"; nm -D "$PLUGIN" | grep -o 'ncclGinPlugin_v[0-9]*' | sort -u; \
           exit 1; }; \
    echo "EFA ${EFA_INSTALLER_VERSION} verified: ncclGinPlugin_v14 present (EFA-GDA capable)"; \
    /opt/amazon/efa/bin/fi_info --version; \
    rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

# ---- EFA runtime env ----
# Baked in rather than left to a launcher: a PD container that comes up with the
# wrong provider does not fail, it silently transfers over TCP.
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1

# On NGC-based images the EFA installer lays down libnccl-ofi-ngc-v3, which ships
# ONLY libnccl-net-ofi.so -- NOT the default name libnccl-net.so that NCCL
# auto-loads. Without this, NCCL logs "NET/Plugin: Could not find: libnccl-net.so"
# and falls back to TCP sockets, which shows up as slow cross-node traffic rather
# than as an error. Use the SHORT name "ofi": NCCL templates it into
# libnccl-net-<value>.so and resolves that through ldconfig, where the installer
# registered /opt/amazon/ofi-nccl/lib. Do NOT use an absolute path -- NCCL 2.27+
# applies the same templating to it, yielding a bogus doubled path.
# Verify: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET should log
# "NET/OFI Selected provider is efa" -- not "NET/Socket".
ENV NCCL_NET_PLUGIN=ofi

ENV PATH="/opt/amazon/efa/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/amazon/efa/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')") && \
    printf '%s\n' "$TORCH_LIB" > /etc/ld.so.conf.d/zz-torch.conf && \
    ldconfig && \
    echo "registered torch lib dir with ldconfig: $TORCH_LIB"

# ---- NCCL ----
# Carried over from the K3 image, where >= 2.31 is what makes NCCL_GIN_TYPE=5
# (EFA-GDA) reachable at all: 2.30.7's host side has no type-5 backend-version
# entry, so it loads the plugin, assigns type 5, and then dies in
# ncclGinDevCommSetup with "Cannot get backend version for invalid GIN type 5".
#
# Nothing in a Hy4 1P1D run needs that -- each side is single-node TP4, so NCCL
# stays on NVLink and the KV wire is Mooncake, not NCCL. So this layer is
# forward-looking, and it has a cost worth stating: it makes this image's
# intra-node TP all-reduce a DIFFERENT NCCL from the stock image the single-node
# arms run on. Any PD-vs-single-node comparison therefore has an NCCL version as
# a second axis. To take that axis away, build with `--build-arg NCCL_PIP_VER=`
# (empty skips the upgrade) and say so in the results header.
#
# --no-deps: this is a leaf CUDA runtime wheel; resolving its deps lets pip walk
# into torch's pin and downgrade it back.
ARG NCCL_PIP_VER=2.31.2
RUN set -eu; \
    if [ -z "${NCCL_PIP_VER:-}" ]; then \
      echo "NCCL_PIP_VER empty -> keeping the base image's NCCL:"; \
      python3 -c "import importlib.metadata as m; print('  nvidia-nccl-cu13', m.version('nvidia-nccl-cu13'))" || true; \
    else \
      pip install --no-cache-dir --no-deps --upgrade "nvidia-nccl-cu13==${NCCL_PIP_VER}"; \
      ldconfig; \
      NCCL_LIB="$(python3 -c 'import nvidia.nccl, os; print(os.path.join(list(nvidia.nccl.__path__)[0], "lib"))')"; \
      test -e "$NCCL_LIB/libnccl.so.2" || { echo "FATAL: no libnccl.so.2 under $NCCL_LIB"; exit 1; }; \
      v=$(python3 -c "import importlib.metadata as m; print(m.version('nvidia-nccl-cu13'))"); \
      echo "pip NCCL: $v -> $NCCL_LIB"; \
      test "$v" = "${NCCL_PIP_VER}" || { echo "FATAL: asked for ${NCCL_PIP_VER}, got $v"; exit 1; }; \
    fi

# ---- Mooncake transfer engine (EFA + CUDA 13) ----
# The base image's mooncake is the non-EFA build (see the file header), so it has
# to be replaced. This used to be a ~15-minute cmake build of a fork; both fixes
# that fork carried are now upstream and published in this wheel, which ships
# prebuilt with USE_EFA=ON + USE_CUDA=ON against CUDA 13:
#   - GPU KV-cache registration binds the CUDA primary context before
#     fi_mr_regattr. Without it registration fails "Operation not supported",
#     startup and PD warmup still pass, and the FIRST REAL REQUEST dies.
#   - registerLocalMemoryBatch()'s thread fan-out is bounded, which is what makes
#     MC_MAX_CONCURRENT_REG_MR=8 (set by the launchers) meaningful: registration
#     serializes on the EFA per-domain lock, so FEWER threads is faster. On K3's
#     1376 buffers: 128 -> 130.1 s, 32 -> 51.0 s, 8 -> 20.7 s, 4 -> 24.8 s.
#
# The uninstall is BY DISCOVERED NAME, not a guessed list. The base image ships
# the distribution `mooncake-transfer-engine-cuda13` -- not `mooncake` and not
# `mooncake-transfer-engine` -- so a hard-coded uninstall list is a silent no-op
# and BOTH wheels end up installed at once. They own the same `mooncake/` package
# directory, so pip overwrites what overlaps and leaves the rest: a 161 MB
# libmooncake_pg.so and four pg_*.so from the non-EFA build survive next to the
# EFA build's files, and both `.libs` dirs sit side by side for RPATH to choose
# from. One package dir, two provenances. So: enumerate every installed mooncake*
# distribution, uninstall them all, then delete the package dir and any *.libs
# dirs outright before installing.
#
# Deliberate consequence: the EFA wheel ships the Python shims ep.py / pg.py /
# mooncake_ep_buffer.py but NOT their compiled extensions, so `import mooncake.ep`
# now fails loudly with "Mooncake EP was not built". That is correct here --
# mooncake is the KV-transfer engine in this kit and nothing else. Do not
# "restore" those files by skipping the uninstall; that resurrects the
# mixed-provenance directory whose EP device library is the non-EFA one.
ARG MOONCAKE_PKG=mooncake-transfer-engine-efa-cuda13
# Empty = take whatever pip resolves, which is right for a single host chasing
# the newest fix. Set it when two hosts must end up with the SAME image -- this
# is the only unpinned input left in the file:
#   docker build --build-arg MOONCAKE_VER=0.3.13.post1 ...
ARG MOONCAKE_VER=
RUN set -eu; \
    SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"); \
    old=$(pip list --format=freeze 2>/dev/null | sed -n 's/^\(mooncake[^=]*\)==.*/\1/p' | tr '\n' ' '); \
    echo "removing pre-installed mooncake distributions: ${old:-<none>}"; \
    if [ -n "$old" ]; then pip uninstall -y $old; fi; \
    rm -rf "$SP/mooncake" "$SP"/mooncake*.libs; \
    SPEC="$MOONCAKE_PKG"; \
    if [ -n "${MOONCAKE_VER:-}" ]; then SPEC="$MOONCAKE_PKG==$MOONCAKE_VER"; fi; \
    echo "installing mooncake: $SPEC"; \
    pip install --no-cache-dir "$SPEC"; \
    python3 -c "import importlib.metadata as m; print('mooncake wheel:', '$MOONCAKE_PKG', m.version('$MOONCAKE_PKG'))"; \
    n=$(pip list --format=freeze 2>/dev/null | grep -c '^mooncake' || true); \
    test "$n" -eq 1 || { echo "FATAL: $n mooncake distributions installed, expected exactly 1:"; \
                         pip list --format=freeze | grep '^mooncake'; exit 1; }

# ---- build-time sanity checks ----
# libcuda.so.1 is absent at build time (no GPU), so use the CUDA stub to let the
# import of the CUDA-linked mooncake engine resolve.
#
# Note what is NOT the discriminator here: `strings engine.so | grep -ci efa`
# returns 389 on the base image's NON-EFA wheel, because it matches the substring
# in "d-efa-ult". The EFA build is the one that LINKS libfabric and carries
# fi_getinfo/fi_mr_reg; the non-EFA build has zero of those. Both have
# ibv_reg_mr, so ibverbs proves nothing either. Each check below fails with its
# own message -- sharing one trailing `||` once reported an import error as a
# link-config error.
RUN set -eu; \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"; \
    python3 -c "from mooncake.engine import TransferEngine; print('mooncake TransferEngine OK')" \
      || { echo "FATAL: cannot import mooncake.engine -- see the traceback above."; exit 1; }; \
    MC=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))"); \
    E="$MC/engine.so"; \
    test -f "$E" || { echo "FATAL: $E missing; the wheel layout changed."; \
                      echo "       found instead:"; ls -1 "$MC" | grep -i engine; exit 1; }; \
    ldd "$E" | grep -q libfabric \
      || { echo "FATAL: $E does not link libfabric -- this is the NON-EFA wheel."; \
           echo "       Check that MOONCAKE_PKG names an ...-efa-... distribution."; \
           echo "       ldd says:"; ldd "$E" | grep -iE "fabric|ibverbs" || true; exit 1; }; \
    fi_syms=$(strings "$E" | grep -cE "^fi_(getinfo|mr_reg)" || true); \
    test "$fi_syms" -ge 2 \
      || { echo "FATAL: $E links libfabric but carries only $fi_syms libfabric"; \
           echo "       entry points (expected >= 2). Suspect a stripped/partial build."; exit 1; }; \
    echo "mooncake EFA verified: links libfabric, $fi_syms libfabric entry points"

# Hy4 preflight: the model class and the DSA attention backend must both still be
# present after the rdma-core/NCCL swaps above. Import-only, no CUDA: this build
# has no GPU, and `import sglang.srt.models.hy4` does not need one.
RUN set -eu; \
    SGL=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))"); \
    grep -rqs "HYV4ForCausalLM" "$SGL/srt/models/" \
      || { echo "FATAL: no HYV4ForCausalLM in $SGL/srt/models -- wrong base image."; exit 1; }; \
    python3 -c "import importlib.util as u, sys; \
      sys.exit(0 if u.find_spec('sglang.srt.layers.attention.flashmla_backend') else 1)" \
      || echo "NOTE: flashmla backend module not found by that name; DSA selection is runtime-resolved."; \
    echo "hy4 preflight OK: HYV4ForCausalLM present"

WORKDIR /sgl-workspace
