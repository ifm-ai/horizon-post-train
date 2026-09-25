# syntax=docker/dockerfile:1
# Public CUDA/PyTorch base used by the reference environment.
ARG BASE_IMAGE=radixark/miles@sha256:df66f39fc7bc4a2ae8796ae14660c9f50a14d15745e4342df857e3693b948c1c
FROM ${BASE_IMAGE} AS gateway-build

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ARG RUST_TOOLCHAIN=stable
ENV CARGO_HOME=/opt/cargo RUSTUP_HOME=/opt/rustup
ENV PATH="/opt/cargo/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential pkg-config libssl-dev protobuf-compiler libprotobuf-dev curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal --default-toolchain "${RUST_TOOLCHAIN}" --no-modify-path \
    && python3 -m pip install --no-cache-dir maturin==1.12.6

COPY components/smg /build/smg
WORKDIR /build/smg/bindings/python
RUN maturin build --release --features vendored-openssl --interpreter python3 --out /build/wheels

FROM ${BASE_IMAGE}
USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/rl360

COPY docker/ /opt/rl360/docker/
COPY components/ /opt/rl360/components/
COPY src/ /opt/rl360/src/
COPY recipes/ /opt/rl360/recipes/
COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES /opt/rl360/
COPY --from=gateway-build /build/wheels/ /tmp/smg-wheels/

# Keep native CUDA packages from the base; install the bundled Python sources.
RUN python3 -c 'import sys; assert sys.version_info[:2] == (3, 12)' \
    && python3 -m pip install --no-cache-dir uv==0.11.3 \
    && uv pip install --system --break-system-packages \
        -r docker/requirements.txt -c docker/constraints.txt --overrides docker/overrides.txt \
        ./components/harbor . /tmp/smg-wheels/*.whl \
    && uv pip install --system --break-system-packages --no-deps \
        -e ./components/miles -e ./components/megatron-lm -e ./components/sglang/python \
    && rm -rf /tmp/smg-wheels /root/.cache/pip /root/.cache/uv

ENV PYTHONPATH="/opt/rl360/src:/opt/rl360/components/miles:/opt/rl360/components/megatron-lm:/opt/rl360/components/sglang/python:/opt/rl360/components/harbor/src"
ENV PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# Import the actual native extensions, not just their top-level packages.
RUN python3 -c 'import torch, ray, transformer_engine.pytorch, flash_attn, smg.smg_rs, sglang, harbor, miles, agent360; import megatron.core; from agent360.harbor.miles import generate, harbor_server'
CMD ["bash"]
