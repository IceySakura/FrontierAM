ARG BASE_IMAGE=mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:latest

FROM ${BASE_IMAGE}

ARG UV_INDEX_URL
ARG UV_EXTRA_INDEX_URL

ENV UV_INDEX_URL=${UV_INDEX_URL}
ENV UV_EXTRA_INDEX_URL=${UV_EXTRA_INDEX_URL}

WORKDIR /nix/runtime

RUN rm -rf /nix/runtime/nexus
COPY . /nix/runtime/nexus

WORKDIR /nix/runtime/nexus

RUN rm -rf .venv-3.12 && \
  uv venv .venv-3.12 --python /nix/runtime/nix-env/bin/python3.12 && \
  uv pip install -p .venv-3.12 -e . && \
  VIRTUAL_ENV=/nix/runtime/nexus/.venv-3.12 .venv-3.12/bin/nexus modules install --all --runtime && \
  uv pip install -p .venv-3.12 \
  -r nexus/extensions/tasks/search_stateful/benchmark/docker/uv-requirements.txt && \
  rm -f .venv-3.12/lib/python3.12/site-packages/playwright/driver/node && \
  ln -s /nix/runtime/nix-env/bin/node .venv-3.12/lib/python3.12/site-packages/playwright/driver/node && \
  rm -rf /root/.cache/uv

ENV VIRTUAL_ENV=/nix/runtime/nexus/.venv-3.12
ENV PATH=/nix/runtime/nexus/.venv-3.12/bin:$PATH
