ARG NEXUS_WORKLOAD_BASE_IMAGE=nexus-distributed-workload:local
FROM ${NEXUS_WORKLOAD_BASE_IMAGE}

ARG NEXUS_WORKLOAD_BASE_IMAGE
ARG NEXUS_PYPI_INDEX_URL=https://mirrors.tencent.com/pypi/simple/
ARG NEXUS_BUILD_REVISION=unknown

ENV PIP_INDEX_URL=${NEXUS_PYPI_INDEX_URL} \
    UV_INDEX_URL=${NEXUS_PYPI_INDEX_URL}

LABEL org.opencontainers.image.revision="${NEXUS_BUILD_REVISION}" \
      io.nexus.base.workload="${NEXUS_WORKLOAD_BASE_IMAGE}" \
      io.nexus.pypi.index="${NEXUS_PYPI_INDEX_URL}"

COPY deployment/distributed/rpm/builder.txt /tmp/builder.txt

RUN dnf -y install $(cat /tmp/builder.txt) \
    && dnf clean all \
    && rm -rf /var/cache/dnf /tmp/builder.txt

RUN uv export --quiet --frozen --no-dev --no-emit-project \
        --extra distributed --extra runtime --extra workload --extra builder \
        --output-file /tmp/nexus-requirements.txt \
    && uv pip install --python /opt/nexus-venv/bin/python --no-cache --no-config --no-deps --require-hashes \
        --requirement /tmp/nexus-requirements.txt \
    && rm -rf /tmp/uv-cache /tmp/nexus-requirements.txt

ENTRYPOINT ["/opt/nexus-venv/bin/python", "-m", "nexus.distributed.dependency_build.worker"]
