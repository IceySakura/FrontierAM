FROM mirrors.tencent.com/tlinux/tlinux4.0-minimal

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/nexus-label-venv/bin:$PATH

WORKDIR /app

RUN dnf -y install python3-pip \
    && dnf clean all \
    && rm -rf /var/cache/dnf \
    && python3 -m venv /opt/nexus-label-venv \
    && python -m pip install --no-cache-dir --upgrade pip uv

COPY pyproject.toml README.md ./
COPY nexus ./nexus
COPY deployment ./deployment
COPY services/packages/infra ./services/packages/infra
COPY services/apps/label ./services/apps/label

RUN uv pip install --python /opt/nexus-label-venv/bin/python --no-cache \
        ".[otel]" \
        ./services/packages/infra \
        ./services/apps/label \
    && rm -rf /tmp/uv-cache

EXPOSE 8080

ENTRYPOINT ["opentelemetry-instrument", "nexus-label"]
