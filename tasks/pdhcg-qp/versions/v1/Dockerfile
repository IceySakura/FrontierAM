FROM python:3.12.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential nodejs npm git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir numpy==2.3.5 scipy==1.16.3
# Nexus reuses the matching preinstalled CLI, so setup also works offline.
RUN npm install -g @openai/codex@0.114.0 && npm cache clean --force

ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 RAYON_NUM_THREADS=1
WORKDIR /workspace
# Build from this version directory. No feedback pool, private data or reference
# implementation enters this public image.
COPY statement.md /workspace/problem.md
COPY data/samples/ /workspace/public-dev/
COPY grader/qpbench/__init__.py grader/qpbench/model.py \
     grader/qpbench/check.py /workspace/public-dev/io/qpbench/
RUN mkdir -p /workspace/submission && chmod 1777 /tmp
