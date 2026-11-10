FROM python:3.12.16-slim-bookworm@sha256:e0b3f8d16a5ccfbe602d5c94297e5dbcda2ec256edca2b80fcd44e39e1ff6feb
ENV DEBIAN_FRONTEND=noninteractive \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LC_ALL=C.UTF-8
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash coreutils procps util-linux build-essential gfortran \
    octave libopenblas-dev liblapack-dev liblapacke-dev libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-linux.txt /opt/frontieram/requirements-linux.txt
RUN python -m pip install --no-cache-dir -r /opt/frontieram/requirements-linux.txt \
    && useradd --uid 1000 --create-home --shell /bin/bash solver \
    && mkdir -p /workspace /submission /opt/frontieram/evaluator \
    && chmod 0755 /workspace /submission
COPY supervisor.py /opt/frontieram/evaluator/supervisor.py
RUN chmod 0444 /opt/frontieram/evaluator/supervisor.py
WORKDIR /submission
ENTRYPOINT ["python", "/opt/frontieram/evaluator/supervisor.py"]
