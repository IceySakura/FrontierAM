# Maintainer-prepared offline context: this Dockerfile, requirements.lock,
# statement.md, samples/ and a matching pre-staged Linux amd64 wheelhouse/.
ARG BASE=frontieram-conic:success-v2-port-v1
FROM --platform=linux/amd64 ${BASE}
USER root
COPY requirements.lock /opt/ihvp/requirements.lock
COPY wheelhouse/ /opt/ihvp/wheelhouse/
RUN python3 -m pip install --no-index --find-links=/opt/ihvp/wheelhouse -r /opt/ihvp/requirements.lock \
    && rm -rf /opt/ihvp/wheelhouse /workspace /root/.cache \
    && mkdir -p /workspace/submission \
    && chown 1000:1000 /workspace/submission
COPY statement.md /workspace/problem.md
COPY samples/ /workspace/samples/
# Samples contain only protocol tools and the tiny smoke fixture. Full-scale
# sample operators must be supplied by the host oracle, never copied here.
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
WORKDIR /workspace
USER 1000:1000
CMD ["/bin/bash"]
