# Build context: scripts/seal/prepare_ihvp_sequence.py --image-context output,
# containing generated sample bodies and a maintainer-staged wheelhouse/.
# The BASE image must already exist locally for linux/amd64. No network install.
ARG BASE=frontieram-conic:success-v2-port-v1
FROM --platform=linux/amd64 ${BASE}
USER root
COPY requirements.lock /opt/ihvp/requirements.lock
COPY wheelhouse/ /opt/ihvp/wheelhouse/
RUN python3 -m pip install --no-index --find-links=/opt/ihvp/wheelhouse -r /opt/ihvp/requirements.lock \
    && rm -rf /opt/ihvp/wheelhouse /workspace /root/.cache \
    && mkdir -p /workspace/submission /workspace/samples \
    && chown 1000:1000 /workspace/submission
COPY statement.md /workspace/problem.md
COPY data/samples/ /workspace/samples/
# The staged offline context must contain actual bodies, not recipes alone.
RUN python3 /workspace/samples/check.py --verify-package
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 LANG=C.UTF-8
WORKDIR /workspace
USER 1000:1000
CMD ["/bin/bash"]
