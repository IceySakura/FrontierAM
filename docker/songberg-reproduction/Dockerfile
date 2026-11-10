# amd64 reconstruction of songberg-reproduction:mcmcse-1.5-0 (shipped as Linux ARM64 in
# mcmc-long-run-covariance-maintainer-v1/images/formal-images.tar), the base of local/trajectory-runtime:b1, which is in
# turn the base of trajectory-covariance and galaxy-gravity.
#
# The image history shows python:3.11.15-slim-bookworm plus pip install -r requirements.lock and user worker; R was
# added in committed containers, so the rest is read from the image's files instead:
#   apt-packages.txt  the 163 Debian packages installed beyond the base image, pinned to the versions in its dpkg status
#                     (arm64-only binutils-aarch64-linux-gnu and libhwasan0 dropped; apt adds the native binutils);
#   requirements.lock copied from /opt/bo/requirements.lock (torch from the CPU wheel index: the PyPI amd64 wheel bundles
#                     CUDA);
#   R source packages the versions under /usr/local/lib/R/site-library (mcmcse 1.5-0, fftwtools 0.9-11,
#                     microbenchmark 1.4.10). momentLS 1.0 is not on CRAN and trajectory-runtime deletes it anyway.
# The vendored Codex layers are the author's agent tooling, not part of the solver environment.
FROM python:3.11.15-slim-bookworm
COPY apt-packages.txt requirements.lock /opt/bo/
RUN sed -i 's#http://deb.debian.org#https://mirrors.tencent.com#g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update && apt-get install -y --no-install-recommends $(cat /opt/bo/apt-packages.txt) \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --index-url https://mirrors.tencent.com/pypi/simple/ \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch==2.2.2+cpu" $(grep -v '^torch==' /opt/bo/requirements.lock) \
    && python3 -m pip check
RUN set -eux; cd /tmp; base=https://mirrors.tencent.com/CRAN/src/contrib; \
    for p in fftwtools_0.9-11.tar.gz Archive/mcmcse/mcmcse_1.5-0.tar.gz Archive/microbenchmark/microbenchmark_1.4.10.tar.gz; do \
        python3 -c "import sys,urllib.request;urllib.request.urlretrieve(sys.argv[1],sys.argv[2])" "$base/$p" "$(basename $p)"; done; \
    R CMD INSTALL --library=/usr/local/lib/R/site-library fftwtools_0.9-11.tar.gz mcmcse_1.5-0.tar.gz microbenchmark_1.4.10.tar.gz; \
    rm -f /tmp/*.tar.gz; Rscript -e 'stopifnot(packageVersion("mcmcse") == "1.5.0")'
RUN groupadd --gid 1000 worker && useradd --create-home --uid 1000 --gid 1000 worker \
    && mkdir -p /workspace/submission && chown worker:worker /workspace /workspace/submission
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/home/worker
WORKDIR /workspace/submission
USER 1000:1000
CMD ["/bin/bash"]
