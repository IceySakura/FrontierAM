# The v4 image already carries the author's target environment (Python 3.12, NumPy 2.3.5, SciPy 1.16.3, psutil 7.1.3,
# threadpoolctl 3.6.0, gcc/g++; requirements.lock); v5 only replaces the model-visible workspace with the five files
# of the MODEL_ONLY package (release ihvp-neutral-statement-20261117).
ARG BASE=frontieram-ihvp-sequence:amd64-v4
FROM --platform=linux/amd64 ${BASE}
USER root
RUN rm -rf /workspace/problem.md /workspace/samples
COPY --chown=root:root --chmod=444 workspace/ /workspace/
RUN chmod 755 /workspace && chown 1000:1000 /workspace/submission
WORKDIR /workspace
USER 1000:1000
CMD ["/bin/bash"]
