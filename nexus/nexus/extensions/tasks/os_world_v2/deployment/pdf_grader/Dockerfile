FROM ubuntu:22.04@sha256:0a9887931b0e38af2e3487ed39e29436b0535c4f3f52e57745baaf79d1898956

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv tesseract-ocr tesseract-ocr-eng sudo util-linux ca-certificates \
    && python3 -m venv /opt/nexus-pdf-grader \
    && /opt/nexus-pdf-grader/bin/pip install --no-cache-dir \
       PyMuPDF==1.26.7 numpy==2.2.6 Pillow==11.3.0 pypdf==6.1.3 \
    && useradd --create-home --user-group --shell /bin/bash user \
    && useradd --create-home --user-group --shell /usr/sbin/nologin nexus-pdf-grader \
    && printf 'user ALL=(ALL) NOPASSWD: ALL\n' > /etc/sudoers.d/pdf-control \
    && chmod 440 /etc/sudoers.d/pdf-control \
    && dpkg-query -W > /opt/pdf-packages.txt \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Build with deployment/ as the context; reuse the evaluator-only transport.
COPY cad_grader/server.py /opt/pdf-server.py
EXPOSE 5000
USER root
CMD ["/usr/bin/python3", "/opt/pdf-server.py"]
