FROM mcr.microsoft.com/playwright/python:v1.52.0-jammy

ARG NODE_VERSION=20.19.1
ARG UV_PYTHON_INSTALL_MIRROR=https://gh-proxy.com/https://github.com/astral-sh/python-build-standalone/releases/download

ENV DEBIAN_FRONTEND=noninteractive \
    BENCH_ROOT=/opt/mcp-benchmarks \
    BENCH_VENV=/opt/mcp-benchmarks/venv \
    BENCH_NPM=/opt/mcp-benchmarks/npm \
    UV_PYTHON_INSTALL_DIR=/opt/mcp-benchmarks/uv-python \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    npm_config_cache=/tmp/npm-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    lsof \
    python3-pip \
    python3-venv \
    ripgrep \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN arch="$(dpkg --print-architecture)" && \
    case "${arch}" in \
      amd64) node_arch='x64' ;; \
      arm64) node_arch='arm64' ;; \
      *) echo "unsupported architecture: ${arch}" >&2; exit 1 ;; \
    esac && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.xz" -o /tmp/node.tar.xz && \
    tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 --no-same-owner && \
    rm -f /tmp/node.tar.xz && \
    node --version && npm --version

RUN mkdir -p ${BENCH_ROOT}/helpers ${BENCH_ROOT}/local_bin ${BENCH_ROOT}/local_servers ${BENCH_ROOT}/datasets

COPY .build/datasets/ ${BENCH_ROOT}/datasets/
COPY helpers/ ${BENCH_ROOT}/helpers/
COPY local_bin/ ${BENCH_ROOT}/local_bin/

RUN python3 -m pip install --no-cache-dir --upgrade pip uv && \
    python3 -m uv python install --mirror ${UV_PYTHON_INSTALL_MIRROR} 3.12.11 && \
    python3 -m uv venv --python 3.12.11 ${BENCH_VENV} && \
    python3 -m uv pip install --python ${BENCH_VENV}/bin/python --upgrade pip setuptools wheel && \
    python3 -m uv pip install --python ${BENCH_VENV}/bin/python \
      requests==2.32.4 \
      jinja2==3.1.6 \
      python-dotenv>=1.1.0 \
      mcp \
      emails-mcp==0.1.12 \
      wikipedia-api==0.8.1 \
      mcp_server_fetch \
      mcp_server_calculator==0.1.1 \
      yfinance==0.2.61 \
      pytz==2024.2 \
      aiohttp==3.12.15 \
      addict \
      openai \
      openai-agents==0.0.15 \
      playwright \
      sympy \
      tenacity \
      termcolor \
      cli-mcp-server==0.2.5 \
      duckduckgo-mcp-server==0.1.1 \
      mcp-server-git==2025.7.1 \
      mysql-mimic==2.6.4 \
      office-powerpoint-mcp-server \
      pyyaml \
      mcp-youtube-transcript \
      mcp-server-time \
      uv

RUN chmod +x ${BENCH_ROOT}/local_bin/*

RUN mkdir -p /data && \
    cp -a ${BENCH_ROOT}/datasets/atlas/services/agent-environment/data/. /data/

RUN git clone --depth 1 https://github.com/geobio/smitheryai-mcp-servers-github.git ${BENCH_ROOT}/local_servers/github-mcp && \
    cd ${BENCH_ROOT}/local_servers/github-mcp && npm install && \
    sed -i 's/console.log("Starting GitHub MCP Server...")/console.error("Starting GitHub MCP Server...")/' \
      ${BENCH_ROOT}/local_servers/github-mcp/src/index.ts \
      ${BENCH_ROOT}/local_servers/github-mcp/dist/index.js && \
    rm -rf ${BENCH_ROOT}/local_servers/github-mcp/.git

RUN git clone --depth 1 https://github.com/geobio/smitheryai-mcp-servers-weather.git ${BENCH_ROOT}/local_servers/atlas-weather && \
    cd ${BENCH_ROOT}/local_servers/atlas-weather && npm install && \
    rm -rf ${BENCH_ROOT}/local_servers/atlas-weather/.git

RUN git clone --depth 1 https://github.com/geobio/PubMed-MCP-Server.git ${BENCH_ROOT}/local_servers/pubmed && \
    git clone --depth 1 https://github.com/Rudra-ravi/wikipedia-mcp.git ${BENCH_ROOT}/local_servers/wikipedia-mcp && \
    git clone --depth 1 https://github.com/takashiishida/arxiv-latex-mcp.git ${BENCH_ROOT}/local_servers/arxiv-latex-mcp && \
    git clone --depth 1 https://github.com/lockon-n/wandb-mcp-server.git ${BENCH_ROOT}/local_servers/wandb-mcp-server && \
    ${BENCH_VENV}/bin/pip install \
      ${BENCH_ROOT}/local_servers/pubmed \
      ${BENCH_ROOT}/local_servers/wikipedia-mcp \
      ${BENCH_ROOT}/local_servers/arxiv-latex-mcp && \
    rm -rf \
      ${BENCH_ROOT}/local_servers/pubmed/.git \
      ${BENCH_ROOT}/local_servers/wikipedia-mcp/.git \
      ${BENCH_ROOT}/local_servers/arxiv-latex-mcp/.git \
      ${BENCH_ROOT}/local_servers/wandb-mcp-server/.git

RUN mkdir -p /usr/lib64 && \
    ln -sf /lib/x86_64-linux-gnu/libutil.so.1 /usr/lib64/libutil.so.1

RUN ${BENCH_VENV}/bin/uv python install --mirror ${UV_PYTHON_INSTALL_MIRROR} 3.11 3.13 && \
    ${BENCH_VENV}/bin/uv venv --python 3.11 ${BENCH_ROOT}/py311 && \
    ${BENCH_VENV}/bin/uv pip install --python ${BENCH_ROOT}/py311/bin/python \
      arxiv-mcp-server==0.2.11 \
      excel-mcp-server \
      mcp-scholarly \
      office-word-mcp-server \
      ${BENCH_ROOT}/local_servers/wandb-mcp-server && \
    ${BENCH_VENV}/bin/uv venv --python 3.12.11 ${BENCH_ROOT}/py312 && \
    ${BENCH_VENV}/bin/uv pip install --python ${BENCH_ROOT}/py312/bin/python pdf-tools-mcp && \
    ${BENCH_VENV}/bin/uv venv --python 3.13 ${BENCH_ROOT}/py313 && \
    ${BENCH_VENV}/bin/uv pip install --python ${BENCH_ROOT}/py313/bin/python osm-mcp-server==0.1.1

RUN while IFS=',' read -r url sha path; do \
      [ -n "$url" ] || continue; \
      echo "Installing $url at $sha to $path"; \
      rm -rf "$path" 2>/dev/null || true; \
      git clone "$url" "$path"; \
      git -C "$path" checkout "$sha"; \
    done < /data/repos/git_submodule_info.csv && \
    cd /data/repos/mcp_code_executor_workspace && \
    ${BENCH_VENV}/bin/uv sync --python ${BENCH_ROOT}/py311/bin/python

RUN mkdir -p ${BENCH_NPM} && \
    cd ${BENCH_NPM} && \
    npm init -y >/dev/null 2>&1 && \
    npm install \
      @modelcontextprotocol/server-filesystem@2025.11.25 \
      @modelcontextprotocol/server-memory@2025.8.4 \
      @notionhq/notion-mcp-server@1.8.1 \
      @playwright/mcp \
      @lockon0927/playwright-mcp-with-chunk \
      @tokenizin/mcp-npx-fetch \
      howtocook-mcp \
      mcp-remote \
      mongodb-mcp-server@0.2.0 \
      clinicaltrialsgov-mcp-server@1.0.8 \
      @upstash/context7-mcp@1.0.14 \
      @geobio/code_execution_server@0.2.1 \
      mcp-server-code-runner@0.1.7 \
      metmuseum-mcp@0.9.2 \
      @geobio/mcp-open-library@0.1.6 \
      @bharathvaj/whois-mcp@1.0.1 \
      @modelcontextprotocol/server-google-maps@0.6.2 \
      chrome-devtools-mcp@latest \
      global-agent \
      undici && \
    npm install --ignore-scripts @wonderwhy-er/desktop-commander@0.2.7 && \
    python3 ${BENCH_ROOT}/helpers/patch_code_runner_logs.py \
      ${BENCH_NPM}/node_modules/mcp-server-code-runner/dist && \
    node ${BENCH_NPM}/node_modules/@playwright/mcp/node_modules/playwright/cli.js install chrome && \
    mkdir -p ${BENCH_NPM}/node_modules/@vscode/ripgrep/bin && \
    ln -sf /usr/bin/rg ${BENCH_NPM}/node_modules/@vscode/ripgrep/bin/rg && \
    npm cache clean --force && \
    rm -rf /root/.npm /tmp/npm-cache

ENV PATH=${BENCH_ROOT}/local_bin:${BENCH_VENV}/bin:${BENCH_ROOT}/py311/bin:${BENCH_ROOT}/py312/bin:${BENCH_ROOT}/py313/bin:${BENCH_NPM}/node_modules/.bin:${PATH} \
    PYTHONPATH=${BENCH_ROOT}/datasets/universe

RUN ${BENCH_VENV}/bin/pip install \
    huggingface_hub==0.33.4 \
    google-api-python-client>=2.171.0 \
    google-auth>=2.40.3 \
    google-auth-httplib2>=0.2.0 \
    google-auth-oauthlib>=1.2.2

RUN cd ${BENCH_ROOT}/datasets/toolathlon && \
    ${BENCH_VENV}/bin/uv sync --python 3.12.11

WORKDIR /workspace
