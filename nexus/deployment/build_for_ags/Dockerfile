FROM nexus-runtime:test AS runtime-data
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ARG NODE_VERSION=20.19.5

# 从 nexus-runtime 拷贝 nix 运行时环境
COPY --from=runtime-data /nix/ /nix/

# 安装系统级依赖：Python、字体、PDF 工具、OCR、文档转换
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ca-certificates \
    curl \
    xz-utils \
    # CJK fonts
    fonts-noto-cjk \
    fonts-wqy-microhei \
    fonts-wqy-zenhei \
    fonts-symbola \
    fontconfig \
    # PDF: poppler tools (pdftotext, pdfimages, pdftoppm)
    poppler-utils \
    # PDF: merge/split/encrypt
    qpdf \
    # PDF/docx: OCR support
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-chi-tra \
    # docx/xlsx/pptx/pdf: headless format conversion and formula recalculation
    libreoffice \
    # docx: text extraction
    pandoc \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js v20 from official binary tarball
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" \
        -o /tmp/node.tar.xz && \
    tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 && \
    rm -f /tmp/node.tar.xz && \
    node --version && npm --version

# Let python3 / pip3 be callable as python / pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Rebuild system font cache so Noto CJK fonts are discoverable
RUN fc-cache -fv

# Use Tencent Cloud package mirrors and fail-fast config for Python and Node installs
RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple && \
    pip config set global.retries 0 && \
    pip config set global.timeout 4 && \
    npm config set registry http://mirrors.tencent.com/npm/ && \
    # npm config set strict-ssl false && \
    npm config set fetch-retries 0 && \
    npm config set fetch-timeout 4000 && \
    npm config set global true

# Data science libraries
RUN pip install --no-cache-dir \
    pandas numpy matplotlib seaborn scipy scikit-learn \
    openpyxl xlrd jupyter ipython && \
    pip install --no-cache-dir \
    pypdf pdfplumber reportlab pdf2image pytesseract && \
    pip install --no-cache-dir \
    "markitdown[pptx]" Pillow html2text

# npm packages for docx and pptx creation
RUN npm install -g pptxgenjs docx react-icons react react-dom sharp

# Make globally installed npm modules resolvable from plain `node -e "require(...)"`
# Node 官方 tar 安装到 /usr/local 后，`npm root -g` 为 /usr/local/lib/node_modules
ENV NODE_PATH=/usr/local/lib/node_modules

# Keep matplotlib config location stable for bash-invoked python scripts too
# Use a writable shared dir so runtime platforms that override UID still work.
ENV MPLCONFIGDIR=/opt/matplotlib
RUN mkdir -p "$MPLCONFIGDIR" && chmod 1777 "$MPLCONFIGDIR"

# Configure matplotlib: set CJK-capable sans/mono defaults and safely refresh font cache
RUN python3 -c "\
import glob, matplotlib, os; \
cfg = matplotlib.get_configdir(); \
cache = matplotlib.get_cachedir(); \
os.makedirs(cfg, exist_ok=True); \
[os.remove(p) for p in glob.glob(os.path.join(cache, 'fontlist-*')) if os.path.isfile(p)]; \
open(os.path.join(cfg, 'matplotlibrc'), 'w').write(\
'font.family: sans-serif\n'\
'font.sans-serif: WenQuanYi Micro Hei, SimHei, Noto Sans CJK SC, Noto Sans CJK JP, Symbola, DejaVu Sans, Arial, sans-serif\n'\
'font.monospace: WenQuanYi Micro Hei, SimHei, Noto Sans CJK SC, Noto Sans CJK JP, Symbola, DejaVu Sans Mono, monospace\n'\
'axes.unicode_minus: False\n')" && \
python3 -c "\
import matplotlib.font_manager as fm; \
cjk = [f.name for f in fm.fontManager.ttflist if any(k in f.name for k in ('Noto', 'WenQuanYi', 'SimHei'))]; \
print('CJK fonts registered:', cjk[:8])"

# 安装envd组件
COPY --from=ccr.ccs.tencentyun.com/ags-image/envd@sha256:2fe2b8621485874d56216d898dbe8cb60ffa177691be7f8e18a7f906fcbc3851  /usr/bin/envd /usr/bin/envd
RUN chmod +x /usr/bin/envd

# Patch files: lazy-load extensions + optimised warmup + runtime patches + yuanbao agent tools
# COPY nexus/extensions/__init__.py                              /nix/runtime/nexus/nexus/extensions/__init__.py
# COPY nexus/extensions/agents/yuanbao_one_agent/               /nix/runtime/nexus/nexus/extensions/agents/yuanbao_one_agent/
# COPY nexus/runtime/                                            /nix/runtime/nexus/nexus/runtime/
# COPY nexus/utils/                                              /nix/runtime/nexus/nexus/utils/
# COPY deployment/runtime/start.sh                               /nix/runtime/nexus/deployment/runtime/start.sh
COPY deployment/build_for_ags/ags_runtime_entrypoint.sh                  /usr/local/bin/ags_runtime_entrypoint.sh

COPY deployment/build_for_ags/skills/       /data/skills/


COPY deployment/build_for_ags/disable-file-locking.xcd /usr/lib/libreoffice/share/registry/disable-file-locking.xcd 

WORKDIR /data/workspace

EXPOSE 32000 49983

 CMD ["/usr/local/bin/ags_runtime_entrypoint.sh"]                                                                                            
# runtime 服务端口 32000
# 对外健康检查端口 8080
# 探针路径 /health
