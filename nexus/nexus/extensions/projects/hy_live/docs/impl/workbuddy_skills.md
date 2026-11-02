# WorkBuddy 预装 Skill 与工具链

带市场 skill 的 runtime 镜像（tag 里带 `-sk`，见 [`runtime_image.md`](runtime_image.md) §5）在 `/nix/runtime/workbuddy` 下预装了一组 WorkBuddy 市场 skill，以及它们需要的 Python、Node、LibreOffice 和浏览器工具链。WorkBuddy 会话启动时自动接上，不需要任何配置。镜像里没有这些目录时，行为与旧镜像完全一致。

当前默认镜像（§1）不带这些 skill 和工具链，只有文档引擎、`slidep` 和最小的 Python / Node 工具链，构建文件见 [`runtime_image.md`](runtime_image.md) §1。

## 1. 预装的 Skill

skill 来自 WorkBuddy 官方市场 `codebuddy-plugins-official`，与 WorkBuddy Desktop「来自市场」里看到的一致。只预装 WorkBuddy 没有自带同类 skill 的那些：

| Skill | 来源插件 | 运行依赖 |
|---|---|---|
| `pdf` | pdf | pypdf、pdfplumber、pypdfium2、reportlab、qpdf、poppler |
| `playwright-cli` | playwright-cli | `playwright-cli` + Chromium |
| `skills-security-check` | skills-security-check | 无 |
| `find-skills` | find-skills | `npx skills`，需要外网 |
| `tmap-lbs-skills` | 腾讯地图·地图助手 | Node；需要 Key，见第 4 节 |
| `tmap-jsapi-gl` | 腾讯地图·地图助手 | 无（JSAPI GL 开发指南） |

市场里的 `pptx`、`xlsx`、`docx`、`agent-browser` 不预装：WorkBuddy 在 ecosystem 下自带 `workbuddy-pptx`、`workbuddy-sheetagent` / `workbuddy-sheet-generation`、`workbuddy-docx`，以及同一个 `agent-browser` 插件，见第 1.1 节。插件里的斜杠命令（如 `/tmap-lbs-plugin:set-key`）是给人用的，也不预装。

### 1.1 内置 office skill（与 Desktop 一致）

WorkBuddy 包本身在 ecosystem 下自带 `workbuddy-office-routing`、`workbuddy-pptx`、`workbuddy-sheetagent`（`workbuddy-sheet-generation`）、`workbuddy-docx`、`workbuddy-local-office-edit` 等 office skill。office 任务先经过 `workbuddy-office-routing`，再交给这些内置 skill，和 Desktop 的行为一样。它们都已在镜像上用真实模型跑通。

内置 `workbuddy-pptx` 通过 `slidep` CLI 编辑演示文稿，而 `slidep` 要连接 wb 的本地腾讯文档引擎（`editor_sdk`，`tencentDocsEngine` 功能）。所以 hy_live 默认开启 `tencentDocsEngine`（见 [`config_reference.md`](config_reference.md)），镜像预装了这两样，见第 2 节。

## 2. 工具链

位置：`/nix/runtime/workbuddy/toolchain`。

- Python 3.12（nix 构建）：openpyxl、pandas、numpy、matplotlib、pypdf、pdfplumber、pypdfium2、pdf2image、reportlab、Pillow、lxml、defusedxml、markitdown
- Node 22（runtime 自带）：pptxgenjs、docx、react、react-dom、react-icons、sharp，`NODE_PATH` 指向 `toolchain/node/node_modules`
- LibreOffice 24.8（`soffice`）、pandoc、poppler（`pdftoppm` / `pdftotext` / `pdfimages`）、qpdf
- Chromium 142，以及 `agent-browser`、`playwright-cli` 两个包装脚本。包装脚本默认指向这个 Chromium，并带上 root 容器需要的参数（`--no-sandbox`、`--remote-allow-origins=*` 等）。调用方设置的 `AGENT_BROWSER_*`、`PLAYWRIGHT_MCP_*` 变量优先
- 腾讯文档引擎 `editor_sdk`：放在 wb 包的 `tencent-docs-ai-engine/bin/linux-x64/`，指向 `toolchain/sys/bin/editor_sdk`。上游二进制依赖 glibc：0.0.0.504 在构建时用 patchelf 改成链接 nix 的 glibc 和 libglvnd；patchelf 处理 0.0.0.533 会段错误，当前镜像改用包装脚本，经 nix 的 `ld-linux` 加 `--library-path` 启动未改动的二进制。两种方式在 Alpine 工作区都能运行
- `slidep`（`@tencent/slidep` 6.6.4）：包装脚本带上 `toolchain/slidep-lib`（libuuid、libX11、libXext）。它的原生模块 canvas、gl、skia-canvas 在构建时下载了 glibc 预编译包；`@swc/core` 在 Alpine 上会因为 `/usr/bin/ldd` 误判为 musl，镜像把 glibc 版本放在了它查找的 `swc.linux-x64-musl.node` 位置
- 中文字体：Noto CJK，`FONTCONFIG_FILE` 指向 `toolchain/etc/fonts/fonts.conf`

## 3. 加载机制

`nexus/extensions/agents/codebuddy/workbuddy_acp.py` 在每个 WorkBuddy 会话启动时做三件事：

1. **链接 skill**：`prepare_workbuddy_config_dir` 把 `/nix/runtime/workbuddy/skills/<name>` 逐个软链到会话的 `<WORKBUDDY_CONFIG_DIR>/skills/<name>`。会话里已有同名 skill 时保留会话自己的。wb 不读 `CODEBUDDY_SESSION_SKILL_DIRS`，只有放在这个目录里的 skill 才能进入模型的 Skill 列表。
2. **工具链加到 PATH 末尾**：`build_workbuddy_acp_env` 把 `toolchain/bin` 追加到 PATH 末尾，并设置 `NODE_PATH` 和 `FONTCONFIG_FILE`。工作区镜像自带的同名工具（比如它自己的 `python3`）仍然优先。
3. **写会话 `HOME` 的 `.profile` / `.zshrc`**：wb 的 Bash 工具不直接用继承来的 PATH，而是用 login shell 快照：Alpine 的 `/etc/profile` 会把 PATH 重置，`NODE_PATH` 也会被丢掉。所以第 2 步还会在会话 `HOME` 的 `.profile` 和 `.zshrc` 里追加一段带 `# nexus: WorkBuddy image toolchain` 标记的配置，以同样的追加方式设置 PATH 和 `NODE_PATH`。会话 `HOME` 是每个会话独立的目录，不影响工作区镜像。

镜像里还修了一处 WorkBuddy 自身的兼容问题：wb 安全删除的 shim（`cli/vendor/shim/safe-bin/rm` 等）写的是 `#!/bin/bash`，在没有 bash 的 Alpine 工作区（如 `claude-code-hle`）里执行不了，模型的 `rm` 和 LibreOffice 启动脚本都会失败。镜像构建时把 shebang 改成了 `/nix/runtime/nix-env/bin/bash`。

## 4. 注入腾讯地图 Key

`tmap-lbs-skills` 从环境变量 `TMAP_WEBSERVICE_KEY`（或 `TMAP_KEY`）读取腾讯位置服务的 WebService Key，在[腾讯位置服务控制台](https://lbs.qq.com/dev/console/key/add)申请。这个变量一路传到 wb 的 Bash 和 Node 里，已在镜像里验证。

**不要** 让 skill 把 Key 写进它目录下的 `config.json`：skill 目录是软链到 `/nix` 的共享副本，写进去会影响同一沙箱的所有会话。

### 4.1 YAML 配置

在后脑配置的 `extra_env` 里用 `__env` 引用 Host 环境变量。runner 运行时从 Host 读取值，数据和产物里只留变量名：

```yaml
task_config:
  work_agent_configs:
    workbuddy:
      extra_env:
        TMAP_WEBSERVICE_KEY__env: TMAP_WEBSERVICE_KEY
```

运行前在 Host 上 `export TMAP_WEBSERVICE_KEY=<key>`。直接写 `TMAP_WEBSERVICE_KEY: <key>` 也能用，但 Key 会进入数据。

### 4.2 example CLI

`--pass-env NAME` 把 Host 上的环境变量 `NAME` 交给当前后脑，效果等同于上面的 `NAME__env: NAME`，并把它的值从 CLI 输出和产物里脱敏。可以重复传入：

```bash
export TMAP_WEBSERVICE_KEY=<key>
uv run python -m nexus.extensions.projects.hy_live.example \
  --case single_snake --output <输出目录>/single_snake/result.json \
  --pass-env TMAP_WEBSERVICE_KEY
```

其他 skill 需要的 Key 也用同样的方式注入。

## 5. 已知限制

- 没有 OCR：tesseract 未预装，`pdf` skill 处理扫描件的部分不可用。
- `find-skills` 需要外网才能搜索和安装新 skill。
- skill 版本在构建镜像时固定，市场更新后需要重建镜像。
- 没有 `FONTCONFIG_FILE` 时，nix 的 Chromium 会卡在 CDP 建立页面这一步（`agent-browser` 报 `CDP command timed out`），和代理变量无关。WorkBuddy 会话总会设置它，包装脚本也会默认设置。
- 带 `rm -rf`、`sudo` 等高危内容的命令需要权限审批。hy_live 会把审批转给用户；没有审批人的单轮 headless 调用里它们会被拒绝，这一轮随之结束。
- 只构建了 x86_64：`agent-browser` 用的是静态链接的 linux-musl-x64 二进制。
- 工作区镜像自带 `python3`、但没有装 openpyxl 等库时，skill 会用到工作区的 `python3` 而报导入失败。`claude-code-hle` 没有自带 Python，不受影响。
- 开启 `tencentDocsEngine` 后，每个会话冷启动多约 12 秒。
- 镜像约 10.4 GB，比不带工具链的 6.7 GB 多出的部分主要是 LibreOffice 和 `editor_sdk`。

## 6. 重建

构建文件不在 nexus 仓库里，暂放在 `/data/work/hylive-runtime-image/workbuddy_skills/`，构建步骤和 Dockerfile 的约束见那里的 README。skill 不进 git（部分带厂商许可），由脚本从 WorkBuddy 市场目录复制。
