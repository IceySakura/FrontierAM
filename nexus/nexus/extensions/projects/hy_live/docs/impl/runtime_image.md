# Runtime 镜像 tag

本文件是 runtime 镜像 tag 的 **唯一来源** 。 其他文档只链接到这里，不要再复制 tag 字符串。

## 1. 当前 tag

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.21-hylive-992fb888-wb8da3c28fa1-py312-slim
```

- 大小：解压后约 2.8 GB（上一版约 9.9 GB）。`/nix` 只装 hy_live 用到的东西：Nexus 的 capture bash、Python 3.12、Node.js 22、基础命令行工具和 Noto 字体，没有 Chromium、Docker-in-Docker、GUI/VNC 和 code-server。需要这些能力的任务用通用 runtime 镜像。
- nexus：`992fb888`（1.8.21），只有 `.venv-3.12`。沙箱按 Host 发来的 `PYTHON_VERSION` 选 venv，所以 hy_live 配置固定 `python_version: "3.12"`，Host 也必须是 Python 3.12，否则序列化可能不兼容。
- WorkBuddy：`5.11.0 (d41e61c1)`，包 sha256 前缀 `8da3c28fa111`，不需要 Host 再上传安装包。
- 文档引擎：包自带的 `editor_sdk` 0.0.0.556，有 `/localapi/sheet/check_formula`，Excel 的 `recalc.py` 直接走引擎。沙箱（gVisor）拒绝执行 patchelf 改过的 ELF，所以 `editor_sdk` 是一个脚本，用 nix 的 glibc loader 启动原始二进制 `editor_sdk.bin`。
- `slidep` 6.6.2：wb 包自带（`vendor/slidep`），内置 `workbuddy-pptx` 只用它建和改演示文稿。`slidep screenshot` 依赖的 libX11 没有装，这个命令会直接报错退出，skill 不用它。
- 工具链 `toolchain/bin`：`python3`（nix 3.12）、`python`/`pip`/`pip3`、`node`/`npm`/`npx`、`curl`。`python` 和 `pip` 跟随 PATH 上第一个 `python3`。没有预装 Python 包，也没有市场 skill、LibreOffice 和浏览器；`pip install` 在只读的 `/nix` 上会自动装到用户目录。
- 已用真实模型跑通 Word、Excel、PPT，以及联网调研后同时产出 HTML、Word 和 PPT 的任务，`workbuddy.features` 是 `{"tencentDocsEngine": true}`。

构建文件不在 nexus 仓库里，放在 `/data/work/hylive-runtime-image/workbuddy_runtime/`，构建步骤见那里的 README；镜像里的 nexus 取 `NEXUS_REPO` 指定的 checkout 的 `HEAD`。

上一版是 `…-wb7a882f1f61ca-esdkw-slidep`，见 §5。

## 2. 使用方式

Gongfeng 两路镜像 **职责不同，不要填成同一个 tag** ：

- `runtime_provider_config.runtime_image`（example CLI 的 `--runtime-image`，环境变量 `NEXUS_RUNTIME_IMAGE`）：沙箱 `/nix` 里跑 invoke 的 nexus，含 CodeBuddy `session_runtime` 和 `host_relay`，填本文件的 tag。
- `runtime_configs.main.image`（example CLI 的 `--workspace-image`，环境变量 `LIVE_AGENT_RUNTIME_IMAGE`）：任务 / 工作区容器，需要任务工具链，填业务镜像。

混用同一个镜像会让两条预热链路互相耦合：任何一侧换 tag 都会让另一侧的 warm pool 失效，且工作区会缺任务工具链。

## 3. 什么时候需要重建

沙箱里的 nexus 是镜像自带的 `/nix/runtime/nexus`， **没有 host → 沙箱的源码同步** ：runtime 侧文件不进镜像就不会生效。 镜像只提供 `/nix` 一个 volume（`Dockerfile.runtime` 末行 `VOLUME /nix`），不重建业务容器镜像。

- hy_live 项目没有沙箱侧文件；只改 hy_live 不需要重建。
- runtime 侧公共文件（`nexus/extensions/agents/codebuddy/`、`nexus/extensions/agents/installed_common/` 等）改动后要重建镜像，或临时用 `worker_venv: true` 把当前 wheel 装进沙箱。
- `backend_adapter='workbuddy'`（`wb --acp`，ACP v2）需要对应 runtime 支持。 当前 tag 已包含；使用缺少修复的旧镜像时才需要 `worker_venv: true`（example CLI 的 `--worker-venv`），或更新镜像。

## 4. WorkBuddy 预装

`/nix/runtime/workbuddy` 是 **预装包查找目录** ，不是运行时安装目录。 镜像构建时将包预置于此，`ensure_workbuddy_cli()` 按 `/nix/runtime/workbuddy/<version>` → `/nix/runtime/workbuddy` 顺序查找，并校验 `package.json` 中的版本。 支持以下布局：

- 解压包：`/nix/runtime/workbuddy/package/bin/wb.js`。
- 去掉解压包顶层 `package/`：`/nix/runtime/workbuddy/bin/wb.js`。
- npm 安装：`/nix/runtime/workbuddy/node_modules/.bin/wb`。
- 多版本预装：上述布局放在 `/nix/runtime/workbuddy/<version>/` 下。

也可以用 `work_agent_configs.workbuddy.workbuddy.cli_path` 指定其他位置的 CLI 入口，不需要修改 runtime 代码。 镜像内 Node.js 必须满足 `^22.16.0 || >=24`。

使用预装版本时，不填写 `workbuddy.package_path`，example CLI 不传 `--workbuddy-package`。 指定 Host 安装包时仍走上传安装流程。 后续只修改 hy-live 的请求清理或追问上下文，不需要重新构建 runtime 镜像。

Host 包上传的分片位于 `/tmp/nexus/workbuddy-acp`，运行时解压安装到 `/tmp/nexus/agents/workbuddy-acp/<version>[-<sha256 前缀>]/package`。 这条安装路径不会写入 `/nix/runtime/workbuddy`。

## 5. 旧 WorkBuddy 测试镜像

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.21-hylive-abf39949-wb7a882f1f61ca-esdkw-slidep
```

§1 的上一版：nexus `abf3994917`，Python 3.10–3.14 五套 venv，WorkBuddy `5.11.0 (250cc48e)`（包 `7a882f1f61ca`）。文档引擎 0.0.0.533 没有 `/localapi/sheet/check_formula`，Excel 的 `recalc.py` 报 Unknown endpoint 后 agent 改用引擎的公式计算接口；`slidep` 6.6.4 另装在工具链里。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb7a882f1f61ca-py312ed
```

§1 的上一版，预装同一个 WorkBuddy 包和市场 skill（[`workbuddy_skills.md`](workbuddy_skills.md)）。里面的 nexus 还是 `8e0a1eb3`，没有追问会话（side question）；`editor_sdk` 是 patchelf 过的二进制，在沙箱（gVisor）里起不来，office skill 用不上文档引擎。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb64f50dab06d1-skc1e4fd35-tr8a216e8e
```

`py312ed` 的上一版。预装 WorkBuddy `5.11.0 (14d3710f)`（包 sha256 前缀 `64f50dab06d1`），文档引擎 `editor_sdk` 0.0.0.533。可以直接使用默认的 `workbuddy.features`。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb450fa8757580-skc1e4fd35-tr8a216e8e
```

与 `wb64f50dab06d1-skc1e4fd35-tr8a216e8e` 相同，但预装的是 `5.11.0 (83e298b0)`（包 sha256 前缀 `450fa8757580`），这个包没有 `ecosystem` 开关。默认的 `workbuddy.features` 带 `ecosystem`，用它时要显式设为 `{"tencentDocsEngine": true}`，否则 wb 报未知功能名，会话一启动就退出。不要再使用这个 tag。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb2988c6d7e6b3-skc1e4fd35-tr8a216e8e
```

`wb64f50dab06d1-skc1e4fd35-tr8a216e8e` 的基础镜像，预装带 `ecosystem` 开关的旧包 `2988c6d7e6b3` 和 0.0.0.504 的文档引擎，可以直接使用默认的 `workbuddy.features`。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb2988c6d7e6b3-skc1e4fd35
```

`tr8a216e8e` 的基础镜像。 沙箱会把单个超过约 1600 字符的正文块和工具字符串截短；配合当前 Host 时，长英文工具输出实时推送写出的 Event 会比重放的短。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb2988c6d7e6b3-sk2ac33dc2
```

与 `skc1e4fd35` 相同，但多预装了 WorkBuddy 已自带的市场版 `pptx`、`xlsx`、`docx`、`agent-browser` 和重复的 `document-skills-pdf`。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb2988c6d7e6b3-sk98010cee
```

已预装 11 个市场 skill 和工具链，但没有腾讯文档引擎和 `slidep`。内置的 `workbuddy-pptx` 在它上面会卡在 `slidep` 连不上文档引擎。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb2988c6d7e6b3
```

这是 `skc1e4fd35` 之前的 tag。按 tag 名它应该预装带 `ecosystem` 开关的包，但实际里面还是 `774c552905d4` 的旧包，传 `--features '{"ecosystem":true}'` 会因未知功能名退出。原因是它用 `RUN` 在基础镜像的 `VOLUME /nix` 下替换安装包，而 legacy builder 会丢弃这类修改。不要再使用这个 tag。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-8e0a1eb3-wc0f44a6-wb774c552905d4
```

该镜像的 nexus 与 `py312ed` 相同（`8e0a1eb3`），也是 §1 的构建基础镜像；预装的是不含 `ecosystem` 开关的 WorkBuddy 5.11.0（包 sha256 前缀 `774c552905d4`）。 传 `--features '{"ecosystem":true}'` 会因未知功能名退出。

```text
mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:1.8.19-hylive-b86e6cc7-wc0bb305-wb774c552905d4
```

该旧镜像已预装 WorkBuddy 5.11.0，但缺少构建后修复的 runtime 搜索凭据传递，仍需 worker venv 才能使用完整修复。 此前已用它加 worker venv 跑完联网调研。 新任务使用 §1 的当前 tag，不再需要 venv。

切换 hy-live 默认后端只是 Host 层配置变更，本身不要求重新构建镜像。
