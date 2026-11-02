# Windows Nexus bundle

将 Nexus 打包成符合 Bootstrap manifest 规范的 tar.gz payload。
**不构建、接收或打包 nexus-bootstrap.exe，不生成 bootstrap.ps1，也不启动服务。**
平台相关实现放在本目录，目前只支持 Windows x86_64；其他平台需各自的构建实现。

## 构建

在 Nexus 仓库根目录执行：

```powershell
make bundle
# 显式指定目标平台和架构（默认值）：
make bundle BUNDLE_OS=windows BUNDLE_ARCH=x86_64
# 或者，不依赖 make：
uv run --no-sync python deployment/bootstrap/windows/main.py
```

默认输出根目录为 `build/bundle/`，产物位于其中的 `windows-x86_64/`。自定义输出根目录：

```powershell
make bundle BUNDLE_OUTPUT="build/bundle-new"
uv run --no-sync python deployment/bootstrap/windows/main.py --output build/bundle-new
```

`BUNDLE_OS` 选择对应平台的构建脚本，`BUNDLE_ARCH` 传入脚本的 `--arch` 参数。
目前仅支持 `windows` / `x86_64`，且必须在 Windows x86_64 环境构建；不支持交叉编译。
不支持的目标会在构建前报错，不会生成错误平台的产物。

目标平台目录必须尚不存在，输出根目录及其他平台目录可以已存在。
失败后保留日志，重试请换新输出根目录；不覆盖旧包或业务运行状态。
不修改注册表、防火墙，不安装 Rust。

## 产物

```text
build/
  bundle/
    windows-x86_64.build.log  # 不放入发布目录或压缩包
    windows-x86_64/
      build.json
      manifest.json
      manifest-<version>.json
      bundles/<version>.tar.gz
      wheels/nexus_agent-<version>-py3-none-any.whl
```

`version` 来自项目版本；源码 SHA 仅记录在 `build.json`，不进入发布文件名。
只生成一份 wheel，存放于 `wheels/`，供本地验证；manifest 仅引用压缩包，附带 SHA-256 和 `extract: true`。
压缩包根目录包含 `entrypoint.py`、`python/` 和 `tools/uv.exe`，不包含 wheel、构建记录、日志或凭证。
构建时在输出目录内准备临时 payload，结束时清理，不保留独立的 python/tools 副本。
外部 Bootstrap 以 `build/bundle/windows-x86_64/manifest.json` 作为 source，需要支持 tar.gz 解压的版本。
自定义输出时，构建日志位于输出根目录的 `windows-x86_64.build.log`。

Manifest 包含 schema_version、version、os=windows、arch=x86_64、入口文件、参数及文件 SHA-256。
入口为 `python/python.exe`，参数为 `-I -B -u ${RELEASE_ROOT}/entrypoint.py`。
payload 接受 Bootstrap 追加的 `--state-root` 和 `--workspace`，启动 Nexus ASGI 服务。
监听 0.0.0.0，端口读取 `NEXUS_RUNTIME_PORT`，默认 18888；仅向可信机器开放访问。

## 打包流程与边界

1. 执行 `uv run --no-sync nexus modules install --all --runtime` 准备模块依赖。
2. 用 `uv build --wheel` 构建当前工作区 Nexus（包含未提交的代码）。
3. 复制当前 CPython 基础安装，排除原 site-packages 和字节码缓存。
4. 从核心依赖、runtime extra、各模块依赖递归筛选已安装包，复制所需 site-packages 文件；
   不复制整个开发 venv，拒绝 editable 第三方依赖。
5. 将 Nexus wheel 标准安装到 bundle Python，再执行 Nexus 模块依赖检查及导入验证。
   `--break-system-packages` 只用于私有 Python 副本，不修改宿主 Python。
6. 加入 payload 入口和 uv，压缩为 tar.gz，再生成 manifest 和构建记录。

build.json 记录 Python/依赖版本、wheel/manifest 摘要、源码 HEAD 和工作区脏状态。
不包含 Bootstrap 二进制的元数据。wheel 不带解释器和第三方依赖，运行时不再安装或 zipimport wheel。

当前复用构建机安装的 Python 和依赖版本，不是隔离、完全可复现的发行流水线，也不跨平台构建。
第三方包安装在 site-packages 外的命令行脚本不随包复制。浏览器、Claude Code、Node、Git 等外部程序
不在交付范围，完整 executor 和 worker venv 能力需另行验证。
Windows 长路径支持须由部署环境设置；Bootstrap 签名、状态管理、进程管理属于外部启动器职责。
状态目录应尽量短（例如 `C:/nexus-state`）；部分原生 DLL 加载仍受 Windows 路径长度限制，
开启长路径支持并不能保证任意深度的部署路径可用。

## 使用外部 Bootstrap 启动

准备支持本平台的 `nexus-bootstrap.exe` 后，在仓库根目录执行（以下假设启动器位于当前目录）：

```powershell
.\nexus-bootstrap.exe --source build/bundle/windows-x86_64/manifest.json `
  --state-root workspace/bootstrap-state --allow-unsigned
```

启动后访问 `http://127.0.0.1:18888/health` 检查服务。
`--allow-unsigned` 仅用于信任来源的本地测试；启动器和运行状态不属于打包产物。

发布到 Generic 仓库参见 [发布说明](../README.md)。
