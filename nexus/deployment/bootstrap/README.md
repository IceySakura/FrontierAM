# Bootstrap bundle 构建与发布

平台构建说明：[Windows x86_64](windows/README.md)、[Linux x86_64](linux/README.md)。

```powershell
make bundle
make bundle-publish
# 不依赖 make 的发布命令：
uv run --no-sync python deployment/bootstrap/publish.py --bundle build/bundle/windows-x86_64
```

构建与发布分开：发布命令不会重新构建，也不操作 Bootstrap 进程。
已存在的目标平台目录不会自动覆盖，重新构建请先备份该目录或设置 `BUNDLE_OUTPUT`。
`make bundle-publish BUNDLE_OS=windows BUNDLE_ARCH=x86_64` 选择本地产物目录；
发布脚本从 manifest 的 `os`、`arch` 推导远端前缀，不依赖发布机器的平台。

## 认证

采用 Mirrors Generic API 的 HTTP Basic Auth。优先读取环境变量 `MIRRORS_USERNAME`、`MIRRORS_TOKEN`，
否则读取用户目录的 `.mirrors_token`，格式如下：

```text
MIRRORS_USERNAME=your_username
MIRRORS_TOKEN=your_token
```

可通过 `--token-file` 指定文件。凭证文件放在仓库外，不提交、不打包、不输出到日志。
无需安装个人 skill 即可运行发布脚本；接口及认证格式与 `mirrors-file-manager` 一致。

## 发布布局

默认仓库为 `nexus-gui-bootstrap-bundle`，可通过 `--repo` 或 Makefile 的 `BUNDLE_REPO` 指定。

```text
nexus-gui-bootstrap-bundle/
  <sys>-<arch>/
    manifest.json                 # latest
    manifest-<version>.json        # immutable version
    bundles/<version>.tar.gz
```

Bootstrap source（支持环境变量的新版 bootstrap）：

```text
${NEXUS_RUNTIME_MANIFEST_URL}
```

发布流程：

1. 校验本地 manifest 路径、版本和压缩包 SHA-256。
2. 查询同名远端包：相同内容复用，不同内容拒绝覆盖。
3. 上传缺失的压缩包，通过 GET 下载流计算 SHA-256 校验；失败不更新 manifest。
4. 生成发布版 manifest，补充压缩包在 Generic 仓库的固定 `url`，先上传并校验不可变的 `manifest-<version>.json`，再更新 `manifest.json`。
   本地保留 `published-manifest.json` 和发布结果 `publish.json`；本地 `manifest.json` 不变，仍可离线启动。

Mirrors 会把 manifest 重定向到按 hash 存储的临时 URL，因此发布版必须包含文件的绝对 `url`，
避免 Bootstrap 以重定向后的存储地址解析 `bundles/...`。`path` 相对于平台目录中的 manifest，
`url` 使用 Generic 固定入口，不含凭证或临时签名，也不写入重定向后的存储地址。

`sys` 对应 manifest 的 `os`（如 `windows`、`linux`、`macos`），`arch` 使用 `x86_64`、`aarch64` 等规范值。
每个平台维护独立的最新 manifest，版本相同的跨平台包不会相互覆盖。
布局及发布脚本可隔离不同平台；目前支持 Windows x86_64 和 Linux x86_64。

只上传 manifest 和它引用的压缩包；wheel、构建记录及日志不会发布。历史包永久保留，不自动删除。
同一平台请串行发布，避免多个发布者竞争更新 latest manifest。发布成功代表 manifest 指向本次版本，
不按版本号推断先后。同一版本只允许相同内容；代码修改后请提升版本号再构建，不覆盖旧版本。

Mirrors 下载可能重定向到带签名的内部 HTTP 存储链接：只允许 GET 跟随这种重定向，
不向 HTTP 或其他站点转发 Basic Auth，并校验下载内容 SHA-256。PUT 不跟随重定向。
这不等同于全链路 TLS 或签名 manifest 校验，部署仍应使用可信网络和合适的签名策略。

构建目前复用本机 Python 和已安装依赖，不保证不同机器构建同一 commit 得到相同压缩包。
manifest 未签名；部署时需按信任策略配置 Bootstrap。若仓库禁止匿名下载，启动器还需要相应读取能力，
不要将上传 token 放进 manifest、URL 或启动命令。

固定版本可在 Bootstrap 的 `--source` 中直接指定完整的 `manifest-<version>.json` 下载地址。
AGRT provider 不选择运行包版本或自动注入 manifest 地址。
手动启动时，PowerShell 请使用单引号包裹 source，保留 `${...}` 给 bootstrap 展开。
新版 `setup-bootstrap.ps1 -Source <source> -StateRoot <path>` 会在登录时读取实例环境，
按 manifest 文件名划分缓存；需要把新版 bootstrap 和注册脚本安装到沙盒模板。
Windows 未开启长路径支持时，建议指定短缓存根目录（如 `-StateRoot C:\nbs`），避免 worker venv 复制依赖时触及路径长度限制。
已有镜像中的旧 bootstrap 不支持变量展开，须升级后重新注册启动任务。
