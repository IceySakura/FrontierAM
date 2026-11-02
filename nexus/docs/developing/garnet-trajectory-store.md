# Garnet trajectory store 使用指南

在 HyTuner V2 中使用 Garnet，需要在每个 Ray 节点安装 Garnet 和支持 Garnet 的 Nexus，
然后设置 `trajectory_store.backend=garnet`。HyTuner 负责启动和停止本次任务的 Garnet 服务，
无需提前手动启动 server。

## 配置与传参

以下配置使用每节点 `64g` 的内存预算；请按节点实际可用内存调整。

```yaml
trajectory_store:
  backend: garnet
  namespace: null
  garnet:
    server_command: [/opt/garnet/2.2.4/net10.0/GarnetServer]
    memory: "64g"

trainer:
  distributed_kvstore_checkpoint: false
```

`trajectory_store.garnet` 只接受以下两个启动参数：

| 参数 | 类型 | 默认值 | 用法 |
| --- | --- | --- | --- |
| `server_command` | 非空字符串列表 | `[garnet-server]` | 指定可执行文件，推荐绝对路径。使用 DLL 发行包时可写为 `[/path/to/dotnet, /path/to/GarnetServer.dll]`。所有节点都必须能执行该命令。 |
| `memory` | 字符串 | `"1024g"` | **每个 server** 的 main-log 内存预算，不是整个集群的预算或进程 RSS 上限。格式为正整数字符串，可带 `k`、`m`、`g`、`t` 后缀，不区分大小写，例如 `"64g"`、`"512m"`。 |

在已有 HyTuner V2 启动命令后追加以下 Hydra 覆盖项，保留原有模型、数据和训练参数：

```bash
trajectory_store.backend=garnet \
  '++trajectory_store.garnet.server_command=[/opt/garnet/2.2.4/net10.0/GarnetServer]' \
  '++trajectory_store.garnet.memory=64g' \
  trainer.distributed_kvstore_checkpoint=False
```

`++` 用于添加或覆盖默认配置中尚未声明的 Garnet 字段。列表参数整体使用单引号，避免 shell
解释其中的特殊字符。

## 安装 Garnet

下面以已验证的 **Garnet 2.2.4、Linux x86_64、.NET 10** 为例。在所有 Ray 节点执行，
包括独立的 head 节点。下面的命令适用于使用 `yum` 的腾讯训练镜像，要求软件源已经提供
`dotnet-runtime-10.0`，没有该包时应先在镜像中配置对应软件源或安装 .NET 10 runtime。

```bash
set -euo pipefail

yum install -y curl tar xz libicu dotnet-runtime-10.0

GARNET_ROOT=/opt/garnet/2.2.4
GARNET_ARCHIVE=/tmp/garnet-2.2.4.tar.xz
mkdir -p "$GARNET_ROOT"

if [[ -n "${GARNET_SHARED_ARCHIVE:-}" ]]; then
    cp "$GARNET_SHARED_ARCHIVE" "$GARNET_ARCHIVE"
else
    curl -fL --retry 3 --connect-timeout 20 \
      https://github.com/microsoft/garnet/releases/download/v2.2.4/linux-x64-based.tar.xz \
      -o "$GARNET_ARCHIVE"
fi

printf '%s  %s\n' \
  c1c464a62378b47d5a0390449ec4733181bb033ec357bde8a6357a7a3b036b11 \
  "$GARNET_ARCHIVE" | sha256sum -c -
tar -xJf "$GARNET_ARCHIVE" -C "$GARNET_ROOT"

dotnet --list-runtimes
"$GARNET_ROOT/net10.0/GarnetServer" --version
```

不能访问 GitHub 的训练节点，可提前将同一安装包放到所有节点可读的共享挂载，
执行脚本前设置 `GARNET_SHARED_ARCHIVE` 为该文件的绝对路径。上面的校验和只对应指定的
2.2.4 安装包；更换版本时需同步调整下载地址、校验和、runtime 和 `server_command`。

安装完成后将 `server_command` 指向实际的 `GarnetServer` 文件。只有在各节点的 `PATH`
中已提供名为 `garnet-server` 的命令时，才可以直接使用默认值 `[garnet-server]`。

## 安装 Nexus

每个节点需要安装 Nexus 的 `garnet` extra，以及 C++17 编译器和训练进程 Python 的开发头文件。
Nexus 在首次创建 Garnet client 时自动编译 native 扩展；普通安装和其他 backend 不触发编译。

以下命令使用 `uv`，在 Nexus 仓库根目录执行。`PYTHON_BIN` 必须指向 Ray worker 实际使用的解释器；
请按镜像环境调整示例中的路径。Ray Python 依赖由训练环境提供，`garnet` extra 不负责安装 Ray。

```bash
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python3

yum install -y gcc-c++ make
uv pip install --python "$PYTHON_BIN" --extra garnet -e .

# 可选：在训练前完成首次编译，并检查当前解释器的运行环境。
"$PYTHON_BIN" -m nexus.gateway.trajectory_store_backends.garnet_native
```

首次初始化会同步等待编译，之后直接加载本机 `/tmp/nexus-garnet-native-<uid>/` 中的缓存。
同一用户的多个 Ray worker 通过文件锁共用一次编译；源码、Python ABI、架构或编译参数变化后会
使用新的缓存目录。 等待编译锁和实际编译各有 300 秒超时。 缓存不写入源码目录或 site-packages，
清理 `/tmp` 后会在下次使用时重建，停止 Garnet server 不删除编译缓存。

`garnet` extra 提供 redis-py、setuptools 和 pybind11；运行时不会自动安装依赖或下载文件。
Conda 环境应使用自身 Python 头文件，不要用其他版本的系统开发包代替。
编译失败会直接终止 Garnet client 初始化，异常包含编译日志路径；其他 backend 仍可使用。

也可打包 wheel 后在各节点安装；wheel 携带 C/C++ 源码，安装时不编译，不需要预先匹配 native ABI：

```bash
uv build --wheel --python "$PYTHON_BIN"
uv pip install --python "$PYTHON_BIN" '/path/to/nexus_agent-<version>-py3-none-any.whl[garnet]'
```

首次使用 Garnet 的节点仍需有编译器和匹配的 Python 头文件，目前只支持 Linux。
如果 Ray 使用独立 runtime environment，应在相同环境中预热；`PYTHONPATH` 或 Ray `working_dir`
指向的 Nexus 包必须包含 `native/build.py`、`garnet_client.cpp` 和 vendored hiredis 源码。

## 在太极任务中使用

将 Garnet 和 Nexus 安装命令放入 **每个节点都会执行的初始化命令**，或预先装入镜像。
安装应在 Ray worker 启动前完成。所有节点使用相同的 Garnet 路径、Nexus 版本和 Python 环境。

启用太极的 Ray 集群后，在已有训练 `start_cmd` 中加入前面的 Hydra 覆盖项。
若平台已通过 Ray 提交 `start_cmd`，直接执行训练命令即可。

运行时需满足：

- Garnet 会在启动时的每个存活 Ray 节点上启动一个实例，包括 head；按节点 IP 去重。
  `memory=64g` 表示每个实例使用该预算，节点还需为训练进程和其他服务留出内存。
- 各节点之间必须能够访问自动分配的 TCP 端口，且 `/tmp` 可写。
- 当前用法面向任务内部的临时存储，不提供持久化、磁盘卸载或已有部署复用配置。
  运行期间应保持节点集合稳定，丢失的内存数据不会自动恢复。
- 正常退出由 HyTuner 停止 Garnet；任务异常终止后的进程清理由任务环境负责。

## 常见配置与安装问题

| 现象 | 检查方法 |
| --- | --- |
| 找不到 `garnet-server` 或可执行文件 | 检查每个节点的安装路径；将 `server_command` 改成绝对路径，并在相同运行环境执行 `--version`。 |
| 提示缺少 .NET runtime | 执行 `dotnet --list-runtimes`，确认已安装所选 Garnet 包要求的 runtime。 |
| `Garnet native compilation failed` | 查看异常中的编译日志；检查 C++17 编译器、Python 开发头文件及 `garnet` extra，并在相同环境运行预热命令。 |
| `Missing Garnet build source` | 检查 Ray working directory 中的 Nexus 是否完整，重新安装或分发包含 native 源码的 Nexus。 |
| `Unsupported garnet configuration keys` | `garnet` 下只保留 `server_command` 和 `memory`；不要把自动生成的 `servers` 当作启动参数。 |
| 提示 checkpoint 只支持 Ray backend | 设置 `trainer.distributed_kvstore_checkpoint=False`。 |
| Garnet 启动失败或超时 | 查看异常中给出的 `/tmp/nexus-garnet-*/server.log`，检查可执行文件、runtime、内存预算和节点间网络。 |
