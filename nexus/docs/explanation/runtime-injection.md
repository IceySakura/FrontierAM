# Runtime Injection

在 Code Agent 场景，LLM 输出的 Action 通常需要在某个具体的环境发生作用，比如文件的增删改查，或是执行某段 shell 指令。 我们需要一个程序在对应的环境中执行这些动作。

这里有两种实现方案，第一种是在目标环境安装这个程序，启动一个服务来接收、执行并返回 Action 的结果；第二种是直接基于 Docker 提供的接口（exec、cp 等）完成一些基础操作。 第一种需要在各种环境中安装服务需要的依赖，但是上限很高，可以对 Action 的行为做细致的控制。 第二种不需要向镜像注入依赖，但是只能进行一些基础的操作。 目前几乎所有的主流 Code Agent 都采用了第一种方案，比如 SWE-Agent 和 OpenHands。

## 大规模 Scale 的难点

我们对第一类方案有很强的需求，但是它在大规模 scale 时有一些阻碍。 以 OpenHands 和 SWE-Gym 为例，要在一个新的 docker 环境中运行 OpenHands，这个环境需要能够启动 OpenHands 的服务，如文档所说，OpenHands 的服务要求目标环境基于 Debian 系统，并且需要构建一个新的镜像：

<figure markdown="span">
  ![OpenHands 镜像要求](/assets/images/runtime-injection/image_1.png)
  <figcaption>OpenHands 镜像要求</figcaption>
</figure>

SWE-Gym 目前较多被用于 Software Agent 相关的强化学习，整个数据集包含了 2438 个实例，每个实例对应一个镜像，总共约 6TB 大小：

<figure markdown="span">
  ![SWE-Gym 数据集](/assets/images/runtime-injection/image_2.png){ width="500" }
  <figcaption>SWE-Gym 数据集</figcaption>
</figure>

如果我们需要基于 OpenHands 在 SWE-Gym 场景做 RL，就要给这 2.4k 个镜像全部安装一遍依赖，即使忽略这带来的时间开销和体积膨胀，不同镜像需要解决的依赖冲突问题、这两千多个新镜像的维护以及服务的版本升级，都需要极其大量的人力来支持。 从认知的角度这个实现是持续膨胀且难以为继的，很容易演化成屎山。

SWE-Agent 团队开发了 [SWE-ReX](https://github.com/SWE-agent/SWE-ReX) 作为其运行时框架，它同样采用了在容器内启动服务的方案。 SWE-ReX 提供了两种注入方式：

1. **动态安装**：在容器启动时通过 pipx 动态安装 swe-rex 包
2. **构建新镜像**：通过 Dockerfile 构建一个包含独立 Python 环境的新镜像

这两种方式都存在明显的局限性：

**动态安装的问题**：每次启动容器时都需要通过网络安装依赖，这不仅增加了启动时间，还要求目标环境具备网络访问能力和 Python/pip 环境。 在离线环境或网络受限的场景下，这种方式完全无法工作。

**构建新镜像的问题**：虽然避免了动态安装，但本质上与 OpenHands 的方案类似——需要基于原始镜像构建新的运行时镜像。 从 SWE-ReX 的实现可以看到，它需要一个基于 Debian 的构建阶段来编译 Python 3.11，并通过 apt-get 安装依赖。 这意味着：

- 仍然需要维护【原始镜像 → 运行时镜像】的映射关系
- 目标镜像必须基于 glibc（如 Debian/Ubuntu），无法支持 musl-based 镜像（如 Alpine）
- 每个新的基础镜像都需要重新构建，在 SWE-Gym 这样的大规模场景下依然面临维护成本高的问题

## 我们的方案

这里我们验证了一个方案，那就是做一个完全自包含（self-contained）的服务，让它不需要任何外部依赖就能运行，这样我们只要把这个服务放在一个公共的 volume 中，任何一个新的 docker 镜像，只要挂载这个 volume，就能启动这个服务，实现上面提到的能力。 这样我们就不再需要有任何【外部镜像->运行时镜像】的管理机制，节省了非常大一块认知成本。

这里最主要的难点就是如何实现完全自包含的服务。 对于编译语言可以尝试构建不需要任何外部依赖的 static binary，但是这样开发的成本会飙升，我们还是希望能基于 python 快速迭代。

下面介绍我们已经验证成功的能够构建自包含 python 服务的方案。

### 通过 Nix 构建基础依赖

关于 Nix 的细节可以学习 [NixOS & Flakes Book](https://nixos-and-flakes.thiscute.world/)，简而言之，我们可以通过 Nix 构建一个环境，并且将 /nix/store 中这个环境的闭包整体拷贝到新环境就能确保它可以正常运行。 因此，我们使用 Nix 来安装所有操作系统级别的依赖，这包括了运行 Python 服务必须的 uv、python，以及执行各类 action 需要的工具，比如 grep、git、sshd、tmux、chromium 等。

### 构建 Python 环境

uv 可以基于 nix 提供的 python 创建 venv，通过环境变量，我们可以控制 uv 将所有依赖存储在指定文件夹，随基础依赖一起分发。 这里我们把所有上层服务和 pip 依赖包都存储在 /nix/runtime 文件夹。

### 借助 Docker 挂载依赖

在上面两步，我们已经成功构建了一个 /nix 文件夹，把它放到任何一个镜像里，只要是 x86 linux，就可以正常运行我们的服务了。 为了让分发和使用的方式更优雅一点，我们构建了一个用于分发 /nix 目录的数据镜像，其中 /nix 被设置为 docker volume。 使用这个镜像启动容器之后，其它容器可以通过只读方式挂载这个数据容器的 /nix volume，这样就完成了拼图的最后一块。

## 方案对比

| 维度 | OpenHands | SWE-ReX | Nexus |
|------|-----------|---------|-------|
| 启动方式 | 构建新镜像 | 动态安装或构建新镜像 | 挂载预构建的 volume |
| 启动时间 | 首次需要构建镜像 | 首次启动需要安装/构建（10-20秒+） | 即时启动（毫秒级） |
| 网络依赖 | 构建时需要网络 | 动态安装需要网络 | 无网络依赖 |
| 镜像兼容性 | 仅支持 Debian 系镜像 | 仅支持 glibc-based 镜像 | 支持任意 x86 Linux 镜像 |
| 维护成本 | 每个基础镜像需要单独构建 | 每个基础镜像需要单独构建 | 单一 volume 适配所有镜像 |
| 离线部署 | 需要预先构建镜像 | 需要预先构建镜像 | 完全支持 |

通过 Nix 构建完全自包含的运行时环境，Nexus 实现了真正的「一次构建，处处运行」。 无论目标镜像是 Ubuntu、CentOS、Alpine 还是任何自定义镜像，只需挂载同一个 volume 即可立即使用所有运行时能力。

### 运行时 Checkpoint 加速

由于 Nexus 的运行时完全基于 Nix 构建，我们可以利用容器运行时 checkpoint 技术进一步优化启动性能。Checkpoint 能够将容器内进程的状态完整保存到磁盘，包括内存、文件描述符、文件系统等，之后可以从这个快照直接恢复进程到 checkpoint 时的状态，节省每次都重新启动 Nexus 服务的开销。

容器 checkpoint 时需要保存完整的容器内进程状态以及进程修改的 rootfs 内容，一般的做法是为每个使用不同基础镜像的容器都做一次 checkpoint，确保完全一致性，但是 Nix 的自包含设计可以实现 checkpoint 文件的 「一次备份，处处恢复」，理由如下：

- **路径确定性**：Nix 的所有依赖都存储在 /nix/store 下，路径基于内容哈希生成（如 `/nix/store/abc123-python-3.11`），完全确定且不可变
- **内容不可变**：Nix store 中的文件一旦构建就不会被修改，保证了 checkpoint 和 restore 时文件内容一致
- **跨容器一致性**：当不同容器挂载同一个 /nix volume 时，所有库文件、Python 解释器、依赖包都在完全相同的路径
- **无外部依赖**：runtime server 不依赖宿主镜像的任何文件，所有依赖都来自 /nix

这正是 Nexus 基于 Nix 的方案能够完美支持 checkpoint 和 restore 的原因。

因此，我们可以：

1. **预热 Runtime Server**：启动 runtime server，等待其完成初始化（加载 Python 解释器、导入依赖包、建立内部状态等）
2. **创建 Checkpoint**：使用容器运行时 checkpoint 技术对已初始化 runtime server 进程做 checkpoint，保存完整的进程状态
3. **快速恢复**：在任意容器中，只要挂载了相同的 /nix volume，就可以通过容器 restore 技术恢复 runtime server，跳过整个初始化过程

相比之下，传统方案（通过系统包管理器安装依赖）无法实现跨镜像的 checkpoint 复用——不同镜像的库路径、版本可能不同，导致 restore 失败。

这种方式可以节省 **几秒到十几秒** 的服务启动时间（取决于硬盘性能），因为：

- Python 解释器启动和字节码编译被完全跳过
- 所有依赖包的 import 和初始化已在 checkpoint 中完成
- FastAPI 服务的路由注册、中间件加载等启动逻辑无需重新执行

这进一步强化了 Nexus 在大规模场景下的优势——不仅无需为每个镜像构建运行时，还能实现近乎瞬时的服务启动。

## 使用效果

### 克隆仓库

首先克隆 Nexus 仓库到本地：

```bash
git clone git@git.woa.com:nexus-agent/nexus.git
cd nexus
```

### 构建 Runtime 镜像

构建 nexus-runtime 镜像：

```bash
docker build -f deployment/docker/Dockerfile.runtime -t nexus-runtime .
```

### 创建数据容器

创建数据容器，这个容器只用于提供 volume 给其它容器挂载，不需要处于运行状态：

```bash
docker run -d --name nexus-runtime nexus-runtime:latest true
```

### 在任意镜像中启动 Runtime 服务

运行一个任意镜像（如 alpine），挂载 runtime volume 并启动服务：

```bash
docker run --volumes-from nexus-runtime -it --rm -p 32000:32000 alpine \
  /nix/runtime/nix-env/bin/bash /nix/runtime/nexus/deployment/runtime/start.sh -v 3.13 -p 32000 -H 0.0.0.0
```

### 调用 Runtime API

服务启动后，可以通过 HTTP API 调用各种 executor。 例如执行命令：

```bash
# 健康检查
curl http://localhost:32000/health
# 输出: {"status":"ok"}

# 执行命令
curl -X POST "http://localhost:32000/api/v1/call_executor/run_command" \
  -H "Content-Type: application/json" \
  -d '{
    "command": "cat /etc/os-release"
  }'

# 输出示例
{
  "status": "Finished",
  "execution_time": 0.01,
  "return_code": 0,
  "stdout": "NAME=\"Alpine Linux\"\nID=alpine\nVERSION_ID=3.22.2\n...",
  "stderr": ""
}
```

Runtime 服务提供了命令执行、文件操作、终端会话、Jupyter notebook、浏览器自动化等多种 executor，详见 [与 Runtime 交互](../how-to/use-runtime/interact-with-runtime.md)。
