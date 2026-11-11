# IHVP v3 增量包使用指南（仅维护者）

这份指南和维护端总包不能作为被评模型的上下文或工作区。本次交付新增 15 条反馈序列和 15 条私有序列，**不包含旧版的 30 条正文**。与旧数据合并后，反馈与私有各 30 条。数据、预算和算法均未在本次指南修订中改变。

## 1. 两个包分别给谁

| 交付物 | 接收者 | 用途 |
| --- | --- | --- |
| `ihvp-hard-v3-with-guide.zip` | 维护者／评测方 | 新增数据、参考答案、私有因子和种子、冻结记录、验证记录、源码快照及本指南 |
| 总包中的 `model-visible.zip` | 被评模型的受限开发环境 | 恰好六个文件：题面、公开小样例及协议工具 |

**不要把总包交给模型后要求它“自觉不看”。先在维护端提取 `model-visible.zip`，模型只拿这个小包。** 文件只读仍然可以泄漏内容；`readonly` 不是信息隔离。

模型应在独立评测会话中开发。不要把当前研究对话、构造说明、PR、仓库搜索结果或维护日志加入其上下文；不要将整个 FrontierAM 仓库设置为模型工作区。模型和它调用的文件、终端等工具，都应受到同一边界约束。

这里的“模型”指参加 benchmark 的求解模型，不包括负责接收和部署题包的维护者。求解模型不能通过宿主终端、仓库／网盘连接器或跨会话搜索工具读取维护材料；只限制最终 `solver.py` 的权限还不够。

## 2. 模型能看的文件：明确白名单

`model-visible.zip` 解压到模型的 `/workspace/`，应当恰好是：

```text
/workspace/
├── problem.md
└── samples/
    ├── README.md
    ├── run_sample.py
    ├── smoke.json
    └── support/
        ├── api.py
        └── worker.py
```

此外允许模型创建、读取和修改自己的 `/workspace/submission/`、自己的临时构建文件，并使用预装的 Python、NumPy、SciPy 等通用运行库。宿主提供的题目材料不应超过上述六个文件。

`smoke.json` 是唯一允许直接暴露矩阵、右端项和参考解的例外：它只有 **4 维、3 步**，仅演示协议，不参与评分，也不承诺代表完整规模系统的构造。

完整规模的 `sample_01`、`sample_02` **不是 NPZ 文件**，而是服务中的两个样例名称。模型只能通过协议取得：

- 当前时刻的 `b_t`、时刻／阶段标识、谱界和截止时间。
- 合法当前 HVP 查询 `v → H_t v` 的回复。
- 完成一条公开样例后的有效性、HVP 次数、用时及样例逐步／平均 loss。

允许访问专用服务的三个 socket 与这些样例的结果文件：`/run/ihvp-samples/{smoke,sample_01,sample_02}.sock`、`/run/ihvp-samples/results/`。该目录不得混入正式反馈或私有集记录。

## 3. 模型不能看的材料：维护端保留

以下均不能复制或挂载到模型／求解进程的可读文件系统，也不能通过编辑器索引、搜索工具、聊天附件或日志传给模型：

| 材料 | 典型位置 | 原因 |
| --- | --- | --- |
| 完整反馈与私有教师正文 | 总包 `data/public/**`、`data/private/**`；导入后的全部 `teacher.npz` | 内含矩阵构造因子、右端项和参考答案；`public` 是反馈角色，不表示可公开阅读 |
| 完整规模样例教师正文 | 导入后的 `data/samples/**` | 样例也只能通过 HVP 服务提供，不把 NPZ 给模型 |
| 生成器、私有因子与实际种子 | `curator/**`、`generation/**`、`plan.json`、`design_freeze.json` | 可恢复或识别实例、生成方式及答案 |
| 评测端实现 | `grader/**`、宿主 oracle 实现 | 不属于白名单；仅 `samples/support/` 中的公开通信协议可见 |
| 参照实现 | `references/**`、`curator/public-worker-snapshots/**` | 包含人类参照与本次筛查程序源码 |
| 验证、分析与标定记录 | `validation/**`、`analysis/**`、运行清单、逐例结果、预测数组及平台内部锚点文件 | 可能暴露构造、答案、逐例信息或实验经验 |
| 完整仓库和历史材料 | `curator/source/**`、FrontierAM 仓库、`.git/`、旧题包、旧模型源码和当前研究对话 | 可绕过最小题包边界恢复上述材料 |
| 维护交接说明与文件校验清单 | 本指南、总包 README、`FILE_SHA256.json`、`MODEL_VISIBILITY.json` | 只供维护者审查和部署，不属于给模型的题面 |

平台可以按评测协议向模型返回允许的**汇总反馈**，例如 public 总分、汇总 loss 或按类别与参照的对比；不能因此开放原始清单、私有记录或整个日志目录。

## 4. 维护端解压与导入

在模型不可访问的维护目录解压。总包采用根目录文件布局，务必明确指定解压目录：

```sh
umask 077
mkdir -p /srv/ihvp-hard-v3-addition
unzip /path/to/ihvp-hard-v3-with-guide.zip -d /srv/ihvp-hard-v3-addition
```

ZIP 旁的 `.sha256` 可在同目录用 `sha256sum -c 文件名.zip.sha256` 校验；macOS 对应 `shasum -a 256 -c 文件名.zip.sha256`。总包内 `FILE_SHA256.json` 校验每个条目，不要把它给模型。

使用包含本次变更的 [PR #2 分支](https://github.com/IceySakura/FrontierAM/pull/2)，在仓库根目录执行：

```sh
python scripts/seal/prepare_ihvp_hard_v3.py \
  --addition-root /srv/ihvp-hard-v3-addition/data \
  --legacy-bundle /path/to/ihvp-sequence-v1
```

如果旧数据已导入，改用 `--legacy-data-root var/raw/ihvp-sequence-v1/data`。默认生成 `var/raw/ihvp-sequence-v3/data/{public,private,samples}/`：30 条反馈、30 条私有，以及两个取自新增反馈的宿主样例。原字节保持不变，不重新生成；角色、SHA、维度和跨集合重复会被校验。

`curator/source/` 是交付源码快照，不能将整个快照给模型。用于平台接线的主入口仍是上述仓库分支。

## 5. 只准备最小模型工作区

在维护端创建一个与私有目录分离的干净目录：

```sh
mkdir -p /srv/ihvp-model-visible
unzip /srv/ihvp-hard-v3-addition/model-visible.zip -d /srv/ihvp-model-visible
mkdir -p /srv/ihvp-model-visible/submission
mkdir -p /srv/ihvp-submission
chmod -R a+rX /srv/ihvp-model-visible
# 在目标 Linux 主机以所需维护权限执行；只调整提交目录，不调整私有目录。
chown 1000:1000 /srv/ihvp-submission
```

核对这里没有 `.git`、任何 NPZ、`curator/`、参照源码或验证记录。为容器非特权用户设置仅这份公开材料所需的读取权限；提交目录单独赋予该用户写权限。不要对私有总包目录递归放宽权限。

需要从仓库重建最小包时，使用固定白名单构建器：

```sh
python scripts/seal/build_ihvp_model_visible.py \
  --out /absolute/path/to/model-visible.zip
```

相邻的 `.manifest.json` 和 `.sha256` 仅供维护者核对，小包本身仍然只有六个文件。不要用 `cp -r tasks/ihvp-sequence` 或 `COPY . /workspace` 代替。

## 6. 启动宿主样例服务

本次补充了明确选择 **v3 后端** 的维护端启动器。不要直接使用默认指向 v1 的旧 oracle 启动方式：

```sh
python scripts/ops/serve_ihvp_hard_samples.py \
  --data-root var/raw/ihvp-sequence-v3/data \
  --socket-dir /tmp/ihvp-v3-samples
```

socket 目录必须不存在，由启动器创建；Unix socket 路径要短。启动器检查样例确实来自反馈集、教师角色不是 private，并核对 SHA 和规模。宿主保留教师文件；只将这个服务目录挂入模型容器的 `/run/ihvp-samples`。服务在前台运行，正常终止后清理自己创建的服务目录。

模型在容器内运行：

```sh
python3 /workspace/samples/run_sample.py smoke
python3 /workspace/samples/run_sample.py sample_01
python3 /workspace/samples/run_sample.py sample_02
```

求解器实现位于自己的 `/workspace/submission/solver.py`。每次 `solve` 返回即提交当前解，只能查询当前系统，不能回看未来系统；每条序列重新启动，不保存实例答案或实例专用状态供后续序列查表。

这个样例服务仅用于开发自检；它不是正式评分器，也不负责完整监控 worker 峰值 RSS 或杀掉超限求解进程。其用时不能替代正式评测，开发容器内存上限也不能替代正式的 worker 256 MiB 规则。

## 7. 目标 Linux 环境的挂载示例

下面仅展示最小开发工作区的挂载方式，使用目标机已经审查过的干净数值运行镜像；本机未运行 Docker，不将此示例描述为已完成部署验收。目录需存在，提交目录对 UID 1000 可写。

```sh
IHVP_IMAGE='the-reviewed-offline-image-tag'
docker run --rm -it --network none --cpus 1 \
  --user 1000:1000 --cap-drop ALL --security-opt no-new-privileges:true \
  --memory 4g --memory-swap 4g \
  --mount type=bind,src=/srv/ihvp-model-visible,dst=/workspace,readonly \
  --mount type=bind,src=/srv/ihvp-submission,dst=/workspace/submission \
  --mount type=bind,src=/tmp/ihvp-v3-samples,dst=/run/ihvp-samples,readonly \
  --workdir /workspace \
  "$IHVP_IMAGE" /bin/bash
```

开发协议的 4 GiB 与正式求解进程 256 MiB 是两个不同限制。单数值线程、正式 worker 的 30 秒／256 MiB／进程约束由平台正式评分链路实施。

这里的 socket 服务和 Docker daemon 必须位于同一个 Linux 宿主机／同一个可传递 Unix socket 的环境。该示例不承诺能把 macOS 宿主 socket 直接传进 Docker Desktop 的 Linux VM。

只允许这三类挂载：最小公开工作区、自己的提交目录、专用样例 socket／结果目录。**不要额外挂载 `/repo`、完整仓库、总包、宿主 home、Docker socket 或任何 teacher 目录；不要共享宿主 PID 空间。** 镜像其他位置也不能预装本题生成器、参照或旧题包。只把最终 `solver.py` 放进容器，而让编程模型的其他工具仍能读宿主私有材料，也不满足边界。

Docker 的 `readonly` 只限制写入，不能隐藏被挂载文件，见 [官方绑定挂载说明](https://docs.docker.com/engine/storage/bind-mounts/)。

## 8. 给模型的初始提示词模板

以下这段可以单独给模型；不要连同整份维护指南一起发送：

> 请完整阅读 `/workspace/problem.md`。本题提供的文件仅限该题面及 `/workspace/samples/` 中的 README、run_sample.py、smoke.json、support/api.py、support/worker.py。你可以使用预装通用运行库，在 `/workspace/submission/` 编写和构建自己的求解器，并通过公开样例服务自检。完整规模实例只提供当前右端项、公开 context 和当前 HVP 查询回复。不得读取工作区之外的题目材料、生成器、教师数组、参考答案、参照源码、私有清单、种子计划、维护日志或其他会话；不得利用网络、目录挂载或工具访问绕过该边界。每条序列从空状态开始。合法的历史解复用和通过 HVP 学习谱信息仍然允许。

## 9. 正式评测的可见范围

正式评分的 worker 只获得冻结提交和公开接口，不获得题目仓库或教师路径。可信评测器、数据、参考答案和标定材料在另一侧保留，通过受控协议交换当前输入与输出。不要照搬旧驱动将整个题目目录挂入 worker 可读路径的做法；只读挂载仍会暴露构造代码。

public 用于后台汇总反馈，private 用于最终评分；正式 private 逐例记录不能进入模型工作区。反馈与私有分别在目标评测机重测锚点，原机器或旧版数字不能复用。

v3 目前仍是独立草稿：本指南和本地样例验证不表示正式驱动已经接线、目标 Linux 隔离已验收或评分已激活。30 条反馈各用满 30 秒已经达到 900 秒，还需明确现行 15 分钟反馈总时限的分批或调整方案。完整部署由维护端在这些条件确定后启用。
