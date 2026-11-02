# Terminal-Bench2 Task 评测

Terminal-Bench2 任务评测模块，支持单任务和批量任务评测，支持本地 Docker 和远程工蜂沙箱两种运行模式。

> 总体流程：读取 task.yaml → 启动容器 → 传输oracle文件 或 与Agent交互 → 执行 test → 收集结果


## 1.Task下载

直接 copy `/apdcephfs_qy4/share_302593112/wxyeszhou/Agent/harbor-main/tasks`

或前往 GitHub 下载: `https://github.com/laude-institute/terminal-bench-2`

## 2.镜像下载[可选]

如果使用本地评测，需要将 `task2image.json` 内的 89 个镜像下载到本地。使用工蜂评测跳过该步骤。

```bash
bash ./nexus/extensions/tasks/terminal/converters/scripts/pull_images.sh 4
```

---

## 3.Run-single评测

本地Debug用，每次评测一个task。先将单个 task 转换为 nexus 的 task.yaml 格式：

```bash
uv run python3 ./nexus/extensions/tasks/terminal/converters/converter.py \
/apdcephfs_qy4/share_302593112/wxyeszhou/Agent/harbor-main/tasks/adaptive-rejection-sampler \
-o ./workspace/tbench2/adaptive-rejection-sampler.yaml \
--run-oracle
```

执行测试：

```bash
uv run nexus run-single ./workspace/tbench2/adaptive-rejection-sampler.yaml
```


## 4.Run-batch批量评测

`generator.py` / `generator_gongfeng.py` 通过环境变量指定参数，在测试时动态生成 task 配置，无需显式生成 task.yaml 文件。

**本地 Docker 批量评测：**

```bash
uv run nexus run-batch \
    -d ./workspace/tbench2_batch.db \
    -g ./nexus/extensions/tasks/terminal/converters/generator.py \
    --max-concurrent 1
```

**远程工蜂沙箱批量评测：**

```bash
uv run nexus run-batch \
    -d ./workspace/tbench2_batch.db \
    -g ./nexus/extensions/tasks/terminal/converters/generator_gongfeng.py \
    --max-concurrent 12
```

以评测Hy2.0为例，需配置如下环境变量：

```bash
export OPENAI_BASE_URL="http://stream-server-online-toolcall-10103.turbotke.production.polaris:81/openapi"
export OPENAI_API_KEY="EMPTY"
export OPENAI_MODEL="openai/服务组名称"
export AGENT_NAME="terminus2"
export GONGFENG_RUNTIME_TOKEN="xxx"
# export TASK_NAMES=DDD,DDD,DDD,  # 如果不设置此参数，则默认评测全量89个task
```

评测蒸馏平台的Opus45，需配置如下环境变量：

```bash
export OPENAI_BASE_URL="http://trpc-gpt-eval.production.polaris:8080"
export OPENAI_API_KEY="xxxxxx:xxxxxx?provider=anthropic&timeout=300"
export OPENAI_MODEL="claude-opus-4-5-20251101"
export AGENT_NAME="terminus2"
export GONGFENG_RUNTIME_TOKEN="xxx"
```

其他task相关配置的默认值都在converters/config.py中，比如最大轮数为200，超时时间为4小时，工蜂runtime版本0.10.16，一般不需要更改。

**查看实时进度：**

```bash
uv run nexus inspect-batch ./workspace/tbench2_batch.db
```

## 5.terminus2脚手架评测SWE任务

只需额外设置`SWE_AGENT_NAME`和`SWE_MAX_ITERATIONS`两个环境变量，并使用swe task的`generator.py`即可。
其他用法参考SWE文档。

```bash
export OPENAI_BASE_URL="http://trpc-gpt-eval.production.polaris:8080"
export OPENAI_API_KEY="xxxxxx:xxxxxx?provider=anthropic&timeout=300"
export OPENAI_MODEL="claude-opus-4-5-20251101"
export GONGFENG_RUNTIME_TOKEN="xxx"
export SWE_AGENT_NAME="terminus2"
export SWE_MAX_ITERATIONS="200"
export SWE_RUN_ORACLE=0  # 是否运行Oracle
export SWE_DATASET="swe-public/SWE-bench_Verified" # 使用远程数据集而不是本地数据集
export SWE_LIMIT="4" # 运行四个任务

# 注意在/nexus/extensions/tasks/swe/generator.py的`_build_openai_clients()`里要改为litellm client。默认是openai client。

uv run nexus run-batch \
    -d ./workspace/swe_terminus2.db \
    -g ./nexus/extensions/tasks/swe/generator.py \
    --max-concurrent 12
```