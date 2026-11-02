# Task Generator 解耦规范

## 1. 框架总览

### 拆分原理

一个 batch generator 回答两类变化频率完全不同的问题，必须分开存放：

| 问题 | 变化频率 | 归属 |
| --- | --- | --- |
| 跑在什么环境、用什么脚手架、接哪个模型 | 按实验换，不按实例换 | `configs/**/*.yaml` |
| 跑哪些实例、每条实例有什么不同 | 每条 item 都不同 | `generator.py` |

把第一类写进 Python，配置的组合爆炸就变成分支的组合爆炸（例如旧 `tasks/swe/generator.py`：近 40 行文档字符串罗列环境变量，`_build_agent_config()` 按脚手架 `if/else`，加一个模型就要改代码）。三条准则：

1. 不随实例变化的东西不进 Python；用环境变量**选文件**，不用环境变量**拼配置**。
2. generator 只保留"看见这条实例才知道"的部分。
3. 三份 YAML 共享同一棵 `RunInput` 树，合并与插值由框架统一做。

### 目录组织

| 路径 | 职责 | 约束 |
| --- | --- | --- |
| `configs/runtime/*.yaml` | `runtime_provider_*`、`task_name`、`task_config`、`processors`、`runtime_configs` | 轴内互斥，一次运行选一个 |
| `configs/agents/*.yaml` | `agent_name`、`agent_config`、脚手架 `imports` | 同上 |
| `configs/llms/*.yaml` | `openai_clients`、后端专属的 `agent_config` 字段 | 同上 |
| `load_config.py` | 按 `RUNTIME_CONFIG` / `AGENT_CONFIG` / `LLM_CONFIG` 选文件并折叠 | 只加载合并，不解析 |
| `generator.py` | 数据集迭代、筛选分页、实例级覆盖、batch 元数据 | 不含环境/模型默认值 |
| `run_batch.<name>.sh` | 选哪套 config + 并发多少 | 不承载可调参数 |
| 其他（`dataset_handler.py`、`preprocess.py`、`utils.py`、`*_hook.py`） | 任务自身逻辑 | 不参与配置组装 |

### 配置流水线

```text
configs/runtime/*.yaml ┐
configs/agents/*.yaml  ├─ load_config() ─→ 通用配置 ─→ deep_merge(实例级覆盖) ─→ RunInput.from_dict()
configs/llms/*.yaml    ┘   (静态折叠)                                             (插值 + 校验)
```

> 该方案类似于 nexus 的 `run-part`，将核心功能提解耦出来，并将各种形式的数据输入部分（文件、目录等）交由 generator.py 来处理。

## 2. 具体规范

### 2.1 Config YAML 与 `load_config.py`

**出发点**：一份配置既要能被 generator 加载，也要能被 `nexus run batch --use` 直接吃，所以用同一个 run-part 信封；选择逻辑只有"按名字找文件"。

每个文件三个顶层键：

```yaml
body:     # 本文件贡献的 partial RunInput
merge:    # 按点分路径声明列表合并策略
source:   # 日志标签，loader 丢弃
```

加载器按固定顺序折叠，**后者覆盖前者**：

```python
ENV_TO_FOLDER = {'AGENT_CONFIG': 'agents', 'LLM_CONFIG': 'llms', 'RUNTIME_CONFIG': 'runtime'}


def load_config() -> dict[str, Any]:
    """Merge runtime, then agent, then LLM. The LLM file wins on ``agent_config`` keys it owns."""
    config: dict[str, Any] = {}
    for env_var in ('RUNTIME_CONFIG', 'AGENT_CONFIG', 'LLM_CONFIG'):
        config = _merge_part(config, env_var)
    return config
```

- 环境变量取值是 `configs/<kind>/` 下的 stem（自动补 `.yaml`）或磁盘路径；**没有默认文件**，不设即跳过该轴
- `${oc.*}` 字符串原样留到 `RunInput.from_dict`，loader 不解析
- 脚手架不认的字段由 pydantic `extra='ignore'` 吃掉，无需防御性判断
- LLM 排最后，因此可改写脚手架的 `agent_config`（`thinking_preset`、`model` 等）
- 文件头注释写清：用途、必填/可选环境变量、使用示例、互斥约束

#### 合并规则

`nexus/runner/compose/merge.py` 的默认语义：两个 dict 逐键递归；`null` 删除该键；其他值（含 list）整体替换。只有 list 需要显式声明策略：

| 策略 | 行为 | 典型用途 |
| --- | --- | --- |
| `append` | 拼接，保留重复 | 累加型列表 |
| `append_unique` | 只追加不存在的项 | `imports` |
| `replace` | 整体替换 | 需要覆盖而非累加时 |

```yaml
merge:
  imports: append_unique
```

#### 参数定义：`oc.env` / `oc.decode` / `oc.select`

**出发点**：可调参数下沉到 YAML 默认值，调参不改代码。

```yaml
working_dir: ${oc.env:AGENT_WORKING_DIR,/testbed}
timeout: ${oc.decode:${oc.env:AGENT_TIMEOUT,14400}}
haiku_model: ${oc.env:HAIKU_MODEL,${oc.select:openai_clients.main.config.request_args.model,null}}
ANTHROPIC_MODEL: 'claude-opus-5[1m]'
```

| 规则 | 说明 |
| --- | --- |
| `oc.env` 一律返回字符串 | `cpu: ${oc.env:RUNTIME_CPU,2}` 会把 `"2"` 喂给下游；数值/布尔必须套 `oc.decode` |
| `oc.select` 路径相对**合并后**的树 | 跨配置读取: agent 文件能读到 LLM 配置的原因 |
| 默认值含 `[` `]` `,` 等特殊字符必须加引号 | 否则被 OmegaConf 解析成列表 |
| 非 `oc` 的 `${...}` | 原样保留字符串 |

#### 秘钥定义：`__env` 后缀

**出发点**：batch 配置会落进 SQLite，Token 不能明文入库。

```yaml
runtime_provider_config:
  token__env: GONGFENG_RUNTIME_TOKEN   # 执行机上才展开成 token: os.environ[...]
```

`foo__env: VARNAME` 是 runner 约定，**generate 机器上不取值**。凡密钥一律用它；例外只有需要 generator 二次加工的凭证（例如 `OPENAI_API_KEY` 要追加 cache 参数），那种才留 `oc.env`。注意，hyway 平台编写 `plan.yaml` 配置时，此类 `__env` 后缀的密钥需要借助 `host-env` 才能在云端 worker 上生效。例如:

```yaml
trajing:
  generator: generator/swe_generator_zkc_mini_swe_agent.py
  dataset_env_name: SWE_DATASET_PATH
  timeout: 172800
  env:
    LLM_CONFIG: claude-opus-5.openai
    AGENT_CONFIG: codex_0.154.0
    RUNTIME_CONFIG: swecosmos_runtime
  args:
    host-env:
      - GONGFENG_RUNTIME_TOKEN=xxxxxxxxxxxx
    # 单个变量也可以写在一行里: host-env: GONGFENG_RUNTIME_TOKEN=xxxxxxxxxxxx
```

### 2.2 `generator.py` 使用规范

通用配置对所有实例相同，只折叠一次；循环体里只叠加实例级覆盖：

```python
def generate_configs() -> Iterator[BatchItem]:
    base = load_config()

    for instance_id, instance, instance_meta in _iter_selected_instances():
        overlay: dict[str, Any] = {'task_config': {'instance_id': instance_id, 'instance': instance}}
        ...
        instance_config = deep_merge(base, overlay)
        yield BatchItem(config=RunInput.from_dict(instance_config), run_id=instance_id, metadata={...})
```

#### 实例加载 (以 SWE 任务为例)

参考 swe-cosmos 的 [`generator.py`](nexus/extensions/tasks/swe/swe_cosmos/generator.py)，从路径读数据集，做筛选和分页，产出 `(instance_id, instance, metadata)`：

| 环境变量 | 作用 |
| --- | --- |
| `DATASET_PATH` | 必填，本地 `.jsonl`；缺失或后缀不对直接报错 |
| `INCLUDE_INSTANCES` / `EXCLUDE_INSTANCES` | 逗号分隔的 id，或 `.json` / `.jsonl` 选择器文件路径 |
| `SWE_SKIP` / `SWE_LIMIT` | 跳过前 N 条，然后最多取 N 条 |

```python
def _iter_selected_instances() -> Iterator[tuple[str, SWEbenchInstance, dict[str, Any]]]:
    """Yield (instance_id, instance, metadata) for every instance that survives the filters and SWE_SKIP/SWE_LIMIT."""
```

- include 先生效，再 exclude；两边都声明同一个 id 视为配置错误，直接抛异常。
- 选择器文件支持 `.jsonl`（逐行对象）和 `.json`（列表或单对象），元素可以是裸字符串或带 `instance_id` 的 dict；取不出 id 要报出文件名和行号。
- `generate_configs` / `get_total_count` / `get_batch_metadata` **共用这一个迭代器**。

#### 参数覆盖

实例级差异统一表达为一个 `overlay` dict，最后一次 `deep_merge(base, overlay)`，不要多处改 `base`。覆盖规则与 2.1 的合并规则完全一致。

有些覆盖要先知道通用配置里的某个值（模型是不是某家的、是否 oracle），但此时整棵树还没解析。做法是把需要的几个值"抬"成小 dict 单独解析，不整体 resolve：

```python
def _resolve_scalars(values: dict[str, Any]) -> dict[str, Any]:
    """Resolve ``oc.env`` / ``oc.decode`` / ``oc.select`` and ``foo__env`` in a lifted dict."""
    resolved = OmegaConf.to_container(OmegaConf.create(values), resolve=True)
    ...
    return _resolve_env_suffix_config({str(k): v for k, v in resolved.items()}, path='probe')
```

注意此时 `oc.select` 的路径相对这个小 dict，不再是完整配置树。探测结果在循环外算一次即可。

具体覆盖写法见[第 3 章](#3-常见参数覆盖场景)。

### 2.3 启动脚本

**出发点**：脚本是入口文档，只声明"选哪套 + 并发多少"；能放进 YAML 默认值的参数不要出现在这里。

```bash
export RUNTIME_CONFIG="${RUNTIME_CONFIG:-swecosmos_runtime}"
export AGENT_CONFIG="${AGENT_CONFIG:-claude_code}"
export LLM_CONFIG="${LLM_CONFIG:-claude-opus-5.litellm}"
export DATASET_PATH="${1:-${DATASET_PATH:-}}"
export MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
export WORKER_PROCESSES="${WORKER_PROCESSES:-$(( (MAX_CONCURRENT + 24) / 25 ))}"
```

## 3. 常见参数覆盖场景

下面介绍一些常见的参数覆盖场景，并给出相应的代码示例。参考源码: [`swe-cosmos/generator.py`](nexus/extensions/tasks/swe/swe_cosmos/generator.py)。

### 3.1 API_KEY 缓存分区

**问题**：网关按 key 切 prompt cache。若 LLM profile 里写一个静态值，并发实例会互相串 prefix cache——所以这个值只能由 generator 逐实例生成。

```python
def _per_instance_cache_id(instance_id: str) -> str:
    """md5 of ``time:pid:instance_id`` — unique per instance and per shard (each shard is its own process)."""
    return hashlib.md5(f'{time.time()}:{os.getpid()}:{instance_id}'.encode()).hexdigest()
```

两条通路，按模型选：

| 通路 | 覆盖位置 | 适用 |
| --- | --- | --- |
| `cache_task_id` | `client_args.api_key` 上追加 query 参数 | 网关从凭证读分区 id 的模型（例如 Claude 系） |
| `prompt_cache_key` | `request_args.prompt_cache_key` | 网关从请求体读分区 key 的模型，维护一张白名单常量 |

```python
client_overlay: dict[str, Any] = {}
if api_key := _api_key_with_cache_task_id(probe, instance_id):
    client_overlay['client_args'] = {'api_key': api_key}
if prompt_cache_key := _prompt_cache_key_for_instance(probe, instance_id):
    client_overlay['request_args'] = {'prompt_cache_key': prompt_cache_key}
if client_overlay:
    overlay['openai_clients'] = {'main': {'config': client_overlay}}
```

两个判定函数都在不适用时返回 `None`，由调用方决定是否叠加——**绝不无条件改写凭证**，否则会把不走该网关的 key 改坏。

### 3.2 `run_oracle` 跳过部分配置

**问题**：oracle 直接应用 golden patch，不走 agent 分支，脚手架与客户端是死重量；但 `run_oracle` 写在 runtime profile 的 `task_config` 里，**折叠完才知道**，所以只能后置裁剪而不是在构造处加分支。

```python
def _load_parts() -> dict[str, Any]:
    config = load_config()
    if not _is_oracle_run(config):
        return config
    return deep_merge(config, {'agent_name': None, 'agent_config': None, 'openai_clients': None})
```

### 3.3 实例级配置特例

**问题**：个别实例的资源需求超出 runtime profile 的默认值。这类例外集中在一张带注释的常量表里，不要散落 `if instance_id == ...`。

```python
# Instances whose resource_spec the runtime profile's default does not cover.
INSTANCE_RESOURCE_OVERRIDE = {
    'scikit-learn__scikit-learn-14710': {'cpu': 16, 'memory': 16},  # Heavy numeric workload, CPU bound.
}
```

```python
if override := INSTANCE_RESOURCE_OVERRIDE.get(instance_id):
    overlay['runtime_provider_config'] = {'resource_spec': override}
```
