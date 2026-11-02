# 共享配置层 `nexus/extensions/configs`

多个 task 共用的 agent / llm / runtime 配置，加上 SWE 和前端两份可以直接跑的 generator 模板。

**换 scaffold 或换模型，只改环境变量，不改代码。** 做法沿用 [Task Generator 解耦规范](../../../docs/developing/decouple-task-generator.md)：按实验变的东西写在 yaml 里，generator 只管实例。本目录把规范里每个 task 各自一份的 `configs/` 提到共享位置，别的 task 不用再各抄一份。

> **你的 task 不是 SWE 或前端？** 下面「快速开始」默认你在跑这两个现成 task；全新 topic 接入的完整步骤在「[新增配置 → 一个全新的 topic，从零接入](#一个全新的-topic从零接入)」。

## 术语

全文只用这几个词，先在这里说清，后面不再解释。

| 词 | 意思 |
| - | - |
| **scaffold** | 跑任务的 agent 脚手架本身（claude_code、codex、codebuddy、workbuddy、Hyness Harness 等）。本文说"选一个 scaffold"就是选 `AGENT_CONFIG` |
| **wire** | 一条**具体的模型接入线路** —— 端点 + 协议 + 模型名的组合，对应 `llms/` 下一份 yaml。同一个模型家族里可能有几条不同 wire（比如 claude 有 litellm / openai / vertex 三条） |
| **family / 模型族** | `gpt` / `claude` / `hunyuan` 三个闭集取值，写在 llm 文件的 `llm.family` 里。它是**合并时的分组键**，不是模型能力的分类 |
| **fact / 事实** | llm 文件在 `llm:` 下声明的、关于**这条 wire 本身**的客观属性（端点读不读 `cache_task_id`、有没有原生 reasoning item…）。scaffold 用 `${llm:键名}` 取用。为什么叫"事实"：它不随 scaffold 变，只能被陈述，不能被 scaffold 决定 |
| **三轴 / 一个 cell** | 一次运行由 `RUNTIME_CONFIG` + `AGENT_CONFIG` + `LLM_CONFIG` 三个名字确定，合起来叫一个 cell（矩阵里的一格） |
| **topic** | 用这套共享配置的一个**业务线**（SWE、前端）。每个 topic 有自己的 generator 模板和 `runtime/` 文件 |
| **overlay** | generator 给每个实例、每次采样额外叠的一层配置（题集、重复次数、镜像等），写在 generator 里而不是 yaml 里 |


## 快速开始

### SWE：跑一个 scaffold × 模型

```bash
export RUNTIME_CONFIG=swe_runtime                  # 沙盒 + 任务（runtime/）
export AGENT_CONFIG=kilo                           # 换脚手架（agents/）
export LLM_CONFIG=gpt-6-sol.openai_responses       # 换模型（llms/）

export DATASET_PATH=/path/to/swe.jsonl             # 完整实例的 jsonl
export GONGFENG_RUNTIME_TOKEN=...                  # 或 GONGFENG_TOKEN
export HR_API_KEY=sk-...                           # 裸 key，llm 文件自己补 ?provider=
export SWE_LIMIT=5                                 # 可选：只跑前 5 条

uv run nexus run-batch -g nexus/extensions/configs/generator_swe_template.py -d out.db --max-concurrent 5
```

### 前端：同样三个名字

```bash
export RUNTIME_CONFIG=frontend_runtime
export AGENT_CONFIG=kilo
export LLM_CONFIG=gpt-6-sol.openai_responses
export DATASET_PATH=/path/to/cases.jsonl           # 每行至少有 instance_id、prompt
export GONGFENG_RUNTIME_TOKEN=... HR_API_KEY=sk-...
export AGENT_TIMEOUT=7200                          # 前端必须设，原因见「环境变量与覆盖关系」

uv run nexus run-batch -g nexus/extensions/configs/generator_frontend_template.py -d out.db
```

### 和生产 / 评测对齐

只换 `RUNTIME_CONFIG`：

| 用途 | SWE | 前端 |
| --- | --- | --- |
| smoke、临时实验（资源和超时都能用环境变量调） | `swe_runtime` | `frontend_runtime` |
| 生产 rollout | `swe_prod_runtime`，配 `RUNNER_TIMEOUT=12900 AGENT_TIMEOUT=12000` | `frontend_prod_runtime`，配 `AGENT_TIMEOUT=12000` |
| 评测 | `swe_prod_runtime`（和生产同一份，见下） | `frontend_eval_runtime`，配 `AGENT_TIMEOUT=12000` |

**SWE 的生产和评测共用一份。** 原来是 `swe_prod_runtime` / `swe_eval_runtime` 两份，但去掉注释后只差 `runner.timeout` 一行、另外 84 行逐字节相同，所以合并了，差的那个值走 `RUNNER_TIMEOUT`（默认 14500 取安全侧，生产设 12900）。前端那两份不一样：镜像、CPU、`rpc_protocol`、`task_config` 键都不同，仍然是两份。

生产和评测的三份要求设 `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`，不设直接报错。smoke 那两份不设就静默关掉 langfuse。每份文件顶部写了它的值抄自哪份生产配置，以及哪些没法对齐。

### 跑之前先校验

```bash
uv run python -m nexus.extensions.configs.validate          # 每份 yaml 是否符合下面的 schema
uv run python -m nexus.extensions.configs.validate --grid   # 再把所有 agent × llm × runtime 合并一遍
uv run python -m nexus.extensions.configs.validate --strict # 有文件没查完（缺可选依赖）就失败；开跑前用这个
```

这台机器连不上 `files.pythonhosted.org`，`uv sync` 会超时。装依赖指向内部镜像，并且不要用 `uv sync`（换源会改写 `uv.lock`）：

```bash
uv pip install --python .venv/bin/python --index-url https://mirrors.tencent.com/pypi/simple/ -e .
uv pip install --python .venv/bin/python --index-url https://mirrors.tencent.com/pypi/simple/ "litellm<1.82.7" langfuse jinja2
```

`litellm`、`langfuse`、`jinja2` 不装的话，用到它们的 llm、runtime、agent 文件只会被查外壳。

脚本会导入每份 yaml 的 `imports` 来查键。解释器里缺 `langfuse`、`litellm` 这类第三方依赖时，这份文件只查不需要导入的部分（外壳、必填键、llm 的 agent_config 白名单），并打一行 `SKIP`，最后汇总有几份只查了外壳。想全查就先装依赖（`uv run nexus modules install <名字>`）再跑。`imports` 里写了 nexus 自己没有的模块仍然直接报错。

nexus 的 agent、task、runtime provider 都会**静默丢弃不认识的键**，`agent_config` 里拼错一个字母，运行时不会报任何错。校验脚本就是用来抓这个的，新增或修改 yaml 之后都跑一次。

## 目录与分工

| 路径 | 内容 | 谁维护 |
| --- | --- | --- |
| `agents/*.yaml` | 脚手架：`agent_name`、`agent_config`、`imports`，以及按模型族分的 `agent_config_by_llm` | scaffold 负责人给推荐值；topic 可以新增文件，但必须遵守下面的 schema |
| `llms/*.yaml` | 模型后端：`openai_clients`，以及 `llm:` 下这个模型的事实 | 跟随 AIHub 的 API，topic 不改 |
| `runtime/*.yaml` | 沙盒与任务：`runtime_provider_*`、`task_name`、`task_config`、`processors` | 各 topic 自己维护，一个用途一份 |
| `load_config.py` | 按三个环境变量选文件并合并 | 共享层 |
| `generator_swe_template.py` | SWE generator 模板 | 共享层，和 `tasks/swe/swe_cosmos/generator.py` 保持同步 |
| `generator_frontend_template.py` | 前端 generator 模板 | 共享层 |
| `validate.py` | 按本文 schema 校验 yaml | 共享层 |
| `cache_id.py` | SWE、前端 generator 共用的 `cache_task_id` 规则。不调用它的 topic 不受影响 | 共享层 |
| `env_table.py` | 生成本文末尾的环境变量大表。运行时不参与合并 | 共享层 |

## 原理：三份 yaml 怎么组合、怎么生效

一个 cell = 三个名字。`load_config()` 按 **runtime → agent → llm** 的顺序把三份文件合并成一棵 `RunInput` 树（`RunInput` 是 runner 读的运行输入结构，见 `nexus/core/`），generator 再给每个实例叠一层 per-instance overlay，最后交给 runner 起沙盒、跑 agent。

```text
  RUNTIME_CONFIG          AGENT_CONFIG              LLM_CONFIG
  runtime/<名>.yaml       agents/<名>.yaml          llms/<名>.yaml
  ┌───────────────┐       ┌──────────────────┐      ┌────────────────────────┐
  │ 沙盒 provider │       │ agent_name       │      │ llm:   <- 模型"事实"   │
  │ task_name     │       │ agent_config     │      │   family / context_    │
  │ task_config   │       │ agent_config_by_ │      │   window / reasoning_  │
  │ processors    │       │   llm.<family>   │      │   effort / ...         │
  │ runner        │       │   (含 ${llm:..}) │      │ openai_clients:        │
  └──────┬────────┘       └────────┬─────────┘      │   client_args          │
         │                         │                │   request_args         │
         ▼ 1. deep_merge，后者覆盖  ▼                └───────────┬────────────┘
         └─────────────►  合并后的一棵树  ◄──────────────────────┘
                                   │
          2. _apply_llm_family     │  取 llm.family 对应的 agent_config_by_llm 节
                                   │  并进 agent_config；把 ${llm:键} 换成 llm: 下的事实
                                   │  （顶层 agent_config 里的 ${llm:键} 也一并替换）
                                   │  然后删掉 llm / agent_config_by_llm 两个辅助键
                                   ▼
          3. _require_consumers    │  拿 agent / task 的 Input 模型逐键核对
                                   │  agent_config / task_config，有人不认的键直接报错
                                   ▼
          4. generator overlay     │  每个实例再叠：instance_id / prompt / cache id /
             (generator_*_template)│  langfuse trace 名 / FRONTEND_TASK_CONFIG_EXTRA
                                   ▼
          5. RunInput.from_dict ─► runner：起沙盒 → 装 agent → 调模型 → 判分
```

要点（每一条都对应一个踩过的坑）：

- **"事实"只写在 llm 文件里，scaffold 只引用，不猜 wire。** 同一个"推理档位"，openai_responses 拼成 `reasoning.effort`，litellm/Anthropic 拼成 `output_config.effort`；llm 文件声明 `llm.reasoning_effort`，scaffold 写 `${llm:reasoning_effort?}`，换 wire 不用改 scaffold。
- **`${llm:键?}` 带问号 = 这个 wire 可以不声明**：不声明时这个 key 被整个省略，scaffold 自己的默认值生效；不带问号则缺了就报错。
- **后合并的覆盖先合并的**：同一个键 llm 文件赢过 agent、agent 赢过 runtime；`oc.env` 只在 yaml 里写了 `${oc.env:名,默认}` 的那个位置生效——环境变量决定的是那个位置的值，不会越过文件之间的覆盖顺序。
- **"没人读的键"在 generate 机器上就报错**（第 3 步），不会等沙盒起了、钱花了才发现被 pydantic `extra='ignore'` 静默丢掉。

## 选文件、查找、合并

- **选文件**：`RUNTIME_CONFIG`、`AGENT_CONFIG`、`LLM_CONFIG` 各给一个文件名（不带 `.yaml`），也可以给磁盘路径。没设的轴直接跳过，没有默认文件。
- **查找顺序**：task 自己的 `configs/<轴>/` 优先，其次本目录，最后把名字当路径。所以一个 task 可以只放一份同名 yaml 盖掉共享的，其余继续用共享的。

  ```python
  from nexus.extensions.configs.load_config import load_config

  config = load_config()                                # 只用共享目录
  config = load_config(task_dir=Path(__file__).parent)  # task 目录优先
  ```

- **合并顺序**：runtime → agent → llm，后者覆盖前者。三份文件写进同一棵 `RunInput` 树。`oc.env` / `oc.decode` / `oc.select` 的写法和合并规则见 `load_config.py` 的文档字符串。
- **按模型族折叠**：合并完之后，loader 把 scaffold 的 `agent_config_by_llm.<llm.family>` 这一节并进 `agent_config`，节里的 `${llm:<键>}` 换成 llm 文件在 `llm:` 下声明的事实，然后把 `llm` 和 `agent_config_by_llm` 两个键都删掉（它们都不是 `RunInput` 字段）。
- **只选了 agent、没选 llm 时**，`${llm:...}` 不会被替换（没有 facts 可用），会原样留在 `agent_config` 里；这只在单测里单独加载一份 scaffold 时出现，真实运行三个轴一定都选。
- **每个键都必须有人读**：折叠完 loader 会拿 agent 和 task 的输入模型逐键核对 `agent_config` / `task_config`，有谁都不认识的键就直接报错，并把键名列出来。这一步在 generate 机器上跑，在起沙盒、发模型请求之前。

  > 这两条防的是同一件事：pydantic 输入模型默认 `extra='ignore'`，键写错名字不会报错、也不会生效，只是静默丢掉。这个错误在沙盒里看不出来，所以要在静态阶段拦掉。

## 参数的作用域和归属

一个参数该放在哪，本质上是回答一个问题：**这个值随什么变化？** 现在分三层，从稳定到具体：

| 作用域 | 随什么变化 | 放哪 | 例子 |
| --- | --- | --- | --- |
| **Scaffold-only** | 不随模型变，只随 scaffold 本身 | `agent_config` 顶层 | `npm_registry`、`timeout`、`version` |
| **Scaffold × family** | 同一模型家族（`gpt`/`claude`/`hunyuan`）内一致，跨家族不同 | `agent_config_by_llm.<family>` | `thinking_preset`（claude/hunyuan 各一个值） |
| **LLM-wire fact** | 这个具体 wire 的客观事实；不随 scaffold 变，**同一家族内也可能不一样** | `llm:`（llm 文件声明），任何 scaffold 用 `${llm:key}` / `${llm:key?}` 取 | `anthropic_model`、`no_thinking_request_args`、`use_azure_provider` |

**最容易搞混的地方：一个参数"看起来"随模型家族变，不代表它真的只随家族变。** 判据很简单——**同一家族里，换一个具体 llm 文件，这个值会不会不一样？** 会，就不是家族的事，是这个 wire 自己的事实，必须声明在 `llm:` 里，由 scaffold 用 `${llm:key}` 去取，不能在 `agent_config_by_llm.<family>` 里写死一个"家族统一值"。

**例子：codex 的 `use_azure_provider`。** 一开始把它当成"GPT 系列都是 Azure 部署"的 scaffold 级开关，整份 codex yaml 共用一个 `${oc.env:CODEX_USE_AZURE_PROVIDER,false}`。实测发现站不住：同是 `family: gpt`，`gpt-6-sol` / `gpt-61-sol` / `gpt-6-astra` 走真 Azure 部署（Bearer 带 `?provider=azure`），`gpt-56-sol` 走的是 aihub2 的"标准"端点（现在钉了 `?provider=azure`，但这条端点上的加密推理回放没单独验证过）——**同一家族内部都不统一**，证明这不是"by family"，是"by 这个具体 wire"。改法：
- 在真 Azure 的那三份 llm 文件的 `llm:` 下声明 `use_azure_provider: true`；`gpt-56-sol` 和所有 claude/hunyuan 文件不声明（省事，也更诚实——不是"false"而是"这事实不适用"）。
- codex 的 `agent_config_by_llm.gpt` 里写 `use_azure_provider: ${llm:use_azure_provider?}`。**位置还在 family 节里，但值来自选中的那份具体 llm 文件的事实，不是家族统一值**；没声明的 llm 文件走顶层 `agent_config` 的安全默认 `false`。
- `CODEX_USE_AZURE_PROVIDER` 环境变量还留着，但只在 llm 文件没声明事实时才生效——一旦声明了，以 llm 文件为准，env var 不能覆盖（写错这个值会实打实破坏行为，不该是一个能被忘记设置的开关）。


**第二个例子：codex 的 `wrap_plaintext_reasoning_content`。** 判据同上——它随"这条 wire 有没有原生 `reasoning_items`"变，而那是**协议**的属性，不是家族的。走 Responses 的 wire 自带原生 reasoning item，所以是 `false`；litellm / openai chat / vertex_anthropic / infer_proxy 只有纯文本 `reasoning_content`，必须 `true`。

**写错方向的代价是丢数据，不是"配置不一致"。** `false` 时 Responses 转换器不再从纯文本 `reasoning_content` 铸 reasoning item（原生 `reasoning_items` 照常透传，见 `runtime_setup.py` 里这个参数的 docstring）。所以给 claude / hunyuan 配 `false`，等于把推理信号整段丢掉——而 SFT 要的就是这个信号。

写法和 `use_azure_provider` 完全同形：

- 7 份 `.openai_responses` 的 `llm:` 下声明 `wrap_plaintext_reasoning_content: false`；其余不声明。
- codex 的 `agent_config_by_llm.gpt` **和** `.claude` 都写 `${llm:wrap_plaintext_reasoning_content?}`。两个家族节都接是故意的：事实属于 wire，将来出现 `claude.openai_responses` 就该直接生效，不用回头改 scaffold。
- 顶层兜底 `true`（与生产、与 codex Input 的 Python 默认值一致），`CODEX_WRAP_PLAINTEXT_REASONING` 只在 llm 文件没声明时生效。

⚠️ **`family=gpt` 今天恰好等价于 `openai_responses`，不要把这个巧合折成家族默认值。** 这正是 `use_azure_provider` 一节讲的那个坑。

⚠️ **查"生产怎么做"时，agent 键可能被 llm profile 覆盖**，只翻 `experiments/*.yaml` 的 `input:` 会得出相反的结论：生产的 `agents/codex_0.160.0.yaml` 默认 `true`，是 4 份 `.openai_responses` 的 llm profile 把它覆盖成 `false` 的。这一类"本层默认值悄悄偏离生产"有 CI 守卫，见「测试」一节。

**留好口子：出现新作用域时怎么加。** 目前只有这三层。如果某个参数真正横跨多个 scaffold、多个 llm 文件，但既不是 family 也不是单个 wire 能决定的（比如将来"按云厂商/按网络环境"变化），加法是：

1. 在相关 llm 文件的 `llm:` 下声明新事实——不需要开新顶层键，`llm:` 本来就是任意事实的口袋。
2. 在 `validate.py` 的 `LLM_FACT_KEYS`（现在是 `family`、`anthropic_model`、`no_thinking_request_args`、`use_azure_provider`、`wrap_plaintext_reasoning_content`、`context_window`、`auto_compact_token_limit`、`reasoning_effort`）里加上这个事实名。**这是故意做成白名单的**：不加就直接报错，逼着改动在这里留一笔，而不是某份 llm 文件悄悄写了一个没人管的事实。
3. scaffold yaml 里用 `${llm:事实名}` / `${llm:事实名?}`：**两种位置都可以**——
   - 顶层 `agent_config`：事实对**所有**家族都适用时（`reasoning_effort` 就是这样：codex 的 CLI 档位跟家族无关，只管这条 wire 认哪个字段名）。
   - `agent_config_by_llm.<family>` 节：只有某个家族关心时（像 `use_azure_provider` 只对 gpt 有意义）。

   `_apply_llm_family`（`load_config.py`）会**两处都扫**：先把选中的 family 节折进 `agent_config`，再对折完的整棵 `agent_config` 扫一遍 `${llm:...}`。所以顶层引用也能解析，scaffold 不需要为了读一个 wire 事实而假装把它塞进某个家族节。

**为什么 `reasoning_effort` 走顶层 fact 而不是让 scaffold 去猜字段名**：同一个"推理档位"，openai_responses wire 拼成 `reasoning.effort`，litellm/Anthropic 原生 wire 拼成 `output_config.effort`。如果让 scaffold 自己 `oc.select` 这两条路径，就相当于把"每种 wire 怎么拼字段"的知识泄漏进了 scaffold——每加一条新 wire 都要回头改 scaffold。改成 llm 文件声明一个协议无关的 `llm.reasoning_effort`（值和它自己 request_args 里那个字段用**同一个 env var**，两者不会不一致），scaffold 只写 `${llm:reasoning_effort?}` 一次，落哪条 wire 都读得到。走 fact 而不是让 scaffold 读扁平字段，是因为要同时覆盖 openai_responses / litellm / openai / hunyuan 四种拼法，并且默认值由 llm 文件自己决定。

## 可选的 agent

| 文件（`AGENT_CONFIG=`） | agent | 默认版本 | 说明 |
| --- | --- | --- | --- |
| `claude_code_eval` | claude_code | 2.1.267 | 评测用：不禁任何工具 |
| `claude_code_rollout` | claude_code | 2.1.267 | rollout 用：带 `--disallowedTools WebSearch WebFetch`。和上一份只差这一处 |
| `codebuddy_eval` | codebuddy | 2.151.0 | 评测用：带无人值守 `prompt_suffix`，不禁工具 |
| `codebuddy_rollout` | codebuddy | 2.151.0 | rollout 用：带 `--disallowedTools=EnterPlanMode --disallowedTools=ExitPlanMode`（等号形式，一个工具一项）。和上一份只差这一处 |
| `codex_0.160.0_codemode` | codex | 0.160.0 | `tool_mode: code_mode_only`，前端 rollout 的默认值。catalog（codex 自带的模型行为表：上下文窗口、推理档位、提示词等）用 `model_catalog_overrides`（MR !1535）叠加在 codex 原生的那条 entry 上，值对齐前端 rollout 的 `gpt6_v7` profile（见下）|
| `codex_0.160.0_toolmode` | codex | 0.160.0 | 同上，只差一个键：`tool_mode: direct` |
| `codebuddy_server` | codebuddy_server | 0.0.23-beta | |
| `hermes` | hermes | 0.19.0 | |
| `kilo` | kilo | 7.7.2 | |
| `opencode` | opencode | 1.18.31 | |
| `cline` | cline | 3.0.62 | |
| `pi` | pi | 0.85.1 | |
| `dsh` | dsh | 0.1.0-rc.6 | |
| `handy` | handy | 0.1.14 | |
| `workbuddy` | workbuddy | 5.9.0 | npm 私有包，需要 `WORKBUDDY_NPM_USERNAME` / `WORKBUDDY_NPM_TOKEN`；见 [wb-cli 示例](../../../docs/examples/configs/wb-cli/README.md) |
| `openclaw` | openclaw | 2026.5.28 | |
| `mini_swe_agent` | mini_swe_agent | 跟 nexus 代码 | nexus 内置实现，不装外部包；上游参考 mini-swe-agent 2.3.0 |
| `swe_agent` | swe_agent | 跟 nexus 代码 | 同上；上游参考 swe-agent 33b8eca |
| `codeact` | codeact | 跟 nexus 代码 | 即 OpenHands 的 CodeAct；上游参考 33b8eca |
| `terminus2` | terminus2 | 跟 nexus 代码 | |
| `hyness` | hyness | 调用方显式指定 | 通用 App 入口；必填 `HYNESS_PACKAGES`、`HYNESS_APP_CONFIG`，Harness/overlay 默认由 App 决定 |
| `hyness_opencc` | hyness | 引擎 0.0.1-alpha.20261020123306 / App 0.2.1-alpha.20261020.1 | OpenCC App + training overlay，完整 SWE deny 已通过安装包启动测试 |
| `hyness_mini_swe_agent` | hyness | 引擎 0.0.1-scaffold-apps.20261021.7 / App 0.1.0-scaffold-apps.20261021.7 | 独立 mini-SWE App，原生任务模板、`bash` 和提交 hooks |
| `hyness_swe_agent` | hyness | 引擎 0.0.1-scaffold-apps.20261021.7 / App 0.1.0-scaffold-apps.20261021.7 | 独立 SWE App，`bash / str_replace_editor / submit` |
| `hyness_workbuddy` | hyness | 引擎 0.0.1-scaffold-apps.20261021.7 / App 0.1.0-scaffold-apps.20261021.11 | 完整 Markdown SP + 原生 WorkBuddy 工具；App 携带依赖并准备 addon |

### Hyness：同一个 adapter，按配置选择 Harness

通用入口和四个命名预设都使用 `nexus.extensions.agents.hyness`，复用本目录的 loader 和 generator。
`AGENT_CONFIG=mini_swe_agent/swe_agent/workbuddy` 仍是原来的原生 Nexus agent；带 `hyness_` 前缀才是 Hyness 的对应实现。

mini/SWE 的独立 App 配置现统一使用 `hyness_mini_swe_agent`、`hyness_swe_agent`。
旧 `hyness_mini_swe_agent_app`、`hyness_swe_agent_app` 文件已删除，调用脚本需要去掉 `_app`；
保留的短名称已经替换旧 alpha 内置 Harness 的配置。相对于原独立 App 配置，安装包、工具、
超时和模型适配保持一致；相对于更早的内置 Harness，Prompt 和工具协议会变化，不能混为同一实验。

```bash
export RUNTIME_CONFIG=swe_runtime
export AGENT_CONFIG=hyness_mini_swe_agent     # 或 hyness_swe_agent
export LLM_CONFIG=hy3.openai
export DATASET_PATH=/path/to/swe.jsonl
export SWE_LIMIT=2
# HR_API_KEY 和 GONGFENG_RUNTIME_TOKEN 按上面的快速开始提供。
uv run nexus run-batch -g nexus/extensions/configs/generator_swe_template.py -d hyness-smoke.db --max-concurrent 2
```

Hyness 参数留在 agent 轴：`HYNESS_PACKAGES` 是精确包名或绝对 `.tgz` 路径组成的 JSON 数组；
`HYNESS_APP_CONFIG`、`HYNESS_HARNESS_ID`、`HYNESS_APP_CONFIG_OVERLAY` 选择 App/Harness/运行策略。
超时、工作目录、轮数分别沿用 `AGENT_TIMEOUT`、`AGENT_WORKING_DIR`、`AGENT_MAX_TURNS`。
模型族参数按 `agent_config_by_llm` 引用所选 llm 的事实，模型地址、密钥和采样参数仍在 llm 轴。

新的 App 使用 `AGENT_CONFIG=hyness`，要求显式提供 `HYNESS_PACKAGES` 和 `HYNESS_APP_CONFIG`。
未设置 `HYNESS_HARNESS_ID` / `HYNESS_APP_CONFIG_OVERLAY` 时均解析为 null，使用 App 自己的默认
Harness、不叠加 OpenCC training overlay。此通用配置不额外禁止工具或关闭压缩/后台任务。
命名预设则保留各自的已声明策略；更换 App 时不要把不相关的预设当通用模板。
固定实验可以把组合保存为独立 run-part YAML，用 `AGENT_CONFIG=/path/to/my_hyness.yaml` 选取。
每个新 Harness 不需要新增 Python 分支，也不需要扩展配置继承语法。

Hyness App overlay 与 generator 的 per-instance overlay 是两个概念。
明确关闭 App overlay 时设 `HYNESS_APP_CONFIG_OVERLAY=null`；使用 App 自己的默认 Harness 时设
`HYNESS_HARNESS_ID=null`。配置用 `oc.decode` 在合并后解析这两个值，避免 YAML `null` 的删除语义
让 OpenCC 旧默认值重新生效。手写 run-part 时也应使用 `${oc.decode:null}` 表达明确的空值。

`hyness_workbuddy` 是最终发布版的唯一 Hyness WorkBuddy 入口，安装精确版本的引擎和 App，
无需额外提供原生程序路径、package root 或 hash。旧 `_app` 名称已删除；原生 `workbuddy` 是独立的 scaffold。
`hyness_opencc` 默认的 engine/App 版本号不同，App 精确依赖上述引擎；完整六项 deny 已通过
真实 CLI、Nexus 代理与本地模型替身的集成测试。这不代表完成真实 SWE/训练验收。
同类型 App extension 的打包、多 Harness 选择、旧配置兼容、模型/轨迹边界见
[Hyness 接入](../../../docs/how-to/use-agent/hyness.md)。

### codex × claude：`context_window` 在不同 wire 上的验证状态

codex 的 catalog 默认 `context_window = 1000000`（gpt 向），而 codex 会直接把它当 `max_tokens` 发给模型。claude-opus-4-8 / 5 / 5.5 的输出上限都是 128000（litellm 成本表）。各 wire 的实际情况：

| llm 文件 | codex 看到的 `context_window` / `auto_compact_token_limit` | 依据 |
| --- | --- | --- |
| `claude-opus-5.litellm` | 128000 / 115000（llm 文件声明） | 默认 1000000 时真沙盒实测 400，声明后 Resolved=True |
| `claude-opus-48.litellm`、`claude-opus-48-aws_third-aihub2.0.litellm` | 128000 / 115000（llm 文件声明） | 同一个 litellm client、同一条路径，按同根因预防性声明；**未单独跑过 codex 真沙盒**（缺 `APP_ID:APP_KEY` 凭证）|
| `claude-opus-5.openai` | 1000000 / 900000（保持默认） | 默认值下 `codex_toolmode` 真沙盒 Resolved=True，没有被拒；**不要为了"一致"去改它**，那会把压缩阈值从 900k 改成 115k |
| `claude-opus-48.openai` | 1000000 / 900000（保持默认） | 默认值下 `codex_toolmode` 真沙盒跑完 24 次 LLM 请求，**没有任何 4xx / max_tokens 类报错**，所以这条 wire 不会被 1000000 拒。判分 `resolved=0`（`test_update_collision` 失败，和 hy4 的未解格子是同一个用例），与 context_window 无关 |
| `claude-opus-5.vertex_anthropic`、`claude-opus-48.vertex_anthropic` | 1000000 / 900000（保持默认） | **没验证过**（缺 `APP_ID:APP_KEY` 凭证）；若实测被拒，照 litellm 那几份补两行事实即可 |

> 命名提醒：这两个事实叫 `context_window`，是因为它们映射到 codex catalog 的同名字段；但在 codex 里这个值同时决定发出去的输出上限，所以 claude 这里填的是**输出上限** 128000，而不是模型的 1M 输入窗口。

### codex 的 catalog 为什么不跟 codex 官方默认值

我们要的是**好训 HY 模型的轨迹**，不是复刻 codex 官方行为，所以这几个字段刻意和 codex 自带的不同，对齐前端 rollout 实际在跑的 `gpt6_v7` profile（`frontend/scaffold` 的 `runtime_setup._override_tool_mode_in_catalog`）：

| 字段 | 我们的值 | codex 0.160.0 自带 |
| --- | --- | --- |
| `use_responses_lite` | `false` | `true` |
| `context_window` / `max_context_window` | 1000000 / 1000000 | 272000 / 872000 |
| `auto_compact_token_limit` | 900000 | 不带这个字段 |
| `supports_search_tool` | `false` | `true` |
| `supports_image_detail_original` | `false` | `true` |
| `default_reasoning_level` | `max` | `medium` |
| `experimental_supported_tools` | `[]` | `[send_user_message_async, clock]` |
| `supports_parallel_tool_calls` | `true` | 不带这个字段 |
| `tool_mode` | 按文件选 `code_mode_only` / `direct` | `code_mode_only` |

用 `model_catalog_overrides` 而不是整份 `model_catalog_json`：override 是**浅合并**到 codex 自己那条 entry 上，所以上表按我们的来，其余（persona / `model_messages` 整块、`truncation_policy`、`supported_reasoning_levels`…）吃 codex 当前版本自带的，codex 升级提示词不用我们手动跟。

两个已知差异：

- `context_window` 这类字段在 catalog 和 `config.toml` 里各有一份，**必须一起给**——codex 把压缩触发点算成 `min(auto_compact_token_limit, context_window * 90%)`，只调一个不生效。所以 `agent_config.model_context_window` / `auto_compact_token_limit` 和 catalog 里的同名字段都设了，统一由 `CODEX_CONTEXT_WINDOW` / `CODEX_AUTO_COMPACT_LIMIT` 驱动。
- `gpt6_v7` 还会 `pop('base_instructions')`，而浅合并**删不掉键**，只能写成 `base_instructions: null`。实测 codex 0.160.0 接受 null 并照常运行。前端那边是用 `catalog_profile: gpt6_v7` 一个开关表达这一整套的，但 `catalog_profile` / `tool_mode` 两个字段只在 `frontend/scaffold` 那份 codex 扩展上有，main 的 `CodexInput` 没有（实测），所以这里逐条写。等它们合进 main，这一整段可以换成一行。

版本都可以用 `AGENT_VERSION` 覆盖（workbuddy 优先读 `WORKBUDDY_VERSION`）。最后三列里"跟 nexus 代码"的几个没有 `version` 字段，"上游参考"是负责人给的对应版本，代码里查不到。

## 可选的 llm

| 文件（`LLM_CONFIG=`） | 客户端 | 模型 | 必须设的环境变量 |
| --- | --- | --- | --- |
| `gpt-6-sol.openai_responses` | openai_responses | gpt-6-sol | `HR_API_KEY` |
| `gpt-61-sol.openai_responses` | openai_responses | gpt-6.1-sol（aihub-azure 网关） | `HR_API_KEY` |
| `gpt-61-sol-aws_third.openai_responses` | openai_responses | us.openai.gpt-6.1-sol（aihub-aws-third 网关） | `HR_API_KEY` |
| `gpt-61-sol-aws_third-eval.openai_responses` | openai_responses | us.openai.gpt-6.1-sol，**对齐评测大表**（reasoning={effort:xhigh, mode:standard}，不发 summary；key 带 per-run cache_task_id，不钉账户） | `HR_API_KEY` |
| `deepseek-v41-flash.openai_responses` | openai_responses | deepseek-v4.1-flash-expires-on-0910（AIHub `/v1/responses` + provider=deepseek，评测大表 DS 列同一条线） | `HR_API_KEY` |
| `gpt-6-astra.openai_responses` | openai_responses | gpt-6-astra | `HR_API_KEY` |
| `gpt-56-sol.openai_responses` | openai_responses | gpt-5.6-sol，AIHub standard，钉 `?provider=azure` | `HR_API_KEY` |
| `claude-opus-5.openai` / `claude-opus-48.openai` | openai | claude-opus-5 / 4-8 | `HR_API_KEY` |
| `claude-opus-5.litellm` | litellm | `anthropic/claude-opus-5`，AIHub standard（`api.aihub.woa.com/standard`），钉 `?provider=aws_third` | `HR_API_KEY` |
| `claude-opus-48-aws_third-aihub2.0.litellm` | litellm | bedrock invoke，AIHub 2.0 | `HR_API_KEY` |
| `claude-opus-48.litellm` | litellm | bedrock，trpc-gpt-eval | `APP_ID`、`APP_KEY`（或 `OPENAI_API_KEY`） |
| `claude-opus-5.vertex_anthropic` / `claude-opus-48.vertex_anthropic` | vertex_anthropic | 同上 | `APP_ID`、`APP_KEY`（或 `OPENAI_API_KEY`） |
| `hy3.openai` / `hy4-preview.openai` | openai | hy3 / hy4-preview | `HR_API_KEY` |
| `hy3-offline.openai` / `hy4-preview-offline.openai` | openai | 太极离线 | `OPENAI_API_KEY` |
| `hy4-taiji-infer-proxy` | infer_proxy | `OPENAI_MODEL` 指定的 HY4 自训 checkpoint，infer-api | `OPENAI_MODEL`、`OPENAI_API_KEY`；可选 `REASONING_EFFORT`（默认 high） |
| `hunyuanapi-rd.litellm` / `infer-api.litellm` / `infer-proxy.litellm` | litellm | `OPENAI_MODEL` 指定 | `OPENAI_MODEL`，以及各自文件头写的 key |

**client（第二列）不是按模型家族分的，是按「这个具体 wire 怎么连」分的**——查了真实生产 13 份 llm 文件（`swe/scaffold/.../scaffold_agent/configs/llm/`）核对过，client 和 family 是两个独立的轴：

| family | 真实生产里走的 client | 本目录对应的文件 |
| --- | --- | --- |
| gpt | 全部 `openai_responses`（AIHub 2.0 的 `/v1/responses`，这几个模型只有这一条线） | 6 份 `gpt-*.openai_responses.yaml`（另有 `deepseek-v41-flash.openai_responses`，同协议但另一个厂商） |
| claude | `litellm`（AIHub standard + bedrock）、`openai`、`vertex_anthropic` | 对应后缀的文件都有 |
| hunyuan | `infer_proxy`（太极 infer-api）、`openai`（AIHub/太极）、`litellm` | 对应后缀的文件都有 |

几点要知道：

- **AIHub 2.0 必须钉厂商。** 标准协议不钉厂商时，每个请求自己挑厂商，同一条轨迹里可能换家（gpt-5.6-sol 有 azure 和 openai；opus-5 的生产配置记过被路由到 google-vertex）。我们的 llm 文件都在 key 上带 `?provider=`（或 `client_args.gateway_routing.provider`），`validate.py` 见到没钉的 AIHub wire 直接报错。响应头 `x-aihub-provider-model` 会写出实际是哪家，排查时看它。
  - **一站式平台的评测任务是另一回事**，不经过这里的 llm 文件：模型来源选「AI Hub 2.0」时，要在任务的参数配置（`parameter_configuration`）里加 `"distill_plat_api_mode": "passthrough"`，否则平台走 `distillation_plat_standard`，丢掉 provider，每题自动选厂商。克隆任务时这个字段是整块覆盖的，要带上原有全部参数。
- 读 `HR_API_KEY` 的文件只读它，自己补 `?provider=`；**不要**再设 `OPENAI_API_KEY` 指望它生效。
- 写成 `${oc.env:OPENAI_API_KEY,'${oc.env:APP_ID}:...'}` 的那几份，**回退部分会提前解析**：即使设了 `OPENAI_API_KEY`，没设 `APP_ID` 也会报错。
- 反过来，**外部残留的 `OPENAI_API_KEY` 会悄悄盖掉**这些文件里拼好的 key。跑之前 `unset OPENAI_API_KEY`，除非你确实要用它。
- `claude-opus-48-aws_third-aihub2.0.litellm` 的 `THINKING_EFFORT` 只能是 `high` / `medium` / `low`。litellm 会在发请求前就拒绝 `xhigh` 和 `max`（`max` 只给 Opus 4.6），然后反复重试，agent 那边表现成卡住或超时。

## 可选的 runtime

| 文件（`RUNTIME_CONFIG=`） | task | 沙盒 | 超时（request / runner） | 用途 |
| --- | --- | --- | --- | --- |
| `swe_runtime` | swe_task | 2 CPU / 10G，环境变量可调 | 18000 / 默认 | smoke、实验 |
| `swe_prod_runtime` | swe_task | 4 / 16 | 15000 / `RUNNER_TIMEOUT`（默认 14500，生产设 12900）| 生产 rollout **和评测共用** |
| `frontend_runtime` | frontend_task | webenv v4，8 / 16，可调 | 10000 / 默认 | smoke、实验 |
| `frontend_prod_runtime` | frontend_task | webenv v4，8 / 16 | 12900 / 默认 | 生产 rollout |
| `frontend_eval_runtime` | frontend_task | webenv v3，4 / 16 | 13500 / 默认 | 评测 |

两份 SWE runtime（`swe_runtime` 与 `swe_prod_runtime`）的 anti-hack（防作弊判定：检测 agent 是否绕过题目直接改判分/测试文件，命中会覆盖分数）都打开。生产和评测那两份还额外有 `hack_reward_override: 0.0` 和两条 `pre_agent_script`（跑 agent 前清 `/tmp` 和编辑器历史），这是照生产配置加的。

前端生产给 codebuddy / dsh / handy / pi / workbuddy 用的是 node22 镜像，要用 `FRONTEND_IMAGE` 指定，具体镜像名写在前端 runtime 的文件头里。

## generator 模板

两份模板只做"看见实例才知道"的事，其余都来自 yaml：

| | SWE | 前端 |
| --- | --- | --- |
| 数据集 | `DATASET_PATH`，完整实例 jsonl，整条内联进 `task_config.instance` | `DATASET_PATH`，每行一个 `instance_id` + `prompt` |
| 筛选 | `INCLUDE_INSTANCES` / `EXCLUDE_INSTANCES`（逗号分隔 id 或 .json/.jsonl 路径），`SWE_SKIP` / `SWE_LIMIT` | 同上，`FRONTEND_SKIP` / `FRONTEND_LIMIT` |
| 按实例叠加 | 缓存分区 id、`INSTANCE_RESOURCE_OVERRIDE` 里的资源、langfuse 命名 | 缓存分区 id、langfuse 命名、**agent 的 `working_dir` 跟 task 一致** |

- **langfuse 命名**：不设 `LANGFUSE_TRACE_NAME` 时，每条 trace 用 instance_id 命名，`trace_metadata` 里带 instance_id、agent、model，这和生产编译器的做法一致。batch 用 `LANGFUSE_SESSION_ID` 归组，用 `LANGFUSE_TAGS` 打标签，例如 `LANGFUSE_TAGS='["agent=kilo","llm=gpt-6-sol"]'`。
- **SWE 模板和同事的 `tasks/swe/swe_cosmos/generator.py` 是两份相同的代码**，只多了 langfuse 叠加。那边改了要同步过来：直接复制新版，再改 import 路径，加回 langfuse 叠加。

## 环境变量与覆盖关系

### 谁覆盖谁

1. **环境变量 > yaml 里 `oc.env` 的默认值。** 一个变量名会在所有读它的文件里同时生效，没法只改一处。
2. **同一个键被多条轴写时：llm > agent > runtime**（合并顺序 runtime → agent → llm，后者赢）。
3. **generator 按实例叠加的内容最后生效**，yaml 管不到。

### 最容易出问题的四件事

| 问题 | 现象 | 怎么办 |
| --- | --- | --- |
| agent 时限要小于 RPC 超时 | agent 读 `AGENT_TIMEOUT`（默认 14400）；前端 smoke 的 `request_timeout` 默认只有 10000，不设就违反 | 前端显式设 `AGENT_TIMEOUT`，并保证 `AGENT_TIMEOUT < request_timeout`，生产还要小于 `runner.timeout` |
| 各 agent 时限字段名不同 | mini 叫 `wall_time_limit_seconds`（已接到 `AGENT_TIMEOUT`，默认 0 = 不限时）；swe_agent、codeact、terminus2 没有时限字段，只能用 `AGENT_MAX_ITERATIONS` 限步数 | 知道这一点，别以为设了 `AGENT_TIMEOUT` 就都生效 |
| `AGENT_WORKING_DIR` 被两条轴读 | agent 默认 `/testbed`，前端 task 默认 `/workspace` | 前端模板把 agent 的 `working_dir`、questions 模式的 pre-start 脚本和 prompt 全部对齐到 task 的，不用再手设；要换目录就设这个变量（存量 generator 的 `FRONTEND_WORKING_DIR` 不到这一层）。自己写 generator 时照着做 |
| 外部残留的 `OPENAI_API_KEY` | 读它的 llm 文件会用它，盖掉文件里拼好的 key（实测会 401） | 不用它时 `unset OPENAI_API_KEY` |

### 全部环境变量

下表由 `python nexus/extensions/configs/env_table.py --write` 从 yaml 生成，**不要手改**。改了 yaml 之后重新生成，`--check` 能检查是否过期（单元测试也会查）。「写入的键」是这个变量最终写进哪个键，也就是它实际控制什么。generator 自己读的变量（`DATASET_PATH`、`*_LIMIT` 等）不在表里，见上一节。

<!-- env-table:start -->
共 80 个变量。其中 **1 个被多条轴同时读取**（见上一节）：`AGENT_WORKING_DIR`。

| 变量 | 轴 | 写入的键 | 默认值 | 文件数 |
| --- | --- | --- | --- | --- |
| `AGENT_CONTEXT_LIMIT` | agents | `agent_config.context_limit` | `null` | 1 |
| `AGENT_CONTEXT_WINDOW` | agents | `agent_config.context_window` | `200000` | 2 |
| `AGENT_ENABLE_SUMMARIZE` | agents | `agent_config.enable_summarize` | `false` | 1 |
| `AGENT_MAX_ITERATIONS` | agents | `agent_config.max_iterations`, `agent_config.max_step_limit`, `agent_config.max_turns` | `20`, `300`, `500` | 7 |
| `AGENT_MAX_OUTPUT_TOKENS` | agents | `agent_config.max_output_tokens`, `agent_config.max_tokens` | `32768`, `64000` | 3 |
| `AGENT_MAX_TURNS` | agents | `agent_config.max_turns` | `600` | 5 |
| `AGENT_PARSER` | agents | `agent_config.parser_name` | `xml` | 1 |
| `AGENT_THINKING` | agents | `agent_config.thinking` | `max` | 1 |
| `AGENT_TIMEOUT` | agents | `agent_config.timeout`, `agent_config.wall_time_limit_seconds`, `extra_env.OPENAI_TIMEOUT` | `0`, `14400` | 22 |
| `AGENT_TOOL_ERROR_RETRY` | agents | `agent_config.completion_retry_on_tool_error` | `0` | 1 |
| `AGENT_VERSION` | agents | `agent_config.acp_version`, `agent_config.version` | `0.0.23-beta`, `0.1.0-rc.6`, `0.1.14`, `0.160.0` 等 13 种 | 16 |
| `AGENT_WORKING_DIR` | agents | `agent_config.working_dir` | `/testbed` | 21 |
| `AGENT_WORKING_DIR` | runtime | `task_config.working_dir` | `/workspace` | 1 |
| `ANTHROPIC_VERSION` | llms | `no_thinking_request_args.anthropic_version`, `request_args.anthropic_version` | `vertex-2023-10-16` | 2 |
| `APP_ID` | llms | `client_args.api_key` | (必填) | 3 |
| `APP_KEY` | llms | `client_args.api_key` | (必填) | 3 |
| `CACHE_CONTROL_ENABLED` | llms | `client_args.cache_control_enabled` | `true` | 6 |
| `CACHE_CONTROL_STRATEGY` | llms | `client_args.cache_control_strategy` | `claude_code` | 6 |
| `CODEX_AUTO_COMPACT_LIMIT` | agents | `agent_config.auto_compact_token_limit`, `model_catalog_overrides.auto_compact_token_limit` | `900000` | 2 |
| `CODEX_CONTEXT_WINDOW` | agents | `agent_config.model_context_window`, `model_catalog_overrides.context_window`, `model_catalog_overrides.max_context_window` | `1000000` | 2 |
| `CODEX_EXTRA_CONFIG_TOML` | agents | `agent_config.extra_config_toml` | `[features] remote_compaction_v2 = false` | 2 |
| `CODEX_MULTI_AGENT_REASONING_EFFORT` | agents | `model_catalog_overrides.multi_agent_reasoning_effort` | `xhigh` | 2 |
| `CODEX_MULTI_AGENT_VERSION` | agents | `model_catalog_overrides.multi_agent_version` | `v2` | 2 |
| `CODEX_USE_AZURE_PROVIDER` | agents | `agent_config.use_azure_provider` | `false` | 2 |
| `CODEX_WRAP_PLAINTEXT_REASONING` | agents | `agent_config.wrap_plaintext_reasoning_content` | `true` | 2 |
| `DROP_FORCED_TOOL_CHOICE_WITH_THINKING` | llms | `request_args.drop_forced_tool_choice_with_thinking` | `true` | 2 |
| `EVAL_TIMEOUT_SECONDS` | runtime | `task_config.eval_timeout_seconds` | `2000` | 2 |
| `FRONTEND_IMAGE` | runtime | `runtime_provider_config.image` | `mirrors.tencent.com/hunyuan_yanguan/webe…` | 3 |
| `FRONTEND_RUN_EVAL` | runtime | `task_config.run_eval` | `false` | 1 |
| `FRONTEND_TEST_TIMEOUT` | runtime | `task_config.test_timeout` | `600` | 1 |
| `GONGFENG_PURPOSE` | runtime | `runtime_provider_config.purpose` | `prod` | 5 |
| `GONGFENG_RUNTIME_TOKEN` | runtime | `runtime_provider_config.token` | `${oc.env:GONGFENG_TOKEN,''}` | 5 |
| `GONGFENG_TOKEN` | runtime | `runtime_provider_config.token` | `''` | 5 |
| `HAIKU_MODEL` | agents | `agent_config.haiku_model` | `${oc.select:openai_clients.main.config.r…` | 3 |
| `HR_API_KEY` | llms | `client_args.api_key` | (必填) | 13 |
| `HYNESS_APP_CONFIG` | agents | `agent_config.app_config` | (必填), `@tencent/hyness-opencc/apps/opencc/app.y…` | 2 |
| `HYNESS_APP_CONFIG_OVERLAY` | agents | `agent_config.app_config_overlay` | `@tencent/hyness-opencc/apps/opencc/overl…`, `null` | 5 |
| `HYNESS_HARNESS_ID` | agents | `agent_config.harness_id` | `null`, `opencc` | 2 |
| `HYNESS_PACKAGES` | agents | `agent_config.packages` | (必填), `["@tencent/hyness@0.0.1-alpha.2026102012…`, `["@tencent/hyness@0.0.1-scaffold-apps.20…` | 5 |
| `HYNESS_STOP_ON_TOOL_ERROR` | agents | `agent_config.stop_on_tool_error` | `false` | 5 |
| `LANGFUSE_BASE_URL` | runtime | `args.host` | `''`, `http://langfuse-taiji-api.woa.com` | 5 |
| `LANGFUSE_PUBLIC_KEY` | runtime | `args.public_key` | `''`, (必填) | 5 |
| `LANGFUSE_SECRET_KEY` | runtime | `args.secret_key` | `''`, (必填) | 5 |
| `LANGFUSE_SESSION_ID` | runtime | `args.session_id` | `''` | 5 |
| `LANGFUSE_TAGS` | runtime | `args.tags` | `[]` | 5 |
| `LANGFUSE_TRACE_NAME` | runtime | `args.trace_name` | `''` | 5 |
| `LLM_MAX_RETRIES` | llms | `client_args.max_retries` | `15`, `32` | 22 |
| `LLM_TIMEOUT` | llms | `client_args.timeout` | `12000`, `3600`, `900` | 22 |
| `LOG_LLM_USAGE` | llms | `client_args.log_llm_usage` | `true` | 7 |
| `MAX_TOKENS` | llms | `enforced_request_args.max_tokens`, `no_thinking_request_args.max_tokens`, `request_args.max_output_tokens` 等 4 种 | `128000`, `16000`, `262144`, `32768` 等 5 种 | 22 |
| `NPM_REGISTRY` | agents | `agent_config.npm_registry` | `http://mirrors.tencent.com/npm/`, `https://mirrors.tencent.com/npm/` | 15 |
| `OPENAI_API_KEY` | llms | `client_args.api_key` | `${oc.env:APP_ID}:${oc.env:APP_KEY}?provi…`, `${oc.env:TAIJI_TOKEN,''}`, `''` | 7 |
| `OPENAI_MODEL` | llms | `enforced_request_args.model`, `no_thinking_request_args.model`, `request_args.model` | (必填) | 4 |
| `OPENCLAW_ONBOARD_TIMEOUT` | agents | `agent_config.onboard_timeout` | `900` | 1 |
| `RAISE_ON_RUNTIME_ERROR` | runtime | `task_config.raise_on_runtime_error` | `true` | 1 |
| `REASONING_EFFORT` | llms | `chat_template_kwargs.reasoning_effort`, `llm.reasoning_effort`, `reasoning.effort` 等 4 种 | `high`, `max`, `xhigh` | 12 |
| `REASONING_MODE` | llms | `reasoning.mode` | `standard` | 4 |
| `REASONING_SUMMARY` | llms | `reasoning.summary` | `auto`, `detailed` | 6 |
| `RPC_PROTOCOL` | runtime | `runtime_provider_config.rpc_protocol` | `http` | 2 |
| `RUNNER_TIMEOUT` | runtime | `runner.timeout` | `14500` | 1 |
| `RUNTIME_BASE_URL` | runtime | `runtime_provider_config.api_base_url` | `http://hyrl-sandbox.prod.woa.com:8052` | 5 |
| `RUNTIME_CPU` | runtime | `resource_spec.cpu` | `2`, `4`, `8` | 5 |
| `RUNTIME_IMAGE` | runtime | `runtime_provider_config.runtime_image` | `current` | 5 |
| `RUNTIME_MEMORY` | runtime | `resource_spec.memory` | `10`, `16` | 5 |
| `RUNTIME_PROVIDER_NAME` | runtime | `runtime_provider_name` | `gongfeng` | 2 |
| `RUNTIME_REQUEST_TIMEOUT` | runtime | `runtime_provider_config.request_timeout` | `10000`, `13500`, `15000`, `18000` | 5 |
| `RUN_ORACLE` | runtime | `task_config.run_oracle` | `false` | 2 |
| `TAIJI_TOKEN` | llms | `client_args.api_key` | `''` | 1 |
| `THINKING_EFFORT` | llms | `llm.reasoning_effort`, `output_config.effort` | `high`, `max`, `xhigh` | 7 |
| `THINKING_TYPE` | llms | `thinking.type` | `adaptive` | 7 |
| `WORKBUDDY_COMPAT_PERMISSION_PATCH` | agents | `agent_config.compat_permission_mode_patch` | `true` | 1 |
| `WORKBUDDY_EFFORT` | agents | `agent_config.effort` | `xhigh` | 1 |
| `WORKBUDDY_NPM_TOKEN` | agents | `agent_config.npm_token` | (有默认值，已隐藏) | 1 |
| `WORKBUDDY_NPM_USERNAME` | agents | `agent_config.npm_username` | `g_nexus-wb-cli-read` | 1 |
| `WORKBUDDY_PERMISSION_MODE` | agents | `agent_config.permission_mode` | `fullAccess` | 1 |
| `WORKBUDDY_PRESET` | agents | `agent_config.preset` | `code` | 1 |
| `WORKBUDDY_ROLLOUT_MODE` | agents | `agent_config.rollout_mode` | `true` | 1 |
| `WORKBUDDY_SHELL` | agents | `extra_env.SHELL` | `/bin/bash` | 1 |
| `WORKBUDDY_TARBALL` | agents | `agent_config.tarball_path` | `''` | 1 |
| `WORKBUDDY_VERSION` | agents | `agent_config.version` | `${oc.env:AGENT_VERSION,5.9.0}` | 1 |
| `WORKER_VENV` | runtime | `eval.worker_venv`, `main.worker_venv` | `true` | 2 |
<!-- env-table:end -->

## 新增配置

### 一个全新的 topic，从零接入

下面是把一个新 topic 接到这套配置层上的最短路径。**八步里有六步是"选共享文件"，真正要自己写的一般只有 runtime 和 generator 两样**；写的时候按第四、五步的规则来，其余几节是它们的细节展开。

先分清你的 topic 属于哪一种，两条路要写的文件不一样：

| 你的情况 | 要自己写 | 直接复用 |
| --- | --- | --- |
| **已经有了 task 实现**（`nexus.extensions.tasks.<你的task>` 能 import，有 Input 模型），只是要换 scaffold / 模型 / 沙盒跑 | runtime yaml、generator（读你 task 的实例格式） | 全部 agent、全部 llm |
| **连 task 都还没有** | 先按 nexus 常规方式写 task，再回到上面那一行 | 同上 |

**第 1 步：确认 task 的 Input 模型长什么样。** 后面所有 `task_config` 的键都要从这里来——loader 会拿它逐键核对，写错键直接报错（见「每个键都必须有人读」）。先打开你 task 的 `task.py` 找 `class XXXInput(BaseModel)`，记下它声明了哪些字段。

**第 2 步：选 runtime，或者写一份。** 先看 `runtime/` 里现有的五份（`swe_runtime`、`swe_prod_runtime`、`frontend_runtime`、`frontend_prod_runtime`、`frontend_eval_runtime`）能不能直接复用——它们绑死了 `task_name` 和 `task_config`，所以**只在你的 task 是 SWE 或前端时能直接复用**。不是的话要自己写一份，规则见下面「给自己的 topic 写 runtime」。

放哪：`nexus/extensions/tasks/<你的task>/configs/runtime/<名字>.yaml`。放自己 task 目录下的好处是它只对你生效，不会和共享目录里同名的打架。

**第 3 步：选 agent 和 llm。** 这两个轴基本都能直接复用共享的：
- `AGENT_CONFIG` 从 `agents/` 里的 20 份中挑，看「可选的 agent」一节的表。
- `LLM_CONFIG` 从 `llms/` 里 22 份中挑。
- **`agent_config_by_llm` 的族折叠是自动的**，你不用管——选定的 agent 文件里已经写好了它对每个模型族该用什么参数，你只负责选一个 agent 和一个 llm，loader 负责把它们对上。

只有确实需要和共享那份不一样的参数时才新开文件，见下面「给某个 topic 单独调一个 agent」。

**第 4 步：写 runtime 的 `task_config`。** 里面的键**必须是第 1 步那个 Input 模型认识的**。写之前对着模型声明抄，别凭记忆。外壳长这样（最小可跑的形状）：

```yaml
body:
  imports:
    - nexus.extensions.tasks.<你的task>
  task_name: <你的task名>
  task_config:
    <第1步记下的字段>: <值>
  runtime_provider_name: gongfeng
  runtime_provider_config:
    <provider 认识的键>: <值>
  processors: [...]
merge:
  imports: append_unique
source: <你的task>_runtime
```

完整规则和该带哪些键，见「给自己的 topic 写 runtime」。

**第 5 步：写 generator。** 复制模板再改：

```bash
cp nexus/extensions/configs/generator_swe_template.py nexus/extensions/tasks/<你的task>/generator.py
```

改三处：实例读取（`_iter_selected_instances` 现在读的是 SWE 的字段和 `SWE_` 前缀，换成你的）、叠加规则（`INSTANCE_RESOURCE_OVERRIDE` 是 SWE 特有的，不需要就删）、以及每行实例的解析。模板的结构和 `--use` / `--data` / `--set` 的关系见「generator 模板」。

**第 6 步：校验。** 跑之前先在 generate 机器上把三个轴合一遍——这一步不用起沙盒、不发请求，纯粹静态检查：

```bash
uv run python -m nexus.extensions.configs.validate --strict
```

**第 7 步：跑一个最小 batch。**

```bash
export RUNTIME_CONFIG=<你的runtime>
export AGENT_CONFIG=<挑一份>
export LLM_CONFIG=<挑一份>
export DATASET_PATH=/path/to/你的实例.jsonl
export GONGFENG_RUNTIME_TOKEN=... HR_API_KEY=sk-...
uv run nexus run-batch -g nexus/extensions/tasks/<你的task>/generator.py -d out.db --max-concurrent 5
```

**第 8 步：如果 runtime/generator 是你新写的，补测试。** 测试放在 `tests/unit/extensions/configs/`（共享层）或你 task 自己的测试目录。`pytest tests/unit/extensions/configs -q` 或全量 `uv run pytest tests/unit -q` 都要绿。

**最容易踩的两个坑**（细节在下面各节）：

1. **`task_config` 里写了 task Input 模型不认识的键**——会被 `validate` 直接拦下来报错，这是对的，不是误报。要么改键名，要么先在 Input 模型上把字段声明出来。
2. **`extra_cli_args` 里带值的参数写成了空格分隔**（`--foo A B`）——提示词走位置参数的 scaffold 会把后面的任务文本一起吞掉，agent 一个请求都不发。用等号形式（`--foo=A`）。走 stdin 的 scaffold（如 `claude_code`）不受影响，判据见「给某个 topic 单独调一个 agent」第 5 条。

### 给某个 topic 单独调一个 agent

**新增一份 yaml，不要改共享的那份。** 命名 `<scaffold>_<版本>_<topic>_<用途>.yaml`，例如 `claude_code_2.1.267_frontend_eval.yaml`。可以放本目录 `agents/`，也可以放自己 task 的 `configs/agents/`（后者只对自己生效）。必须满足：

1. 外壳是 `body` / `merge` / `source`。`body` 里只放 `imports`、`agent_name`、`agent_config`、`agent_config_by_llm`；`merge.imports` 是 `append_unique`。
2. `agent_config` 和 `agent_config_by_llm` 每一节的键，都必须是这个 agent 的输入模型认识的（节是并进 `agent_config` 的，所以规则一样）。
3. 数值和布尔值用 `oc.decode` 包 `oc.env`；默认值取推荐值，调参靠环境变量，不靠改文件。
4. 文件头写清环境变量，以及**适用的 llm 系列**（例如 codex 那几份只配 GPT-6）。
5. **提示词走位置参数的 scaffold**，`extra_cli_args` 里带值的参数必须用等号写，一个值一项：`--disallowedTools=EnterPlanMode`。写成 `--disallowedTools A B` 时，可变长参数会把后面的任务文本也吞掉，agent 一个请求都不发（codebuddy 实测过，0 次请求）。
   提示词走 stdin 的 scaffold 不受这条约束——例如 `claude_code`，它的提示词由 `_run_claude_streaming` 从子进程 stdin 喂进去，argv 末尾没有东西可吞（见 `nexus/extensions/agents/claude_code/runtime_setup.py` 里 `The prompt is NOT appended to argv` 那段注释），所以 `claude_code_rollout.yaml` 用可变长形式写 `--disallowedTools WebSearch WebFetch` 是安全的。
   **拿不准就查那个 agent 的 `runtime_setup.py`**：提示词进 argv 的用等号形式，进 stdin 的两种都行。
6. **凡是随模型族变的参数，写进 `agent_config_by_llm`，不要写死在 `agent_config` 里**，族名只能是 `gpt` / `claude` / `hunyuan`。**先按「参数的作用域和归属」判断清楚它到底是不是真的随家族变**——同一家族换一份具体 llm 文件，这个值会不会不一样？会的话它是 llm-wire 事实，不是家族统一值，但引用方式一样：

   ```yaml
   agent_config_by_llm:
     gpt:
       use_azure_provider: ${llm:use_azure_provider?} # 不是家族统一值，来自选中的那份 llm 文件
       # 同理：随"这条 wire 有没有原生 reasoning_items"变，是协议的属性，不是家族的
       wrap_plaintext_reasoning_content: ${llm:wrap_plaintext_reasoning_content?}
     claude:
       thinking_preset: bedrock-claude
       no_thinking_request_args: ${llm:no_thinking_request_args?}
       extra_env:
         ANTHROPIC_MODEL: '${llm:anthropic_model}'
   ```

   `${llm:<键>}` 取 llm 文件 `llm:` 下的事实；带 `?` 表示这个事实可以没有，没有时整个键不写进去，scaffold 自己的默认值继续生效。没写的族什么也拿不到。自己不声明这些字段的 scaffold（`codeact`、`hermes`、`openclaw`、`swe_agent`、`terminus2`）整节都不要写。

### 接入一个新模型

新增 `llms/<模型>.<客户端>.yaml`。只写两样：`openai_clients`，和 `llm:` 下这个模型的事实——`family`（`gpt` / `claude` / `hunyuan`）、`anthropic_model`、`no_thinking_request_args`，以及（如果适用）`use_azure_provider`。`llm:` 允许的事实名是 `validate.py` 里 `LLM_FACT_KEYS` 的白名单，新事实要先加进那里，见「参数的作用域和归属」最后一条。**llm 文件不写 `agent_config`**，也不提任何 scaffold 的名字；哪个 scaffold 怎么用这些事实，由 scaffold 自己的 `agent_config_by_llm` 决定。scaffold 自己会带、而这条路径不收的参数（例如 bedrock 下的 `reasoning_effort`、`temperature`），用 `additional_drop_params` 丢掉，参照 `claude-opus-48-aws_third-aihub2.0.litellm.yaml`。

#### `request_args` 里的新键该放顶层还是 `extra_body`

接新llm wire时踩出来的一类坑，**和具体模型无关，是所有 topic 接新模型时都会碰到的**，所以单独写一条判断流程：

1. **先看是不是 OpenAI 官方 Responses API 的标准字段**（`model`、`input`、`store`、`stream`、`max_output_tokens`、`reasoning`、`tools`、`tool_choice`、`parallel_tool_calls`、`instructions`、`metadata`、`previous_response_id`、`truncation`、`user`、`text`、`include`）——是标准字段，放 `request_args` 顶层，装的 SDK 一定认。
2. **不是标准字段**（provider 自己扩展的，比如这次的 `thinking`）——**不要因为"生产的配置/别的 topic 的配置这么写"就直接抄**，先在这台机器装的 SDK 上直接试一遍：
   ```python
   await client.responses.create(model=..., 你要加的字段=...)
   ```
   跑通就放顶层；报 `TypeError: ... got an unexpected keyword argument '<字段>'`，说明这版 SDK 的 typed 方法签名里没有这个参数，挪进 `extra_body: {<字段>: <值>}`——这是 SDK 专门留给"官方没收录但后端认"的字段的逃生口，不做类型检查，原样转发进请求体（litellm 的 `additional_drop_params`/`allowed_openai_params` 等参数也走此路径，见 `claude-opus-5.litellm.yaml`）。
3. **为什么"抄别人已经在跑的配置"这次不保险**：生产某份配置把非标准字段写在 `request_args` 顶层，大概率是在另一个 SDK 版本上调通的——各 topic 自己的检出锁的包版本不一定和这边一致。backend 本身很可能两种写法都收（实测 curl 直接打，顶层和 `extra_body` 两种写法服务端都认），**但本地 SDK 的类型检查在请求发出去之前就会拦下来**——这种失败不报 HTTP 错误，`capture_raw_io` 也抓不到任何东西（纯 Python `TypeError` 没有 `.body` 属性可捕获），症状就是日志里一句没有任何细节的 `"OpenAI Responses request failed"`，Langfuse 的 trace/observation 也是空的。
4. **查这类"什么细节都没有"的失败，第一反应应该是绕开 nexus、直接拿同一个 SDK 复现**，而不是在 nexus 日志和 Langfuse 里继续翻——这次翻了很久才翻到这条路。复现脚本就是上面第 2 步那三行，不需要起沙盒。
5. 这条流程不是 `claude-opus-5` 专属的：**给自己 topic 接一个新模型、或者直接复用别的 topic（哪怕是生产）已经在跑的 llm 配置时，只要 `request_args` 里有一个不在上面标准字段列表里的键，都先过一遍这个流程。**

#### 复用 harbor_bridge 的 llm 配置

`nexus/extensions/harbor_bridge/configs/llm/*.yaml` 和本目录是**同一种格式**，实测可以原样放进 `llms/` 直接用：把 `taiji-hy4-preview.yaml` 原封不动复制进来，`validate` 45/45 份文件、grid 全组合（那是当时的文件集合，1995 个；现在是 47 份文件 / 2200 个）全过，`RunInput` 校验通过，连它那条跨层引用的 `${oc.select:openai_clients.main.config.request_args.max_tokens,null}` 都能在合并后的树里正确解析。它用的 `enforced_request_args` 也不是 harbor 私有的，是 nexus 核心功能（`nexus/runner/single.py`、`nexus/gateway/client.py` 都消费它）。

但原样复制之前要改三处：

1. **改名**，本目录是 `<模型>.<客户端>.yaml`，harbor 那边没有客户端后缀。
2. **key 换成 `HR_API_KEY`**，harbor 那 11 份全都是 `api_key__env: OPENAI_API_KEY`，和同事 MR !1609 统一后的约定相反。
3. **`agent_config` 改成 `llm:`**，harbor 那边还是旧形状（直接往 `agent_config` 里写 `thinking_preset`）；按上面「接入一个新模型」改。

另外先查重：`taiji-hy4-preview.yaml` 和本目录已有的 `hy4-preview-offline.openai.yaml` 是同一个端点（`http://api.taiji.woa.com/openapi/inner/v2`）同一个模型（`hy4-preview-offline`），只是客户端一个 `infer_proxy` 一个 `openai`。复制进来会变成两份打架的配置。

#### 反过来：这边的 llm/agent 文件能直接给 harbor_bridge 用吗

**不完全能，是单向的——这个不对称是真的，查过 harbor_bridge 实际的加载路径才确认的，不是猜的。**

harbor_bridge 自己的 `generator.py` 不读 `AGENT_CONFIG` / `LLM_CONFIG` / `RUNTIME_CONFIG`：它按 `AGENT_NAME`、`OPENAI_MODEL` 这类环境变量在 Python 里**直接拼 `RunInput`**，`--use` 只用来叠加 policy/runtime 这类附加片段，从来不经过本目录的 `nexus.extensions.configs.load_config.load_config()`。

两个方向因此不对称：
- **harbor 的 llm 文件 → 这边**：安全。harbor 那边是旧形状（`agent_config` 直接写死值，没有 `llm:` / `agent_config_by_llm`），不管走哪条合并路径都一样，上面「改三处」之后能直接用。
- **这边的 llm/agent 文件 → harbor 的原生加载路径**：**不安全**。只要这份文件用了 `llm:` 或 `agent_config_by_llm`（现在所有 llm 文件、以及任何声明了家族相关字段的 agent 文件都用了），这两个键只有 `load_config()` 里的 `_apply_llm_family` 会折叠、替换 `${llm:...}`；走 harbor 原生路径（或者任何不经过 `load_config()` 的 `--use`/手拼 `RunInput`），这两个键会原样留在合并结果里，而 `RunInput`/`AgentInput` 都没有这两个字段，`extra='ignore'` 会把它们**静默丢掉**——家族相关的行为（`thinking_preset`、`ANTHROPIC_MODEL`、`wrap_plaintext_reasoning_content`、`use_azure_provider`……）整个不生效，而且不会报错。这正是这整套配置层想防的那类坑，只是发生在了"被另一个系统绕过 loader 直接读文件"这个场景里。

真要让 harbor_bridge 复用这边的文件并且行为对，它需要调用这边的 `load_config()`（可以传 `task_dir` 指向 harbor 自己的目录），而不是走它现在这条"环境变量 + Python 拼 RunInput + `--use` 叠加片段"的路径；或者只复用那 5 个不声明任何家族相关字段的 scaffold（`codeact`、`hermes`、`openclaw`、`swe_agent`、`terminus2`）和不需要家族折叠的 llm 文件——这些文件没有 `agent_config_by_llm`，走哪条路径结果都一样。

### 给自己的 topic 写 runtime

可以用自己的 schema，**一个用途一份**：生产、评测参数不同就各写一份，用 `RUNTIME_CONFIG` 选，不要用一份文件加一堆开关。文件头注明对应哪条链路、值抄自哪里。最小合同：外壳 `body` / `merge` / `source`；写全 `task_name`、`task_config`、`runtime_provider_*`、`processors`；`task_config` 和 `runtime_provider_config` 的键必须是 task 和 provider 认识的（`validate.py` 会查）。

### 接入一个新 task

1. 复制模板到自己的 task 目录：`cp nexus/extensions/configs/generator_swe_template.py nexus/extensions/tasks/<你的task>/generator.py`（前端类任务复制前端模板）。
2. 改实例读取：`_iter_selected_instances` 读的是 SWE 的字段和 `SWE_` 前缀，换成你的。
3. 写自己的 runtime，文件名带 task 名，放进 task 的 `configs/runtime/` 或本目录 `runtime/`。
4. 改叠加：`INSTANCE_RESOURCE_OVERRIDE` 是 SWE 特有的，不需要就删；缓存分区的两个函数按模型情况留或删。
5. 设三个环境变量启动。agent 和 llm 直接选共享的。

## 测试

```bash
uv run pytest tests/unit/extensions/configs -q
```

覆盖查找顺序、合并顺序、每份 yaml 的 schema、环境变量表是否最新，以及几处实测踩过的坑：workbuddy 的权限模式能否真的送达、codebuddy 禁用工具的写法、opus-48 的 effort 取值、mini 的时限、生产和评测 runtime 的差异、前端模板的 `working_dir` 和 langfuse 命名，以及缺可选依赖时校验脚本的降级行为。

**两道专门防"值写了但没生效"的守卫**，都做过变异验证（把 bug 改回去必须变红）：

- `test_no_shared_agent_profile_silently_diverges_from_production`：把 20 份共享 agent profile 映射到 `origin/main:nexus/extensions/tasks/swe/swe_cosmos/configs/agents/` 下对应的生产 profile（claude_code / codebuddy / codex 在本层拆成 eval/rollout、codemode/toolmode 变体，多对一），逐个比对两边都有的 `${oc.env:NAME,DEFAULT}`。**允许的差异写成显式表并带理由**（`AGENT_VERSION` 本层钉更新的版本、`AGENT_TIMEOUT` 归 runtime 轴、`AGENT_WORKING_DIR` 前端不用 `/testbed`），没列进去的就是 finding。**改默认值时如果这道变红，先确认生产是不是也该跟着改，而不是直接往允许表里加一行。**
- `test_workbuddy_permission_mode_is_actually_delivered` / `test_workbuddy_permission_default_matches_production`：wb 5.9.0 的 CLI 没有 `--permission-mode`，`permission_mode` 只能靠 compat 补丁改写压缩产物才送得出去，所以这两个键必须成对钉住；后者直接从生产配置读默认值做比对，不复述常量。

这两道都要能跑 `git show origin/main:...`；拿不到远端时会 `skip` 而不是假通过。

## 已知限制

- **GPT 工具调用的 `strict` 字段，已经在 client 代码里修过，这份配置层不用再管。** AIHub 2.0 的 `/v1/responses` 对缺 `strict` 的 tool 会强制补 `strict: true`，连带把 `required` 展开成全部字段，模型因此学出"倾向于填满所有参数"的坏习惯（哪怕是可选参数也塞空字符串）。修法在 `nexus/extensions/openai_clients/openai_responses.py` 的 `_with_default_function_strict`：凡是 tool 定义里没写 `strict`，发请求前客户端直接补成 `false`（OpenAI 文档里的默认值），不是 AIHub 默认的 `true`。这个修复对**所有**走 `openai_responses` 客户端的请求生效，和选哪份 `llms/*.openai_responses.yaml` 无关，不需要在 llm 文件里加任何字段。
- **本目录对 task 代码零 import 依赖，已核实。** `load_config.py`、`validate.py`、`env_table.py`、两份 generator 模板只 import `nexus.runner.*` 这类核心基建；yaml 里的 `nexus.extensions.tasks.swe` / `nexus.extensions.tasks.frontend` 只是模块名字符串，交给 nexus 自己的 import 机制在运行时解析（见 `grep -n '^import\|^from' nexus/extensions/configs/*.py` 和各 runtime yaml 的 `imports:`）。所以这份配置层合进 main 之后，哪个分支跑它、`nexus.extensions.tasks.frontend` 底下是 main 的实现还是那个分支自己改过的实现，和这份配置层本身无关——前端分支 merge/rebase 一下 main 就能直接拿到这套基建，同时继续用自己分支上的 task 代码跑。能跑通，不等于字段对齐：下一条就是这件事的另一半。
- **前端生产 / 评测的 `task_config` 有 5 个字段，只有 `frontend/scaffold` 检出认。** 生产用的是那份检出里的 `frontend_task`，它的 `FrontendTaskInput` 比 main 多声明 5 个字段。`validate.py` / `_require_consumers` 是**按当前检出**核对的：同一份 yaml 在 `frontend/scaffold` 上这些键是被认的，在 main 上会被判为「会被静默丢弃」而报错（这是对的，不是误报）。

  | 字段 | 类型 / 默认 | 作用 | 生产取值 | 该放哪 |
  | --- | --- | --- | --- | --- |
  | `cache_task_id` | `str` / `''` | 32 位 cache 作用域 id，嵌进 COS 归档 key；`run_id == cache_task_id[:16]` | 每个实例每一轮重新生成一个 `uuid4().hex`；aws_third 网关上还要从 client_args 里剔掉 | **generator 的 per-instance overlay**（每实例都不同，不是常量，不能进 yaml 也不进 EXTRA） |
  | `tar_exclude_paths` | `list[str]` / `[]` | 打包归档时跳过的**路径前缀**（精确匹配或目录前缀，不是 glob） | 默认空；显式导出 `FRONTEND_TAR_EXCLUDE_PATHS` 才注入 | `FRONTEND_TASK_CONFIG_EXTRA` |
  | `tar_exclude_extensions` | `list[str]` / `[]` | 后缀黑名单，大小写不敏感，和上一条是「或」关系 | 默认空 | `FRONTEND_TASK_CONFIG_EXTRA` |
  | `task_subdir` | `str` / `'environment'` | `task_dir` 下要上传到 `working_dir` 的子目录；空串表示直接上传 `task_dir/` | — | `FRONTEND_TASK_CONFIG_EXTRA` |
  | `is_identity_task` | `bool` / `False` | — | — | `FRONTEND_TASK_CONFIG_EXTRA` |

  **怎么在不改 yaml 的前提下写这些值**：前端 generator 模板认一个环境变量 `FRONTEND_TASK_CONFIG_EXTRA`（JSON 对象），合并进每个实例的 `task_config`：

  ```bash
  FRONTEND_TASK_CONFIG_EXTRA='{"tar_exclude_extensions": [".log", ".map"], "task_subdir": ""}' \
    RUNTIME_CONFIG=frontend_prod_runtime AGENT_CONFIG=kilo LLM_CONFIG=gpt-6-sol.openai_responses \
    nexus run-batch -g .../generator_frontend_template.py ...
  ```

  注入后会立刻对**当前检出**跑一遍 `_require_consumers`：在 `frontend/scaffold` 上通过；在 main 上因为 `FrontendTaskInput` 没声明这些字段，**在 generate 阶段就报错**，而不是跑完才发现设置被忽略。所以 yaml 里仍然不写这几个键（写了 main 上整份文件校验不过），要用时由环境注入。已经声明的字段（如 `test_timeout`）两边都能注入。
  **跑评测题集（turns / assets / repo）**：默认一行一个 `prompt`（rollout 用）。设 `FRONTEND_QUESTIONS=<题集 jsonl>` 后切成评测模式，转换逻辑逐字取自 `frontend/eval/runs/humaneval_*/generator.py`：`turns` 拼成 prompt（多轮加"第 N 轮"标题）、`assets` 打包成 pre-start 脚本解到 `/workspace`（分块写 base64，避开 128KB 的 E2BIG）、`repo` 按 commit 检出；附件与仓库说明写进 prompt 抬头。

  ```bash
  FRONTEND_QUESTIONS=$PWD/frontend/eval/questions/benchmark_0928-0930/questions_0928-0930.jsonl \
    RUNTIME_CONFIG=frontend_eval_runtime AGENT_CONFIG=codebuddy_eval ... nexus run-batch -g .../generator_frontend_template.py ...
  ```
  `pre_start_scripts` 是 **agent** 输入（codebuddy / opencode）上的字段，不是 `frontend_task` 的，所以它写进 `agent_config`；放 `task_config` 会在没声明这个字段的检出上被静默丢掉。
  **每道题重复采样**：`EVAL_TIMES`（默认 1）表示每个实例跑几次，做法照 `nexus/extensions/tasks/hitl/recipe/terminal/generator.py`：多于 1 次时 `run_id` 变成 `<instance_id>__try<N>`，各占一行；每轮的 cache id 也按尝试次数重新算，不然各轮会共用第一轮的缓存作用域。

  ```bash
  EVAL_TIMES=3 RUNTIME_CONFIG=frontend_runtime AGENT_CONFIG=kilo ... nexus run-batch -g .../generator_frontend_template.py ...
  ```
  **按脚手架换镜像**：生产给 codebuddy / dsh / handy / pi / workbuddy 用 node22 镜像，其余用默认镜像；一次运行只能设一个 `FRONTEND_IMAGE`，所以按脚手架区分要用 `FRONTEND_IMAGE_BY_AGENT`（JSON `{"<agent_name>": "<image>"}`），只覆盖列出的脚手架，其余沿用 runtime 的镜像：

  ```bash
  FRONTEND_IMAGE_BY_AGENT='{"codebuddy": "mirrors.tencent.com/g_kechizhang/webenv.claudecode.testset-node22:20260606", "handy": "..."}' \
    RUNTIME_CONFIG=frontend_prod_runtime AGENT_CONFIG=codebuddy ... nexus run-batch -g .../generator_frontend_template.py ...
  ```
  **`cache_task_id`：一条轨迹一个 id，同时上线和进 metadata**（iwiki 4031597247）。AIHub 2.0 已支持按 `cache_task_id` 动态调度账号，**`account_id2` 已下线，不要再指定**（本层原先的 `FRONTEND_ACCOUNT_IDS2` 账户池也已删除）。规则：

  - 同一条轨迹的所有请求用**同一个** id，不同轨迹用**不同**的；id 必须放在 key 的 `?` 查询串里（`key?cache_task_id=...`），放别处无效。
  - 这个 id 还要写进 nexus output 的 **`metadata.cache_task_id`**：HyWay 读它交给 CoT hack（rethink），因为加密推理只能由产出它的账号解码，rollout 与 hack 用不同的 id 就解不出思考。
  - 两件事由 generator **一次决定**（`cache_id.py`，SWE 与前端两个模板共用）。不能让客户端自己派生：`cache_task_id_from_run_id` 派生的 id 生成器看不到，metadata 就没有可记录的；而写进 key 查询串的 id 会被派生值**静默替换**（`normalize_cache_task_id` 会丢掉已有的 `cache_task_id` 参数），metadata 会记下一个线上从没发过的 id。所以对接受 `cache_task_id` 参数的客户端（`openai_responses`、`litellm`）显式设 `client_args.cache_task_id`（runner 只用 `setdefault`，显式值优先）；其余客户端直接写进 key 查询串。
  - 只有线上真带了 id 才记进 metadata（key 里没有 `?` 查询串的非 AIHub 裸 key，如 api.taiji.woa.com 离线路由，不会被改写）。
  - 实测（2026-10-16，gpt-6.1-sol aws_third）：只带 per-session `cache_task_id`、不带任何账号信息，加密推理回放 0/9 失败；既不带 id 也不钉账号 8/10 失败。
  前端评测的 21 题（turns / assets / repo）已由 `FRONTEND_QUESTIONS` 表达，见上。
- **本目录的 agents / llms 和 `tasks/swe/swe_cosmos/configs/` 里同名文件是两份**。同名的 llm 已经和那边的最新版同步；agents 里有几份按负责人的推荐值改过版本。
- **token 会写进 batch DB**：runtime 的 token 现在和 `swecosmos_runtime.yaml` 一样用 `oc.env`（这样 `GONGFENG_TOKEN` 回退才生效），不再用 `__env`。
