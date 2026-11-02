# Context Agent

本 RL 分支只保留 `context_agent_general` / `context_agent_general_task` 和
`context_agent_enterprisebench_native_task`，以及它们使用的共享评分与工具实现。
旧预加载文档 Agent、SFT 评测和 Harbor Enterprise 已移除。

支持：

- HTTP(S) URL 输入；
- 本地 TaskFiles 输入；
- 同一条任务混合 URL 和 TaskFiles；
- 单轮文件问答；
- CLB、SurgeOODV21 等多轮 `context_dialogue`；
- rule、reference 和 rubric 三类 Judge；
- GDPval 交付物持久化。

## 当前主线集成范围

ContextAgent 扩展提供 General / EnterpriseBench native、AgentIF 约束评分、工具名错误策略、
坏模式检测、MCP 工具搜索和 Judge 回放。 此次集成不包含 CLB 示范回放或 HyTuner 主仓库的奖励塑形。

公共已有代码的改动仅涉及 HyTuner client 的思考字段与 token 统计。
终止原因由 ContextAgent 任务通过 tracing 上报，HyTuner runner 保持主线实现。
Judge 请求捕获由 Enterprise 扩展内部的任务级 SDK 包装层实现。 当前主线的 Gateway、轨迹存储、视频输入、原生 Responses、
重试和 tracing 实现继续使用现有版本。 必填参数提示仅由 ContextAgent 工具执行器添加。

Judge 捕获只在任务拥有的 client 副本上开启，保留区域响应头和每次重试的请求记录。
评分和审计信息通过任务输出返回；HyTuner 专用汇总指标是否展示取决于训练端的消费逻辑。

## 可选 bad-pattern 检测与扣分

以下任务支持同一份 `task_config.tool_call_bad_pattern_policy`：
`context_agent_general_task`、`context_agent_enterprisebench_native_task`。
默认 `null`，不检测、不改变原有奖励。配置放在 Nexus RunInput / 数据集行的
`task_config` 中，不是 HyTuner 训练 YAML 的顶层字段。

```yaml
task_config:
  tool_call_bad_pattern_policy:
    id: context-agent-think-v1
    penalty: 0.5
    reward_threshold: 0.5
    max_hits: 3
    ignore_incomplete_final_assistant: true
    patterns:
      think_protocol_leak_hy:
        enabled: true
        penalize: true
      think_protocol_leak_all_family:
        enabled: true
        penalize: false
```

规则复用 [Harbor bad-pattern-v2](../../harbor_bridge/configs/policies/bad-pattern-v2.yaml)，
只运行 `patterns` 中启用的规则；`penalize: false` 只记录。
原分达到 `reward_threshold` 后才扣 `penalty`；规则可覆盖这两个值。
同一条轨迹多次命中、多规则命中不累加，取符合条件的最大扣分，最低到零。
`max_hits` 只限制明细数量，不限制统计或扣分。

检测使用 Agent 的 `main_trajectory`（包含正文、reasoning、工具调用和结果）以及
`tool_schemas`。`context_agent_general` 输出实际使用的工具 schema；
Enterprise 使用其他 Agent 时也应提供这两个字段。缺少轨迹或检测器异常时
保留原分并记录错误，不伪造“检测通过”。动态工具规则的准确性依赖工具注册信息完整度。
Judge 的对话不参与检测；不会对原始 chat-template 字符串或 `raw_response` 扫描。
正常思考分隔符已由 parser 消耗，检测的是解析后字段中的残留标签。

扣分只修改任务 `score`，不修改 Agent 的终止原因、General 的 `answer_score` /
`answer_judge` / `constraint_judge`，也不修改 Enterprise 结算前的 `reward` /
`rewards` / `raw_reward` / criteria。现有工具名错误清零先执行，再结算 bad-pattern。
HyTuner runner 已识别检测明细，在二值化 reward policy 下也会保留扣分。

审计信息位于 `extra_fields.tool_call_bad_pattern`，包含原分、扣分、最终分和命中明细；
监控指标写入 `extra_fields.reward_extra_info.extra_reward_metrics`，与现有
AgentIF 轴分数及工具错误计数合并。启用规则的命中率会同时报告未命中的零值。
提前返回的 General 任务（例如截断、没有最终答案）也会检测；其零分不变。
截断末轮的协议泄漏遵循 `ignore_incomplete_final_assistant` 豁免规则。

此接入不会自动启用任何规则，也不会修改已有训练脚本或数据集。
轨迹截断由任务层直接跳过 Judge 并给零分，不依赖 bad-pattern 规则。

## 通用与 Native EnterpriseBench 的终止原因

`context_agent_general_task` 和 `context_agent_enterprisebench_native_task` 共用
Agent 退出状态映射。正常完成映射为 `ENV_DONE`，截断、轮数耗尽等状态保留对应分类。
任务结束时通过 task span 的 `agent.termination_reason` 上报最终结果，普通和 Gateway
两条 runner 路径使用现有 tracing 消费逻辑读取，不需要修改公共 runner：

| 原因 | 含义 |
| --- | --- |
| `ENV_DONE` | 正常完成；得分可以为 0 |
| `MAX_STEPS` | 工具轮数耗尽；强制收尾得到答案后仍保留此原因 |
| `TRUNCATION` | Agent 上下文、总响应预算或单次模型输出达到长度限制 |
| `FORMAT_ERROR` | Agent 输出格式不合法或最终回答为空 |
| `INITIALIZATION_ERROR` | 非超时的环境初始化失败 |
| `ENV_ERROR` | 运行中的非超时环境故障 |
| `ENV_TIMEOUT` | 环境启动超时或 Runtime RPC 超时 |
| `REWARD_ERROR` | Judge / verifier 未能产生有效评估 |
| `TASK_TIMEOUT` | Agent 执行阶段或整个 task 超时 |
| `UNHANDLED_ERROR` | provider 拒绝或其他未捕获异常 |

`MAX_ITERATIONS` 兼容映射为 `MAX_STEPS`，`PROVIDER_ERROR` 映射为 `UNHANDLED_ERROR`。
Agent 的 `TRUNCATION` 和 `OUTPUT_LENGTH_EXCEEDED` 在 task output 和 HyTuner runner
输出中统一为 `TRUNCATION`，原始 Agent `exit_status` 保留各自的值。
`INITIALIZATION_ERROR`、`ENV_ERROR` 和 `ENV_TIMEOUT` 分别保留，`TIMEOUT` 兼容映射为
`TASK_TIMEOUT`。原始 Agent `exit_status` 和 runner
`error.type/message/traceback` 保留，异常类本身没有合并。
ContextAgent 任务捕获并上报异常分类后原样抛出异常：`EnterpriseRuntimeStartupTimeoutError`
和 `EnterpriseRuntimeRPCError` 映射为 `ENV_TIMEOUT`，`EnvironmentInitError` 映射为
`INITIALIZATION_ERROR`。外层 runner 的超时取消上报为 `TASK_TIMEOUT`，其他取消保持
`UNHANDLED_ERROR` 并继续传播。任务开始前的框架异常仍由原 runner 兜底。
环境超时维持环境故障的默认 mask 行为，不纳入 Agent / task 的自适应超时统计。

模型的轮数、格式错误和长度限制分别保留；两类长度限制统一跳过 Judge、给零分并标记 overlong。
环境故障、环境超时、评估故障和任务超时仍属于默认 mask 集合；Agent / task 超时参与
`TASK_TIMEOUT + UNHANDLED_ERROR` 自适应 mask 统计。若为这两个 task 单独配置了
`TIMEOUT` 或 `OUTPUT_LENGTH_EXCEEDED` 的 reward postprocess
或 additional mask 规则，需要改用上表对应的新标签；`TRUNCATION` 规则同时覆盖两类长度限制。

Judge 跳过条件保持原行为：两种长度限制及格式错误直接跳过，轮数耗尽且没有最终答案时
跳过。有最终答案的 `MAX_STEPS` 仍可评分。Judge 前抛出的异常会直接终止 task。

核心文件：

```text
nexus/extensions/tasks/context_agent/
├── base.yaml
├── remote_taskfiles.yaml
├── configs/
│   ├── examples/
│   └── remote/
└── eval/
```

## 1. 默认运行参数

`base.yaml` 是所有 `configs/*.yaml` 共用的 Runtime 和模型配置：

```text
Runtime: Gongfeng / suanli
Policy:  gpt-6-astra, reasoning effort xhigh
Judge:   gpt-5.6-sol, reasoning effort high
Tools:   curl, bash, python, file_search_msearch,
         file_search_mclick, grep, read_lines
```

这是正式评测参数。快速 smoke 测试不要修改 `base.yaml`，而是临时设置：

```bash
export CTXAGENT_POLICY_REASONING_EFFORT=low
export CTXAGENT_JUDGE_REASONING_EFFORT=low
```

不设置这两个环境变量时，所有数据集配置都使用正式的 `xhigh` / `high`。

### Claude Opus 4.8 基座

`base_opus4-8.yaml` 是可与同一批本地或远程 Dataset 组合使用的
Claude Opus 4.8 基座：

```text
Policy:  anthropic.claude-opus-4-8, adaptive thinking, effort xhigh
Judge:   gpt-5.6-sol, reasoning effort high
```

它使用 `vertex` client 的 Anthropic protocol。每轮原始 Anthropic
`content` blocks 都保存在 `provider_specific_fields.anthropic_content`，
下一轮会按原始顺序回传，因此 `thinking_blocks` 及其 `signature` 不会丢失。
同时启用 Anthropic ephemeral prompt caching，在最近一条 user/tool
消息和最后一个工具定义处设置 cache breakpoint。首轮通常产生 cache
creation tokens，后续工具轮可复用稳定前缀并产生 cache read tokens。

主模型凭据沿用 `base.yaml` 的 `CTXAGENT_POLICY_API_KEY`，不引入新的环境
变量名。使用 Opus 4.8 时，该变量的值应为 AIHub 接受的完整 Bearer
token，需要时包含 `?provider=aws_third`：

```bash
export CTXAGENT_POLICY_API_KEY='<opus-api-key>?provider=aws_third'
```

运行本地 Dataset：

```bash
uv run nexus run-single \
  --use nexus/extensions/tasks/context_agent/base_opus4-8.yaml \
  --data /apdcephfs_gy6/share_303171455/yufeihuang/align/ctx_agent_data/nexus_context_agent_eval \
  --split aalcr \
  --id aalcr-00000 \
  -o outputs/context_agent/aalcr-opus4-8-local/result.json
```

运行远程 Dataset：

```bash
uv run nexus run-single \
  --use nexus/extensions/tasks/context_agent/base_opus4-8.yaml \
  --data yufeihuang/context_agent_aalcr@v1 \
  --split aalcr \
  --id aalcr-00000 \
  -o outputs/context_agent/aalcr-opus4-8-remote/result.json
```

`base_opus4-8.yaml` 已包含远程 ZIP TaskFiles 所需的 WeData、Dataset IO
和 COS egress。对于带 HTTP(S) 文档的数据，item 自身的精确 egress
配置仍会按正常 compose 规则生效。

默认任务参数：

```yaml
task_config:
  evaluate: true
  max_tool_rounds: 999
  max_parallel_calls: 0
  file_search_top_k: 5
  file_search_chunk_lines: 50
  python_timeout: 120
  curl_timeout: 180
```

运行前设置凭据：

```bash
export CTXAGENT_GONGFENG_TOKEN='<gongfeng-token>'
export CTXAGENT_POLICY_API_KEY='<policy-api-key>'
export CTXAGENT_JUDGE_API_KEY='<judge-api-key>'
```

不要将真实凭据提交到仓库。

## 2. 统一 Dataset

默认配置使用本地统一 Dataset：

```text
/apdcephfs_gy6/share_303171455/yufeihuang/align/ctx_agent_data/nexus_context_agent_eval
```

Dataset item 使用统一的数据结构，但文件传输方式按 URI 自动区分：

### URL 文件

```yaml
context_files:
  - name: document.pdf
    uri: https://example.com/document.pdf
```

URL 原样保留，item 不带 `files` payload。Runtime 中由 Agent 调用 `curl`
下载，不经过 TaskFiles 上传。

每个 URL item 自带精确的 Gongfeng `host:port` egress allowlist。

### 本地文件

```yaml
files:
  type: directory
  path: files
body:
  task_config:
    context_files:
      - name: doc_1.md
        uri: task-files://doc_1.md
```

文件由 Nexus TaskFiles materialize 到 Runtime：

```text
./workspace/doc_1.md
```

### 混合输入

同一条 item 可以同时包含 URL 和本地文件。只有本地文件进入 TaskFiles，
URL 不下载、不打包、不上传。

## 3. 每个数据集的配置

每个 split 都有一个 compose YAML：

```text
nexus/extensions/tasks/context_agent/configs/
```

| 配置 | Split | 输入 | Policy 输入形式 | 评分 |
| --- | --- | --- | --- | --- |
| `gdppdf500_train.yaml` | `gdppdf500_train` | URL + TaskFiles | 单轮 | GDPpdf rubric |
| `gdppdf500_eval.yaml` | `gdppdf500_eval` | URL | 单轮 | GDPpdf rubric |
| `aagdppdf.yaml` | `aagdppdf` | TaskFiles | 单轮 | GDPpdf rubric |
| `gdpval800_train.yaml` | `gdpval800_train` | URL | 单轮 | agentic deliverable Judge |
| `gdpval800_eval.yaml` | `gdpval800_eval` | URL | 单轮 | agentic deliverable Judge |
| `clb_200.yaml` | `clb_200` | TaskFiles | 多轮 user/assistant | rubric Judge |
| `clblife.yaml` | `clblife` | TaskFiles | 多轮 user/assistant | rubric Judge |
| `l3_clb_l25.yaml` | `l3_clb_l25` | TaskFiles | 多轮 user/assistant | rubric Judge |
| `l3_clb_real.yaml` | `l3_clb_real` | TaskFiles | 多轮 user/assistant | rubric Judge |
| `l3_clb_v1.yaml` | `l3_clb_v1` | TaskFiles | 多轮 user/assistant | rubric Judge |
| `l3_clb_v2.yaml` | `l3_clb_v2` | TaskFiles | 多轮 user/assistant | rubric Judge |

`AAGDPpdf` 已转换为 `aagdppdf` split，共 100 条，源 JSONL 是：

```text
/apdcephfs_gy6/share_303171455/yufeihuang/align/context_agent_general/data/AAGDPpdf.files.contextagent.jsonl
```

附件来自：

```text
/apdcephfs_gy6/share_303171455/yufeihuang/align/ctx_agent_data/
hyctx_data/data/eval/files/GDPpdf100
```

所有 100 个附件均已加入本地 Dataset。

除三个 train split 外，每个 split 还具有独立的远程配置：

```text
nexus/extensions/tasks/context_agent/configs/remote/
```

大多数远程数据集使用：

```text
yufeihuang/context_agent_<split>@v1
```

`gdppdf500_eval` 在原数据更新为 43 条 URL 加 7 条本地 Markdown
之后重新发布为：

```text
yufeihuang/context_agent_gdppdf500_eval@v2
```

远程 TaskFiles 所需的鉴权、Dataset IO 和 COS 网络规则集中在：

```text
nexus/extensions/tasks/context_agent/remote_taskfiles.yaml
```

一站式平台可以把 WeData key 放在公共配置的 `task_config.WEDATA_KEY`：

```json
{
  "nexus_run_input": {
    "task_config": {
      "WEDATA_KEY": "${oc.env:WEDATA_KEY}"
    }
  }
}
```

该字段仅由 `context_agent_general_task` 用于远程 TaskFiles 鉴权，不会传给
Agent。优先级为：

```text
task_config.WEDATA_KEY
→ 进程环境 WEDATA_KEY
→ WEDATA_KEY_FILE
→ ~/.nexus/auth/wedata/key
```

### 使用远程 Dataset

运行前设置 Runtime、Policy 和 Judge 凭据：

```bash
export CTXAGENT_GONGFENG_TOKEN='<gongfeng-token>'
export CTXAGENT_POLICY_API_KEY='<policy-api-key>'
export CTXAGENT_JUDGE_API_KEY='<judge-api-key>'
```

查看远程数据集版本：

```bash
uv run nexus dataset versions yufeihuang/context_agent_aalcr
```

查看某个 split 中可用的 item ID：

```bash
uv run nexus dataset items \
  yufeihuang/context_agent_aalcr@v1 \
  --split aalcr \
  --limit 20
```

运行远程数据集中的一条：

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/remote/gdppdf500_eval.yaml \
  --id gdppdf500_eval-00000 \
  -o outputs/context_agent/gdppdf500_eval-remote/gdppdf500_eval-00000/result.json
```

批量运行整个远程 split：

```bash
uv run nexus run-batch \
  nexus/extensions/tasks/context_agent/configs/remote/gdppdf500_eval.yaml \
  --database outputs/context_agent/gdppdf500_eval-remote/batch.db \
  --max-concurrent 8 \
  --timeout 3600 \
  --export-format jsonl \
  --export-path outputs/context_agent/gdppdf500_eval-remote/results.jsonl
```

远程配置固定到明确版本，例如：

```yaml
compose:
  - use: ../../base.yaml
  - data:
      uri: yufeihuang/context_agent_aalcr@v1
      split: aalcr
  - use: ../../remote_taskfiles.yaml
```

其中：

- Dataset item 和元数据由 Host 从远程 Dataset 加载；
- `task-files://` 附件由 Runtime 从远程 ZIP 按需读取并解压到
  `./workspace/<name>`；
- HTTP(S) 附件继续保留为 URL，由 Agent 调用 `curl` 下载；
- `remote_taskfiles.yaml` 使用 `append_unique` 追加远程签名和 COS 下载所需
  的网络规则，不会覆盖 item 自身的 URL allowlist。

当前没有发布以下三个 train split：

```text
gdppdf500_train
ctxagent1000_train
gdpval800_train
```

它们继续使用 `configs/` 下的本地 Dataset 配置。其余配置可以从
`configs/remote/` 运行独立远程 `v1` 数据集。

## 4. 单条运行

使用配置内的 Dataset 和 split，只需要指定 item ID。

通用格式：

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/<dataset>.yaml \
  --id <item-id> \
  -o outputs/context_agent/<dataset>/result.json
```

### GDPpdf URL

`gdppdf500_eval` 全部是 URL：

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/gdppdf500_eval.yaml \
  --id gdppdf500_eval-00000 \
  -o outputs/context_agent/gdppdf-url/result.json
```

### GDPpdf 本地 TaskFiles

`gdppdf500_train-00001` 是本地文件 case：

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/gdppdf500_train.yaml \
  --id gdppdf500_train-00001 \
  -o outputs/context_agent/gdppdf-local/result.json
```

该 item 的 URI 是：

```text
task-files://OG00020-03.md
```

### GDPval

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/gdpval800_eval.yaml \
  --id gdpval800_eval-00000 \
  -o outputs/context_agent/gdpval800-eval/result.json
```

GDPval 配置默认：

```yaml
task_config:
  evaluate: true
  judge_backend: deliverable
  artifact_root: outputs/context_agent/gdpval800_eval
```

GDPval 使用 agentic deliverable Judge。Policy 完成后，Judge 会在同一个
Runtime 内创建 `workspace/` 的私有副本，并使用无网络的
`bash`、`python`、`file_search`、`grep`、`read_lines` 工具检查实际生成的
XLSX、PPTX、DOCX、PDF 和文本文件。Judge 不会仅凭最终消息中“已创建文件”的
陈述给分。

`artifact_root` 是是否持久化 Runtime workspace 的显式开关。为了让
`result.json` 和 workspace 位于同一个 run 目录，需要给本次运行指定稳定的
`run_id`，并将输出路径写到 `<artifact_root>/<run_id>/result.json`：

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/gdpval800_eval.yaml \
  --id gdpval800_eval-00000 \
  --set run_id=gdpval800_eval-00000 \
  -o outputs/context_agent/gdpval800_eval/gdpval800_eval-00000/result.json
```

最终目录：

```text
outputs/context_agent/gdpval800_eval/gdpval800_eval-00000/
├── result.json
├── manifest.json
└── workspace/
```

不配置 `artifact_root` 时不会下载 workspace。批量运行也使用相同的
`<artifact_root>/<run-id>/workspace/` 布局。

Agentic Judge 返回：

```text
graded_on: deliverables
criteria_passed
criteria_total
judge_rounds
criteria[]: 每条 rubric 的 pass/fail 和文件证据
```

如果只想生成交付物、不运行 Judge，可以覆盖：

```bash
--set task_config.evaluate=false
```

### 多轮 benchmark

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/clb_200.yaml \
  --id clb_200-00000 \
  -o outputs/context_agent/clb/result.json

uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/l3_clb_v1.yaml \
  --id l3_clb_v1-00000 \
  -o outputs/context_agent/l3-clb-v1/result.json
```

多轮模式发送：

```text
system
context_dialogue[0]
context_dialogue[1]
...
context_dialogue[-1]  # 当前 user 问题
```

不会再次追加 `context_question`。对话中的旧附件路径会按文件名改写为：

```text
./workspace/<name>
```

## 5. 整个 split 批量运行

配置文件已经包含 Dataset URI 和 split：

```bash
uv run nexus run-batch \
  nexus/extensions/tasks/context_agent/configs/gdppdf500_eval.yaml \
  --database outputs/context_agent/gdppdf500_eval/batch.db \
  --max-concurrent 8 \
  --timeout 3600 \
  --export-format jsonl \
  --export-path outputs/context_agent/gdppdf500_eval/results.jsonl
```

运行另一个数据集时只需替换配置路径和输出目录。

先跑少量样本：

```bash
uv run nexus run-batch \
  nexus/extensions/tasks/context_agent/configs/clblife.yaml \
  --limit 5 \
  --database outputs/context_agent/clblife-smoke/batch.db \
  --max-concurrent 5 \
  --timeout 3600 \
  --export-format jsonl \
  --export-path outputs/context_agent/clblife-smoke/results.jsonl
```

恢复已有 batch：

```bash
uv run nexus run-batch \
  nexus/extensions/tasks/context_agent/configs/clblife.yaml \
  --database outputs/context_agent/clblife-smoke/batch.db \
  --max-concurrent 5 \
  --timeout 3600
```

## 6. 不使用 Dataset 的单条 mock

临时构造一条数据时不需要创建 `dataset.yaml`。可以通过 compose 把
`base.yaml` 和一段内联 `input` 合并。仓库中提供了一个从早期
GDPpdf500 单条样例整理出的无密钥示例：

```text
nexus/extensions/tasks/context_agent/configs/examples/gdppdf500_single.yaml
```

运行：

```bash
uv run nexus run-single \
  nexus/extensions/tasks/context_agent/configs/examples/gdppdf500_single.yaml \
  -o outputs/context_agent/gdppdf500-single/result.json
```

这个示例直接在 YAML 中声明 `context_question`、`context_files`、rubrics 和
metadata，不使用 Dataset，也不需要 `--id`。URL 输入仍需在
`runtime_provider_config.network_policy.egress` 中放行对应的 `host:port`。

正式 benchmark 和批量任务统一使用 Nexus Dataset。原始
`*.contextagent.jsonl` 应先离线转换为 Nexus Dataset，不再由运行时
`generator.py` 逐行读取。

## 7. Judge 路由

评分依据是 `metadata.dataset_name`，不是 JSONL 文件名。

### Rule

```text
LongBenchV2
MRCR_400 / MRCR_800
oolong
```

### Reference Judge

```text
AALCR
LongBenchPro
BrowseCompLongContext
TableMixV2
DocLonger
```

### Rubric Judge

```text
GDPpdf
surge_docs
SurgeOODV21
CLB
CLBLife
outside_L3_high
Pluto20
TableMixLonger
```

### Agentic deliverable Judge

```text
GDPval
```

GDPval 的 reward 是通过项数除以 rubric 总项数；证据来自实际交付物文件，
而不是只读取 `final_answer`。因此部分完成的交付物也能为 GRPO 提供梯度。

`SurgeOODV21` 虽然带非空 `answer`，仍按 rubrics 评分。历史 dialogue 只作为
Policy 上下文，Judge 只检查本轮新生成的最终回答。

GDPval 的 deliverable Judge 与普通 rubric Judge 的分数含义相同：所有 rubric
全部通过才得 `1`，否则得 `0`。区别仅在证据来源：普通 rubric Judge 只读取最终
文本，deliverable Judge 会实际打开 Runtime 中生成的文件。

## 8. 输出

单条 `result.json`：

```text
success
error
task_output
agent_output.main_trajectory
processor_results.chat
processor_results.tokens
workflow_states
```

重点字段：

| 字段 | 含义 |
| --- | --- |
| `task_output.final_answer` | 最终答案 |
| `task_output.score` | 评分 |
| `task_output.grading_style` | `rule` / `reference` / `rubric` |
| `task_output.judge_unavailable` | Judge 请求或解析是否失败 |
| `task_output.evaluation_skipped` | 是否跳过 Judge |
| `task_output.requirement_status` | rubric 逐项结果 |
| `task_output.graded_on` | `deliverables` 或降级后的 `final_message` |
| `task_output.criteria` | Agentic Judge 的逐项 verdict 和文件证据 |
| `task_output.criteria_passed` | 满足的 rubric 数量 |
| `task_output.criteria_total` | rubric 总数 |
| `task_output.judge_rounds` | Agentic Judge 使用的模型轮数 |
| `task_output.workspace_artifact` | GDPval 等任务的持久化文件 |
| `agent_output.main_trajectory` | system/user/assistant/tool 完整轨迹 |

## 9. Runtime 镜像

默认主镜像：

```text
mirrors.tencent.com/hunyuan_yanguan/context-agent:v1.2
```

模型可见的 `curl`、`bash`、`python` 和文档处理依赖来自主镜像。依赖列表：

```text
nexus/extensions/agents/context_agent/requirements-main-image.txt
```

`worker_venv.modules: []` 会上传并安装当前 Nexus wheel，使 `nexus/` 下的
Runtime 侧改动生效，但不会把文档处理依赖安装到模型使用的主镜像 Python。

## 10. EnterpriseBench

EnterpriseBench 代码位于 `tasks/context_agent/enterprisebench_native/`，通过
`context_agent_enterprisebench_native_task` 直接使用 Nexus Runtime 编排，在
Gongfeng world 镜像中运行 `context_agent_general`。数据集生成阶段默认从 `WORLDBENCH_TOOL_SETS`
移除 Docker world 的 `core` 工具集；通用的 `curl`、`bash`、`python`、
文件搜索和读取能力由 Nexus agent 提供，MCP 只保留业务应用工具。

完整构建、镜像和批量运行说明：

```text
nexus/extensions/tasks/context_agent/enterprisebench_native/README.md
```

入口：

```bash
python -m nexus.extensions.tasks.context_agent.eval.enterprisebench \
  --delivery /path/to/tencent_aug24_delivery \
  --workdir /path/to/eval_run \
  --image-tag v2 --dry-run
```

### Reward System environment variables

The `rm_system` scorer uses `NEXUS_REWARD_SYSTEM_TOKEN` and
`NEXUS_REWARD_SYSTEM_EXP_ID` as its primary credential and experiment variables.
Resolution order is the uppercase Nexus name, its lowercase compatibility name,
then `RM_TOKEN` / `RM_EXP_ID`. An unset experiment ID retains the existing
`ctxagent_rm_eval` default. These values are read when the scorer module is imported,
so configure them in the worker environment before startup.
`RM_REWARD_URL` and `RM_MODEL_TYPE` retain their existing behavior; request payloads,
score parsing, and retry handling are unchanged.
