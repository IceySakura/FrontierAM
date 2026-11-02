# Taiji Replay — Nexus Benchmark 接入方案

将太极实例分析 replay 接入 Nexus 评估框架。

---

## 总览

```
Benchmark = Generator × Task × Agent × Evaluator × Runtime(gongfeng sandbox)

┌────────────────────────────────────────────────────────────────┐
│  Generator                                                     │
│  dataset.json → Parquet → yield BatchItem                      │
└────────────────────────┬───────────────────────────────────────┘
                         ▼
┌────────────────────────────────────────────────────────────────┐
│  Task: taiji_analysis                                          │
│  1. 检查实例过期 → 2. run_agent() → 3. evaluate()             │
└──────────┬─────────────────────────────────────────────────────┘
           ▼                                 ▼
┌──────────────────────┐          ┌──────────────────────────────┐
│  Agent: claude_code  │          │  Evaluator: TaijiEvaluator   │
│  - 全量 38 个 skills │          │  - LLM Judge: 完整报告比较   │
│  - 自主选择工具      │          │    根因是否一致 (0/1)        │
│  - CLAUDE.md 引导    │          └──────────────────────────────┘
└──────────────────────┘

Runtime: gongfeng sandbox（claude_code 需在容器内运行）
```

---

## 第 1 层：Generator

### 职责

从 `dataset.json` 产出 `BatchItem`，每条太极实例一个。

### 数据源

现有 `episodes/dataset.json`，转为 Parquet：

```python
# columns:
#   instance_id: str    — instId（去重键）
#   question: str       — 原始用户 query（直接从历史对话提取，不做拼接）
#   answer: str         — historical_report（参考报告）
#   inst_id: str        — 太极 instId
#   task_id: str        — 太极 taskID
#   ws_id: str          — 太极 wsId
#   url: str            — 太极平台完整 URL
#   sender: str         — 原始请求人
#   started_at: int     — 原始执行时间戳
```

### 实现

```python
# nexus/extensions/tasks/taiji_analysis/generator.py
import os
from typing import Any, Iterator

import pandas as pd

from nexus.core.batch import BatchItem
from nexus.core.config import RunInput


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Required environment variable '{name}' is not set.")
    return value


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    return int(value) if value else default


def _build_openai_clients() -> dict[str, dict[str, Any]]:
    """从环境变量构建 LLM client 配置."""
    # Agent 模型（被评估的 LLM，驱动容器内 Claude Code）
    clients: dict[str, dict[str, Any]] = {
        "main": {
            "name": "litellm",
            "config": {
                "client_args": {
                    "api_key": _require_env("TAIJI_OPENAI_API_KEY"),
                    "base_url": _require_env("TAIJI_OPENAI_BASE_URL"),
                },
                "request_args": {"model": _require_env("TAIJI_OPENAI_MODEL")},
            },
        },
    }

    # Judge 模型（评分 LLM，判断根因是否一致）
    # 不设则复用 Agent 模型的凭证
    judge_model = os.getenv("TAIJI_JUDGE_MODEL", os.getenv("TAIJI_OPENAI_MODEL", ""))
    clients["judge"] = {
        "name": "litellm",
        "config": {
            "client_args": {
                "api_key": os.getenv("TAIJI_JUDGE_API_KEY", _require_env("TAIJI_OPENAI_API_KEY")),
                "base_url": os.getenv("TAIJI_JUDGE_BASE_URL", _require_env("TAIJI_OPENAI_BASE_URL")),
            },
            "request_args": {"model": judge_model},
        },
    }

    return clients


RUNTIME_IMAGE = os.getenv("TAIJI_RUNTIME_IMAGE", "current")


def generate_configs() -> Iterator[BatchItem]:
    dataset = pd.read_parquet(os.getenv("TAIJI_DATASET_PATH", "./episodes/dataset.parquet"))
    openai_clients = _build_openai_clients()

    skip = _env_int("TAIJI_SKIP", 0) or 0
    limit = _env_int("TAIJI_LIMIT")
    instance_ids = os.getenv("TAIJI_INSTANCE_IDS", "").split(",") if os.getenv("TAIJI_INSTANCE_IDS") else None

    for idx, (_, row) in enumerate(dataset.iterrows()):
        if idx < skip:
            continue
        if limit and idx >= skip + limit:
            break
        if instance_ids and row["inst_id"] not in instance_ids:
            continue

        config = RunInput.model_validate({
            "imports": ["nexus.extensions.tasks.taiji_analysis"],
            "task_name": "taiji_analysis",
            "task_config": {
                "question": row["question"],
                "answer": row["answer"],
                "inst_id": row["inst_id"],
                "task_id": row["task_id"],
                "ws_id": row["ws_id"],
                "url": row["url"],
            },
            "agent_name": "claude_code",
            "agent_config": {
                "working_dir": "/data/workspace",
                "timeout": _env_int("CLAUDE_CODE_TIMEOUT", 600),
                "allowed_tools": "Bash,Read",
                "max_turns": 120,
                "extra_env": {
                    "TAIJI_API_TOKEN": os.getenv("TAIJI_API_TOKEN", ""),
                    "GF_TOKEN": os.getenv("GF_TOKEN", ""),
                    "SWANLAB_API_KEY": os.getenv("SWANLAB_API_KEY", ""),
                    "MCPORTER_IWIKI_TOKEN": _require_env("MCPORTER_IWIKI_TOKEN"),
                    "MCPORTER_GONGFENG_TOKEN": _require_env("MCPORTER_GONGFENG_TOKEN"),
                    "MCPORTER_TAPD_TOKEN": _require_env("MCPORTER_TAPD_TOKEN"),
                },
                "pre_start_scripts": [
                    {"script": "npm install -g @tencent/mcporter-internal --registry=https://mirrors.tencent.com/npm/ --silent"},
                    {"script": "git clone --depth 1 -b debug_agent_taiji_replay https://oauth2:$GF_TOKEN@git.woa.com/nexus-agent/nexus.git /tmp/ws && cp -r /tmp/ws/.agent /data/workspace/.agent && cp /tmp/ws/CLAUDE.md /data/workspace/CLAUDE.md && rm -rf /tmp/ws"},
                    {"script": 'mkdir -p /data/workspace/config && cat > /data/workspace/config/mcporter.json <<EOFJ\n{"mcpServers":{"iwiki":{"baseUrl":"https://prod.mcp.it.woa.com/app_iwiki_mcp/mcp3","headers":{"Authorization":"Bearer $MCPORTER_IWIKI_TOKEN"}},"gongfeng":{"baseUrl":"https://mcpgw.knot.woa.com/gongfeng","headers":{"Authorization":"Bearer $MCPORTER_GONGFENG_TOKEN"}},"tapd":{"baseUrl":"http://mcp-idc.tapd.woa.com/mcp","headers":{"Authorization":"Bearer $MCPORTER_TAPD_TOKEN"}}}}\nEOFJ'},
                ],
            },
            "openai_clients": openai_clients,
            "runtime_provider_name": "gongfeng",
            "runtime_provider_config": {
                "api_base_url": os.getenv("TAIJI_RUNTIME_BASE_URL", "http://hyrl-sandbox.prod.woa.com:8052"),
                "token": _require_env("GONGFENG_RUNTIME_TOKEN"),
                "runtime_image": RUNTIME_IMAGE,
                "request_timeout": 600,
                "resource_spec": {"cpu": 2, "memory": 4},
            },
        })

        yield BatchItem(
            config=config,
            run_id=row["instance_id"],
            metadata={"inst_id": row["inst_id"], "sender": row["sender"]},
        )
```

### 环境变量

#### 数据集与过滤

| 变量 | 说明 | 默认值 |
|---|---|---|
| `TAIJI_DATASET_PATH` | Parquet 数据集路径 | `./episodes/dataset.parquet` |
| `TAIJI_SKIP` | 跳过前 N 条 | 0 |
| `TAIJI_LIMIT` | 最多跑 N 条 | 无限制 |
| `TAIJI_INSTANCE_IDS` | 逗号分隔的 instId 过滤 | 全部 |

#### Agent LLM（被评估模型，驱动容器内 Claude Code）

| 变量 | 说明 | 示例值 |
|---|---|---|
| `TAIJI_OPENAI_API_KEY` | LLM API Key | **必填**（HY3 可设为 `EMPTY`） |
| `TAIJI_OPENAI_BASE_URL` | LLM API Endpoint | **必填**（如 `http://infer-proxy-yb-test.production.polaris:8000/openapi`） |
| `TAIJI_OPENAI_MODEL` | 模型 ID | **必填**（如 `openai/HY3-dev0413`） |

#### Judge LLM（Evaluator 评分用，判断根因是否一致）

| 变量 | 说明 | 示例值 |
|---|---|---|
| `TAIJI_JUDGE_API_KEY` | Judge API Key | 默认复用 Agent；如用 GPT 则设 Azure key |
| `TAIJI_JUDGE_BASE_URL` | Judge Endpoint | 默认复用 Agent；如 `http://llm-api.model-eval.woa.com/v1` |
| `TAIJI_JUDGE_MODEL` | Judge 模型 ID | 默认复用 Agent；如 `api_azure_openai_gpt-5.6-sol` |
| `TAIJI_JUDGE_API_STYLE` | API 风格：`chat` 或 `responses` | GPT-5.6-sol 用 `responses` |
| `TAIJI_JUDGE_REASONING_EFFORT` | Reasoning effort | `none/low/medium/high/xhigh/max` |

#### Runtime（Nexus sandbox 容器调度）

| 变量 | 说明 | 默认值 |
|---|---|---|
| `GONGFENG_RUNTIME_TOKEN` | 工蜂 Sandbox API Token（管理容器生命周期） | **必填** |
| `TAIJI_RUNTIME_BASE_URL` | Sandbox API 地址 | `http://hyrl-sandbox.prod.woa.com:8052` |
| `TAIJI_RUNTIME_IMAGE` | Runtime 容器镜像 | `current`（Nexus 标准镜像） |

#### Skill 依赖 Token（注入容器 extra_env，供 agent 调用 MCP 工具）

| 变量 | 说明 | 默认值 |
|---|---|---|
| `TAIJI_API_TOKEN` | 太极 MCP 查询 Token | `h3kcqqyYy5Uosyuyx1E5mw`（内置只读） |
| `GF_TOKEN` | 工蜂 PAT（mcporter-internal 调用） | **必填** |
| `SWANLAB_API_KEY` | SwanLab 图表 API Key | 可选 |
| `MCPORTER_IWIKI_TOKEN` | iWiki MCP 的 Bearer Token | **必填** |
| `MCPORTER_GONGFENG_TOKEN` | 工蜂 MCP 的 Bearer Token | **必填** |
| `MCPORTER_TAPD_TOKEN` | TAPD MCP 的 Bearer Token | **必填** |

#### Claude Code

| 变量 | 说明 | 默认值 |
|---|---|---|
| `CLAUDE_CODE_VERSION` | Claude Code CLI 版本 | `latest` |
| `CLAUDE_CODE_TIMEOUT` | Agent 超时（秒） | 600 |

### LLM 调用机制

```
┌─── Sandbox 容器 ──────────────────────────────────────────────────────┐
│  Claude Code CLI（Anthropic API 格式）                                  │
│    ANTHROPIC_BASE_URL = http://127.0.0.1:{port}                        │
│    ANTHROPIC_API_KEY  = "dummy-key-for-proxy"                          │
│         │                                                              │
│         ▼                                                              │
│  [Nexus Anthropic Proxy]                                               │
│    anthropic_to_openai() → Socket.IO → Host LLM Handler                │
└────────────────────────────────────────────────────────────────────────┘
         │
         ▼  openai_clients.main（OpenAI 兼容格式）
┌────────────────────────────────────────────────────────────────────────┐
│  任意 OpenAI-Compatible 后端                                            │
│  · HY3 vLLM:  http://infer-proxy-yb-test.production.polaris:8000       │
│  · Bedrock:   bedrock/anthropic.claude-opus-4-6-v1                     │
│  · GPT:       http://trpc-gpt-eval.production.polaris:8080             │
└────────────────────────────────────────────────────────────────────────┘
```

Claude Code CLI 始终发 Anthropic 格式请求。Nexus Proxy 在容器内做 `anthropic_to_openai()` 格式转换后转发到 `openai_clients.main` 指定的后端。**Agent 模型可以是任何 OpenAI-compatible 后端**（HY3、Claude、GPT 等），不限于 Anthropic 模型。

Judge 模型独立运行在 **Task 进程**（调度节点）中，不在容器内。Evaluator 通过 `ctx.get_openai_client("judge")` 访问。Agent 和 Judge 可以使用完全不同的模型和 endpoint。

**推荐配置**：Agent 用 HY3（被评估模型），Judge 用 GPT-5.5（评分稳定且独立于被评估模型）。

### 辅助脚本

```python
# nexus/convert_dataset.py
# 将 dataset.json 转为 Parquet（一次性）
```

---

## 第 2 层：Task

### 职责

编排：过期检查 → 调 Agent → 调 Evaluator → 返回结构化结果。

### Input/Output 定义

```python
class TaijiTaskInput(TaskRunInput, WithAgentRunner):
    question: str       # 用户 query（含 URL）
    answer: str         # 参考报告（historical_report）
    inst_id: str
    task_id: str
    ws_id: str
    url: str

class TaijiTaskOutput(TaskRunOutput, WithScore):
    prediction: str         # agent 输出的完整诊断报告
    status: str             # completed | expired | error
```

### 实现

```python
@task_registry.register('taiji_analysis')
async def taiji_analysis_task(task_input: TaijiTaskInput) -> TaijiTaskOutput:
    # Step 1: 过期检查（调度节点直接 HTTP 查太极 API）
    if await check_instance_expired(task_input.inst_id, task_input.task_id, task_input.ws_id):
        return TaijiTaskOutput(score=0.0, prediction="", status="expired")

    # Step 2: 执行 Agent
    agent_output = await task_input.run_agent(user_prompt=task_input.question)
    prediction = agent_output.final_text  # claude_code agent 返回的完整文本输出

    # Step 3: 评估（LLM Judge 直接比较完整报告）
    evaluator = TaijiEvaluator()
    score = await evaluator.evaluate(
        prediction=prediction,
        reference=task_input.answer,
        question=task_input.question,
    )

    # Step 4: 返回
    return TaijiTaskOutput(score=score, prediction=prediction, status="completed")
```

### 过期检查

通过太极 MCP Server 的 HTTP 端点查询实例是否还存在。MCP Server 是标准 JSON-RPC over HTTP，调度节点直接 POST 即可，不需要 `connect_mcp.py` 脚本或 sandbox 环境。

```python
import httpx

TAIJI_MCP_URL = os.getenv("TAIJI_MCP_URL", "http://21.46.122.32:80")
TAIJI_API_TOKEN = os.getenv("TAIJI_API_TOKEN", "h3kcqqyYy5Uosyuyx1E5mw")


async def check_instance_expired(inst_id: str, task_id: str, ws_id: str) -> bool:
    """通过太极 MCP Server 查实例是否还存在（在调度节点执行，无需 sandbox）.

    太极 MCP Server 协议：POST /mcp，JSON-RPC 2.0 格式，
    token 通过 HTTP header 'taiji_api_token' 传递。
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "taiji_api_token": TAIJI_API_TOKEN,
    }
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "query_task_instance_list",
            "arguments": {"task_id": task_id, "ws_id": ws_id},
        },
    }

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(f"{TAIJI_MCP_URL}/mcp", headers=headers, json=payload)
        if resp.status_code != 200:
            return False  # API 不可达时不标记过期，让 agent 自己发现

        data = resp.json()
        # MCP 响应格式：{"result": {"content": [{"type": "text", "text": "<json>"}]}}
        content = data.get("result", {}).get("content", [])
        for item in content:
            if item.get("type") == "text":
                import json as _json
                try:
                    instances_data = _json.loads(item["text"])
                    # 在返回的实例列表中查找目标 inst_id
                    instances = instances_data if isinstance(instances_data, list) else instances_data.get("instances", [])
                    return not any(
                        inst.get("inst_id") == inst_id or inst.get("instId") == inst_id
                        for inst in instances
                    )
                except (ValueError, KeyError):
                    continue
        return False  # 解析失败时不标记过期
```

过期实例返回 `status="expired"`，最终统计时排除。

---

## 第 3 层：Agent

### 选择

使用 `claude_code` agent type（Nexus 已支持）。

### 为什么不用其他 agent type

| Agent Type | 不选的原因 |
|---|---|
| `swe_agent` | 面向 patch 生成，不适合诊断分析 |
| `codeact` | 需要自定义工具定义，太重 |
| `search_agent` | 没有 shell 执行能力 |
| **`claude_code`** | **直接调 CLI，自动加载 CLAUDE.md 和全量 skills** |

### 全量 Skill 环境（增加难度）

Agent 可见 `/data/workspace/.agent/skills/` 下的**全部 38 个 skill**，而不是只暴露太极相关的几个。这模拟了真实生产环境——agent 必须从众多可用工具中自主判断哪些适用于当前任务。

**全量 Skill 列表**：

| 类别 | Skills | 与太极分析的关系 |
|---|---|---|
| **核心诊断** | taiji-log-analyzer, check-machine-status, taiji-config-checker, taiji-official-skills, taiji-token-provider | ✅ 应该使用 |
| **辅助诊断** | angelft-db-query, ray-job-checker, ray-code-explorer, swanlab-chart-viewer, taiji-cluster-task-checker | ⚠️ 视情况使用 |
| **代码/CI** | fp8-qat-rebase, hytuner-nightly-ci, ci-bisect-debugger, nightly-ci-image-upgrade, code-reviewer, hytuner-pyspy-diagnose | ❌ 不应使用（干扰项） |
| **平台工具** | gongfeng, iwiki, tapd, mcporter-internal, knot-cli, knot-agent-editor, knot-skill-finder, skill-creator-for-knot | ❌ 不应使用 |
| **通信** | wecom-bot, wecom-webhook-sender, feedback-collector | ❌ 不应使用 |
| **文档/办公** | docx, pptx, xlsx, pdf, obsidian, apple-notes, apple-reminders, km | ❌ 不应使用 |
| **其他** | weather, diagrams-generator, isearch | ❌ 不应使用 |

### 评估难度维度

这引入了**工具选择能力**的评估：

1. **正确选择**：是否使用了 taiji-log-analyzer + check-machine-status 这套核心组合？
2. **抗干扰**：是否避免了无关 skill 的调用？（如不应该去跑 CI、不应该去查 iwiki）
3. **组合判断**：遇到 NCCL hang 时是否主动调用 angelft-db-query 做误杀排查？

### Agent 行为路径

Agent 启动后，CLAUDE.md 中的规则引导它：

```
1. 扫描 skills 列表，判断适用的 skill
2. 从 URL 提取 instId / taskID / wsId
3. 选择 taiji-log-analyzer → 获取全量日志
4. 如有崩溃 → 选择 check-machine-status
5. 如怀疑 AngelFT 误杀 → 选择 angelft-db-query（可选）
6. 如无明确错误 → 选择 taiji-config-checker
7. 综合输出诊断报告（按 SKILL.md 格式）
```

关键：agent 需要**自主判断**走哪条路径，而不是被硬编码。全量 skill 暴露测试的就是这个判断能力。

### 配置

```yaml
agent_name: claude_code
agent_config:
  working_dir: /data/workspace    # CLAUDE.md + 全量 skills 在此目录
  timeout: 600                    # 太极分析通常 2-5 分钟
  allowed_tools: "Bash,Read"      # 不需要 Edit/Write
  max_turns: 40
```

### Agent 不感知 benchmark

Agent 不知道自己在被评估——它收到的 prompt 和真实用户请求完全一样：

```
"帮我看下这个已完成的实例跑的怎么样：https://hunyuanaide.taiji.woa.com/web/..."
```

不提供任何"你应该用 taiji-log-analyzer"的 hint，完全靠 agent 自主决策。

---

## 第 4 层：Evaluator

### 设计原则

**只评估最终产出，不评估中间过程。**

工具选择是手段，不是目的。如果 agent 用了一条非常规路径（比如直接 grep 日志而没走 taiji-log-analyzer）但最终结论正确，不应扣分。反之，如果调了所有正确的 skill 但结论错误，也不应加分。

因此评估维度只有一个：**根因结论是否正确。**

### 评分

```
score = LLM_Judge(replay_report, historical_report)  → 0.0 或 1.0
```

Binary：一致 = 1，不一致 = 0。不做 partial credit——诊断要么对要么错。

### 实现

```python
class TaijiEvaluator(Evaluator):
    """单维度评估：根因结论是否与历史报告一致.

    直接将完整报告交给 Judge LLM 比较，不做预提取——
    避免正则匹配失败导致 false negative。
    """

    async def evaluate(self, prediction: str, reference: str, question: str | None = None) -> float:
        if not prediction.strip():
            return 0.0

        prompt = TAIJI_JUDGE_PROMPT.format(
            question=question,
            report_a=reference,
            report_b=prediction,
        )
        ctx = require_context()
        client = ctx.get_openai_client("judge")
        response = await client.create_completion(messages=[
            {"role": "system", "content": TAIJI_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        return self._parse_judge_output(response.choices[0].message.content)

    def _parse_judge_output(self, output: str) -> float:
        match = re.search(r'<is_consistent>\s*([01])\s*</is_consistent>', output)
        return float(match.group(1)) if match else 0.0
```

### Judge Prompt

```python
TAIJI_JUDGE_SYSTEM_PROMPT = """你是训练任务诊断报告的评审专家。
判断两份报告的根因结论是否语义一致。

你需要从每份报告中自行提取核心根因结论（通常在"一句话总结"或开头段落），
然后判断两者是否指向同一问题。只看核心结论，忽略格式、详略和辅助信息。"""

TAIJI_JUDGE_PROMPT = """判断两份诊断报告的根因是否一致。

## 用户问题
{question}

## 参考报告（完整）
{report_a}

## 待评估报告（完整）
{report_b}

## 评判标准
- 两份报告指向相同根因（如都说"OOM"或"节点故障"）→ 1
- 待评估报告包含参考的根因 + 额外正确分析 → 1
- 根因不同（如一个说 OOM，另一个说 NCCL hang）→ 0
- 待评估报告无有效结论或纯废话 → 0

请先从两份报告中各自提取根因结论，再做比较。

<result>
<reasoning>一句话理由</reasoning>
<is_consistent>1或0</is_consistent>
</result>"""
```

### 辅助指标（不参与打分，仅用于观测）

以下信息记录在 `TaskOutput.metadata` 中，用于离线分析 agent 行为，但不影响 score：

```python
metadata = {
    "skills_used": ["taiji-log-analyzer", "check-machine-status"],
    "latency_seconds": 180,
    "turns_used": 12,
}
```

从 agent 输出中提取这些信息：

```python
def extract_observability_metadata(agent_output) -> dict:
    """从 agent 的工具调用记录中提取观测指标."""
    skills_used = set()
    for tool_call in agent_output.tool_calls:
        # 匹配 skill 脚本调用：python3 /data/workspace/.agent/skills/<name>/...
        match = re.search(r'/\.agent/skills/([^/]+)/', tool_call.command)
        if match:
            skills_used.add(match.group(1))
    return {
        "skills_used": sorted(skills_used),
        "latency_seconds": agent_output.elapsed_seconds,
        "turns_used": agent_output.turns_used,
    }
```

这些指标可以回答"agent 是否效率低下"或"格式遵守得如何"，但不应该混入打分——它们是 debug 信号，不是评估标准。

---

## 第 5 层：Runtime

### 为什么需要 Runtime

太极分析是**纯读取**操作——Agent 不写持久化状态，多个 agent 并发不互相干扰。但 Agent（`claude_code`）仍然需要在 **Nexus Runtime sandbox** 中执行（与 SWE-bench 的 claude_code 一致）。原因：
- Nexus 的 `claude_code` agent 通过 `call_runtime_agent()` → `runtime.invoke()` 在 sandbox 容器内运行
- 框架设计如此，不能绕过

不需要 `conflict_groups`——每个 episode 独立容器，互不影响。

### Sandbox 环境需求

Agent 在 sandbox 容器内需要：

| 依赖 | 说明 | 来源 |
|------|------|------|
| Claude Code CLI | agent 执行体 | Nexus 标准镜像已预装 |
| mcporter-internal | MCP 工具调用 | `pre_start_scripts` 安装 |
| skills 目录 | taiji-log-analyzer 等脚本 | `pre_start_scripts` 从 git clone |
| CLAUDE.md | Agent 行为引导 | `pre_start_scripts` 从 git clone |
| `TAIJI_API_TOKEN` | 太极 API 认证 | `extra_env` 注入 |
| `GF_TOKEN` | 工蜂 API（mcporter / git clone） | `extra_env` 注入 |
| mcporter.json | MCP 服务端点配置 | `pre_start_scripts` 组装 |
| Python 3.12+ | skill 脚本运行 | Nexus 标准镜像已有 |

### Runtime 镜像

**使用 Nexus 标准 Runtime 镜像，不定制镜像。**与 SWE-bench 一致：

```python
RUNTIME_IMAGE = os.getenv("TAIJI_RUNTIME_IMAGE", "current")
# "current" = 当前 Nexus 版本对应的标准镜像
# 也可指定 tag: "mirrors.tencent.com/hunyuan_yanguan/nexus-runtime:{tag}"
```

所有额外依赖（mcporter-internal、skills、CLAUDE.md、mcporter.json）通过 `pre_start_scripts` 在容器启动时动态安装，无需维护自定义镜像。

### pre_start_scripts（容器启动时执行）

```python
"pre_start_scripts": [
    # 1. 安装 mcporter-internal
    {"script": "pip install mcporter-internal -i https://mirrors.tencent.com/pypi/simple/ -q"},
    # 2. Clone skills + CLAUDE.md（用 GF_TOKEN 认证）
    {"script": "git clone --depth 1 https://oauth2:$GF_TOKEN@git.woa.com/nexus-agent/debug-agent-workspace.git /tmp/ws && cp -r /tmp/ws/.agent /data/workspace/.agent && cp /tmp/ws/CLAUDE.md /data/workspace/CLAUDE.md && rm -rf /tmp/ws"},
    # 3. 组装 mcporter.json（token 从环境变量展开）
    {"script": "mkdir -p /data/workspace/config && cat > /data/workspace/config/mcporter.json <<EOFJ\n{\"mcpServers\":{\"iwiki\":{\"baseUrl\":\"https://prod.mcp.it.woa.com/app_iwiki_mcp/mcp3\",\"headers\":{\"Authorization\":\"Bearer $MCPORTER_IWIKI_TOKEN\"}},\"gongfeng\":{\"baseUrl\":\"https://mcpgw.knot.woa.com/gongfeng\",\"headers\":{\"Authorization\":\"Bearer $MCPORTER_GONGFENG_TOKEN\"}},\"tapd\":{\"baseUrl\":\"http://mcp-idc.tapd.woa.com/mcp\",\"headers\":{\"Authorization\":\"Bearer $MCPORTER_TAPD_TOKEN\"}}}}\nEOFJ"},
],
```

**好处**：
- 不需要维护自定义 Docker 镜像
- 更新 skills/CLAUDE.md 只需 push 到 git，下次跑自动拉最新
- token 全部走环境变量，不泄露到镜像层
- 与 SWE-bench 使用同一套镜像，减少运维负担

---

## Token 获取与配置

### 获取方式

| Token | 获取途径 | 备注 |
|-------|---------|------|
| `GONGFENG_RUNTIME_TOKEN` | 工蜂 Sandbox 资源平台申请 | 用于启动/停止 sandbox 容器；如已跑过 SWE-bench 则复用同一 token |
| `TAIJI_OPENAI_*`（Agent） | HY3 推理服务 | `API_KEY=EMPTY`，`BASE_URL` 为 infer-proxy 地址，`MODEL=openai/HY3-dev0413` |
| `TAIJI_JUDGE_*` | GPT-5.6-sol via Responses API | `MODEL=api_azure_openai_gpt-5.6-sol`, `API_STYLE=responses`, `REASONING_EFFORT=xhigh` |
| `GF_TOKEN` | 工蜂个人 PAT（Settings → Access Tokens） | 需对目标仓库有 read 权限 |
| `TAIJI_API_TOKEN` | 内置只读 Token | 固定值 `h3kcqqyYy5Uosyuyx1E5mw`，直接写死 |
| `MCPORTER_IWIKI_TOKEN` | Knot 平台 → iWiki MCP 应用 → PAT | Bearer token for `https://prod.mcp.it.woa.com/app_iwiki_mcp/mcp3` |
| `MCPORTER_GONGFENG_TOKEN` | Knot 平台 → 工蜂 MCP 应用 → PAT | Bearer token for `https://mcpgw.knot.woa.com/gongfeng` |
| `MCPORTER_TAPD_TOKEN` | Knot 平台 → TAPD MCP 应用 → PAT | Bearer token for `http://mcp-idc.tapd.woa.com/mcp` |
| `SWANLAB_API_KEY` | SwanLab 平台 → Settings → API Keys | 可选，不影响核心评估 |

### 调度节点环境配置

在运行 `nexus run-batch` 的机器上 export：

```bash
# ── Agent LLM（HY3，被评估模型）──
export TAIJI_OPENAI_API_KEY="EMPTY"
export TAIJI_OPENAI_BASE_URL="http://infer-proxy-yb-test.production.polaris:8000/openapi"
export TAIJI_OPENAI_MODEL="openai/HY3-dev0413"

# ── Judge LLM（GPT-5.6-sol，评分）──
export TAIJI_JUDGE_API_KEY="<your-judge-api-key>"
export TAIJI_JUDGE_BASE_URL="http://llm-api.model-eval.woa.com/v1"
export TAIJI_JUDGE_MODEL="openai/api_azure_openai_gpt-5.6-sol"
export TAIJI_JUDGE_API_STYLE="responses"
export TAIJI_JUDGE_REASONING_EFFORT="xhigh"

# ── Runtime ──
export GONGFENG_RUNTIME_TOKEN="<sandbox API token>"

# ── Skill Token ──
export TAIJI_API_TOKEN="h3kcqqyYy5Uosyuyx1E5mw"
export GF_TOKEN="<工蜂 PAT>"
export SWANLAB_API_KEY="<可选>"

# ── MCP Token（注入容器，pre_start_scripts 组装为 mcporter.json）──
export MCPORTER_IWIKI_TOKEN="<iWiki MCP Bearer Token>"
export MCPORTER_GONGFENG_TOKEN="<工蜂 MCP Bearer Token>"
export MCPORTER_TAPD_TOKEN="<TAPD MCP Bearer Token>"
```

### Token 流向

```
调度节点 shell env
    │
    ├─ TAIJI_OPENAI_* ──→ openai_clients.main ──→ Nexus Proxy ──→ HY3 vLLM（驱动 agent）
    ├─ TAIJI_JUDGE_*  ──→ openai_clients.judge ──→ Evaluator 直接调用 ──→ GPT-5.6-sol（评分）
    ├─ GONGFENG_RUNTIME_TOKEN ──→ runtime_provider_config.token ──→ 启动 sandbox 容器
    ├─ GF_TOKEN / TAIJI_API_TOKEN / SWANLAB_API_KEY
    │         ──→ agent_config.extra_env ──→ 容器内环境变量 ──→ skill 脚本读取
    └─ MCPORTER_IWIKI_TOKEN / MCPORTER_GONGFENG_TOKEN / MCPORTER_TAPD_TOKEN
              ──→ agent_config.extra_env ──→ 容器内环境变量
                      ──→ pre_start_scripts 组装为 /data/workspace/config/mcporter.json
                              ──→ mcporter-internal 读取 MCP 端点配置
```

---

## 运行方式

```bash
# 转换数据集（一次性，在开发机执行）
python3 convert_dataset.py

# 批量运行（100 路并发，匹配容器配额）
# run-batch 进程在调度节点执行，agent 在远程 sandbox 容器中运行
uv run nexus run-batch \
    -g nexus/extensions/tasks/taiji_analysis/generator.py \
    -d ./results/taiji_bench.db \
    --max-concurrent 100 \
    --timeout 600

# 查看进度
uv run nexus inspect-batch ./results/taiji_bench.db

# 断点续跑（自动跳过已成功的，不需要 -g）
uv run nexus run-batch \
    -d ./results/taiji_bench.db \
    --max-concurrent 100

# 只重跑失败的
uv run nexus run-batch \
    -d ./results/taiji_bench.db \
    --retry-only \
    --max-concurrent 100

# 计算指标
python -m nexus.extensions.tasks.taiji_analysis.compute_metrics --db ./results/taiji_bench.db
```

---

## 指标计算（compute_metrics.py）

### 实现

```python
# nexus/extensions/tasks/taiji_analysis/compute_metrics.py
"""从 batch DB 计算 taiji_analysis 指标.

Usage:
    python -m nexus.extensions.tasks.taiji_analysis.compute_metrics --db results/taiji_bench.db
"""
import json
import sqlite3
from collections import defaultdict
from pathlib import Path


def load_results(db_path: str) -> list[dict]:
    """从 SQLite DB 加载全部 run 结果."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT run_id, status, start_time, end_time, error,
               config, metadata, task_output
        FROM runs
    """)
    results = []
    for row in cursor.fetchall():
        r = {
            "run_id": row["run_id"],
            "status": row["status"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
        }
        for field in ["config", "metadata", "task_output"]:
            value = row[field]
            r[field] = json.loads(value) if value else {}
        results.append(r)
    conn.close()
    return results


def compute_metrics(results: list[dict]) -> dict:
    """计算聚合指标."""
    total = len(results)
    completed = []
    expired = 0
    errors = 0

    for r in results:
        task_output = r.get("task_output", {})
        status = task_output.get("status", r.get("status", ""))
        if status == "expired":
            expired += 1
        elif status in ("completed", "success"):
            completed.append(task_output)
        else:
            errors += 1

    # Score（只算 completed 的）
    scores = [r["score"] for r in completed if "score" in r]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    pass_count = sum(1 for s in scores if s >= 1.0)

    # Observability（从 metadata 提取）
    latencies = []
    turns = []
    skill_counts = defaultdict(int)

    for r in completed:
        meta = r.get("metadata", {})
        if meta.get("latency_seconds"):
            latencies.append(meta["latency_seconds"])
        if meta.get("turns_used"):
            turns.append(meta["turns_used"])
        for skill in meta.get("skills_used", []):
            skill_counts[skill] += 1

    n = len(completed) or 1

    return {
        "total": total,
        "completed": len(completed),
        "expired": expired,
        "errors": errors,
        "score": {
            "avg": round(avg_score, 4),
            "pass_rate": round(pass_count / n, 4),
        },
        "observability": {
            "avg_latency_seconds": round(sum(latencies) / len(latencies), 1) if latencies else None,
            "avg_turns": round(sum(turns) / len(turns), 1) if turns else None,
            "skill_usage_rate": {
                k: round(v / n, 3) for k, v in sorted(skill_counts.items(), key=lambda x: -x[1])
            },
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Batch results DB path")
    parser.add_argument("--output", default=None, help="Output JSON path (default: stdout)")
    args = parser.parse_args()

    results = load_results(args.db)
    metrics = compute_metrics(results)

    output = json.dumps(metrics, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(output)
        print(f"Metrics saved to {args.output}")
    else:
        print(output)
```

### 输出示例

```json
{
  "total": 800,
  "completed": 750,
  "expired": 112,
  "errors": 0,
  "score": {
    "avg": 0.72,
    "pass_rate": 0.72
  },
  "observability": {
    "avg_latency_seconds": 180.0,
    "avg_turns": 14.0,
    "skill_usage_rate": {
      "taiji-log-analyzer": 0.98,
      "check-machine-status": 0.82,
      "taiji-config-checker": 0.15,
      "angelft-db-query": 0.08
    }
  }
}
```

---

## 代码部署方式

**代码放入 Nexus 仓库 `nexus/extensions/tasks/taiji_analysis/`，跟随仓库版本迭代。**

这样 task/evaluator 代码与 Nexus 框架保持同步更新，agent 类型或 runner API 变更时可以一起适配。

数据集和运行结果不入仓库——这是 Nexus 所有任务的统一做法（SWE、Search 等都一样）。

**数据集**：通过环境变量 `TAIJI_DATASET_PATH` 指定本地 Parquet 路径。Generator 从中读取。

**运行结果**：存 SQLite DB，路径由 `-d` 参数指定。框架内置 `nexus inspect-batch` TUI 查看、`nexus export` 导出 JSONL/CSV。

**指标分析**：`compute_metrics.py` 放在仓库内（与 search、mm_agent 等任务的做法一致），从 DB 读取结果计算聚合指标。

---

## 文件结构

### Nexus 仓库内（`nexus/extensions/tasks/taiji_analysis/`）

```
nexus/extensions/tasks/taiji_analysis/
├── __init__.py            # 注册 task
├── generator.py           # 第1层：数据 → BatchItem
├── task.py                # 第2层：编排逻辑（@task_registry.register）
├── evaluator.py           # 第4层：LLM Judge
├── prompt.py              # Judge prompt 模板
├── compute_metrics.py     # 指标统计（从 DB 读取，输出 JSON summary）
└── pyproject.toml         # 模块元数据（依赖声明）
```

### 部署环境本地（不入仓库）

```
/data/workspace/taiji_replay/
├── episodes/
│   ├── dataset.json       # 原始数据
│   └── dataset.parquet    # 转换后的数据集
├── convert_dataset.py     # dataset.json → Parquet（一次性脚本）
└── results/
    └── taiji_bench.db     # 运行结果（SQLite）
```

---

## 依赖

| 依赖 | 用途 |
|---|---|
| Nexus 框架 | `run-batch` 引擎 |
| Claude CLI | `claude_code` agent 执行 |
| mcporter-internal | MCP 工具调用 |
| skills 目录 | taiji-log-analyzer 等脚本 |
| OpenAI-Compatible LLM 后端 | LLM 调用（agent: HY3 + judge: GPT-5.5） |
| 太极 MCP Server | 过期检查（调度节点 POST `http://21.46.122.32:80/mcp`，JSON-RPC） |
| httpx | 异步 HTTP 客户端（过期检查用） |

---

## 风险与缓解

| 风险 | 缓解 |
|---|---|
| 数据过期率高 | 定期重新收集；过期率纳入监控 |
| LLM Judge 不准 | 先跑 50 条人工标注对比 |
| 并发 MCP 限流 | `--max-concurrent` + 工具层重试 |
| Agent 不走 skill 流程 | CLAUDE.md 强制规则已覆盖 |
