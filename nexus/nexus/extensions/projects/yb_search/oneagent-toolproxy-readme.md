# Yb OneAgent + ToolProxy 技术文档

## 文件结构

```
nexus/extensions/
├── projects/yb_search/
│   ├── agent.py                 # Agent loop：LLM 调用 + tool_calls 分发
│   ├── task.py                  # Task 入口：调 agent + 计算 reward
│   ├── log_utils.py
│   ├── tool_proxy_dispatcher.py # 请求构建 + 响应解析 + session 管理
│   ├── tool_proxy_client.py     # HTTP JSON 传输 + 限流 + 重试 + 超时
│   ├── tools/                   # 工具注册（参数 schema + build_extra）
│       ├── registry.py
│       ├── web_search.py
│       ├── picture_search.py
│       ├── video_search.py
│       └── context.py           # ToolProxyContextIn 定义
│   └── reward/                  # 奖励计算模块
│       ├── total_reward.py      # 总分整合（路由到各 judge）
│       ├── pairwise_judge.py    # BT-RM / APP RM 评分
│       ├── grm_judge.py         # GRM 多参考答案评分
│       ├── rule_based_judge.py  # 规则检查
│       ├── format_judge.py      # 格式协议检查
│       ├── closed_qa_judge.py   # 闭卷 QA 评估
│       ├── closed_qa_rubric_judge.py
│       └── tool_call_utils.py   # query discount 计算
```

---


## 参数传递链路

```yaml
# YAML 配置（由 HyTuner 传入）
task_config:
  task_name: yb_one_agent_toolproxy
  agent_name: yb_one_agent_toolproxy
  agent_config:
    max_iterations: 2
    tool_proxy_service: "trpc.yb_agent.tool_proxy.http"
    tool_proxy_namespace: "Production"
    tool_proxy_timeout: 30
    tool_proxy_caller: "nexus"      # 兼容旧配置，HTTP transport 不使用
    tools: [...]                    # OpenAI-format tool schemas
    tool_choice: "auto"
    enable_multimodal: false
    tool_proxy_context_in:
      user: {user_id: "...", client_version: "..."}
      caller: {call_source: "...", model_name: "..."}
      session: {conversation_id: "...", task_id: "..."}
```

**传递路径**：

```
YAML → YbOneAgentInput (pydantic)
  ├── messages            → LLM 对话历史
  ├── tools               → 传给 LLM（tool schema）+ 提取 tool_versions
  ├── tool_proxy_service  → ToolProxyConfig → Polaris HTTP 服务发现
  ├── tool_proxy_context_in → dispatcher 构建 /batch_tool_call 的 context_in
  └── max_iterations      → agent loop 最大轮数
```

---

## 超时设计

### 参数

| 参数 | 默认值 | 环境变量 | 说明 |
|------|--------|----------|------|
| `TOOL_PROXY_TOTAL_TIMEOUT` | 1000s | `NEXUS_SEARCH_TOTAL_TIMEOUT` | 总时间基数 |
| `PER_ATTEMPT_TIMEOUT` | 1000s | 自动：`TOTAL / RETRIES` | 单次 HTTP 请求超时 |
| `MAX_TOOL_PROXY_RETRIES` | 1 | `NEXUS_SEARCH_MAX_RETRIES` | 重试次数 |
| `TOOL_PROXY_RATE_LIMIT` | 1 | `NEXUS_SEARCH_RATE_LIMIT` | per-worker 并发数 |
| `TOOL_PROXY_RETRY_SLEEP` | 10s | `NEXUS_SEARCH_RETRY_SLEEP` | 重试间隔 |

### 时序

```
rate_limiter 排队        ← 不计入 timeout
    │
    ▼
asyncio.wait_for(        ← PER_ATTEMPT_TIMEOUT
    batch_tool_call(req)     只有 HTTP 请求本身受超时限制
)
    │
    ▼ (失败 → retry sleep → 回到 rate_limiter)
```

**核心规则**：rate_limit 等待不触发超时。超时只管实际 HTTP 调用时间。

### 重试条件

| 触发重试 | 不触发重试 |
|----------|-----------|
| `asyncio.TimeoutError`（HTTP 超时） | `rsp.results[i].status.code != 0`（业务错误） |
| 网络异常 / 连接失败 | `ValueError`（参数校验失败，dispatcher 层抛出） |

---


## Reward 限流

各 judge 独立限流，互不干扰：

| Judge | rate_limiter key | 默认并发 | 环境变量 |
|-------|-----------------|----------|----------|
| pairwise_judge | `pairwise_rm:{model_name}` | 6 | `PAIRWISE_JUDGE_RATE_LIMIT` |
| closed_qa_judge | `llm_judge` | 6 | `TAIJI_JUDGE_RATE_LIMIT` |
| format_judge | `format_judge` | 6 | `TAIJI_JUDGE_RATE_LIMIT` |
| hy_reward | `hy_judge` | 64 | `HY_JUDGE_RATE_LIMIT` |
| tool_proxy | `tool_proxy` | 1 | `NEXUS_SEARCH_RATE_LIMIT` |

### Reward 鉴权环境变量

以下 token 不在代码中提供默认值，运行对应 reward 前必须由环境变量注入：

| Reward | 环境变量 |
|--------|----------|
| pairwise_judge | `PAIRWISE_AUTH_TOKEN` |
| grm_judge | `GRM_AUTH_TOKEN` |
| hy_reward | `REWARD_SERVICE_TOKEN` |

---

## Session 管理

`ToolProxyDispatcher` 内部维护 `_SessionState`，跨轮自动传递：

- `cursors`：资源编号游标（web_doc / image_doc / video_doc 跨轮递增）
- `visited_urls`：已访问 URL 列表（去重用）
- `sandbox_state`：沙箱状态（如有）

每次 HTTP 响应的 `context_out.session` 会覆写更新 session state，下次请求自动带入 `context_in.session`。

---

## 日志排查

| 现象 | grep 关键字 | 示例 |
|------|------------|------|
| tool call 排队等待 | `rate-limit wait` | `[ToolProxy][abc123] rate-limit wait 2.350s before attempt 1/1, tools=['web_search']` |
| tool call HTTP 超时 | `timed out after` | `[ToolProxy][abc123] attempt 1/1 timed out after 1000s, tools=['web_search']` |
| tool call 网络异常 | `failed (` | `[ToolProxy][abc123] attempt 1/1 failed (ConnectionError), tools=['web_search']: ...` |
| agent 汇总耗时 | `AGENT_DONE` | `[YB-OneAgent][abc123] AGENT_DONE \| exit=completed, iters=2, tool_calls=3, obs_time=12.3s, rate_limit_wait=2.1s` |
| reward 排队等待 | `RATE_LIMIT_WAIT` | `[REWARD][abc123] PAIRWISE_RATE_LIMIT_WAIT \| 12.50s` |
| reward 计算完成 | `REWARD_CALC_DONE` | `[TASK][abc123] REWARD_CALC_DONE & TASK_DONE \| score=0.72, reward_time=30.12, termination=ENV_DONE` |
| reward 超时失败 | `REWARD_CALC_FAILED` | `[TASK][abc123] REWARD_CALC_FAILED \| TimeoutError: ...` → score=-200, termination=REWARD_ERROR |
| agent 工具超时 | `TOOL_TIMEOUT` | `[YB-OneAgent][abc123] TOOL_TIMEOUT \| iter=1, elapsed=1002.3s, TimeoutError: ...` |
| agent 工具异常 | `TOOL_ERROR` | `[YB-OneAgent][abc123] TOOL_ERROR \| iter=1, elapsed=0.5s, ConnectionError: ...` |
