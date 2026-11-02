"""direct_llm 后端连接配置。

把原 litellm_router/proxy_csig.py / proxy_taiji.py 中硬编码的平台地址、
API Key、超时、重试等参数统一收敛到这里，并且 **优先从环境变量读取**
（secrets env-only）。

⚠️ 安全约束：
    - CSIG 的 API Key 视为 **私有密钥**，必须通过环境变量 / 命令行 /
      `config/.env`（已被 .gitignore 忽略）注入，**严禁** 把真实 key
      写回本文件或提交到 git；本文件只保留占位值，运行时若未注入则
      调用会失败，以此提醒使用者必须配置自己的 key。
          export ONE_AGENT_CSIG_API_KEY=sk-xxx
          # 或在启动脚本前: source config/.env
    - Taiji 的 API Key 是团队/平台层面的 **共享 key**，与仓库其他模块
      （taiji.py / tau3 / bfclv4 / custom_web 等）一致，可以保留为
      内置默认值；如需替换，仍可通过 ONE_AGENT_TAIJI_API_KEY 覆盖。

环境变量一览：
    # CSIG
    ONE_AGENT_CSIG_BASE
    ONE_AGENT_CSIG_API_KEY
    ONE_AGENT_CSIG_TIMEOUT
    ONE_AGENT_CSIG_MAX_RETRIES

    # Taiji
    ONE_AGENT_TAIJI_URL
    ONE_AGENT_TAIJI_API_KEY
    ONE_AGENT_TAIJI_TIMEOUT
    ONE_AGENT_TAIJI_MAX_RETRIES

    # Taiji prompt cache（一致性哈希路由，命中推理实例 prefix/KV cache）
    ONE_AGENT_TAIJI_STRATEGY      # 默认 ConsistentHash；置空字符串可关闭
    ONE_AGENT_TAIJI_SESSION_ID    # 留空时由 client 用 run_id 兜底

    # Taiji 慢思考
    ONE_AGENT_TAIJI_REASONING_EFFORT  # 如 high/medium/low；未设置则不下发该参数
"""

from __future__ import annotations

import os


# ── CSIG ────────────────────────────────────────────────────────
CSIG_BASE: str = os.getenv('ONE_AGENT_CSIG_BASE', 'http://csig.litellm.prod.sgpolaris')
CSIG_API_KEY: str = os.getenv('ONE_AGENT_CSIG_API_KEY', 'REPLACE_WITH_YOUR_CSIG_API_KEY')
CSIG_TIMEOUT: float = float(os.getenv('ONE_AGENT_CSIG_TIMEOUT', '300'))
CSIG_MAX_RETRIES: int = int(os.getenv('ONE_AGENT_CSIG_MAX_RETRIES', '2'))
CSIG_INITIAL_BACKOFF: float = float(os.getenv('ONE_AGENT_CSIG_INITIAL_BACKOFF', '1.0'))

# CSIG prompt cache：用请求体 user 字段做一致性哈希粘连路由，让同一会话/同一前缀
# 的请求落到同一推理实例，命中该实例上由 cache_control 创建的 Anthropic prefix cache。
# 实测：固定 user 可达 ~7/8 命中；Taiji 式 StrategyType/SessionID header 对 CSIG 无效
# （≈随机路由），故 CSIG_STRATEGY 仅作无害冗余 header 保留。SessionID 留空时由 client
# 用 run_id 兜底（见 client._csig_routing），该值会被写入请求体 user 字段。
CSIG_STRATEGY: str = os.getenv('ONE_AGENT_CSIG_STRATEGY', 'ConsistentHash')
CSIG_SESSION_ID: str = os.getenv('ONE_AGENT_CSIG_SESSION_ID', '')

# ── Taiji ───────────────────────────────────────────────────────
TAIJI_URL: str = os.getenv(
    'ONE_AGENT_TAIJI_URL',
    'http://stream-server-online-sbs-11644.turbotke.production.polaris:1081/openapi/chat/completions',
)
TAIJI_API_KEY: str = os.getenv('ONE_AGENT_TAIJI_API_KEY', '7auGXNATFSKl7dF')
TAIJI_TIMEOUT: float = float(os.getenv('ONE_AGENT_TAIJI_TIMEOUT', '450'))
TAIJI_MAX_RETRIES: int = int(os.getenv('ONE_AGENT_TAIJI_MAX_RETRIES', '3'))
TAIJI_INITIAL_BACKOFF: float = float(os.getenv('ONE_AGENT_TAIJI_INITIAL_BACKOFF', '2.0'))

# Taiji prompt cache：通过 StrategyType=ConsistentHash + 稳定 SessionID，
# 让同一会话的多轮请求路由到同一推理实例，从而命中该实例的 prefix/KV cache。
# 默认开启 ConsistentHash；显式设为空串可回退到默认轮询。
TAIJI_STRATEGY: str = os.getenv('ONE_AGENT_TAIJI_STRATEGY', 'ConsistentHash')
TAIJI_SESSION_ID: str = os.getenv('ONE_AGENT_TAIJI_SESSION_ID', '')

# Taiji 慢思考：设置后以 chat_template_kwargs.reasoning_effort 下发给推理侧。
# 留空（默认）表示不下发该参数，行为与开启前完全一致。
TAIJI_REASONING_EFFORT: str = os.getenv('ONE_AGENT_TAIJI_REASONING_EFFORT', '')

# ── httpx 连接池（与原 proxy_server 保持一致）────────────────────
HTTP_MAX_CONNECTIONS: int = int(os.getenv('ONE_AGENT_HTTP_MAX_CONNECTIONS', '300'))
HTTP_MAX_KEEPALIVE: int = int(os.getenv('ONE_AGENT_HTTP_MAX_KEEPALIVE', '100'))
HTTP_CONNECT_TIMEOUT: float = float(os.getenv('ONE_AGENT_HTTP_CONNECT_TIMEOUT', '10.0'))
