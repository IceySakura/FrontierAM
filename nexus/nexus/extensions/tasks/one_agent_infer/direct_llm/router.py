"""模型路由表：model 名 → 后端 + provider + 真实 model 名。

从 litellm_router/proxy_router.py 内联而来，仅保留 one_agent_infer 直连所需的
后端（csig / taiji）。路由策略（按优先级）：
  1. 精确匹配 ROUTE_TABLE
  2. 前缀匹配 FALLBACK_RULES（如 "taiji/" 前缀，模型名透传）
  3. 默认 fallback（路由到 DEFAULT_BACKEND，模型名透传）

新增模型只需改这里，无需重启任何服务。
"""

from __future__ import annotations

from dataclasses import dataclass


# 直连只实现 csig / taiji 两个后端
SUPPORTED_BACKENDS = ('csig', 'taiji')


@dataclass(frozen=True)
class RouteInfo:
    backend: str
    provider: str
    model: str


# ── 默认 fallback：未匹配时路由到太极，模型名透传（自部署模型直接用名字）──
DEFAULT_BACKEND = 'taiji'
DEFAULT_PROVIDER = 'hunyuan'

# ── 前缀规则（前缀会被剥离，模型名透传）──────────────────────────
FALLBACK_RULES: dict[str, tuple[str, str]] = {
    'taiji/': ('taiji', 'hunyuan'),
    'csig/': ('csig', 'anthropic'),
}

# ── 精确匹配路由表 ──────────────────────────────────────────────
ROUTE_TABLE: dict[str, RouteInfo] = {
    # ---- CSIG (litellm proxy) ----
    'deepseek-v3.2': RouteInfo('csig', 'deepseek', 'deepseek-v3.2'),
    'claude-opus-4-6': RouteInfo('csig', 'anthropic', 'claude-opus-4.6'),
    'claude-opus-4-7': RouteInfo('csig', 'anthropic', 'claude-opus-4.7'),
    # yaml/DB 里带点、带横线两种写法都常见，都要精确进 CSIG，否则会落到默认 Taiji
    'claude-opus-4-8': RouteInfo('csig', 'anthropic', 'claude-opus-4.8'),
    'claude-opus-4.8': RouteInfo('csig', 'anthropic', 'claude-opus-4.8'),
    'claude-opus-5': RouteInfo('csig', 'anthropic', 'claude-opus-5'),
    'claude-sonnet-4.6': RouteInfo('csig', 'anthropic', 'claude-sonnet-4.6'),
    'claude-sonnet-4-6': RouteInfo('csig', 'anthropic', 'claude-sonnet-4.6'),
    'claude-sonnet-5': RouteInfo('csig', 'anthropic', 'claude-sonnet-5'),
    'glm-5': RouteInfo('csig', 'zai', 'glm-5'),
}


def _strip_provider_prefix(model: str) -> str:
    """'openai/gpt-5.2' → 'gpt-5.2'；'gpt-5.2' → 'gpt-5.2'。

    不剥 ``csig/`` / ``taiji/``：这两个是选后端的前缀，留给 FALLBACK_RULES 处理。
    """
    if '/' not in model:
        return model
    prefix, rest = model.split('/', 1)
    if f'{prefix}/' in FALLBACK_RULES:
        return model
    return rest


def resolve(model: str) -> RouteInfo:
    """查路由表，支持带/不带 provider 前缀的 model 名。"""
    if not model:
        raise ValueError("missing 'model' for routing")

    stripped = _strip_provider_prefix(model)

    # 1) 精确匹配
    if model in ROUTE_TABLE:
        return ROUTE_TABLE[model]
    if stripped in ROUTE_TABLE:
        return ROUTE_TABLE[stripped]

    # 2) 前缀规则
    for prefix, (backend, provider) in FALLBACK_RULES.items():
        if stripped.startswith(prefix):
            real_model = stripped[len(prefix) :]
            return RouteInfo(backend, provider, real_model)

    # 3) 默认 fallback
    return RouteInfo(DEFAULT_BACKEND, DEFAULT_PROVIDER, stripped)
