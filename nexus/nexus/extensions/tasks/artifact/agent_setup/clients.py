"""Build named OpenAI-compatible clients and bind task-specific model roles."""

from __future__ import annotations

import os
import uuid

from typing import Any

from nexus.extensions.tasks.artifact import config


DEFAULT_MAX_TOKENS = 128000
"""各 provider 未被 ``ARTIFACT_MAX_TOKENS`` 覆盖时的统一兜底输出预算。

统一成一个值是为了少一类「某个模型悄悄用着 16000 兜底、长输出被截断」的意外。
唯一例外是 gemini,见 :data:`GEMINI_MAX_TOKENS`。
"""

GEMINI_MAX_TOKENS = 65536
"""gemini 系的硬上限。 **实测**:polaris 网关对 128000 直接 400,报错原文给出区间——
``maxOutputTokens value of 128000 but the supported range is from 1 (inclusive) to
65537 (exclusive)``。 所以 gemini 既要单独兜底,也要夹住 ``ARTIFACT_MAX_TOKENS``:
否则为别家设的 128000 会把三个 gemini alias 全打成 400(litellm 还会重试 7 次)。
"""


def _pin_cache_task_id(api_key: str, session_id: str | None) -> str:
    """Append ``&cache_task_id=<sid>`` to the key query when not already pinned."""
    if 'cache_task_id' in api_key:
        return api_key
    sid = (session_id or uuid.uuid4().hex)[:32]
    return f'{api_key}&cache_task_id={sid}'


def rebind_cache_task_id(clients: dict[str, dict[str, Any]], task_id: str) -> dict[str, dict[str, Any]]:
    """Return a deep-ish copy of ``clients`` with every key's cache_task_id set to
    ``task_id`` (32-char). Used to give each batch instance its own rate-limit
    bucket: build the named clients once, then per-instance rebind so N concurrent
    instances spread across N buckets instead of sharing one (→ 429 storm).

    Only rewrites keys that already carry a ``cache_task_id=`` (i.e. providers
    that opted in via :func:`_pin_cache_task_id`); other providers pass through.
    """
    import copy
    import re

    sid = (task_id or uuid.uuid4().hex)[:32]
    out: dict[str, dict[str, Any]] = {}
    for alias, spec in clients.items():
        spec2 = copy.deepcopy(spec)
        ca = (spec2.get('config') or {}).get('client_args') or {}
        key = ca.get('api_key')
        if isinstance(key, str) and 'cache_task_id=' in key:
            ca['api_key'] = re.sub(r'cache_task_id=[^&]*', f'cache_task_id={sid}', key)
        out[alias] = spec2
    return out


# ── Provider builders (explicit params; return one {name, config} spec) ──────
# Each accepts model/base_url/api_key/effort (falling back to a provider default
# when empty) plus enable_thinking/session_id, and returns a single client spec.


def _spec_hy3(*, model: str, base_url: str, api_key: str, effort: str, enable_thinking: bool, session_id: str | None):
    model = model or 'openai/HY3-dev0413'
    base_url = base_url or 'http://infer-proxy-yb-test.production.polaris:8000/openapi'
    api_key = api_key or 'EMPTY'
    # 与 frontend/generator_midtrain._build_openai_clients_hy3 对齐: max_tokens=65536,
    # chat_template_kwargs.reasoning_effort 为字符串 + preserved_thinking。
    request_args: dict[str, Any] = {
        'model': model,
        'timeout': 1800.0,
        'max_retries': 0,
        'temperature': config.get().models.openapi_temp,
        'top_p': 1.0,
        'top_k': -1,
        'max_tokens': config.get().models.max_tokens or DEFAULT_MAX_TOKENS,
        'repetition_penalty': 1.0,
        'extra_body': {
            'chat_template_kwargs': {
                'reasoning_effort': effort or 'high',
                'preserved_thinking': True,
            },
        },
    }
    if session_id:
        request_args['extra_headers'] = {'SessionID': session_id, 'StrategyType': 'ConsistentHash'}
    if not enable_thinking:
        request_args['chat_template_kwargs'] = {'reasoning_effort': 'no_think'}
    return {
        'name': 'litellm',
        'config': {'client_args': {'api_key': api_key, 'base_url': base_url}, 'request_args': request_args},
    }


def _spec_opus(*, model: str, base_url: str, api_key: str, effort: str, enable_thinking: bool, session_id: str | None):
    model = model or 'bedrock/anthropic.claude-opus-4-8'
    base_url = base_url or 'http://trpc-gpt-eval.production.polaris:8080'
    api_key = api_key or 'EMPTY'
    # 网关 key(含 query string)pin cache_task_id,让 rebind_cache_task_id 能给每个 instance
    # 重绑独立 rate-limit bucket。裸 key 不动。
    if '?' in api_key or '&' in api_key:
        api_key = _pin_cache_task_id(api_key, session_id)
    # 与 frontend/generator_midtrain._build_openai_clients_openai_opus 对齐:
    # bedrock/converse/ 透传路径(如 aws_third 透传 opus-5)必须显式发 max_tokens——不发会落
    # 网关默认 4096,opus-5 的 max-effort thinking 直接吃满、finish_reason=length 截断在
    # thinking 里、正文无输出 → Claude Code CLI 第一轮就 stop。summarized display 会加剧
    # idle timeout,故 converse 路不发。
    is_bedrock_converse = model.startswith('bedrock/converse/')
    request_args: dict[str, Any] = {
        'model': model,
        'output_config': {'effort': effort or 'xhigh'},
        # drop_params 对所有 opus 路径都必要: bedrock Invoke transformer
        # (bedrock/anthropic.claude-opus-4-8) 不认 thinking 参数,不 drop 会 400
        # UnsupportedParamsError;converse 路也需要它。litellm 只丢 transformer 不认的,不影响其余。
        'drop_params': True,
    }
    if is_bedrock_converse:
        request_args['max_tokens'] = config.get().models.max_tokens or DEFAULT_MAX_TOKENS
    if enable_thinking:
        request_args['thinking'] = {'type': 'adaptive'}
        if not is_bedrock_converse and config.get().models.summarized_thinking:
            request_args['thinking']['display'] = 'summarized'
    if 'fable-5' in model or 'opus-5' in model:
        request_args['max_tokens'] = DEFAULT_MAX_TOKENS
    return {
        'name': 'litellm',
        'config': {
            'client_args': {
                'api_key': api_key,
                'base_url': base_url,
                'timeout': config.get().models.litellm_timeout,
                'cache_control_enabled': True,
                'cache_control_strategy': 'claude_code',
                'log_llm_usage': True,
                # opus-5 走 bedrock/converse/ 单次调用 >5min 非流式撞 CC CLI stream idle
                # timeout(~300s) → 第一轮就 stop。开流式后持续 chunk 不再 idle timeout。
                'stream': config.get().models.litellm_stream,
            },
            'request_args': request_args,
        },
    }


def _spec_openai(
    *, model: str, base_url: str, api_key: str, effort: str, enable_thinking: bool, session_id: str | None
):
    """OpenAI-compatible passthrough (gpt / kimi / qwen / glm / deepseek / grok / doubao-ish …).

    与 frontend/generator_midtrain._build_openai_clients_openai 对齐: 按 model 名分派各家
    thinking 协议(glm-5 / kimi / qwen3.7 / deepseek-v4 / MiniMax-M3 / gpt-5-azure / grok),
    带 max_tokens(gpt-5 用 max_completion_tokens) + drop_params;grok 自动补 openai/ 前缀。
    """
    model = model or 'openai/xxxx'
    base_url = base_url or 'http://llm-api.model-eval.woa.com/v1'
    api_key = api_key or 'EMPTY'
    # xAI grok-* 透传: litellm 需要 openai/ 前缀路由到 OpenAI-compatible handler。
    if 'grok-' in model.lower() and not model.startswith('openai/'):
        model = f'openai/{model}'
    # 网关 key(含 query string)pin cache_task_id → 每 instance 独立 rate-limit bucket。裸 key 不动。
    if '?' in api_key or '&' in api_key:
        api_key = _pin_cache_task_id(api_key, session_id)

    is_grok = 'grok-' in model.lower()
    _eff = (effort or '').strip().lower()

    extra_body: dict[str, Any] = {}
    if 'glm-5' in model:
        extra_body['thinking'] = (
            {'type': 'enabled', 'clear_thinking': False} if enable_thinking else {'type': 'disabled'}
        )
    elif 'kimi' in model:
        extra_body['thinking'] = {'type': 'enabled', 'keep': 'all'} if enable_thinking else {'type': 'disabled'}
    elif 'qwen3.7' in model:
        extra_body = (
            {'enable_thinking': True, 'preserve_thinking': True} if enable_thinking else {'enable_thinking': False}
        )
    elif 'deepseek-v4' in model:
        extra_body = (
            {'thinking': {'type': 'enabled'}, 'reasoning_effort': _eff or 'max'}
            if enable_thinking
            else {'thinking': {'type': 'disabled'}}
        )
    elif 'MiniMax-M3' in model:
        extra_body = (
            {'thinking': {'type': 'adaptive'}, 'reasoning_split': False}
            if enable_thinking
            else {'thinking': {'type': 'disabled'}}
        )
    elif 'gpt-5' in model and 'azure' in model.lower():
        # Azure gpt-5.x 在 /v1/chat/completions 上拒绝 function tools + reasoning_effort
        # (HTTP 400)。claude_code 总带 tools,所以不能推 reasoning_effort;Azure 仍会默认 reasoning。
        extra_body = {}
    elif is_grok:
        # grok reasoning_effort 是 extra_body 里的纯字符串(low|medium|high;默认 high)。
        # 白名单外(xhigh/max/none)→ high。ENABLE_THINKING=false 省略字段(grok 关不掉 reasoning)。
        if enable_thinking:
            eff = _eff if _eff in ('low', 'medium', 'high') else 'high'
            extra_body['reasoning_effort'] = eff
    else:
        extra_body['enable_thinking'] = enable_thinking

    request_args: dict[str, Any] = {
        'model': model,
        'extra_body': extra_body,
        'timeout': config.get().models.litellm_timeout,
        # litellm 自动丢弃模型不支持的参数(azure gpt-5 不接受顶层 reasoning_effort;
        # grok 不接受 frequency_penalty/presence_penalty/stop)。
        'drop_params': True,
    }
    # gpt-5 系列弃用 max_tokens,改用 max_completion_tokens。
    _max_tokens = config.get().models.max_tokens or DEFAULT_MAX_TOKENS
    if 'gpt-5' in model:
        request_args['max_completion_tokens'] = _max_tokens
    else:
        request_args['max_tokens'] = _max_tokens
    return {
        'name': 'litellm',
        'config': {
            'client_args': {
                'api_key': api_key,
                'base_url': base_url,
                # grok 有自己的自动 prompt cache(无 Anthropic cache_control);其余 openai 家
                # 保留 claude_code cache_control 标注。
                'cache_control_enabled': not is_grok,
                'stream': config.get().models.litellm_stream,
            },
            'request_args': request_args,
        },
    }


def strip_anthropic_cache_control(messages: Any, **_kwargs: Any) -> Any:
    """Remove Anthropic cache-control breakpoints before provider submission."""
    if not isinstance(messages, list):
        return messages

    stripped: list[Any] = []
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            stripped.append(message)
            continue
        content = message.get('content')
        has_block_breakpoint = isinstance(content, list) and any(
            isinstance(block, dict) and 'cache_control' in block for block in content
        )
        if 'cache_control' not in message and not has_block_breakpoint:
            stripped.append(message)
            continue
        changed = True
        clean = {key: value for key, value in message.items() if key != 'cache_control'}
        if has_block_breakpoint:
            clean['content'] = [
                {key: value for key, value in block.items() if key != 'cache_control'}
                if isinstance(block, dict)
                else block
                for block in content
            ]
        stripped.append(clean)
    return stripped if changed else messages


_GEMINI_STRIP_CACHE_CONTROL = 'nexus.extensions.tasks.artifact.agent_setup.clients::strip_anthropic_cache_control'


def _spec_gemini(
    *, model: str, base_url: str, api_key: str, effort: str, enable_thinking: bool, session_id: str | None
):
    """Build a native Gemini client with provider-specific generation limits."""
    raw = model or 'gemini-3.5-flash'
    for prefix in ('openai/', 'google/', 'gemini/', 'vertex_ai/'):
        while raw.startswith(prefix):
            raw = raw[len(prefix) :]
    base_url = (base_url or 'http://trpc-gpt-eval.production.polaris:8080').rstrip('/')
    if not base_url.endswith('/v1beta'):
        base_url = f'{base_url}/v1beta'
    # Pin a cache_task_id so concurrent requests land in independent rate-limit
    # buckets (the gateway rate-limits per task id). Without it, N concurrent
    # gemini calls share one bucket → 429 storm under high concurrency.
    api_key = _pin_cache_task_id(api_key or 'EMPTY', session_id)
    request_args: dict[str, Any] = {
        'model': f'gemini/{raw}',
        # 夹住而不是直接取 config: 为别家设的 ARTIFACT_MAX_TOKENS=128000 会让 gemini 全量 400。
        'max_tokens': min(config.get().models.max_tokens or GEMINI_MAX_TOKENS, GEMINI_MAX_TOKENS),
        'timeout': config.get().models.litellm_timeout,
        'reasoning_effort': effort or 'low',
        # Some gemini model routes (e.g. gemini-3.5-flash on the polaris gateway)
        # reject reasoning_effort outright. Let litellm drop params the model does
        # not accept instead of erroring — matches the opus/openai specs above.
        'drop_params': True,
    }
    return {
        'name': 'litellm',
        'config': {
            'client_args': {
                'api_key': api_key,
                'api_base': base_url,
                # 这条路是 vlm judge 的唯一通道，之前没有 max_retries —— 一次网络抖动就让
                # aesthetic 判不出分，实测 26585 次调用里 926 次(3.5%)失败。 对齐 opus 的 3 次。
                'max_retries': 3,
                # Enabled only to reach the strategy hook: the sandbox agent's inbound
                # cache_control breakpoints have to be stripped before they hit a gateway
                # that does not implement Anthropic prompt caching. See
                # strip_anthropic_cache_control.
                'cache_control_enabled': True,
                'cache_control_strategy': _GEMINI_STRIP_CACHE_CONTROL,
            },
            'request_args': request_args,
        },
    }


def _spec_vertex(
    *, model: str, base_url: str, api_key: str, effort: str, enable_thinking: bool, session_id: str | None
):
    """GCP Vertex AI Anthropic passthrough. Needs the vertex_anthropic client type."""
    model = model or 'claude-opus-4-8'
    base_url = base_url or 'http://trpc-gpt-eval.production.polaris:8080'
    api_key = _pin_cache_task_id(api_key or 'EMPTY', session_id)
    request_args: dict[str, Any] = {
        'model': model,
        'anthropic_version': config.get().models.vertex_anthropic_version,
        'max_tokens': config.get().models.max_tokens or DEFAULT_MAX_TOKENS,
        'drop_forced_tool_choice_with_thinking': True,
    }
    if enable_thinking:
        thinking: dict[str, Any] = {'type': config.get().models.vertex_thinking_type}
        budget_tokens = config.get().models.vertex_thinking_budget_tokens
        if budget_tokens is not None:
            thinking['budget_tokens'] = budget_tokens
        # summarized 展示默认开(与 opus Anthropic passthrough 路径一致);IS_SUMMARIZED_THINKING=false 才关。
        if config.get().models.summarized_thinking:
            thinking['display'] = 'summarized'
        request_args['thinking'] = thinking
        request_args['output_config'] = {'effort': effort or 'xhigh'}
    if 'fable-5' in model or 'opus-5' in model:
        request_args['max_tokens'] = DEFAULT_MAX_TOKENS
    return {
        'name': 'vertex_anthropic',
        'config': {
            'client_args': {
                'api_key': api_key,
                'base_url': base_url,
                # vertex 历史默认 900s(短于其他 provider),仅当显式设了 env 才覆盖。
                'timeout': config.get().models.litellm_timeout
                if 'models.litellm_timeout' in config.get().provenance
                else 900.0,
                'max_retries': 3,
                'cache_control_enabled': True,
                'cache_control_strategy': 'claude_code',
                'log_llm_usage': True,
            },
            'request_args': request_args,
        },
    }


def _spec_openai_responses(
    *, model: str, base_url: str, api_key: str, effort: str, enable_thinking: bool, session_id: str | None
):
    """Build an Azure OpenAI Responses client using an openai/responses/ model."""
    model = model or 'openai/responses/gpt-5.5'
    base_url = base_url or 'http://trpc-gpt-eval.production.polaris:8080/openai/v1'
    api_key = api_key or 'EMPTY'
    if '?' in api_key or '&' in api_key:
        api_key = _pin_cache_task_id(api_key, session_id)
    if not enable_thinking:
        eff = 'none'
    else:
        eff = (effort or '').strip().lower()
        if eff not in ('low', 'medium', 'high', 'xhigh'):
            eff = 'xhigh'
    return {
        'name': 'litellm',
        'config': {
            'client_args': {
                'api_key': api_key,
                'base_url': base_url,
                'timeout': config.get().models.litellm_timeout,
                'cache_control_enabled': True,
                'log_llm_usage': True,
                'stream': config.get().models.litellm_stream,
            },
            'request_args': {
                'model': model,
                'drop_params': True,
                'reasoning_effort': {'effort': eff, 'summary': 'auto'},
            },
        },
    }


def _spec_doubao(
    *, model: str, base_url: str, api_key: str, effort: str, enable_thinking: bool, session_id: str | None
):
    """Build a Doubao passthrough client with normalized reasoning effort."""
    model = model or 'doubao-seed-2-1-pro-260628'
    base_url = base_url or 'http://trpc-gpt-eval.production.polaris:8080/api/v3'
    api_key = _pin_cache_task_id(api_key or 'EMPTY', session_id)
    extra_body: dict[str, Any] = {'thinking': {'type': 'enabled' if enable_thinking else 'disabled'}}
    eff = (effort or '').strip().lower()
    if eff and eff not in ('minimal', 'low', 'medium', 'high'):
        eff = 'high'
    if eff:
        extra_body['reasoning_effort'] = eff
    return {
        'name': 'litellm',
        'config': {
            'client_args': {'api_key': api_key, 'base_url': base_url, 'cache_control_enabled': True},
            'request_args': {
                'model': f'openai/{model}',
                'extra_body': extra_body,
                'max_tokens': config.get().models.max_tokens or DEFAULT_MAX_TOKENS,
                'timeout': config.get().models.litellm_timeout,
                'drop_params': True,
            },
        },
    }


# ── Dispatch ─────────────────────────────────────────────────────────────────

_SPEC_BUILDERS = {
    'hy3': _spec_hy3,
    'opus': _spec_opus,
    'openai': _spec_openai,
    'gemini': _spec_gemini,
    'vertex': _spec_vertex,
    'openai_responses': _spec_openai_responses,
    'doubao': _spec_doubao,
}


# Which anthropic-proxy thinking preset a provider's responses need, so the
# response's ``signature`` is preserved and roundtripped on the NEXT request —
# extended thinking requires the signature to come back or the model treats the
# thinking chain as broken.
#   'bedrock-claude' → reads ``thinking_blocks`` and preserve_signature=True
#                      (opus/vertex — and gemini, verified 2026-07-17: even with
#                      thinking DISABLED at the request level, gemini still returns
#                      ``thought_signature`` in every response for multi-turn
#                      continuity; llm_handler's Gemini fallback synthesizes empty
#                      thinking_blocks from ``provider_specific_fields.thought_signatures``
#                      so ``bedrock-claude`` preset picks up the signature.)
#   'hunyuan'        → reads ``reasoning_content`` and preserve_signature=False
#                      (hy3 and openai-compatible upstreams that don't sign)
_THINKING_PRESET_BY_PROVIDER = {
    'gemini': 'bedrock-claude',
    'opus': 'bedrock-claude',
    'vertex': 'bedrock-claude',
}


def thinking_preset_for_provider(provider: str) -> str:
    """Anthropic-proxy thinking preset for a provider (default ``hunyuan``)."""
    return _THINKING_PRESET_BY_PROVIDER.get((provider or '').strip().lower(), 'hunyuan')


# Providers whose client type needs an extra nexus import to be registered.
# (gemini now routes through the built-in litellm client — no extra import.)
PROVIDER_IMPORTS: dict[str, list[str]] = {
    'vertex': ['nexus.extensions.openai_clients.vertex_anthropic'],
}


def build_client_spec(
    provider: str,
    *,
    model: str = '',
    base_url: str = '',
    api_key: str = '',
    effort: str = '',
    enable_thinking: bool = False,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build a single ``{'name': ..., 'config': {...}}`` openai_clients entry.

    Unknown providers fall back to ``openai`` (plain litellm passthrough).
    """
    builder = _SPEC_BUILDERS.get(provider.strip().lower(), _spec_openai)
    return builder(
        model=model,
        base_url=base_url,
        api_key=api_key,
        effort=effort,
        enable_thinking=enable_thinking,
        session_id=session_id,
    )


# ── Shared model table (M_<alias>_* env groups) → nexus openai_clients ───────
# One source of truth for BOTH the offline generator (the generator) AND the RL
# parquet builder: define models once as env groups, get nexus-format clients.

_ModelRow = tuple[str, str, str, str, str]  # (provider, model, base_url, api_key, effort)
_MODEL_ENV_PREFIX = 'M_'
_MODEL_ENV_SUFFIX = '_PROVIDER'


def parse_model_table_from_env() -> dict[str, _ModelRow]:
    """Read models.table, or discover M_<alias>_* groups when no table is supplied.

    Return alias-to-provider/model/endpoint/credential/effort entries.
    """
    table_cfg = config.get().models.table
    if table_cfg:
        return {
            alias: (
                (entry.get('provider') or '').strip(),
                (entry.get('model') or '').strip(),
                (entry.get('base') or '').strip(),
                (entry.get('key') or '').strip(),
                (entry.get('effort') or '').strip(),
            )
            for alias, entry in table_cfg.items()
            if (entry.get('provider') or '').strip()
        }

    aliases = sorted(
        k[len(_MODEL_ENV_PREFIX) : -len(_MODEL_ENV_SUFFIX)]
        for k in os.environ
        if k.startswith(_MODEL_ENV_PREFIX) and k.endswith(_MODEL_ENV_SUFFIX)
    )
    table: dict[str, _ModelRow] = {}
    for alias in aliases:
        provider = os.getenv(f'M_{alias}_PROVIDER', '').strip()
        if not provider:
            continue
        table[alias] = (
            provider,
            os.getenv(f'M_{alias}_MODEL', '').strip(),
            os.getenv(f'M_{alias}_BASE', '').strip(),
            os.getenv(f'M_{alias}_KEY', '').strip(),
            os.getenv(f'M_{alias}_EFFORT', '').strip(),
        )
    return table


def build_named_clients_from_env(*, enable_thinking: bool | None = None) -> dict[str, dict[str, Any]]:
    """Build named clients and bind main, judge, and vlm role aliases.

    Use the configured model table or environment groups, not a mixture of both.
    """
    table = parse_model_table_from_env()
    if not table:
        return {}
    if enable_thinking is None:
        enable_thinking = config.get().models.enable_thinking
    clients: dict[str, dict[str, Any]] = {}
    for alias, (provider, model, base_url, api_key, effort) in table.items():
        clients[alias] = build_client_spec(
            provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            effort=effort,
            enable_thinking=enable_thinking,
            session_id=uuid.uuid4().hex,
        )
    # Role defaults come from config, so a YAML profile can set them; config still reads the
    # ARTIFACT_*_MODEL env names underneath, which is why an all-env run is unaffected.
    models_cfg = config.get().models
    for role, field_name, alias in (
        ('main', 'models.infer_alias', models_cfg.infer_alias),
        ('judge', 'models.judge_alias', models_cfg.judge_alias),
        ('vlm', 'models.vlm_alias', models_cfg.vlm_alias),
    ):
        alias = (alias or '').strip()
        if alias:
            if alias not in clients:
                raise ValueError(f'{field_name}={alias!r} is not in the model table (aliases: {sorted(table)})')
            clients[role] = dict(clients[alias])
    return clients


def table_provider_imports(table: dict[str, _ModelRow]) -> list[str]:
    """Return registry imports required by the configured client providers."""
    imports: list[str] = []
    for provider, *_ in table.values():
        for imp in PROVIDER_IMPORTS.get(provider, []):
            if imp not in imports:
                imports.append(imp)
    return imports


__all__ = [
    'build_client_spec',
    'build_named_clients_from_env',
    'parse_model_table_from_env',
    'rebind_cache_task_id',
    'table_provider_imports',
    'thinking_preset_for_provider',
    'PROVIDER_IMPORTS',
]
