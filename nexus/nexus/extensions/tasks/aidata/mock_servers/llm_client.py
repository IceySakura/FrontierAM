"""供 LLM 后端 mock 使用的极简 OpenAI 兼容 chat-completion 助手。

读取三个环境变量（均可选 —— 若缺失， llm_chat() 会抛出
LLMUnavailableError，调用方可优雅回退到脚本化回复）：

  NEXUS_MOCK_LLM_BASE_URL  例如 http://llm-api.model-eval.woa.com
  NEXUS_MOCK_LLM_API_KEY   APP_ID:APP_KEY
  NEXUS_MOCK_LLM_MODEL     例如 api_naci_default_gemini-3.1-pro-preview

若需要指向与 agent / mock 服务器 **不同** 的 LLM （例如 rubric judge
在 ``JUDGE_OPENAI_BASE_URL`` 独立于 ``SEARCH_OPENAI_BASE_URL`` 时），
调用方可通过 ``model_env`` / ``base_url_env`` / ``api_key_env`` 关键字
参数覆盖一个或多个环境变量名 —— 当覆盖变量未设置时会回退到对应的
``NEXUS_MOCK_LLM_*``，因此单 LLM 部署无需任何配置改动。

仅使用标准库（urllib 、 json）—— 不依赖 litellm / openai。
"""

from __future__ import annotations

import http.client
import json
import os
import re
import sys
import threading
import urllib.error
import urllib.request


class LLMUnavailableError(RuntimeError):
    pass


class LLMBudgetExceededError(LLMUnavailableError):
    """单进程 LLM 调用预算已耗尽。"""


def llm_available() -> bool:
    return bool(
        os.getenv('NEXUS_MOCK_LLM_BASE_URL')
        and os.getenv('NEXUS_MOCK_LLM_API_KEY')
        and os.getenv('NEXUS_MOCK_LLM_MODEL')
    )


_BUDGET_LOCK = threading.Lock()
_BUDGET_USED = 0
_BUDGET_LIMIT: int | None = None


def _budget_limit() -> int:
    global _BUDGET_LIMIT
    if _BUDGET_LIMIT is None:
        try:
            _BUDGET_LIMIT = max(1, int(os.environ.get('NEXUS_MOCK_LLM_MAX_CALLS', '50')))
        except ValueError:
            _BUDGET_LIMIT = 50
    return _BUDGET_LIMIT


def _consume_budget() -> int:
    global _BUDGET_USED
    limit = _budget_limit()
    with _BUDGET_LOCK:
        if limit <= _BUDGET_USED:
            raise LLMBudgetExceededError(f'LLM call budget exhausted ({_BUDGET_USED}/{limit})')
        _BUDGET_USED += 1
        return _BUDGET_USED


def llm_call_budget_status() -> tuple[int, int]:
    return _BUDGET_USED, _budget_limit()


def llm_chat(
    *,
    system: str,
    user: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    timeout_s: float = 60.0,
    model_env: str = 'NEXUS_MOCK_LLM_MODEL',
    base_url_env: str | None = None,
    api_key_env: str | None = None,
) -> str:
    """调用 chat/completions 并返回 assistant 消息内容。

    ``*_env`` 关键字参数指定用于路由的环境变量名； 每个都会在未设置时
    回退到 ``NEXUS_MOCK_LLM_*`` 默认值，因此常见的 mock 服务器场景
    仍是无参调用。 rubric judge 会传入
    ``base_url_env='NEXUS_JUDGE_BASE_URL'`` / ``api_key_env=
    'NEXUS_JUDGE_API_KEY'`` / ``model_env='NEXUS_JUDGE_MODEL'``，
    以便独立的 ``JUDGE_OPENAI_BASE_URL`` 不会被预解析的
    ``NEXUS_MOCK_LLM_BASE_URL`` 覆盖（后者历史上从
    ``SEARCH_OPENAI_BASE_URL`` 派生并会破坏 judge 的路由 ——
    参见 mcp_catalog.normalize_host_env）。
    """
    base_url = (os.getenv(base_url_env) if base_url_env else None) or os.getenv('NEXUS_MOCK_LLM_BASE_URL')
    api_key = (os.getenv(api_key_env) if api_key_env else None) or os.getenv('NEXUS_MOCK_LLM_API_KEY')
    model = os.getenv(model_env) or os.getenv('NEXUS_MOCK_LLM_MODEL')
    if not (base_url and api_key and model):
        raise LLMUnavailableError('NEXUS_MOCK_LLM_* env vars not set')

    used = _consume_budget()
    if used == _budget_limit():
        sys.stderr.write(f'[llm_client] LLM call budget reached limit ({used})\n')

    for prefix in ('openai/', 'anthropic/', 'gemini/'):
        if model.startswith(prefix):
            model = model[len(prefix) :]
            break

    # base_url 可能已经带有 '/v1' 前缀
    # （如 JUDGE_OPENAI_BASE_URL=http://host:8000/v1）。 仅在缺失时
    # 追加 '/v1'，避免出现重复的 '/v1/v1/chat/completions'，
    # 该形式在 OpenAI 兼容代理上会 404 （曾在 judge 端点 base_url
    # 已以 /v1 结尾时观察到）。
    _base = base_url.rstrip('/')
    url = _base + ('/chat/completions' if _base.endswith('/v1') else '/v1/chat/completions')
    payload = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}',
        },
        method='POST',
    )
    # 网络 / HTTP / 解码失败统一转为 ``LLMUnavailableError``，
    # 这样调用方（mock 服务器 、 rubric judge）可通过单一异常类型
    # 分支来处理回退行为。 若不做统一，短暂的 gongfeng 中断会传播
    # 原始 urllib 异常，回退将无法触发。
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode('utf-8', errors='replace')
    except urllib.error.HTTPError as exc:
        head = exc.read()[:300] if hasattr(exc, 'read') else b''
        raise LLMUnavailableError(f'LLM HTTP {exc.code} from {url}: {head!r}') from exc
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError) as exc:
        raise LLMUnavailableError(f'LLM transport error to {url}: {exc!r}') from exc

    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError(f'LLM returned non-JSON from {url}: {body[:300]!r}') from exc
    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
    if not content:
        raise LLMUnavailableError(f'empty content from LLM: {body[:300]}')
    return content.strip()


def extract_json(text: str):
    """从 ``text`` 中稳健地提取第一个 JSON 对象或数组。"""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith('```'):
        nl = cleaned.find('\n')
        if nl > 0:
            cleaned = cleaned[nl + 1 :]
        if cleaned.rstrip().endswith('```'):
            cleaned = cleaned.rstrip()[:-3]

    first_brace = cleaned.find('{')
    first_bracket = cleaned.find('[')
    candidates = []
    if first_brace >= 0:
        candidates.append((first_brace, '{', '}'))
    if first_bracket >= 0:
        candidates.append((first_bracket, '[', ']'))
    if not candidates:
        return None
    start_idx, _open_ch, close_ch = min(candidates, key=lambda x: x[0])
    last = cleaned.rfind(close_ch)
    if last <= start_idx:
        return None
    body = cleaned[start_idx : last + 1]

    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    repaired = re.sub(r',(\s*[}\]])', r'\1', body)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None


def llm_chat_json(*, system: str, user: str, **kwargs):
    """调用 chat/completions 并尽力从响应中解析出 JSON。"""
    text = llm_chat(system=system, user=user, **kwargs)
    parsed = extract_json(text)
    if parsed is None:
        raise ValueError(f'could not extract JSON from LLM response: {text[:200]!r}')
    return parsed
