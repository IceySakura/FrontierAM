"""LLM 判定器（Y / N ）辅助模块。

同步 urllib+time.sleep 的判定器， 由 ``_v_llm_judge`` 与
``_eval_judge_claim`` 通过 ``asyncio.to_thread`` 调用。 自包含 —— 无
其他新增文件的导入。
"""

import logging
import os
import time


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM 判定器（Y / N ）
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM_PROMPT = (
    'You are an impartial judge. Given a SOURCE (structured / textual evidence) '
    'and a CLAIM about that source, decide whether the source supports the claim. '
    'Output EXACTLY one character:\n'
    '  Y — the source supports the claim (substantive evidence, not just a mention).\n'
    '  N — the source does not support the claim, is empty, or only name-drops it.\n\n'
    'No punctuation, no explanation, no other words — just Y or N.'
)


_RAW_RESPONSE_HEAD_CHARS = 200


def _judge_once(
    source_text: str,
    claim: str,
    timeout_s: float,
    *,
    max_source_chars: int = 48_000,
) -> tuple[str, str]:
    """单次 LLM 判定尝试。 返回 ``(verdict, raw_head)`` 。

    `verdict` ∈ {'Y','N','UNAVAILABLE'}； `raw_head` 是模型字面
    响应的前 ~200 个字符（当调用在读取文本前失败时为空），
    捕获这个字段是为了让 rubric_results 即便在 parser 走到
    UNAVAILABLE 时也带有可检查的信号。

    ``max_source_chars`` 会截断传入判定器的 SOURCE 块， 使得
    某个病态 trajectory 无法把每次判定调用都撑爆。 默认（48k ）
    在 :func:`_gather_trajectory` 的 32k 用户初始消息上限之外，
    留出足以承载 assistant 回合 + 工具调用 + 工具结果的余量。
    """
    # 将 JUDGE_OPENAI_* → NEXUS_JUDGE_* 映射（judge 专用路由）。 这些
    # 是 有意 的 硬 赋值 （非 setdefault ）： 当同时设置了
    # JUDGE_OPENAI_BASE_URL 与 SEARCH_OPENAI_BASE_URL 时，
    # mcp_catalog.normalize_host_env 会先用 SEARCH_OPENAI_BASE_URL
    # 播下 NEXUS_MOCK_LLM_BASE_URL ， 因此 judge 命名空间必须
    # 独立填充， 否则会静默继承 search URL 。
    # 权威解析路径与本次修复的 bug 历史见
    # engine/mcp_catalog.py::resolve_judge_creds 。
    for src, dst in [
        ('JUDGE_OPENAI_API_KEY', 'NEXUS_JUDGE_API_KEY'),
        ('JUDGE_OPENAI_BASE_URL', 'NEXUS_JUDGE_BASE_URL'),
        ('JUDGE_OPENAI_MODEL', 'NEXUS_JUDGE_MODEL'),
    ]:
        val = os.environ.get(src)
        if val:
            os.environ[dst] = val
    # 当 NEXUS_JUDGE_* 未由 normalize_host_env 预填充（例如
    # 宿主机侧的单元测试、 临时 CLI 运行）时的回退链。
    # 三个 NEXUS_JUDGE_* 必须在 llm_chat 读取它们时都已设置 ——
    # 它的 base_url_env / api_key_env / model_env 只有在 judge
    # 环境未设置时才会回退到 NEXUS_MOCK_LLM_* ， 因此在此处
    # 播种要比把回退策略泄漏到 llm_chat 表面便宜。
    if not os.environ.get('NEXUS_JUDGE_BASE_URL'):
        fallback_base = os.environ.get('NEXUS_MOCK_LLM_BASE_URL') or 'http://llm-api.model-eval.woa.com'
        os.environ['NEXUS_JUDGE_BASE_URL'] = fallback_base
    if not os.environ.get('NEXUS_JUDGE_API_KEY'):
        for cand in ('NEXUS_MOCK_LLM_API_KEY', 'LLM_API_KEY'):
            v = os.environ.get(cand)
            if v:
                os.environ['NEXUS_JUDGE_API_KEY'] = v
                break
    os.environ.setdefault(
        'NEXUS_JUDGE_MODEL',
        'api_naci_default_gemini-3.1-pro-preview',
    )
    # 也把这些回退镜像到 NEXUS_MOCK_LLM_* 上， 使得任何先导入本
    # 模块的非 judge 调用者（例如在 normalize_host_env 之前运行的
    # 宿主侧 mock 客户端）仍能看到合理的默认值。 这里纯粹是
    # setdefault —— 绝不会覆盖显式设置。
    if os.environ.get('NEXUS_MOCK_LLM_API_KEY') is None and os.environ.get('LLM_API_KEY'):
        os.environ['NEXUS_MOCK_LLM_API_KEY'] = os.environ['LLM_API_KEY']
    os.environ.setdefault('NEXUS_MOCK_LLM_BASE_URL', 'http://llm-api.model-eval.woa.com')
    os.environ.setdefault('NEXUS_MOCK_LLM_MODEL', 'api_naci_default_gemini-3.1-pro-preview')

    # 供下方 UNAVAILABLE 诊断使用的共享上下文（是哪个 judge 模型
    # 产出了失败判定， 判定的又是哪个 claim ）。 UNAVAILABLE 会被
    # 记为一次（假的）失败， 因此每条返回它的路径都要记录原因。
    judge_model = os.environ.get('NEXUS_JUDGE_MODEL', '?')
    claim_preview = claim.replace('\n', ' ').strip()[:160]

    try:
        from ..mock_servers.llm_client import LLMUnavailableError, llm_chat
    except ImportError as exc:
        _logger.warning(
            '[rubric] judge UNAVAILABLE: llm_client import failed (%s) | model=%s claim=%r',
            exc,
            judge_model,
            claim_preview,
        )
        return 'UNAVAILABLE', ''

    trimmed = (
        source_text
        if len(source_text) <= max_source_chars
        else (source_text[:max_source_chars] + f'\n…(source truncated at {max_source_chars} chars)')
    )
    user = f'SOURCE:\n{trimmed}\n\nCLAIM:\n{claim}'
    try:
        text = llm_chat(
            system=_JUDGE_SYSTEM_PROMPT,
            user=user,
            temperature=0.0,
            # 为推理型 judge 模型留出 headroom ： 它们会先把 token 预算
            # 烧在 reasoning_content 上， 因此小上限（原为 800 ）会得到
            # 空 content ， finish_reason=length → LLMUnavailableError → 该
            # check 会被记为（假的）失败。 判定本身只是一个单字符 Y/N ；
            # 额外预算纯粹是 reasoning 的 headroom 。
            max_tokens=8192,
            timeout_s=timeout_s,
            model_env='NEXUS_JUDGE_MODEL',
            base_url_env='NEXUS_JUDGE_BASE_URL',
            api_key_env='NEXUS_JUDGE_API_KEY',
        )
    except LLMUnavailableError as exc:
        # 最常见的 UNAVAILABLE 原因： 空 content / finish_reason=length
        # （耗光 token 预算的推理型 judge 模型）。 ``exc`` 携带响应体
        # 头部 —— 模型名 + finish_reason —— 因此原样记录。
        _logger.warning(
            '[rubric] judge UNAVAILABLE: llm_chat unavailable (%s) | model=%s claim=%r',
            exc,
            judge_model,
            claim_preview,
        )
        return 'UNAVAILABLE', ''
    except Exception as exc:
        _logger.warning(
            '[rubric] judge UNAVAILABLE: judge call raised %s: %s | model=%s claim=%r',
            type(exc).__name__,
            exc,
            judge_model,
            claim_preview,
        )
        return 'UNAVAILABLE', ''

    raw_head = (text or '')[:_RAW_RESPONSE_HEAD_CHARS]
    cleaned = (text or '').strip().upper()
    if not cleaned:
        _logger.warning(
            '[rubric] judge UNAVAILABLE: blank verdict text | model=%s claim=%r raw=%r',
            judge_model,
            claim_preview,
            raw_head,
        )
        return 'UNAVAILABLE', raw_head
    for ch in (cleaned[0], cleaned[-1]):
        if ch == 'Y':
            return 'Y', raw_head
        if ch == 'N':
            return 'N', raw_head
    if 'Y' in cleaned and 'N' not in cleaned:
        return 'Y', raw_head
    if 'N' in cleaned and 'Y' not in cleaned:
        return 'N', raw_head
    _logger.warning(
        '[rubric] judge UNAVAILABLE: unparseable verdict (no decisive Y/N) | model=%s claim=%r raw=%r',
        judge_model,
        claim_preview,
        raw_head,
    )
    return 'UNAVAILABLE', raw_head


def _llm_judge(
    source_text: str,
    claim: str,
    *,
    timeout_s: float = 45.0,
    max_retries: int = 3,
    retry_sleep_s: float = 1.5,
    max_source_chars: int = 48_000,
) -> tuple[str, str]:
    """带 UNAVAILABLE 重试的 LLM 判定器。 返回 ``(verdict, raw_head)`` 。

    当判定为 UNAVAILABLE 时最多重试 `max_retries` 次 —— 任何决定性的
    Y / N 都会立即返回。 首次 决定性 （或最后一次） 尝试 的
    raw_head 会被保留。 ``max_source_chars`` 会传入
    :func:`_judge_once` 用作 SOURCE 上限。
    """
    # 环境变量覆盖 —— 大规模 fan-out 下 （例如 model-eval 网关在负载下变慢时）
    # 运维可以通过它调高 judge 韧性， 而不动默认值。 未设置 → 保留传入 / 默认值。
    max_retries = max(1, int(os.environ.get('NEXUS_JUDGE_MAX_RETRIES', '') or max_retries))
    last_verdict = 'UNAVAILABLE'
    last_raw = ''
    for attempt in range(1, max_retries + 1):
        verdict, raw_head = _judge_once(
            source_text,
            claim,
            timeout_s,
            max_source_chars=max_source_chars,
        )
        if verdict in ('Y', 'N'):
            if attempt > 1:
                _logger.info('[rubric] judge recovered on retry %d/%d', attempt, max_retries)
            return verdict, raw_head
        last_verdict, last_raw = verdict, raw_head
        if attempt < max_retries:
            _logger.warning('[rubric] judge returned UNAVAILABLE (attempt %d/%d); retrying', attempt, max_retries)
            time.sleep(retry_sleep_s)
    _logger.error(
        '[rubric] judge exhausted %d retries — returning UNAVAILABLE (last_raw=%r)',
        max_retries,
        last_raw,
    )
    return last_verdict, last_raw
