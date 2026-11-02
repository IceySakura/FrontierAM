"""v2 verifier 注册表 + shell/eq/regex/llm_judge verifier + v2 分发 shim
（``_run_v2_check``）。

``_v_eq`` 和 ``_run_v2_check`` 会取用 ``dispatch.py`` 中的 ``_dotted_get`` /
``_result_record`` / ``_accepts_limits``——这些是延迟导入（在函数体内），
因为 ``dispatch.py`` 会从本模块导入 ``_run_v2_check``。
"""

import asyncio
import inspect
import logging
import re

from collections.abc import Callable
from typing import Any

from .config import _DEFAULT_LIMITS, RubricLimits, VerifierFn, VerifierResult
from .judge import _llm_judge
from .selectors import _gather_source


_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verifier 注册表
# ---------------------------------------------------------------------------


_VERIFIERS: dict[str, VerifierFn] = {}


def register_verifier(kind: str) -> Callable[[VerifierFn], VerifierFn]:
    """装饰器：以 `kind` 注册一个 verifier。

    该函数接收 ``(state, check, *, runtime)`` 并必须返回
    ``(passed, verdict, detail, raw_head)``。 ``runtime`` 是沙箱运行时句柄，
    用于 shell/REST 类检查；纯 state 类 verifier 会忽略它。 同步或
    ``async def`` 均可。
    """

    def deco(fn: VerifierFn) -> VerifierFn:
        _VERIFIERS[kind] = fn
        return fn

    return deco


@register_verifier('llm_judge')
async def _v_llm_judge(
    state: dict,
    check: dict,
    *,
    runtime: Any,
    limits: RubricLimits | None = None,
) -> VerifierResult:
    """v2 LLM-judge verifier。 async 让多个 judge 调用可在
    :func:`run_rubric_async` 的信号量下并发扇出——底层的 ``_llm_judge``
    是同步的（urllib + ``time.sleep`` 重试），因此我们通过
    ``asyncio.to_thread`` 交给 worker 线程，避免阻塞事件循环。 分发器
    已经通过 ``inspect.isawaitable`` 等待返回的协程，因此这个改名对
    调用方是透明的。
    """
    source = check.get('source', 'all')
    source_ref = check.get('source_ref') or {}
    claim = check.get('claim', '')
    if not claim:
        return False, 'INVALID', 'empty_claim', ''
    lim = limits or _DEFAULT_LIMITS
    source_text = _gather_source(state, source, source_ref, limits=lim)
    if not source_text:
        return False, 'EMPTY_SOURCE', f'empty_source ({source})', ''
    verdict, raw_head = await asyncio.to_thread(
        _llm_judge,
        source_text,
        claim,
        max_source_chars=lim.judge_source_max_chars,
    )
    return verdict == 'Y', verdict, f'verdict={verdict}', raw_head


@register_verifier('eq')
def _v_eq(state: dict, check: dict, *, runtime: Any) -> VerifierResult:
    """针对原始 state 的点分路径进行结构化相等检查。

    ``source`` 是点分路径（``a.b.c`` 或 ``items.0.name``）。 ``expected``
    可以是 JSON 字面值（string/int/bool/None），也可以是带
    ``one_of: [...]`` 的映射来接受列表中的任一值。
    """
    from .dispatch import _dotted_get

    path = check.get('source', '')
    expected = check.get('expected')
    if not path:
        return False, 'INVALID', 'empty_source_path', ''
    if 'expected' not in check:
        return False, 'INVALID', 'missing_expected', ''
    found, got = _dotted_get(state, path)
    if not found:
        return False, 'MISSING', f'path_not_found: {path}', ''
    if isinstance(expected, dict) and 'one_of' in expected:
        allowed = expected['one_of']
        passed = got in allowed if isinstance(allowed, list) else False
        detail = f'got={got!r} one_of={allowed!r}'
    else:
        passed = got == expected
        detail = f'got={got!r} expected={expected!r}'
    return passed, 'Y' if passed else 'N', detail, ''


@register_verifier('regex')
def _v_regex(
    state: dict,
    check: dict,
    *,
    runtime: Any,
    limits: RubricLimits | None = None,
) -> VerifierResult:
    """在 LLM judge 会看到的同一份扁平化文本上做正则搜索。

    ``source`` 复用 :func:`_gather_source` 的选择器（默认为 ``all``）。
    ``pattern`` 是 Python 正则。 默认开启 DOTALL，因此 ``.`` 会跨越换行；
    可在 check 上设 ``dotall: false`` 关闭。
    """
    pattern = check.get('pattern', '')
    if not pattern:
        return False, 'INVALID', 'empty_pattern', ''
    source = check.get('source', 'all')
    source_ref = check.get('source_ref') or {}
    text = _gather_source(state, source, source_ref, limits=limits)
    if not text:
        return False, 'EMPTY_SOURCE', f'empty_source ({source})', ''
    flags = re.DOTALL if check.get('dotall', True) else 0
    try:
        match = re.search(pattern, text, flags=flags)
    except re.error as exc:
        return False, 'INVALID', f'bad_regex: {exc}', ''
    passed = match is not None
    detail = f'matched={bool(match)}'
    if match:
        detail += f' at={match.start()}:{match.end()}'
    return passed, 'Y' if passed else 'N', detail, ''


@register_verifier('shell')
async def _v_shell(state: dict, check: dict, *, runtime: Any) -> VerifierResult:
    """在沙箱 runtime 内运行命令；退出码匹配则通过。

    必填：``command``。 可选：``expect_exit_code``（默认 0）、
    ``timeout_s``（默认 30）。 仅在 ``runtime`` 已被穿到
    :func:`run_rubric_async` 时可用——否则该 verifier 会直接短路为 INVALID。
    """
    cmd = check.get('command', '')
    if not cmd:
        return False, 'INVALID', 'empty_command', ''
    if runtime is None:
        return False, 'NO_RUNTIME', 'shell_verifier_needs_runtime', ''
    want_code = int(check.get('expect_exit_code', 0))
    timeout_s = int(check.get('timeout_s', 30))
    try:
        res = await runtime.run_command(cmd, timeout=timeout_s)
    except Exception as exc:
        return False, 'ERROR', f'run_command_failed: {exc}', ''
    # Nexus runtime 返回 RunCommandResponse.return_code；subprocess 风格的
    # 包装则用 `exit_code`；两者都接受，这样同一个 verifier 既能作用于单元
    # 测试里的裸 asyncio.subprocess Mock，也能作用于真正的 nexus 沙箱
    # runtime。
    exit_code = getattr(res, 'return_code', None)
    if exit_code is None:
        exit_code = getattr(res, 'exit_code', None)
    stdout = getattr(res, 'stdout', '') or ''
    passed = exit_code == want_code
    head = stdout[:160].replace('\n', ' ⏎ ')
    detail = f'exit={exit_code} want={want_code} stdout_head={head!r}'
    return passed, 'Y' if passed else 'N', detail, ''


async def _run_v2_check(
    check: dict,
    state: dict,
    *,
    runtime: Any,
    limits: RubricLimits,
) -> dict[str, Any]:
    """通过 ``verify:`` 注册表分发单个 v2 check。

    行为对齐 ``run_rubric_async`` 中原本的内联实现（连
    UNKNOWN_KIND / ERROR 兜底也一并保留），因此把 for 循环替换为
    ``asyncio.gather`` 时不会改变任何单个 check 的语义。 verifier 内部
    抛出的任何异常都会被捕获成一条 ERROR 记录，而不是向上冒泡——否则
    一个 verifier 出错就会把整个 gather 取消掉，丢失所有其它进行中
    check 的判定。
    """
    from .dispatch import _accepts_limits, _result_record

    kind = check.get('verify') or 'llm_judge'
    fn = _VERIFIERS.get(kind)
    if fn is None:
        return _result_record(
            check,
            kind,
            False,
            'UNKNOWN_KIND',
            f'unknown verify kind: {kind!r}',
            '',
        )
    cid = check.get('id', '')
    try:
        # 早于该 kwarg 注册的外部 v2 verifier 可能不接受 ``limits``。
        # 只在 callee 能接收时才穿透，这样自定义 verifier 保留原本的
        # ``(state, check, *, runtime)`` 签名。
        kwargs: dict[str, Any] = {'runtime': runtime}
        if _accepts_limits(fn):
            kwargs['limits'] = limits
        ret = fn(state, check, **kwargs)
        if inspect.isawaitable(ret):
            ret = await ret
        passed, verdict, detail, raw_head = ret
    except Exception as exc:
        _logger.exception('[rubric] verifier %r raised on check id=%s', kind, cid)
        passed, verdict, detail, raw_head = (
            False,
            'ERROR',
            f'verifier_exception: {exc}',
            '',
        )
    return _result_record(check, kind, passed, verdict, detail, raw_head)
