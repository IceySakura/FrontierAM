"""``test_result`` 原语 —— 通过 check.id 解析 junit 测试用例。"""

from typing import Any

from ..config import JunitLookupContext, RubricLimits, VerifierResult


async def _eval_test_result(
    cfg: dict, state: dict, runtime: Any, junit_ctx: JunitLookupContext | None, *, limits: RubricLimits | None = None
) -> VerifierResult:
    """通过稳定的 check.id（ ``cfg['id']`` ）或显式的 ``test_name`` 解析 junit 测试用例。

    要求 rubric 在调用时提供了 :class:`JunitLookupContext` 。
    """
    if junit_ctx is None:
        return False, 'NO_JUNIT', 'rubric_invoked_without_junit', ''
    # 优先使用 assert cfg 中的显式 id；否则回退到分发器为我们透传的 check.id
    # （参见 `run_rubric_async` ）。
    check_id = cfg.get('id') or cfg.get('check_id') or cfg.get('_check_id', '')
    test_name = cfg.get('test_name')
    from ..check_map import resolve_testcase

    if test_name:
        tc = junit_ctx.report.find(test_name)
    else:
        if not check_id:
            return False, 'INVALID', 'test_result_needs_id_or_test_name', ''
        tc = resolve_testcase(check_id, junit_ctx.report, explicit_map=junit_ctx.explicit_map)
    if tc is None:
        available = junit_ctx.report.all_names()[:8]
        return (
            False,
            'MISSING',
            f'no junit testcase matched check_id={check_id!r} test_name={test_name!r}; '
            f'first 8 names in report: {available!r}',
            '',
        )
    passed = tc.passed
    detail = f'junit_name={tc.name!r} passed={passed} msg={tc.message[:160]!r}'
    return passed, 'Y' if passed else 'N', detail, ''
