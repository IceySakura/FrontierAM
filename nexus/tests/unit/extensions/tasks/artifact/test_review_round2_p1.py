"""Review 第二轮的三个 P1：分数可信度的三条独立失效路径。

每条都能产出一个**看起来完全正常的分数**，所以都用「修复前的真实行为」当反例
钉住：

* R1 合法零分被子项故障改成 maskout —— 真零分（模型确实做得不对）被当成评测
  故障丢掉，训练信号消失，而 Harbor 侧同一结果保留 0.0。
* R2 已随 ``static_chenxu`` / ``static_general`` 一起删除（那个 judge 不再存在）。
  教训留档：它先 ``float()`` 再 clamp，``max(0.0, min(5.0, nan))`` 返回 ``5.0``
  —— NaN 参与比较全为 False，于是「模型没给出分数」被洗成满分。**任何把模型
  输出转成分数的地方，都要在 clamp 之前校验类型 / 有限性 / 范围。**
* R3 上传失败不抛异常而是写在响应里 —— 判分读空目录仍给分。
"""

from __future__ import annotations

import asyncio
import base64
import io
import tarfile

from dataclasses import dataclass

import pytest

from nexus.extensions.tasks.artifact import task as artifact_task
from nexus.extensions.tasks.artifact.harbor_entry import verdict_to_reward
from nexus.extensions.tasks.artifact.judges.base import JudgeResult


# ══════════════════════ R1: 合法零分 vs 子项故障 ══════════════════════


def test_legal_zero_with_failing_child_is_not_masked_out():
    """顶层合法零分 + 可降级子项故障 → ``ALL_ZERO``，不是评测故障。

    复合 judge 已经决定了「这条腿失败可以降级，总分算 0」。task 越过它重扫
    ``child_results`` 就推翻了这个决定，把真零分当成评测故障屏蔽 —— 而 Harbor
    的 ``verdict_to_reward`` 只读顶层，给 0.0 保留在训练里。同一个结果两端语义
    相反，正是重构要消除的那类分歧。
    """
    child = JudgeResult(judge_name='leg', score=0.0, error='bad json', failure_kind='parse_failed')
    top = JudgeResult(judge_name='comp', score=0.0, child_results=[child])

    score, _resolved, zero_reason = artifact_task.resolve_eval_outcome(top)

    assert score == 0.0
    assert zero_reason == 'ALL_ZERO', '合法零分只看顶层'


def test_task_and_harbor_agree_on_a_legal_zero():
    """同一个结果，两条路径的 reward 语义必须一致。"""
    child = JudgeResult(judge_name='leg', score=0.0, error='e', failure_kind='vlm_failed')
    top = JudgeResult(judge_name='comp', score=0.0, child_results=[child])

    _score, _resolved, zero_reason = artifact_task.resolve_eval_outcome(top)
    harbor_reward = verdict_to_reward(top, judge_name='comp')['reward']

    assert zero_reason == 'ALL_ZERO'
    assert harbor_reward == 0.0
    # 两边都把它当成「保留在训练里的真零分」。
    assert harbor_reward != -1.0


def test_top_level_failure_still_masks_out():
    """顶层自己带 failure_kind 时仍须屏蔽 —— 修 R1 不能把这条也放过。"""
    top = JudgeResult(judge_name='x', score=0.0, error='boom', failure_kind='parse_failed')

    _score, resolved, zero_reason = artifact_task.resolve_eval_outcome(top)

    assert zero_reason == 'EVAL_PARSE_FAILED'
    assert resolved is False


@pytest.mark.parametrize(
    ('label', 'kwargs'),
    [
        ('满分 + 故障', {'score': 1.0, 'failure_kind': 'parse_failed', 'error': 'e'}),
        ('正分 + 故障', {'score': 0.6, 'failure_kind': 'vlm_failed', 'error': 'e'}),
        ('显式 passed + 故障', {'score': 0.0, 'failure_kind': 'no_client', 'error': 'e', 'passed': False}),
    ],
)
def test_failure_forces_resolved_false(label: str, kwargs: dict):
    """评测故障 ⇒ 任务不可能「已解决」。

    ``score=1.0`` + ``passed=None`` + ``parse_failed`` 曾推导出 ``resolved=True``：
    奖励被屏蔽了，任务却记成已解决 —— 统计和复盘都会被污染，而且这种条目最难
    发现，因为它看起来是一次成功。
    """
    top = JudgeResult(judge_name='x', **kwargs)

    _score, resolved, _zero_reason = artifact_task.resolve_eval_outcome(top)

    assert resolved is False, f'{label}: 故障结果不得标记为已解决'


def test_child_diagnostics_are_still_available():
    """子项诊断不参与 reward 分类，但不能因此丢失 —— 排查还得靠它。"""
    child = JudgeResult(judge_name='leg', score=0.0, error='bad json', failure_kind='parse_failed')
    top = JudgeResult(judge_name='comp', score=0.0, child_results=[child])

    payload = verdict_to_reward(top, judge_name='comp')

    assert payload['child_scores'] == {'leg': 0.0}
    assert top.child_results[0].failure_kind == 'parse_failed'


def test_legal_nonzero_score_passes_through():
    """没有故障的正常分数不受影响。"""
    top = JudgeResult(judge_name='x', score=0.75, passed=True)

    score, resolved, zero_reason = artifact_task.resolve_eval_outcome(top)

    assert (score, resolved, zero_reason) == (0.75, True, None)


# ══════════════════════ R3: 上传失败被当成成功 ══════════════════════


@dataclass
class _FileResult:
    """``FileOperationResult`` 的形状。"""

    path: str
    success: bool
    message: str


@dataclass
class _UploadResponse:
    """``UploadFileResponse`` 的形状。"""

    results: list


def _tiny_archive() -> str:
    """一个含单个文件的真实 tar.gz —— 走真实解包路径，不 mock 掉它。"""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        data = b'<html>hi</html>'
        info = tarfile.TarInfo('index.html')
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


class _Runtime:
    def __init__(self, response):
        self._response = response

    async def upload_file(self, **_kwargs):
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _upload(response) -> tuple[int, str | None]:
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import upload_archive_to_runtime

    return asyncio.run(upload_archive_to_runtime(_Runtime(response), _tiny_archive(), '/workspace'))


def test_per_file_failure_blocks_scoring():
    """``results[].success=False`` 必须阻断判分。

    真实 executor 把权限/内容/覆盖类失败记在响应里**并正常返回**，不抛异常。
    只 try/except 的话这里返回 ``(1, None)``，调用方认为源码已就位 ——
    ``code_quality`` 的 agent 去评一个空目录，照样给出自信的分数。
    """
    count, error = _upload(
        _UploadResponse(results=[_FileResult('/workspace/index.html', False, 'permission denied')])
    )

    assert count == 0
    assert error is not None
    assert 'permission denied' in error


def test_successful_upload_reports_the_count():
    count, error = _upload(_UploadResponse(results=[_FileResult('/workspace/index.html', True, 'ok')]))

    assert (count, error) == (1, None)


def test_missing_results_is_not_assumed_successful():
    """拿不到逐文件结果时不假设成功 —— 静默放过正是这个 bug 的成因。"""
    count, error = _upload(None)

    assert count == 0
    assert error is not None
    assert 'no per-file results' in error


def test_dict_shaped_response_is_understood():
    """远端 runtime 走序列化，可能是 dict —— 两种形状都要能读出失败。"""
    count, error = _upload({'results': [{'path': '/workspace/index.html', 'success': False, 'message': 'disk full'}]})

    assert count == 0
    assert 'disk full' in (error or '')

    ok_count, ok_error = _upload({'results': [{'path': '/workspace/index.html', 'success': True, 'message': ''}]})
    assert (ok_count, ok_error) == (1, None)


def test_result_count_mismatch_is_an_error():
    """数量对不上意味着有文件被静默丢弃，判分读到的源码不完整。"""
    count, error = _upload(_UploadResponse(results=[]))

    assert count == 0
    assert error is not None


def test_upload_exception_still_reported():
    """异常路径不能因为新增检查而丢失。"""
    count, error = _upload(RuntimeError('connection reset'))

    assert count == 0
    assert 'connection reset' in (error or '')
