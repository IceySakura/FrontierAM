import base64
import inspect
import io
import json
import re

from types import SimpleNamespace

import pytest

from nexus.extensions.tasks.artifact.judges.capabilities.deploy import deploy_failure_results
from nexus.extensions.tasks.artifact.judges.strategies import web_judger
from nexus.extensions.tasks.artifact.judges.strategies.web_judger import executor, flow, verifier
from nexus.runtime.executors.models.file import FileItem


def _png_b64(color: tuple[int, int, int]) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new('RGB', (8, 8), color).save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


@pytest.mark.asyncio
async def test_download_shots_matches_files_by_path() -> None:
    class Runtime:
        async def invoke(self, _fn, _request):
            return SimpleNamespace(
                files=[FileItem(path='/tmp/b.png', content='BBBB', encoding='base64')],
                errors=[SimpleNamespace(path='/tmp/a.png', success=False, message='missing')],
            )

    shots = [
        {'name': 'a', 'label': 'A', 'path': '/tmp/a.png'},
        {'name': 'b', 'label': 'B', 'path': '/tmp/b.png'},
    ]

    result = await executor._download_shots(Runtime(), shots, None, 'F1')

    assert result == [
        {'name': 'a', 'label': 'A', 'b64': '', 'download_failed': True},
        {'name': 'b', 'label': 'B', 'b64': 'BBBB'},
    ]


@pytest.mark.asyncio
async def test_download_shots_marks_invoke_failure() -> None:
    class Runtime:
        async def invoke(self, _fn, _request):
            raise RuntimeError('download unavailable')

    shots = [{'name': 'a', 'label': 'A', 'path': '/tmp/a.png'}]

    result = await executor._download_shots(Runtime(), shots, None, 'F1')

    assert result == [{'name': 'a', 'label': 'A', 'b64': '', 'download_failed': True}]


def test_enforce_rejects_empty_evidence_reference() -> None:
    item = {
        'status': 'completed',
        'code': 'await count_of(".item")',
        'observations': [{'label': 'count_of .item', 'value': 1}],
        'shots': [],
    }
    verdict = {
        'verdict': 'pass',
        'reason': 'criterion satisfied',
        'evidence_refs': [''],
        'inconclusive_reason': 'none',
        'confidence': 1.0,
    }

    result = verifier._enforce(verdict, item)

    assert result['verdict'] == 'inconclusive'
    assert result['overridden'] == 'unciteable'


@pytest.mark.parametrize('status', ['missing_script', 'invoke_failed', 'no_result', 'harness_error'])
def test_enforce_marks_collection_without_artifact_evidence_inconclusive(status: str) -> None:
    verdict = {
        'verdict': 'fail',
        'reason': 'criterion not satisfied',
        'evidence_refs': [],
        'inconclusive_reason': 'none',
        'confidence': 1.0,
    }

    result = verifier._enforce(verdict, {'status': status, 'error': 'collection failed', 'shots': []})

    assert result['verdict'] == 'inconclusive'
    assert result['overridden'] == 'collection_unavailable'


@pytest.mark.asyncio
async def test_run_source_stops_when_candidate_directory_is_missing() -> None:
    class Runtime:
        async def invoke(self, _fn, request):
            encoded = re.search(r'^echo ([A-Za-z0-9+/=]+) \| base64', request.command).group(1)
            body = base64.b64decode(encoded).decode()
            assert 'exit 125' in body
            assert body.index('exit 125') < body.index('echo should-not-run')
            return SimpleNamespace(
                return_code=125,
                stdout='=== 源码目录不存在 ===\n',
                stderr='',
            )

    result = await executor._run_source(
        runtime=Runtime(),
        script={'fid': 'F1', 'kind': 'source_code', 'code': 'echo should-not-run', 'note': ''},
        timeout=3,
        source_dir='/workspace',
    )

    assert result['status'] == 'script_error'
    assert result['return_code'] == 125
    assert '源码目录不存在' in result['error']


@pytest.mark.asyncio
async def test_run_source_marks_shell_error_as_script_error() -> None:
    class Runtime:
        async def invoke(self, _fn, _request):
            return SimpleNamespace(return_code=127, stdout='', stderr='command not found')

    result = await executor._run_source(
        runtime=Runtime(),
        script={'fid': 'F1', 'kind': 'source_code', 'code': 'missing-command', 'note': ''},
        timeout=3,
        source_dir='/workspace',
    )

    assert result['status'] == 'script_error'
    assert result['return_code'] == 127
    assert result['error'] == 'command not found'


@pytest.mark.asyncio
async def test_flow2code_stops_when_the_artifact_has_no_source_files() -> None:
    """空产物不跑 agent —— 没有源码可读，flow2code 无从写脚本。

    判分与 agent 同容器（Harbor verifier），源码原地读，所以 ``file_list`` 为空
    就是「没有产物」。曾经这里会先把 archive 解包到判分自己的目录再判断，那条
    路径只有宿主判分时才需要，已随 artifact_task 一起弃用。
    """

    async def run_agent_and_read_reward(**_kwargs):
        raise AssertionError('agent should not run')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            'nexus.extensions.tasks.artifact.judges.capabilities.agent.run_agent_and_read_reward',
            run_agent_and_read_reward,
        )
        scripts, metadata = await executor.flow2code(
            {'task_type': 'web', 'flows': [{'fid': 'F1', 'check_via': 'browser'}]},
            runtime=object(),
            runtime_name='test',
            archive_b64='',
            problem_statement='build a page',
            file_list=[],
            openai_client=object(),
            source_dir='/workspace',
        )

    assert scripts == []
    assert metadata['upload_error']
    assert metadata['missing_fids'] == ['F1']


@pytest.mark.asyncio
async def test_flow2code_retries_a_zero_script_run_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """零脚本重试一次 —— 两次都零脚本才返回，把失败交给上层 maskout。

    曾经只跑一次：624 条历史跑批里零脚本失败 68% 跑够 300 秒以上，重试成本高。A49B 生产中
    61-70% 的前端 REWARD_ERROR 来自 flow2code 跑到时限被杀、零脚本，因此放宽到 4800 秒并保留
    2 次尝试；前端任务把 T3_VERIFIER_TIMEOUT_SEC 和 test_timeout 调到 11000 秒，两次都放得进 judge 预算。
    """
    calls = []

    async def upload_archive_to_runtime(_runtime, _archive_b64, _working_dir):
        return 3, None

    async def run_agent_and_read_reward(**_kwargs):
        calls.append(1)
        raw = '{}'  # 永远零脚本
        return SimpleNamespace(
            raw=raw,
            agent_error=None,
            read_error=None,
            cc=SimpleNamespace(execution_time=480.0, output=raw, trajectories=[]),
        )

    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.capabilities.agent.run_agent_and_read_reward',
        run_agent_and_read_reward,
    )

    scripts, metadata = await executor.flow2code(
        {'task_type': 'web', 'flows': [{'fid': 'F1', 'check_via': 'browser'}]},
        runtime=object(),
        runtime_name='test',
        archive_b64='archive',
        problem_statement='build a page',
        file_list=['index.html'],
        openai_client=object(),
        source_dir='/workspace',
    )

    assert len(calls) == 2, f'零脚本应重试一次 agent，实际跑了 {len(calls)} 次'
    assert scripts == []
    assert metadata['n_attempts'] == 2
    assert metadata['missing_fids'] == ['F1']


@pytest.mark.asyncio
@pytest.mark.parametrize('proxy_url', ['http://proxy', '', None])
async def test_web_reward_preserves_scoring_and_proxy_urls(monkeypatch, tmp_path, proxy_url):
    """The retired report's URL coverage now follows the current reward/evidence protocol."""
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.artifact.harbor_entry import verdict_to_reward, write_reward
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import SandboxContext

    refs = ['F1/main']
    _pipeline_stubs(monkeypatch, evidence=[], verify_result={
        'summary': _summary(),
        'verdicts': [{'fid': 'F1', 'verdict': 'pass', 'evidence_refs': refs}],
        'aesthetic': {'available': True, 'score': 0.4},
    })
    execute = AsyncMock(return_value=[])
    monkeypatch.setattr(web_judger.executor, 'execute', execute)
    result = await web_judger.compute_score(_judge_input(
        sandbox_ctx=SandboxContext(sandbox_url='http://real', sandbox_proxy_url=proxy_url),
    ))

    execute.assert_awaited_once()
    assert execute.await_args.kwargs['page_url'] == 'http://real'
    assert result.failure_kind is None
    assert result.score == 0.7
    assert result.metadata['real_url'] == 'http://real'
    assert result.metadata['proxy_url'] == (proxy_url or '')
    assert result.metadata['verdicts'][0]['evidence_refs'] == refs
    payload = verdict_to_reward(result, judge_name='web_judger')
    write_reward(tmp_path, payload)
    saved = json.loads((tmp_path / 'reward.json').read_text())
    assert saved['reward'] == result.score
    assert saved['judge_metadata'] == result.metadata


def test_parse_aesthetic_output_accepts_any_dimension_set() -> None:
    """维度照单全收 —— 各品类 prompt 可以定义自己的维度集，数量也可以不同。

    这是「按品类分流」的前提：解析不比对固定清单，所以加一份新 prompt 不必改代码。
    """
    raw = json.dumps(
        {
            'hud_readability': {'reason': 'HUD 清晰', 'score': 4},
            'visual_feedback': {'reason': '打击感足', 'score': 3},
            'composition': {'reason': '布局合理', 'score': 5},
            'summary': '游戏画面整体不错',
        },
        ensure_ascii=False,
    )

    dims, reasons, summary = verifier.parse_aesthetic_output(raw)

    assert dims == {'hud_readability': 4.0, 'visual_feedback': 3.0, 'composition': 5.0}
    assert reasons['hud_readability'] == 'HUD 清晰'
    assert summary == '游戏画面整体不错'


def test_parse_aesthetic_output_reads_strict_json() -> None:
    shared_dims = sorted(verifier.AESTHETIC_DIMS_BY_VARIANT[''])
    raw = json.dumps(
        {name: {'reason': f'{name} reason', 'score': index + 0.5} for index, name in enumerate(shared_dims)}
        | {'summary': 'overall summary'}
    )

    dims, reasons, summary = verifier.parse_aesthetic_output(raw)

    assert set(dims) == set(shared_dims)
    assert reasons['composition'] == 'composition reason'
    assert reasons['polish'] == 'polish reason'
    assert summary == 'overall summary'


def test_parse_aesthetic_output_rejects_xml() -> None:
    dims, reasons, summary = verifier.parse_aesthetic_output(
        '<composition><reason>布局</reason><result>3</result></composition><summary>总结</summary>'
    )

    assert dims == {}
    assert reasons == {}
    assert summary == ''


@pytest.mark.asyncio
async def test_aesthetic_judge_retries_when_dimensions_disagree_with_the_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """维度和 prompt 声明的对不上 = 这次 VLM 没照协议输出 → 重试，第二次对上就照常给分。

    这类失败返回的是 200 + 合法 JSON，不抛异常，所以校验必须放在被重试的那一层里 ——
    否则 ``retry_with_backoff`` 只会覆盖网络抖动，而漏掉最常见的判分侧故障。
    """
    shared = sorted(verifier.AESTHETIC_DIMS_BY_VARIANT[''])
    calls = []

    async def vlm_score(*_args, **_kwargs):
        calls.append(1)
        dims = shared[:2] if len(calls) == 1 else shared  # 第一次少给几维
        return json.dumps({name: {'reason': 'r', 'score': 3} for name in dims} | {'summary': 's'}, ensure_ascii=False)

    monkeypatch.setattr('nexus.extensions.tasks.artifact.judges.capabilities.llm.vlm_score', vlm_score)

    result = await verifier.aesthetic_judge(
        [{'fid': 'F1', 'shots': [{'name': 'main', 'label': '首屏', 'b64': _png_b64((128, 128, 128))}]}],
        object(),
        problem_statement='build a page',
    )

    assert len(calls) == 2, '维度残缺必须触发重试'
    assert result['available'] is True
    assert set(result['dims']) == set(shared)
    assert 'partial' not in result, '要么全对要么重试，不再有降级打分'


@pytest.mark.asyncio
async def test_aesthetic_judge_masks_out_after_retries_are_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    """每次都对不上 → 重试耗尽 → available=False，由 pipeline 整条 maskout。"""
    calls = []

    async def vlm_score(*_args, **_kwargs):
        calls.append(1)
        return '{"polish": {"reason": "只有一维", "score": 5}, "summary": "总结"}'

    monkeypatch.setattr('nexus.extensions.tasks.artifact.judges.capabilities.llm.vlm_score', vlm_score)
    # tenacity 的退避是 1s 起指数抖动，4 次重试约 1+2+4 秒 —— 单测里把它压掉。
    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.strategies.web_judger.verifier.AESTHETIC_MAX_ATTEMPTS', 2
    )

    result = await verifier.aesthetic_judge(
        [{'fid': 'F1', 'shots': [{'name': 'main', 'label': '首屏', 'b64': _png_b64((128, 128, 128))}]}],
        object(),
        problem_statement='build a page',
    )

    assert len(calls) == 2, '每次都对不上就一直重试到上限'
    assert result['available'] is False
    assert '维度' in result['error']


@pytest.mark.asyncio
async def test_every_aesthetic_prompt_declares_its_dimension_set() -> None:
    """每份 prompt 都必须能从输出模板里解析出维度集 —— 那是校验完整性的唯一依据。

    模块加载时已经会因空集炸掉；这里再断言一次，顺带锁住「各品类维度数可以不同」。
    """
    by_variant = verifier.AESTHETIC_DIMS_BY_VARIANT

    assert set(by_variant) == set(verifier.AESTHETIC_PROMPTS)
    for variant, dims in by_variant.items():
        assert dims, f'{variant or "shared"} 解析不出维度'
    # 数目不同是设计如此：svg 2 维、game 3 维、共用那份 5 维
    assert by_variant['svg'] == {'structure', 'aesthetics'}
    assert len({len(dims) for dims in by_variant.values()}) > 1


def test_every_task_type_has_its_own_aesthetic_prompt() -> None:
    """flow 认的每个 task_type 都得有对应 prompt，否则 by_type 会静默回落到共用那份。

    加第五个品类时这条先红，提醒补 prompt 而不是让线上悄悄按 web 的维度去判 3D。
    """
    assert set(verifier.AESTHETIC_PROMPTS) >= flow.TASK_TYPES


def test_summary_score_counts_missing_function_as_zero_component() -> None:
    result = verifier.summary_score(
        [{'verdict': 'inconclusive', 'tier': 'core'}],
        {'available': True, 'score': 1.0},
    )

    assert result['func_available'] is False
    assert result['aesthetic_available'] is True
    assert result['final_score'] == 0.5
    assert result['basis'] == 'func_unavailable'


def test_summary_score_keeps_missing_aesthetic_as_zero_weighted_component() -> None:
    result = verifier.summary_score(
        [{'verdict': 'pass', 'tier': 'core'}],
        {'available': False, 'score': 0.0},
    )

    assert result['func_score'] == 1.0
    assert result['aesthetic_score'] == 0.0
    assert result['final_score'] == 0.5
    assert result['basis'] == 'func_only'
    assert result['weights'] == {'func': 0.5, 'aesthetic': 0.5}


def test_summary_score_adds_half_weighted_function_and_aesthetic_scores() -> None:
    result = verifier.summary_score(
        [{'verdict': 'pass', 'tier': 'core'}, {'verdict': 'fail', 'tier': 'core'}],
        {'available': True, 'score': 0.6},
    )

    assert result['func_score'] == 0.5
    assert result['aesthetic_score'] == 0.6
    assert result['final_score'] == 0.55
    assert result['basis'] == 'func+aesthetic'


def test_choose_images_takes_first_shot_of_each_flow_in_order() -> None:
    """按 flow 顺序取首图 —— F1 首屏必须进判分。

    旧实现按缩略图差异挑「最不相似」的图，会优先选中动画中间态；实测 51.2% 的 case 首屏
    因此被排除，同一份产物换三张图美学分能从 0.04 判到 0.91。
    """
    evidence = [
        {'fid': 'F1', 'shots': [{'name': 'f1_initial', 'b64': 'a'}, {'name': 'f1_later', 'b64': 'b'}]},
        {'fid': 'F2', 'shots': [{'name': 'f2_first', 'b64': 'c'}, {'name': 'f2_second', 'b64': 'd'}]},
        {'fid': 'F3', 'shots': [{'name': 'f3_first', 'b64': 'e'}]},
        {'fid': 'F4', 'shots': [{'name': 'f4_first', 'b64': 'f'}]},
    ]

    picked = verifier.choose_images(evidence, maximum=3)

    # 前 3 条 flow 各贡献首图，同 flow 的后续图和第 4 条 flow 都不取。
    assert [shot['name'] for shot in picked] == ['f1_initial', 'f2_first', 'f3_first']
    assert [shot['fid'] for shot in picked] == ['F1', 'F2', 'F3']


def test_choose_images_skips_flows_without_shots_and_allows_fewer() -> None:
    """无图的 flow 跳过；候选不足 maximum 时给几张算几张，不补齐。"""
    evidence = [
        {'fid': 'F1', 'shots': []},
        {'fid': 'F2', 'shots': [{'name': 'f2_first', 'b64': 'a'}]},
        {'fid': 'F3'},
    ]

    picked = verifier.choose_images(evidence, maximum=3)

    assert [shot['name'] for shot in picked] == ['f2_first']


def test_choose_images_dedupes_identical_shots_across_flows() -> None:
    """跨 flow 完全相同的截图只取一次，避免重复图占掉配额。"""
    evidence = [
        {'fid': 'F1', 'shots': [{'name': 'f1_initial', 'b64': 'same'}]},
        {'fid': 'F2', 'shots': [{'name': 'f2_dup', 'b64': 'same'}, {'name': 'f2_other', 'b64': 'other'}]},
        {'fid': 'F3', 'shots': [{'name': 'f3_first', 'b64': 'third'}]},
    ]

    picked = verifier.choose_images(evidence, maximum=3)

    # F2 的首图与 F1 重复被跳过，取其第二张。
    assert [shot['name'] for shot in picked] == ['f1_initial', 'f2_other', 'f3_first']


def test_parse_aesthetic_output_matches_dimension_names_case_insensitively() -> None:
    """模型偶尔把维度名写成 ``"Typography"``，五维齐全时不该整条判废。"""
    raw = json.dumps({
        'composition': {'score': 2, 'reason': 'a'},
        'color': {'score': 3, 'reason': 'b'},
        'Typography': {'score': 4, 'reason': 'c'},
        'coherence': {'score': 3, 'reason': 'd'},
        'polish': {'score': 1, 'reason': 'e'},
        'Summary': 's',
    })

    dims, reasons, summary = verifier.parse_aesthetic_output(raw)

    assert dims['typography'] == 4.0
    assert reasons['typography'] == 'c'
    assert len(dims) == len(verifier.AESTHETIC_DIMS_BY_VARIANT[''])
    assert summary == 's'


def test_parse_aesthetic_output_tolerates_raw_newlines_in_reason() -> None:
    """美学输出全是分数和评语，``reason`` 里的裸换行可以放过。

    共用的 ``loads_json_lenient`` 仍然拒收裸换行 —— flow2code 的 ``code`` 字段要 exec，
    见 ``test_artifact_parse.py::test_real_newline_inside_string_still_fails``。
    """
    raw = (
        '{"composition":{"score":2,"reason":"第一行\n第二行"},'
        '"color":{"score":3,"reason":"b"},'
        '"typography":{"score":4,"reason":"c"},'
        '"coherence":{"score":3,"reason":"d"},'
        '"polish":{"score":1,"reason":"e"},}'  # 顺带带一个尾逗号
    )

    dims, reasons, _ = verifier.parse_aesthetic_output(raw)

    assert len(dims) == len(verifier.AESTHETIC_DIMS_BY_VARIANT[''])
    assert '\n' in reasons['composition']


def _pipeline_stubs(monkeypatch: pytest.MonkeyPatch, *, evidence, verify_result) -> None:
    """把 judge 的外部依赖替换成固定返回值。

    部署不在这里 stub：``web_judger`` 声明 ``capabilities=('deploy',)``，由
    wrapper 负责，所以测试直接把现成的 ``sandbox_ctx`` 放进 ``JudgeInput``
    （见 :func:`_run_judge`），走的是 judge 自己的代码路径。
    """

    async def flow2code(*_args, **_kwargs):
        return [{'fid': 'F1'}], {}

    async def execute(*_args, **_kwargs):
        return evidence

    async def verify(*_args, **_kwargs):
        return verify_result

    monkeypatch.setattr(web_judger.executor, 'flow2code', flow2code)
    monkeypatch.setattr(web_judger.executor, 'execute', execute)
    monkeypatch.setattr(web_judger.verifier, 'verify', verify)


def _summary(**overrides):
    base = {
        'func_available': True,
        'aesthetic_available': True,
        'weighted_coverage': 1.0,
        'final_score': 0.7,
        'func_score': 1.0,
        'aesthetic_score': 0.4,
        'counts': {'pass': 1, 'fail': 0, 'inconclusive': 0},
    }
    return base | overrides


def _judge_input(**overrides):
    """``JudgeInput`` with the sandbox already prepared, as the wrapper would."""
    from nexus.extensions.tasks.artifact.judges.base import JudgeInput
    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import sandbox_context_from_screenshot

    defaults = {
        'instance_id': 'case-1',
        'problem_statement': 'build a page',
        'openai_clients': {'agent': object(), 'vlm': object()},
        'archive_b64': 'archive',
        'image': 'img',
        'runtime': object(),
        'sandbox_ctx': sandbox_context_from_screenshot(sandbox_url='http://x', screenshot_b64=''),
        'instance': {'flow': {'task_type': 'web', 'flows': [{'fid': 'F1'}]}},
    }
    defaults.update(overrides)
    return JudgeInput(**defaults)


async def _run_judge(**overrides) -> object:
    """Call the judge directly (not through ``run_judge``) so the stubbed
    internals are exercised without the wrapper re-preparing capabilities."""
    return await web_judger.compute_score(_judge_input(**overrides))


@pytest.mark.asyncio
@pytest.mark.parametrize('nested', [False, True])
async def test_web_contract_formats_reach_scoring(monkeypatch: pytest.MonkeyPatch, nested: bool) -> None:
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.artifact.judges.registry import run_judge

    contract = {'task_type': 'web', 'flows': [{'fid': 'F1'}]}
    row = {'flow': contract} if nested else dict(contract)
    row['aesthetic_mode'] = 'absolute'
    _pipeline_stubs(
        monkeypatch,
        evidence=[],
        verify_result={
            'summary': _summary(),
            'verdicts': [],
            'aesthetic': {'available': True, 'score': 0.4},
        },
    )
    flow2code = AsyncMock(return_value=([{'fid': 'F1'}], {}))
    monkeypatch.setattr(web_judger.executor, 'flow2code', flow2code)

    agent, vlm, model_config = object(), object(), object()
    verify = AsyncMock(return_value={
        'summary': _summary(), 'verdicts': [], 'aesthetic': {'available': True, 'score': 0.4},
    })
    monkeypatch.setattr(web_judger.verifier, 'verify', verify)
    result = await run_judge('web_judger', _judge_input(
        instance=row, openai_clients={'agent': agent, 'vlm': vlm}, model_configs={'agent': model_config},
    ))

    assert result.failure_kind is None
    assert result.score == 0.7
    received = flow2code.await_args.args[0]
    assert received['flows'] == contract['flows']
    assert received['task_type'] == 'web'
    assert flow2code.await_args.kwargs['openai_client'] is agent
    assert flow2code.await_args.kwargs['model_config'] is model_config
    assert verify.await_args.kwargs['judger'] is vlm


@pytest.mark.asyncio
@pytest.mark.parametrize('roles', [('main',), ('agent',), ('vlm',), ('flow2code', 'scoring')])
async def test_missing_roles_stop_before_deployment_and_collection(monkeypatch, roles):
    from unittest.mock import AsyncMock

    from nexus.extensions.tasks.artifact.judges.registry import run_judge

    deploy = AsyncMock(side_effect=AssertionError('must not deploy'))
    collect = AsyncMock(side_effect=AssertionError('must not collect'))
    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.capabilities.deploy.prepare_sandbox_context', deploy,
    )
    monkeypatch.setattr(web_judger.executor, 'flow2code', collect)
    inp = _judge_input(openai_clients={role: object() for role in roles}, sandbox_ctx=None)
    result = await run_judge('web_judger', inp)
    assert result.failure_kind == 'no_client'
    assert result.score == 0.0
    deploy.assert_not_awaited()
    collect.assert_not_awaited()


@pytest.mark.parametrize('invalid_flow', [None, [], 'invalid', {}])
def test_explicit_invalid_flow_does_not_fall_back_to_top_level(invalid_flow) -> None:
    inp = _judge_input(instance={'flow': invalid_flow, 'flows': [{'fid': 'F1'}]})

    assert web_judger._validate_input(inp) is not None


@pytest.mark.asyncio
async def test_pipeline_masks_out_when_the_aesthetic_score_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """美学分没出来 → 整条 maskout，func 分也不要。

    此前按 0 分计入（final = 0.5×func + 0.5×0），把判分侧的故障当成产物难看喂进训练。
    """
    _pipeline_stubs(
        monkeypatch,
        evidence=[{'fid': 'F1', 'shot_calls': 1, 'shots': [{'name': 'a', 'b64': 'img'}]}],
        verify_result={
            'summary': _summary(aesthetic_available=False, aesthetic_score=0.0, final_score=0.5),
            'verdicts': [{'fid': 'F1', 'verdict': 'pass'}],
            'aesthetic': {'available': False, 'error': 'VLM 调用失败'},
        },
    )

    result = await _run_judge()

    assert result.failure_kind == 'vlm_failed'
    assert result.score == 0.0
    assert 'VLM 调用失败' in (result.error or '')


@pytest.mark.asyncio
async def test_pipeline_scores_normally_when_only_some_flows_lost_screenshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """单条 flow 少截图不影响出分 —— 采到几张就用几张，只看最终分数在不在。"""
    _pipeline_stubs(
        monkeypatch,
        evidence=[
            {'fid': 'F1', 'shot_calls': 1, 'shots': []},  # 这条一张图都没拿到
            {'fid': 'F2', 'shot_calls': 1, 'shots': [{'name': 'b', 'b64': 'img'}]},
        ],
        verify_result={
            'summary': _summary(),
            'verdicts': [{'fid': 'F1', 'verdict': 'pass'}, {'fid': 'F2', 'verdict': 'pass'}],
            'aesthetic': {'available': True, 'score': 0.4},
        },
    )

    result = await _run_judge()

    assert result.failure_kind is None
    assert result.score == 0.7


def test_browser_harness_omits_data_uri_payloads() -> None:
    script = executor._build_browser_script(
        "record('image_src', 'data:image/png;base64,AAAA')",
        'http://example.test',
        '/tmp/result.json',
        '/tmp/shots',
        0,
    )

    assert 'DATA_URI_RE.sub' in script
    assert '<base64 内容已省略:' in script


def test_browser_harness_isolates_generated_variables_and_accepts_locators() -> None:
    generated = (
        "status = 'page-owned-status'\nos = 'page-owned-os'\nitems = page.locator('.item')\nawait count_of(items)"
    )
    script = executor._build_browser_script(
        generated,
        'http://example.test',
        '/tmp/result.json',
        '/tmp/shots',
        0,
    )

    assert '        async def run_generated_script():' in script
    assert "            status = 'page-owned-status'" in script
    assert '                await run_generated_script()' in script
    assert 'return page.locator(sel) if isinstance(sel, str) else sel' in script


def test_browser_harness_keeps_screenshot_failure_out_of_record() -> None:
    """截图失败的原因必须进 shot_errors，不能走 record。

    ``record`` 会按 VERDICT_LABEL 过滤名字带判定语义的 label，而
    ``screenshot_failed_*`` 正好命中 ``failed`` —— 失败原因会被静默塞进 suppressed，
    事后只能看到「0 张截图」而无从归因。
    """
    script = executor._build_browser_script(
        "await shot('main', '首屏')",
        'http://example.test',
        '/tmp/result.json',
        '/tmp/shots',
        0,
    )

    assert 'screenshot_failed' not in script
    assert "'shot_errors': shot_capture_failures[:10]" in script
    assert f'timeout={executor.SHOT_TIMEOUT_MS}' in script


def test_browser_harness_settles_before_shot_instead_of_freezing_animations() -> None:
    """截图前静置，且不得冻结动画。

    ``animations='disabled'`` 会把滚动入场动画（IntersectionObserver 加 class +
    transition 显现）冻在起始的 ``opacity: 0``。本地实测同一页面同一滚动位置：
    冻结时非背景像素占比 0.0000（纯空屏），静置 400ms 后 0.0028（内容可见）。
    """
    script = executor._build_browser_script(
        "await shot('main', '首屏')",
        'http://example.test',
        '/tmp/result.json',
        '/tmp/shots',
        0,
    )

    # 查实际的 screenshot 调用（注释里提到 animations 是在讲历史，不算）
    call = re.search(r'await page\.screenshot\([^)]*\)', script).group(0)
    assert 'animations' not in call
    assert f'await page.wait_for_timeout({executor.SETTLE_SHOT_MS})' in script
    # 静置必须在截图之前，否则等于没等
    assert script.index(f'wait_for_timeout({executor.SETTLE_SHOT_MS})') < script.index(call)


def test_summary_score_reports_weighted_coverage() -> None:
    result = verifier.summary_score(
        [
            {'verdict': 'pass', 'tier': 'feature'},
            {'verdict': 'inconclusive', 'tier': 'core'},
            {'verdict': 'inconclusive', 'tier': 'feature'},
        ],
        {'available': False},
    )

    assert result['func_score'] == 1.0
    assert result['weighted']['total'] == 2.0
    assert result['contract_weight'] == 8.0
    assert result['weighted_coverage'] == 0.25


@pytest.mark.asyncio
async def test_pipeline_masks_out_low_weighted_coverage(monkeypatch: pytest.MonkeyPatch) -> None:
    async def flow2code(*_args, **_kwargs):
        return [{'fid': 'F1', 'kind': 'browser', 'code': 'pass'}], {'missing_fids': []}

    async def execute(*_args, **_kwargs):
        return [{'fid': 'F1', 'status': 'completed', 'shots': [], 'shot_calls': 0}]

    async def verify(*_args, **_kwargs):
        return {
            'summary': {
                'func_available': True,
                'aesthetic_available': False,
                'weighted_coverage': 0.25,
                'final_score': 1.0,
            },
            'verdicts': [],
        }

    monkeypatch.setattr(web_judger.executor, 'flow2code', flow2code)
    monkeypatch.setattr(web_judger.executor, 'execute', execute)
    monkeypatch.setattr(web_judger.verifier, 'verify', verify)

    result = await _run_judge()

    assert result.score == 0.0
    # 覆盖率过低是契约质量问题（flow2code 没为足够多 flow 产出可执行脚本，或契约
    # 本身太严），不是 agent 运行挂了 —— 标 agent_failed 会把排查方向带偏。
    assert result.failure_kind == 'config_error'
    assert '加权覆盖率过低' in (result.error or '')


def test_shared_deploy_failure_splits_infra_and_artifact() -> None:
    """``deploy_failure_results`` must attribute a deploy failure to the right side.

    ``capabilities/deploy.py`` prefixes ``result_status`` with ``artifact:`` (the
    delivered code does not build/start) or ``infra:`` (sandbox request/poll failed).
    Only the artifact case may reach training: it carries no ``failure_kind`` and sets
    ``metadata['deploy_failed']``, which task.py converts into a DEPLOY_ARTIFACT
    penalty. An ``infra:`` failure must keep ``failure_kind='no_sandbox'`` so the
    rollout is masked out instead of being blamed on the model.

    All 5 pipelines share this helper, so the split is uniform (previously only
    web_judger implemented it and the other 4 masked artifact faults).
    """
    artifact_fault = deploy_failure_results(('web_judger',), 'artifact: npm run build failed')[0]
    assert artifact_fault.failure_kind is None, 'artifact fault must not look like an eval failure'
    assert artifact_fault.metadata['deploy_failed'] is True
    assert artifact_fault.score == 0.0  # negative reward can only be expressed at the task layer

    for infra_error in ('infra: poll timeout', 'no sandbox url', ''):
        infra_fault = deploy_failure_results(('web_judger',), infra_error)[0]
        assert infra_fault.failure_kind == 'no_sandbox', f'{infra_error!r} should mask out'
        assert not infra_fault.metadata.get('deploy_failed')

    # one result per judge name, so multi-judge pipelines stay consistent
    assert len(deploy_failure_results(('a', 'b', 'c'), 'infra: x')) == 3


# ── aesthetic prompt routing (config.judge.aesthetic_prompt) ──────────────────


async def _run_aesthetic(monkeypatch, *, mode, task_type, dims=None):
    """Run aesthetic_judge with a stubbed VLM, returning its result dict.

    ``dims=None`` makes the stub answer with whatever the resolved prompt requires, so a
    routing test exercises routing rather than tripping the dimension check.
    """
    from nexus.extensions.tasks.artifact import config

    if mode is None:
        monkeypatch.delenv('ARTIFACT_JUDGE_AESTHETIC_PROMPT', raising=False)
    else:
        monkeypatch.setenv('ARTIFACT_JUDGE_AESTHETIC_PROMPT', mode)
    config.reset()

    expected_variant = task_type if mode == 'by_type' else (mode or '')
    if expected_variant not in verifier.AESTHETIC_PROMPTS:
        expected_variant = ''
    answer = dims if dims is not None else sorted(verifier.AESTHETIC_DIMS_BY_VARIANT[expected_variant])

    async def vlm_score(*_args, **_kwargs):
        body = {name: {'score': 3.0, 'reason': f'{name} r'} for name in answer}
        body['summary'] = 's'
        return json.dumps(body, ensure_ascii=False)

    monkeypatch.setattr('nexus.extensions.tasks.artifact.judges.capabilities.llm.vlm_score', vlm_score)
    try:
        return await verifier.aesthetic_judge(
            [{'fid': 'F1', 'shots': [{'name': 'main', 'label': '首屏', 'b64': 'X'}]}],
            object(),
            problem_statement='q',
            flow={'task_type': task_type},
        )
    finally:
        config.reset()


@pytest.mark.asyncio
async def test_aesthetic_prompt_defaults_to_the_shared_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """开关不设时用共用 prompt —— 默认行为与分流上线前一致，A/B 的对照组才干净。"""
    result = await _run_aesthetic(monkeypatch, mode=None, task_type='game')

    assert result['prompt_variant'] == 'shared'
    assert result['available'] is True


@pytest.mark.asyncio
async def test_aesthetic_prompt_by_type_routes_on_flow_task_type(monkeypatch: pytest.MonkeyPatch) -> None:
    for task_type in ('web', 'game', '3d', 'svg'):
        result = await _run_aesthetic(monkeypatch, mode='by_type', task_type=task_type)
        assert result['prompt_variant'] == task_type, task_type


@pytest.mark.asyncio
async def test_aesthetic_prompt_falls_back_for_an_unknown_task_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """认不出的品类回落共用那份，而不是判不出分。

    新品类进数据集时判分不该整条 maskout —— 共用 prompt 一直在，回落是安全的。
    """
    result = await _run_aesthetic(monkeypatch, mode='by_type', task_type='brand_new_type')

    assert result['prompt_variant'] == 'shared'
    assert result['available'] is True


@pytest.mark.asyncio
async def test_aesthetic_prompt_can_be_pinned_for_ablation(monkeypatch: pytest.MonkeyPatch) -> None:
    """填死某个品类 → 忽略 task_type，用于单变量消融。"""
    result = await _run_aesthetic(monkeypatch, mode='svg', task_type='web')

    assert result['prompt_variant'] == 'svg'


@pytest.mark.asyncio
async def test_each_variant_is_scored_against_its_own_dimension_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """每份 prompt 按它自己声明的维度集校验 —— svg 两维就该两维，不拿共用的五维去比。

    ``mode='svg'`` 下答两维即通过；同样答两维在共用模式（要五维）下会被判失败并重试，
    这正是「判分侧对维度数无感知、但要求与 prompt 一致」的含义。
    """
    svg_dims = sorted(verifier.AESTHETIC_DIMS_BY_VARIANT['svg'])
    assert len(svg_dims) == 2

    ok = await _run_aesthetic(monkeypatch, mode='svg', task_type='svg')
    assert ok['available'] is True
    assert set(ok['dims']) == set(svg_dims)
    assert 'partial' not in ok

    # 同一份两维答案喂给共用 prompt（要五维）→ 判失败
    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.strategies.web_judger.verifier.AESTHETIC_MAX_ATTEMPTS', 1
    )
    mismatched = await _run_aesthetic(monkeypatch, mode=None, task_type='svg', dims=svg_dims)
    assert mismatched['available'] is False


# ── pairwise 美学 ─────────────────────────────────────────────────────────────

_ANCHOR = 'A' * 600  # >500 字符走 base64 分支，见 visual_anchored._resolve_anchor_b64


def _pairwise_reply(winner: str) -> str:
    return json.dumps(
        {
            'per_dimension': {'AI模板化程度': {'better': winner, 'reason': '候选更克制'}},
            'winner': winner,
            'reason': f'winner={winner}',
        },
        ensure_ascii=False,
    )


async def _run_pairwise(monkeypatch: pytest.MonkeyPatch, raw: str, *, instance=None, mode='pairwise'):
    """跑一次 pairwise 美学判定，返回 ``(结果, VLM 调用次数)``。

    打桩打在 client 上而不是某个函数上 —— pairwise 复用 ``visual_anchored`` judge，
    它自己拼 A/B 双图的 content，桩要接在它真正发请求的那一层才数得准调用次数。
    """
    calls: list[object] = []

    class FakeClient:
        async def create_completion(self, messages=None, **_kwargs):
            calls.append(messages)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])

    result = await verifier.aesthetic_judge(
        [
            {'fid': 'F1', 'shots': [{'name': 'main', 'label': '首屏', 'b64': 'CANDIDATE'}]},
            {'fid': 'F2', 'shots': [{'name': 'main', 'label': '第二屏', 'b64': 'OTHER'}]},
        ],
        FakeClient(),
        problem_statement='q',
        mode=mode,
        instance={'anchor_screenshot': _ANCHOR} if instance is None else instance,
        archive_b64='archive',
    )
    return result, len(calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('winner', 'score', 'verdict'),
    [('candidate', 1.0, 'win'), ('reference', 0.0, 'lose'), ('tie', 0.5, 'tie')],
)
async def test_pairwise_maps_the_winner_to_a_score(
    monkeypatch: pytest.MonkeyPatch, winner: str, score: float, verdict: str
) -> None:
    """winner=candidate（候选胜参考）→ 1.0，reference → 0.0，tie → 0.5。"""
    result, n_calls = await _run_pairwise(monkeypatch, _pairwise_reply(winner))

    assert result['available'] is True
    assert result['score'] == score
    assert result['verdict'] == verdict
    assert result['mode'] == 'pairwise'
    # tie 是合法结论，不该被当成协议违规去重试。
    assert n_calls == 1


@pytest.mark.asyncio
async def test_pairwise_compares_only_the_first_flows_first_shot() -> None:
    """只送第一条 flow 的首图 —— 对比 prompt 是按两张图写的，且首图语义最稳定。"""
    result, _ = await _run_pairwise(None, _pairwise_reply('candidate'))

    assert result['n_images'] == 1
    assert result['images'] == ['F1/main']


@pytest.mark.asyncio
async def test_pairwise_records_which_anchor_was_used() -> None:
    """anchor 是 per-case 常量，喂错会让该 instance 的所有 rollout 同向偏移。

    分数上看不出来，所以要留指纹 —— 事后能对着它查是不是配错了 anchor。
    """
    result, _ = await _run_pairwise(None, _pairwise_reply('candidate'))

    assert len(result['anchor_sha256']) == 12


@pytest.mark.asyncio
@pytest.mark.parametrize('raw', ['{"reason": "忘了写 winner"}', '抱歉，我无法比较这两张图。'])
async def test_pairwise_retries_a_malformed_reply_and_never_invents_a_tie(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """输出解不出 winner → 重试到上限 → 判不可用。**绝不能落成 0.5。**

    ``parse_visual_verdict`` 的契约是「永不失败」，解析不出就返回 tie/0.5。那个 0.5 是凭空
    捏造的，喂进 RL 比判 0 更坏（0 至少和失败相关）。校验必须在被重试的那一层里，否则
    judge 内的 retry_with_backoff 永远看不到协议违规 —— 只报错不重试。
    """
    # 降到 2 次：证明「会重试」不需要跑满 4 次，退避真会睡（4 次要 8s+）。
    monkeypatch.setattr(
        'nexus.extensions.tasks.artifact.judges.strategies.web_judger.verifier.AESTHETIC_MAX_ATTEMPTS', 2
    )
    result, n_calls = await _run_pairwise(monkeypatch, raw)

    assert result['available'] is False
    assert result['score'] == 0.0
    assert n_calls == 2


@pytest.mark.asyncio
async def test_pairwise_without_an_anchor_does_not_call_the_vlm() -> None:
    """没 anchor 重试也没用，直接判不可用，一次 VLM 都不发。"""
    result, n_calls = await _run_pairwise(None, _pairwise_reply('candidate'), instance={})

    assert result['available'] is False
    assert n_calls == 0


@pytest.mark.asyncio
async def test_an_unknown_aesthetic_mode_fails_loudly() -> None:
    """认不出的模式判不可用，不静默回落绝对打分。

    静默回落会让「以为在跑 pairwise」的整轮训练白跑，而日志里看不出任何异常。
    """
    result, n_calls = await _run_pairwise(None, _pairwise_reply('candidate'), mode='pairwize')

    assert result['available'] is False
    assert 'pairwize' in result['error']
    assert n_calls == 0


@pytest.mark.asyncio
async def test_absolute_mode_labels_its_own_score(monkeypatch: pytest.MonkeyPatch) -> None:
    """绝对分也带 mode —— 一个连续、一个三档，混在一列里会静默污染离线统计。"""
    result = await _run_aesthetic(monkeypatch, mode=None, task_type='web')

    assert result['mode'] == 'absolute'
    assert result['available'] is True


@pytest.mark.asyncio
async def test_pairwise_without_an_anchor_is_rejected_before_deploying(monkeypatch: pytest.MonkeyPatch) -> None:
    """pairwise 缺 anchor 是数据错误，要在**部署之前**就拦掉。

    走 ``run_judge`` 而不是直接调 ``compute_score``：这个检查现在在注册的
    ``validate`` 钩子里，由 wrapper 在准备 capability 之前执行。直接调
    ``compute_score`` 的测试覆盖不到这个顺序 —— 而顺序正是问题所在：放在
    ``compute_score`` 里就跑在 deploy 之后，部署失败会先短路，数据错误被
    记成产物错误，变成一个假惩罚进训练。
    """
    import nexus.extensions.tasks.artifact.judges.capabilities.deploy as deploy_mod

    from nexus.extensions.tasks.artifact import config
    from nexus.extensions.tasks.artifact.judges import registry as judge_registry

    monkeypatch.setenv('ARTIFACT_JUDGE_AESTHETIC_MODE', 'pairwise')
    config.reset()

    deployed: list[str] = []
    ran_flow2code: list[str] = []

    async def prepare_sandbox_context(_archive_b64, *, instance_id=''):
        deployed.append(instance_id)
        raise AssertionError('deploy should not run when the anchor is missing')

    async def flow2code(*_args, **_kwargs):
        ran_flow2code.append('called')
        raise AssertionError('flow2code should not run when the anchor is missing')

    monkeypatch.setattr(deploy_mod, 'prepare_sandbox_context', prepare_sandbox_context)
    monkeypatch.setattr(web_judger.executor, 'flow2code', flow2code)
    try:
        result = await judge_registry.run_judge('web_judger', _judge_input(sandbox_ctx=None))
    finally:
        config.reset()

    assert result.failure_kind == 'config_error'
    assert 'anchor_screenshot' in (result.error or '')
    assert not deployed, '数据错误不该先烧一次部署'
    assert not ran_flow2code, '数据错误不该先烧一次 agent 运行'


@pytest.mark.asyncio
async def test_a_row_level_aesthetic_mode_overrides_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """数据行的 aesthetic_mode 优先于 profile。

    容器侧判分读不到 profile，
    所以数据行是 RL 里唯一能生效的开关载体。
    """
    from nexus.extensions.tasks.artifact import config

    monkeypatch.delenv('ARTIFACT_JUDGE_AESTHETIC_MODE', raising=False)  # profile = absolute
    config.reset()

    seen: dict[str, object] = {}

    async def verify(*_args, **kwargs):
        seen['mode'] = kwargs.get('aesthetic_mode')
        return {'summary': _summary(), 'verdicts': [], 'aesthetic': {'mode': kwargs.get('aesthetic_mode')}}

    _pipeline_stubs(monkeypatch, evidence=[], verify_result=None)
    monkeypatch.setattr(web_judger.verifier, 'verify', verify)
    try:
        await _run_judge(
            instance={
                'flow': {'task_type': 'web', 'flows': [{'fid': 'F1'}]},
                'aesthetic_mode': 'pairwise',
                'anchor_screenshot': _ANCHOR,
            },
        )
    finally:
        config.reset()

    assert seen['mode'] == 'pairwise'


def test_every_judge_prompt_renders_without_leftover_brace_escapes() -> None:
    """prompt 里不能留 ``{{`` —— ``render_prompt`` 用正则替换，不会把它还原成 ``{``。

    留着的话 VLM 收到的是坏 JSON 模板，输出自然解不出来。judge_svg.txt 曾经就是这样，
    而它是真在线上跑的（svg.py 渲染它）。
    """
    from nexus.extensions.tasks.artifact.judges.capabilities.prompts import PROMPTS_DIR, load_prompt, render_prompt

    for path in sorted(PROMPTS_DIR.iterdir()):
        if path.suffix not in ('.txt', '.md'):
            continue
        rendered = render_prompt(load_prompt(path.name), query='q', shots='s')
        assert '{{' not in rendered, f'{path.name} 渲染后仍含字面 {{{{'


def test_the_pairwise_prompt_and_parser_agree_on_the_winner_vocabulary() -> None:
    """prompt 声明的 winner 取值必须与解析器认的那套完全一致。

    这是 pairwise 最危险的耦合：取值对不上时判分不报错、不崩，只是所有分数悄悄取反
    （原先用 A/B 时，prompt 里两张图一换顺序，字母含义就反了）。改用 reference/candidate
    是为了自解释，这条测试则把「prompt 和代码同源」钉死。

    图片标签也一起查 —— 标签用的词就是 VLM 判断哪张是哪张的依据。
    """
    from nexus.extensions.tasks.artifact.judges.capabilities.prompts import load_prompt
    from nexus.extensions.tasks.artifact.judges.strategies import visual_anchored

    prompt = load_prompt('judge_visual_anchored.txt')

    for word in visual_anchored.WINNER_VERDICTS:
        assert word in prompt, f'prompt 没声明 winner 取值 {word!r}'
    assert 'A或B' not in prompt, 'prompt 还在用 A/B 字母，与解析器的语义词对不上'

    # _compare 里的图片标签必须和 prompt 的措辞同源，否则 VLM 认不出哪张是候选。
    labels = inspect.getsource(visual_anchored._compare)
    assert "('参考', references)" in labels
    assert "('候选', candidates)" in labels
    assert '【参考' in prompt
    assert '【候选' in prompt


def _multi_evidence(n: int) -> list[dict]:
    """n 条 flow，每条一张首图 —— 与绝对模式的 choose_images 取图方式一致。"""
    return [
        {'fid': f'F{i}', 'shots': [{'name': 'main', 'label': f'状态{i}', 'b64': f'CAND{i}'}]} for i in range(1, n + 1)
    ]


async def _capture_pairwise_content(evidence, anchors, *, winner='candidate'):
    """跑一次 pairwise，返回发给 VLM 的 content 列表。"""
    sent: list[list[dict]] = []

    class FakeClient:
        async def create_completion(self, messages=None, **_kwargs):
            sent.append(messages[0]['content'])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=_pairwise_reply(winner)))])

    result = await verifier.aesthetic_judge(
        evidence,
        FakeClient(),
        problem_statement='q',
        mode='pairwise',
        instance={'anchor_screenshot': anchors},
        archive_b64='archive',
    )
    return sent[0], result


@pytest.mark.asyncio
async def test_pairwise_sends_each_side_as_a_labelled_group() -> None:
    """先给完整一组参考、再给完整一组候选，每张带 i/N 标签。

    分组而不是交错：判的是「哪套实现更好」，每一侧要能被当成一套连贯的设计来看。
    顺序和标签是 VLM 区分两侧的唯一依据，也是 winner 语义的锚点，钉死它。
    """
    anchors = [f'{"R" * 600}{i}' for i in (1, 2, 3)]
    content, result = await _capture_pairwise_content(_multi_evidence(3), anchors)

    kinds = [part['type'] for part in content]
    assert kinds == ['text'] + ['text', 'image_url'] * 6, kinds  # prompt + 3 参考 + 3 候选

    urls = [part['image_url']['url'] for part in content if part['type'] == 'image_url']
    texts = [part['text'] for part in content if part['type'] == 'text']
    # 参考组整块在前，候选组整块在后
    assert texts[1:] == [f'【参考 {i}/3】' for i in (1, 2, 3)] + [f'【候选 {i}/3】' for i in (1, 2, 3)]
    assert all(anchors[i] in urls[i] for i in range(3)), '前 3 张必须是参考图'
    assert all(f'CAND{i + 1}' in urls[3 + i] for i in range(3)), '后 3 张必须是候选图'
    assert all(part['image_url']['detail'] == 'high' for part in content if part['type'] == 'image_url')
    assert result['n_images'] == 3
    assert result['images'] == ['F1/main', 'F2/main', 'F3/main']


@pytest.mark.asyncio
async def test_pairwise_uses_the_same_shots_as_absolute_mode() -> None:
    """候选图与绝对模式同源 —— 两种模式看的是同一批状态，判据才可比。

    此前 pairwise 只送 images[0]（初始态），绝对模式送 3 张，等于两种模式在评不同的东西。
    """
    evidence = _multi_evidence(3)
    anchors = [f'{"R" * 600}{i}' for i in (1, 2, 3)]
    content, _ = await _capture_pairwise_content(evidence, anchors)

    absolute_shots = [i['b64'] for i in verifier.choose_images(evidence)]
    urls = [part['image_url']['url'] for part in content if part['type'] == 'image_url']
    for shot in absolute_shots:
        assert any(shot in url for url in urls), f'{shot} 没进 pairwise'


@pytest.mark.asyncio
async def test_pairwise_aligns_both_sides_to_the_shorter_one() -> None:
    """参考只有 1 张时，候选也只送 1 张。

    否则一侧 3 张、一侧 1 张，VLM 会把「状态更多」当成视觉优势 —— 那是采图数量的差异，
    不是设计质量的差异，而且这个偏向对该 instance 的所有 rollout 同向生效。
    """
    content, result = await _capture_pairwise_content(_multi_evidence(3), 'R' * 600)

    urls = [part['image_url']['url'] for part in content if part['type'] == 'image_url']
    assert len(urls) == 2, f'两侧各 1 张，共 2 张，实际 {len(urls)}'
    assert 'CAND1' in urls[1]
    assert 'CAND2' not in ''.join(urls)
    assert result['n_images'] == 1
    assert result['images'] == ['F1/main']


@pytest.mark.asyncio
async def test_pairwise_still_accepts_a_single_string_anchor() -> None:
    """数据行给单张字符串（老写法）仍然能跑，不是只认 list。"""
    _, result = await _capture_pairwise_content(_multi_evidence(1), 'R' * 600)

    assert result['available'] is True
    assert result['score'] == 1.0


@pytest.mark.asyncio
async def test_the_shots_placeholder_is_filled_with_per_group_labels() -> None:
    """prompt 的 {shots} 要写清第 i 组对应什么状态，否则 VLM 不知道各组在比什么。"""
    anchors = [f'{"R" * 600}{i}' for i in (1, 2)]
    content, _ = await _capture_pairwise_content(_multi_evidence(2), anchors)

    prompt_text = content[0]['text']
    assert '{shots}' not in prompt_text, '占位符没被填上'
    assert '第 1 组：状态1' in prompt_text
    assert '第 2 组：状态2' in prompt_text
