"""模型给的数字进入分数之前必须校验 —— clamp 不是校验。

`max(low, min(high, float(v)))` 把不可用的输出变成一个像样的分数，而最坏的情况
恰好最好看：

    min(5.0, nan) → nan        NaN 与任何值比较都是 False
    max(0.0, nan) → 5.0        上界 clamp 交还自己的上限

于是「模型没给出评价」变成**满分**。到了 ``JudgeResult`` 已经是普通 float，那层
的严格校验查不出来。

这个坑在三个地方各出现过一次（``static_chenxu`` 已下线，剩下两处在这里钉住），
所以校验收敛到 ``capabilities.parse.coerce_score`` 一个函数。
"""

from __future__ import annotations

import json
import math

import pytest

from nexus.extensions.tasks.artifact.judges.capabilities.parse import coerce_score


# ────────────────────── coerce_score 本身 ──────────────────────


@pytest.mark.parametrize(
    ('label', 'value'),
    [
        ('NaN', float('nan')),
        ('NaN 字符串', 'NaN'),
        ('+inf', float('inf')),
        ('-inf', float('-inf')),
        ('inf 字符串', 'Infinity'),
        ('溢出成 inf', 1e400),
        ('bool True', True),
        ('bool False', False),
        ('超上界', 99),
        ('低于下界', -5),
        ('非数值 list', [1]),
        ('非数值 dict', {'a': 1}),
        ('None', None),
        ('空串', ''),
        ('乱码', 'abc'),
    ],
)
def test_unusable_values_are_rejected(label: str, value):
    """全部返回 ``None`` —— 调用方据此走 parse_failed 或跳过这一维。"""
    assert coerce_score(value, low=0.0, high=5.0) is None, label


@pytest.mark.parametrize(
    ('value', 'expected'),
    [(3.5, 3.5), ('3.5', 3.5), (0, 0.0), (5, 5.0), ('0', 0.0), (2, 2.0)],
)
def test_legal_values_pass_through(value, expected: float):
    """合法输入不受影响，含两个边界值。"""
    assert coerce_score(value, low=0.0, high=5.0) == expected


def test_bounds_are_inclusive():
    assert coerce_score(1.0, low=1.0, high=5.0) == 1.0
    assert coerce_score(5.0, low=1.0, high=5.0) == 5.0
    assert coerce_score(0.999, low=1.0, high=5.0) is None
    assert coerce_score(5.001, low=1.0, high=5.0) is None


def test_nan_would_have_become_top_marks():
    """钉住这个机制本身：裸 clamp 对 NaN 给满分。

    有人以后想「简化」成一行 clamp 时，这条会红。
    """
    nan = float('nan')
    naive = max(0.0, min(5.0, nan))

    assert naive == 5.0, '裸 clamp 的行为变了？'
    assert math.isnan(nan)
    assert coerce_score(nan, low=0.0, high=5.0) is None


# ────────────────────── 美学维度（web_judger） ──────────────────────


def _aesthetic(value):
    from nexus.extensions.tasks.artifact.judges.strategies.web_judger.verifier import parse_aesthetic_output

    dims, _reasons, _summary = parse_aesthetic_output(json.dumps({'layout': {'score': value}}))
    return dims


@pytest.mark.parametrize(
    ('label', 'value'),
    [('NaN', float('nan')), ('inf', float('inf')), ('越界 99', 99), ('bool', True), ('负数', -5)],
)
def test_aesthetic_dimension_rejects_unusable_scores(label: str, value):
    """非法 → 不写这一维，下游按缺维处理。

    实测过：修复前五个维度全是 NaN 时，解析结果是五个 5.0（满分）。
    """
    assert _aesthetic(value) == {}, label


def test_aesthetic_dimension_keeps_legal_scores():
    assert _aesthetic(3.5) == {'layout': 3.5}
    assert _aesthetic(0) == {'layout': 0.0}
    assert _aesthetic(5) == {'layout': 5.0}


def test_all_dimensions_nan_yields_no_dimensions():
    """review 里的原始场景：五维全 NaN。"""
    from nexus.extensions.tasks.artifact.judges.strategies.web_judger.verifier import parse_aesthetic_output

    raw = json.dumps({d: {'score': float('nan'), 'reason': 'x'} for d in
                      ('layout', 'color', 'typography', 'detail', 'overall')})
    dims, reasons, _ = parse_aesthetic_output(raw)

    assert dims == {}, f'五维全 NaN 却得到分数: {dims}'
    # 理由仍然保留 —— 它们是诊断信息，不是分数。
    assert len(reasons) == 5


# ────────────────────── flow 判定的 confidence ──────────────────────


def _verdict(confidence):
    import inspect
    import re

    from nexus.extensions.tasks.artifact.judges.strategies.web_judger import verifier as V

    source = inspect.getsource(V)
    start = source.rindex('def ', 0, source.index("'confidence': confidence,"))
    name = re.match(r'def (\w+)', source[start:]).group(1)
    return getattr(V, name)(json.dumps({'verdict': 'pass', 'confidence': confidence}))


@pytest.mark.parametrize(
    ('label', 'value'),
    [('NaN', float('nan')), ('inf', float('inf')), ('越界 9', 9), ('bool', True)],
)
def test_confidence_rejects_unusable_values(label: str, value):
    """非法 confidence 归 0，而不是被 clamp 成 1.0（满信心）。

    ``isinstance(nan, float)`` 为 True，所以原来那个只查类型的守卫挡不住它。
    """
    assert _verdict(value)['confidence'] == 0.0, label


def test_confidence_keeps_legal_values():
    assert _verdict(0.8)['confidence'] == 0.8
    assert _verdict(0)['confidence'] == 0.0
    assert _verdict(1)['confidence'] == 1.0


# ────────────────────── svg 的两个维度 ──────────────────────


def _svg_parse(structure, aesthetics=3):
    from nexus.extensions.tasks.artifact.judges.strategies.svg import _parse_scrutiny_json

    return _parse_scrutiny_json(json.dumps({'structure': structure, 'aesthetics': aesthetics}))


@pytest.mark.parametrize(
    ('label', 'value'),
    [('NaN', float('nan')), ('inf', float('inf')), ('越界 99', 99), ('bool', True), ('低于下界 0', 0)],
)
def test_svg_rejects_unusable_dimension(label: str, value):
    """任一维非法 → 整个解析失败 → ``parse_failed``。

    不替一个中位分：编出来的分数会进训练，而且没人知道它是编的。修复前
    NaN 会变成 5.0，归一化后是 0.75（两维之一满分）。
    """
    assert _svg_parse(value) is None, label


def test_svg_keeps_legal_dimensions():
    parsed = _svg_parse(4, 3)

    assert parsed is not None
    assert (parsed[0], parsed[1]) == (4.0, 3.0)


def test_svg_both_dimensions_must_be_usable():
    """只有一维合法也不行 —— 合成分需要两维。"""
    assert _svg_parse(4, float('nan')) is None
    assert _svg_parse(float('nan'), 4) is None


# ────────────────────── 重复 fid ──────────────────────


def _validate_flows(flows):
    from nexus.extensions.tasks.artifact.judges.base import JudgeInput
    from nexus.extensions.tasks.artifact.judges.registry import get_judge

    return get_judge('web_judger').validate(
        JudgeInput(
            instance_id='i',
            problem_statement='p',
            openai_clients={},
            instance={'flow': {'flows': flows}},
        )
    )


def test_duplicate_fid_is_rejected():
    """按 fid 建字典时后一条覆盖前一条，声明 N 条只判 N-1 条。

    而 ``contract_weight`` 仍按 N 算 —— 覆盖率虚高，分数上看不出异常。
    """
    reason = _validate_flows([{'fid': 'F1'}, {'fid': 'F1'}])

    assert reason is not None
    assert '重复' in reason


def test_duplicate_fid_detected_after_stripping():
    """``' F1 '`` 和 ``'F1'`` 是同一个键 —— 下游 strip 后索引。"""
    assert _validate_flows([{'fid': 'F1'}, {'fid': ' F1 '}]) is not None


def test_unique_fids_pass():
    assert _validate_flows([{'fid': 'F1'}, {'fid': 'F2'}, {'fid': 'F3'}]) is None


def test_duplicate_check_does_not_mask_the_missing_fid_error():
    """缺 fid 的报错要保留 —— 两种问题的修法不同。"""
    reason = _validate_flows([{'fid': 'F1'}, {}])

    assert reason is not None
    assert '缺 fid' in reason


# ────────────────────── 上传还原保真 ──────────────────────
#
# 二进制值在 ``{path: content}`` 里是 ``data:<mime>;base64,<payload>``（为了在
# JSON 里无损传给 web-sandbox）。写回文件系统时必须解回字节 —— 按 utf8 写会把
# 那串**文本**存成 logo.png：上传逐文件都报成功，文件也存在，但内容不是图片。
# 判分读到的源码于是与产物不一致，而这件事没有任何报错。


def _upload_captured(files: dict[str, bytes]):
    import asyncio
    import base64
    import io
    import tarfile
    from dataclasses import dataclass

    from nexus.extensions.tasks.artifact.judges.capabilities.deploy import upload_archive_to_runtime

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as tf:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    archive = base64.b64encode(buf.getvalue()).decode()

    @dataclass
    class _Result:
        path: str
        success: bool
        message: str

    @dataclass
    class _Response:
        results: list

    captured: list = []

    class _Runtime:
        async def upload_file(self, files, **_kwargs):
            captured.extend(files)
            return _Response(results=[_Result(f.path, True, 'ok') for f in files])

    count, error = asyncio.run(upload_archive_to_runtime(_Runtime(), archive, '/workspace'))
    # 两种视图：按文件名取（方便断言编码），以及完整路径列表（嵌套路径会被
    # 文件名映射压平，所以不能只靠前者）。
    return count, error, {f.path.rsplit('/', 1)[-1]: f for f in captured}, [f.path for f in captured]


def test_binary_files_are_restored_as_bytes():
    """PNG 要以 base64 编码上传，还原后字节完全一致。"""
    import base64

    png = b'\x89PNG\r\n\x1a\n' + bytes(range(60))
    count, error, items, _paths = _upload_captured({'logo.png': png, 'index.html': b'<html/>'})

    assert (count, error) == (2, None)
    logo = items['logo.png']
    assert logo.encoding == 'base64', 'PNG 被当文本上传了'
    assert base64.b64decode(logo.content) == png, '字节还原不一致'


def test_text_files_stay_utf8():
    """文本文件不受影响 —— 不要为了修二进制把文本也 base64 化。"""
    _count, _error, items, _paths = _upload_captured({'index.html': b'<html/>', 'src/a.js': b'let x=1'})

    assert items['index.html'].encoding == 'utf8'
    assert items['index.html'].content == '<html/>'
    assert items['a.js'].content == 'let x=1'


def test_directory_structure_is_preserved():
    """嵌套路径要保留 —— 判分按相对路径找文件。"""
    _count, _error, _items, paths = _upload_captured({'src/nested/a.js': b'x', 'index.html': b'y'})

    assert '/workspace/src/nested/a.js' in paths, paths
    assert '/workspace/index.html' in paths, paths


# ────────────────────── reward 文件清理顺序 ──────────────────────


def test_reward_cleanup_runs_after_pre_run_setup():
    """顺序必须是 pre_run_setup → 清 reward → agent。

    ``code_quality`` 的 reward 文件在 ``working_dir`` 里，而它的 pre_run_setup
    把产物归档解回同一目录。先清后解包时，只要归档带着上一轮的 reward 文件，
    它就被原样恢复 —— agent 正常退出但没重写时，读到的是上一轮的分数，看不出
    任何异常。
    """
    import asyncio

    from nexus.extensions.tasks.artifact.agent_setup.claude_config import default_judge_claude_input
    from nexus.extensions.tasks.artifact.judges.capabilities import agent as agent_capability

    order: list[str] = []

    class _Remote:
        def on(self, *_a, **_k): ...
        def off(self, *_a, **_k): ...
        async def ws_disconnect(self): order.append('ws_disconnect')
        async def ws_connect(self): order.append('ws_connect')

        async def call_executor(self, _name, args):
            if 'rm -f' in str(args):
                order.append('rm_reward')
            return {}

        async def invoke(self, _fn, **_kwargs):
            order.append('agent')
            return {'success': True, 'output': 'ok', 'stderr': '', 'execution_time': 1.0}

    async def _setup(_runtime):
        order.append('pre_run_setup')

    asyncio.run(
        agent_capability.run_claude_code_on_prepared_runtime(
            _Remote(),
            'rt',
            task_prompt='p',
            openai_client=object(),
            claude_input=default_judge_claude_input(with_playwright=False),
            reward_file='/workspace/reward.txt',
            pre_run_setup=_setup,
        )
    )

    assert order.index('pre_run_setup') < order.index('rm_reward') < order.index('agent'), order


def test_ws_reconnect_is_explicit():
    """ws 重连要显式做，不能靠「下一个操作恰好会自动连」。

    哪个操作排在最前面是会变的细节（reward 清理刚刚就挪了位置），一旦那个操作
    不再碰 runtime，trajectory 会静默变空，而且没有任何线索指向原因。
    """
    import inspect

    from nexus.extensions.tasks.artifact.judges.capabilities import agent as agent_capability

    source = inspect.getsource(agent_capability.run_claude_code_on_prepared_runtime)

    assert 'await runtime.ws_connect()' in source, '重连必须显式'
    disconnect_at = source.index('ws_disconnect()')
    connect_at = source.index('ws_connect()')
    assert disconnect_at < connect_at, '先断后连'
