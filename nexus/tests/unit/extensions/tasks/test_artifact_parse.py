"""``loads_json_lenient`` 的转义修复回归测试。

修复的动机：agent 在 JSON 字符串里写正则/路径时会漏转义一层（``grep 'x\\.jpg'`` 里的
``\\.`` 在 JSON 里必须是 ``\\\\.``），整份输出会因为一处手滑全丢。

**这里真正要守住的是「不破坏本来合法的输入」** —— ``loads_json_lenient`` 有 20+ 个调用方
（probe / atomic_probe / orbit_probe / svg / visual_anchored / web_judger…），一个过度
热心的修复会静默改坏它们的 JSON。所以修复只在 ``json.loads`` 已经失败后才触发，且必须
成对扫描：用 ``re.sub`` 对已经正确的 ``\\\\.`` 会从第二个反斜杠开始匹配，把它破坏成
``\\\\\\.``。
"""

from __future__ import annotations

import json

from nexus.extensions.tasks.artifact.judges.capabilities.parse import loads_json_lenient


def test_legal_escapes_are_untouched():
    """合法 JSON 一律原样解析 —— 修复路径根本不该被走到。"""
    cases = [
        r'{"a": "x\\.jpg"}',  # 已经正确的 \\.
        r'{"a": "line\nbreak"}',
        r'{"a": "quote\"q"}',
        r'{"a": "tab\tsep"}',
        r'{"a": "unié"}',
        r'{"a": "path\\to\\file"}',
    ]
    for raw in cases:
        obj, err = loads_json_lenient(raw)
        assert err is None, f'{raw!r} 应该直接解析成功，却报 {err}'
        assert obj == json.loads(raw), f'{raw!r} 解析结果被改动了'


def test_missing_escape_is_repaired():
    """漏转义一层的 ``\\.`` / ``\\d`` 能救回来。

    注意 ``\\'`` 在 JSON 里同样是非法转义，所以它也会被补成字面的 ``\\'``（两个字符）。
    这对下游无害 —— 单引号外的 ``\\'`` 在 shell 里和 ``'`` 等价 —— 关键是整份脚本不再丢。
    """
    obj, err = loads_json_lenient(r'{"code": "grep \'x\.jpg\' f"}')
    assert err is None
    assert obj is not None
    assert r'x\.jpg' in obj['code'], '正则里的 \\. 必须原样留着'

    obj, err = loads_json_lenient(r'{"re": "\d+"}')
    assert err is None
    assert obj == {'re': r'\d+'}


def test_mixed_correct_and_missing_in_one_string():
    """同一份输出里有的写对有的写漏（实测的真实形态）—— 两者最终要落到同一个字面值。

    这是修复最关键的一条：``re.sub`` 会从已经正确的 ``\\\\.`` 的第二个反斜杠开始匹配，
    把它破坏成 ``\\\\\\.``（解析出来变成两个反斜杠），下游 grep 就匹配不到了。
    """
    obj, err = loads_json_lenient(r'{"cmd": "a\\.jpg|b\.jpg"}')
    assert err is None
    assert obj is not None
    # 写对的 a\\.jpg 和写漏的 b\.jpg，解析后都应该是单个反斜杠 + 点
    assert obj['cmd'] == r'a\.jpg|b\.jpg', f'两处应等价，实际 {obj["cmd"]!r}'


def test_unescaped_quote_still_fails():
    """裸引号让字符串提前闭合，结构已经错乱 —— 必须继续报错而不是猜。

    ``code`` 字段是要拿去 exec 的，事后重建结构比丢掉这条危险得多。
    """
    obj, err = loads_json_lenient('{"code": "attr_of(x, "aria-expanded")"}')
    assert obj is None
    assert err and 'JSON parse failed' in err


def test_real_newline_inside_string_still_fails():
    """字符串里的真实换行同理 —— 报错，不猜。"""
    obj, err = loads_json_lenient('{"code": "open(f)\n    write(x)"}')
    assert obj is None
    assert err and 'JSON parse failed' in err


def test_empty_and_fenced_input():
    """原有行为不变：空输入报错，```json 围栏能剥掉。"""
    assert loads_json_lenient('') == (None, 'empty output')
    obj, err = loads_json_lenient('```json\n{"a": 1}\n```')
    assert err is None
    assert obj == {'a': 1}
