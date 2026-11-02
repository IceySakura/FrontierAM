"""Prompt contract tests — 确保 prompt 文件的占位符和代码实际传的 key 对得上,
以及 render_prompt 的替换行为正确(不注入、不误伤字面花括号)。

这几个测试就是"确保 prompt 里面没问题"的自动化保障:改了 prompt 文件加/删占位符,
或改了 judge 传的 key,任一边对不上立刻在这里报错,不用等真跑 batch 才发现。
"""

from __future__ import annotations

import re

from nexus.extensions.tasks.artifact.judges.capabilities.prompts import PROMPTS_DIR, load_prompt, render_prompt
from nexus.extensions.tasks.artifact.judges.strategies.general_agent import (
    STRICT_PROMPT_FILES,
    resolve_strict_prompt_type,
)


_PLACEHOLDER_RE = re.compile(r'\{(\w+)\}')


def _placeholders(text: str) -> set[str]:
    return set(_PLACEHOLDER_RE.findall(text))


# 每个 prompt 文件 → 代码在 render_prompt 时实际会传入的 key 集合(见各 strategy/pipeline)。
# 文件里的占位符必须 ⊆ 这个集合,否则运行时会留下没填的 {key}。
_PROMPT_SUPPLIED_KEYS: dict[str, set[str]] = {
    'judge_aesthetic.txt': {'query'},
    'judge_baseline_check.txt': {'query', 'url', 'reward_file'},
    'judge_general_agent.txt': {'query', 'url', 'reward_file'},
    'judge_general_agent_strict_3d.txt': {'query', 'url', 'reward_file'},
    'judge_general_agent_strict_game.txt': {'query', 'url', 'reward_file'},
    'judge_general_agent_strict_svg.txt': {'query', 'url', 'reward_file'},
    'judge_general_agent_strict_web.txt': {'query', 'url', 'reward_file'},
    'judge_static_code_review.txt': {'query', 'working_dir', 'reward_file'},
    'judge_svg.txt': {'query'},
    # ``shots`` 逐组说明每组截图对应哪个交互状态，由 ``_shots_desc`` 生成
    # （见 strategies/visual_anchored.py）。
    'judge_visual_anchored.txt': {'query', 'shots'},
}


def test_every_prompt_placeholder_is_supplied():
    """prompt 文件里出现的每个 {占位符} 都必须在代码传入的 key 集合里。"""
    for fname, supplied in _PROMPT_SUPPLIED_KEYS.items():
        used = _placeholders(load_prompt(fname))
        missing = used - supplied
        assert not missing, f'{fname} 用了未被代码填充的占位符: {missing} (代码只传 {supplied})'


def test_prompt_files_all_covered():
    """prompts/ 下的每个 judge_*.txt 都在上表里登记了(防止新增文件漏测)。"""
    on_disk = {p.name for p in PROMPTS_DIR.glob('judge_*.txt')}
    assert on_disk == set(_PROMPT_SUPPLIED_KEYS), (
        f'prompt 文件与测试登记不一致: 只在磁盘 {on_disk - set(_PROMPT_SUPPLIED_KEYS)}, '
        f'只在测试 {set(_PROMPT_SUPPLIED_KEYS) - on_disk}'
    )


def test_render_fills_known_and_keeps_unknown():
    assert render_prompt('Q={query}', query='hi') == 'Q=hi'
    # 未填的占位符原样保留,不 crash、不清空
    assert render_prompt('{query}/{missing}', query='Q') == 'Q/{missing}'


def test_render_no_reinjection():
    """填进去的值若含 {另一个key},不被二次替换(回归 bug: rubric 文本含 {query})。"""
    out = render_prompt('R={rubrics_text} Q={query}', rubrics_text='显示 {query} 字样', query='PS')
    assert out == 'R=显示 {query} 字样 Q=PS'


def test_render_preserves_literal_braces():
    """JSON / CSS 花括号不匹配 {word},原样保留(不像 str.format 会崩)。"""
    assert render_prompt('ex {"winner":"A"} q={query}', query='Q') == 'ex {"winner":"A"} q=Q'
    assert render_prompt('a {{ margin:0 }} {query}', query='Q') == 'a {{ margin:0 }} Q'


def test_strict_prompt_type_prefers_instance_metadata():
    assert resolve_strict_prompt_type('auto', {'_task_type': 'svg'}, 'Build a Three.js game') == 'svg'
    assert resolve_strict_prompt_type('auto', {'task_type': 'webgl'}, 'Build a dashboard') == '3d'


def test_strict_prompt_type_infers_from_query():
    assert resolve_strict_prompt_type('auto', None, 'Build a Phaser game with two players') == 'game'
    assert resolve_strict_prompt_type('auto', None, 'Draw a single SVG vector illustration') == 'svg'
    assert resolve_strict_prompt_type('auto', None, 'Create a Three.js WebGL shader scene') == '3d'
    assert resolve_strict_prompt_type('auto', None, 'Build an ecommerce dashboard with a form') == 'web'


def test_strict_prompt_type_allows_explicit_override():
    assert resolve_strict_prompt_type('game', {'_task_type': 'svg'}, 'Draw an SVG') == 'game'


def test_strict_prompt_files_all_exist_and_have_no_generic_fallback():
    """strict 只按品类路由,没有 generic 兜底 —— 表里每个文件都必须真的在磁盘上。

    回归 bug: 原来推断不出品类时 fallback 到 ``judge_general_agent_strict.txt``,
    该文件删掉后 judge 会在 load_prompt 抛 FileNotFoundError 而不是给出结构化失败。
    """
    assert set(STRICT_PROMPT_FILES) == {'web', 'svg', '3d', 'game'}
    for fname in STRICT_PROMPT_FILES.values():
        assert (PROMPTS_DIR / fname).is_file(), f'strict prompt 缺文件: {fname}'
    assert not (PROMPTS_DIR / 'judge_general_agent_strict.txt').exists(), (
        '已删除的 generic strict prompt 又回来了 —— 要么恢复 fallback,要么删掉它'
    )


def test_strict_prompt_type_unresolvable_returns_none():
    """认不出品类时返回 None,由 judge 转成 config_error(而不是去 load 不存在的文件)。"""
    assert resolve_strict_prompt_type('auto', None, 'do something nice') is None
    assert resolve_strict_prompt_type('auto', {'_task_type': 'cad'}, 'build it') is None
