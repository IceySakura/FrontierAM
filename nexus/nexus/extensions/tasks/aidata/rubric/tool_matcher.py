"""共享的 tool-call step 匹配器 （被 ``tool_called`` +
``tool_called_sequence`` 使用）。

设计上与用例无关。 step 规格在两个 primitive 中形状相同的 dict， 因此
``tool_called.with_text = {...}`` 和 ``tool_called_sequence.steps[i].with_text = {...}``
的求值完全一致。

step 规格上支持的谓词 （除了 ``tool`` / ``tool_prefix`` /
``acceptable_tools`` 三选一之外全部可选）：

    tool:         精确的业务名 （裸名 ``email_send`` 或完全限定的
                  ``email_email_send`` —— server 前缀通过 ``_<exact>``
                  后缀匹配自动兼容， 与经典 ``_eval_tool_called`` 行为一致）。
    acceptable_tools:
                  可接受工具名的 OR 列表。 每个条目使用与 ``tool``
                  相同的匹配语义。 命名空间分隔符 ``/`` 和 ``.`` 会被
                  当作 ``_``， 因此 ``wecom/im_search`` 匹配 ``wecom_im_search``。
    tool_prefix:  对 MCP 前缀化的工具名的原始 startswith 匹配。
    with_text:    可以是纯字符串 （对整个 ``arguments`` JSON blob 做
                  大小写不敏感的子串匹配 —— v2 行为）， 或者是
                  一个 dict， 支持以下键：
                      field:         调用中的点式路径 （默认
                                     ``arguments.keyword``）。 支持的
                                     根： ``arguments``、 ``tool_name``、
                                     ``name``。
                      equals:        精确字符串相等。
                      contains:      子串 （大小写不敏感）。
                      contains_all:  子串列表 —— 全部必须命中。
                      contains_any:  子串列表 —— 任一命中即可。
                      regex:         Python 正则， 对原始 （保留大小写的）
                                     字段值调用 ``re.search``。
    arg_empty:    必须缺失 / falsy / 空字符串的参数键列表。
    arg_not_in:   映射 ``{key: [disallowed_values]}`` —— 参数的标量
                  值必须不等于任一列出的值。
    all_of / any_of / not:
                  布尔组合子 —— 每个子规格是完整 step 规格， 递归求值。
                  让标注者可以表达 “鞋子赛道但不是耳机” 等。
    matcher:      ``MATCHER_REGISTRY`` 中注册的名字 —— 面向罕见用例专用
                  谓词的逃生舱。 默认让注册表为空， 以保持 YAML 是唯一
                  真源； 只有当某个谓词被许多用例需要时才注册。

一个不含上述任何字段的 step 规格是纯粹的工具身份检查 （即最初 ``tool_called``
所支持的行为）。
"""

import json
import logging
import re

from collections.abc import Callable
from typing import Any


_logger = logging.getLogger(__name__)


# 与用例无关的具名 hook 注册表。 有意保持为空。
MATCHER_REGISTRY: dict[str, Callable[[dict], bool]] = {}


def _tool_call_args(call: dict) -> dict:
    """返回 tool_call 已解析的参数 dict （处理原始 JSON 字符串）。"""
    fn = call.get('function') or {}
    args = fn.get('arguments', {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    return args if isinstance(args, dict) else {}


def _coerce_to_str(v: Any) -> str:
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)
    return '' if v is None else str(v)


def _tool_call_field(call: dict, field_path: str) -> str:
    """解析 tool_call 内部的点式路径。

    ``arguments.keyword`` → call['function']['arguments']['keyword'] 的 json 字符串。
    ``tool_name`` / ``name`` → call['function']['name']。
    """
    if not field_path:
        return ''
    parts = field_path.split('.')
    fn = call.get('function') or {}
    node: Any = call
    if parts[0] == 'arguments':
        node = _tool_call_args(call)
        parts = parts[1:]
    elif parts[0] in ('tool_name', 'name'):
        return str(fn.get('name') or '')
    for p in parts:
        if isinstance(node, dict):
            node = node.get(p, '')
        else:
            node = ''
            break
    return _coerce_to_str(node)


def _eval_with_text(call: dict, spec: Any) -> bool:
    """在 tool_call step 规格上求值 ``with_text`` 谓词。

    ``None`` 通过。 纯字符串 = 对原始 ``arguments`` 字段做大小写不敏感的
    子串匹配， 与 v2 求值器 （``str(fn.get('arguments', ''))``） 逐字节兼
    容。 dict 形式支持 ``field`` + ``equals`` / ``contains`` /
    ``contains_all`` / ``contains_any`` / ``regex``。
    """
    if spec is None:
        return True
    if isinstance(spec, str):
        # 与原 ``_eval_tool_called`` 主体逐字节等价：
        # ``str(fn.get('arguments', ''))`` —— JSON 字符串参数原样保留； dict
        # 参数变为其 Python repr。 有意收紧， 使得任何现有 yaml 迁移到该
        # 辅助时都不会回归。
        fn = call.get('function') or {}
        raw = str(fn.get('arguments', ''))
        needle = spec.lower()
        if needle in raw.lower():
            return True
        # CJK / escaped-args 修复： tool_call 的 ``arguments`` 常常被序
        # 列化为 ASCII 转义 JSON （``\uXXXX``）， 因此字面的非 ASCII
        # ``with_text`` （例如中文关键词） 永远不可能是原始 blob 的子串。
        # 也匹配以 ``ensure_ascii=False`` 渲染的解码后参数。 上面的原始
        # 匹配保持了对 ASCII 规格的逐字节兼容； 此处仅新增匹配。
        parsed = _tool_call_args(call)
        if parsed:
            try:
                decoded = json.dumps(parsed, ensure_ascii=False)
            except (TypeError, ValueError):
                decoded = ''
            if needle in decoded.lower():
                return True
        return False
    if not isinstance(spec, dict):
        return False
    field_path = str(spec.get('field') or 'arguments.keyword')
    value = _tool_call_field(call, field_path)
    lower = value.lower()
    ok = True
    if 'equals' in spec:
        ok = ok and value == _coerce_to_str(spec['equals'])
    if 'contains' in spec:
        ok = ok and str(spec['contains']).lower() in lower
    if 'contains_all' in spec:
        needles = [str(x).lower() for x in (spec['contains_all'] or [])]
        ok = ok and all(n in lower for n in needles)
    if 'contains_any' in spec or 'any_of' in spec:
        src = spec.get('contains_any') or spec.get('any_of') or []
        needles = [str(x).lower() for x in src]
        ok = ok and (not needles or any(n in lower for n in needles))
    if 'regex' in spec:
        try:
            ok = ok and re.search(str(spec['regex']), value) is not None
        except re.error as exc:
            _logger.warning('[rubric.tool_called.with_text] bad regex %r: %s', spec.get('regex'), exc)
            return False
    return ok


def _eval_arg_empty(args: dict, spec: Any) -> bool:
    """``arg_empty: [a, b]`` —— 列出的字段必须缺失 / falsy / 空。"""
    if not spec:
        return True
    fields = list(spec) if isinstance(spec, (list, tuple)) else [spec]
    for k in fields:
        v = args.get(k)
        truthy = bool(v) if not isinstance(v, str) else bool(v.strip())
        if truthy:
            return False
    return True


def _eval_arg_not_in(args: dict, spec: Any) -> bool:
    """``arg_not_in: {field: [v1, v2]}`` —— 字段的标量值必须不等于
    任何列出的值 （字符串强转后精确相等）。
    """
    if not spec:
        return True
    for k, disallowed in spec.items():
        got = _coerce_to_str(args.get(k))
        choices = [str(x) for x in (disallowed if isinstance(disallowed, (list, tuple, set)) else [disallowed])]
        if got in choices:
            return False
    return True


def _normalise_tool_name(name: Any) -> str:
    return str(name or '').replace('/', '_').replace('.', '_')


def _tool_name_matches(actual: str, expected: Any) -> bool:
    """将轨迹中的工具名与 rubric 的工具别名进行匹配。

    保留既有的精确或 ``_<tool>`` 后缀语义， 然后在归一化编写 rubric 时常
    用的命名空间分隔符 （``wecom/im_search`` / ``wecom.im_search``） 后，
    再执行相同检查。
    """
    expected_s = str(expected or '')
    if not expected_s:
        return False
    if actual == expected_s or actual.endswith('_' + expected_s):
        return True
    actual_n = _normalise_tool_name(actual)
    expected_n = _normalise_tool_name(expected_s)
    return actual_n == expected_n or actual_n.endswith('_' + expected_n)


def _acceptable_tool_names(spec: dict) -> list[str]:
    tools: list[str] = []
    primary = spec.get('tool')
    if primary:
        tools.append(str(primary))
    aliases = spec.get('acceptable_tools')
    if isinstance(aliases, str):
        if aliases.strip():
            tools.append(aliases)
    elif isinstance(aliases, (list, tuple, set)):
        tools.extend(str(x) for x in aliases if str(x or '').strip())
    return tools


def _acceptable_tools_shape_valid(spec: dict) -> bool:
    aliases = spec.get('acceptable_tools')
    return aliases is None or isinstance(aliases, (str, list, tuple, set))


def _match_tool_call(spec: dict, call: dict) -> bool:
    """``tool_called`` / ``tool_called_sequence`` 的统一 step 匹配器。

    支持工具身份 （``tool`` / ``acceptable_tools`` /
    ``tool_prefix``） + 丰富的参数谓词 （``with_text`` /
    ``arg_empty`` / ``arg_not_in``） + 布尔组合子 （``all_of`` /
    ``any_of`` / ``not``） + ``matcher`` 逃生舱。 所有字段均可选 ——
    空 dict 匹配任何工具调用。
    """
    if not isinstance(spec, dict):
        return False

    # 布尔组合子 （递归）。 优先求值以便短路能够在支付工具身份 /
    # 参数抽取成本之前尽早返回。
    if 'all_of' in spec and not all(_match_tool_call(sub, call) for sub in (spec['all_of'] or [])):
        return False
    if 'any_of' in spec:
        subs = spec['any_of'] or []
        if subs and not any(_match_tool_call(sub, call) for sub in subs):
            return False
    if 'not' in spec and _match_tool_call(spec['not'], call):
        return False

    # 工具身份。 ``tool`` 是裸名形式； 我们通过 ``_<tool>`` 后缀匹配
    # 接受 MCP 前缀化的变体 ``<server>_<tool>``
    # （与经典 _eval_tool_called 语义一致）。
    fn = call.get('function') or {}
    name = fn.get('name') or ''
    if not _acceptable_tools_shape_valid(spec):
        return False
    acceptable = _acceptable_tool_names(spec)
    if acceptable:
        if not any(_tool_name_matches(name, want) for want in acceptable):
            return False
    else:
        prefix = spec.get('tool_prefix')
        if prefix and not name.startswith(prefix):
            return False

    # 参数谓词
    if not _eval_with_text(call, spec.get('with_text')):
        return False
    args = _tool_call_args(call)
    if not _eval_arg_empty(args, spec.get('arg_empty')):
        return False
    if not _eval_arg_not_in(args, spec.get('arg_not_in')):
        return False

    # 与用例无关的具名 matcher 逃生舱
    matcher_name = spec.get('matcher')
    if matcher_name:
        fn_matcher = MATCHER_REGISTRY.get(matcher_name)
        if fn_matcher is None:
            _logger.warning(
                '[rubric.tool_called] unknown matcher %r (MATCHER_REGISTRY is case-neutral)',
                matcher_name,
            )
            return False
        try:
            if not fn_matcher(call):
                return False
        except Exception as exc:  # noqa: BLE001
            _logger.warning('[rubric.tool_called] matcher %r raised %s', matcher_name, exc)
            return False

    return True
