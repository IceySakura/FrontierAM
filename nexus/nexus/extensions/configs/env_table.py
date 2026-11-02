"""Generate the environment-variable table in README.md from configs/{agents,llms,runtime}/*.yaml.

    python env_table.py           # print the table
    python env_table.py --write   # rewrite the region between the markers in README.md
    python env_table.py --check   # exit 1 if README.md is stale

The table is derived from every ``${oc.env:NAME,default}`` in the yaml files, so it cannot drift from them.
The "key" column is the yaml key the variable feeds, which is what the variable means in practice.
"""

import bisect
import re
import sys

from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).parent
README = HERE / 'README.md'
AXES = ('agents', 'llms', 'runtime')
START, END = '<!-- env-table:start -->', '<!-- env-table:end -->'
SECRET_RE = re.compile(r'(?:^|_)(?:TOKEN|SECRET|PASSWORD|KEY)$')

# A default may itself hold ``${...}``; the match consumes it, so every default is rescanned
# (``_nested``) instead of trusting one level of nesting.
ENV_RE = re.compile(r'\$\{oc\.env:([A-Z][A-Z0-9_]*)(?:,((?:[^{}]|\{[^{}]*\})*))?\}')
KEY_RE = re.compile(r'^(\s*)(?:- )?([A-Za-z_][\w.\-]*):')


def _line_index(text: str):
    """Per line: start offset, the dotted key it sits under, and whether it is a comment."""
    starts: list[int] = []
    keys: list[str] = []
    comments: list[bool] = []
    stack: list[tuple[int, str]] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip('\r\n')
        body = line.lstrip()
        comment = bool(body) and body.startswith('#')
        if body and not comment:
            m = KEY_RE.match(line)
            if m:
                indent = len(m.group(1))
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                stack.append((indent, m.group(2)))
        names = [k for _, k in stack if k != 'body']
        starts.append(offset)
        keys.append('.'.join(names[-2:]) if names else '')
        comments.append(comment)
        offset += len(raw)
    return starts, keys, comments


def _nested(default: str, key: str):
    """Variables inside a default: the enclosing match already consumed them."""
    for env in ENV_RE.finditer(default):
        inner = env.group(2)
        yield env.group(1), key, None if inner is None else inner.strip("'")
        if inner:
            yield from _nested(inner, key)


def _scan_file(path: Path):
    """Yield (variable, key, default) for each ``oc.env`` in a yaml file.

    Scans the whole text rather than line by line: a default can span lines
    (``agents/codex_0.160.0_codemode.yaml``'s ``CODEX_EXTRA_CONFIG_TOML``), and a
    per-line regex never sees the closing brace. The key column comes from a line
    index, and a match that starts on a commented-out line is skipped.
    """
    text = path.read_text(encoding='utf-8')
    starts, keys, comments = _line_index(text)
    for env in ENV_RE.finditer(text):
        line = bisect.bisect_right(starts, env.start()) - 1
        if comments[line]:
            continue
        key = keys[line]
        default = env.group(2)
        yield env.group(1), key, None if default is None else default.strip("'")
        if default:
            yield from _nested(default, key)


def collect() -> dict:
    """variable -> axis -> {'keys': set, 'defaults': set, 'files': set}"""
    table: dict = defaultdict(lambda: defaultdict(lambda: {'keys': set(), 'defaults': set(), 'files': set()}))
    for axis in AXES:
        for path in sorted((HERE / axis).glob('*.yaml')):
            for name, key, default in _scan_file(path):
                cell = table[name][axis]
                cell['files'].add(path.stem)
                if key:
                    cell['keys'].add(key)
                cell['defaults'].add('(必填)' if default is None else default or "''")
    return table


def _cell(items: set, limit: int = 3) -> str:
    shown = sorted(items)[:limit]
    text = ', '.join(f'`{s}`' if not s.startswith('(') else s for s in shown)
    return (text + f' 等 {len(items)} 种' if len(items) > limit else text).replace('|', '\\|')


def _default(name: str, item: str) -> str:
    """Do not copy a literal credential default into the README; shorten long fallbacks."""
    if item.startswith('('):
        return item
    if SECRET_RE.search(name) and item != "''" and not item.startswith('${'):
        return '(有默认值，已隐藏)'
    item = ' '.join(item.split())  # a default can span lines; one table row cannot
    return item if len(item) <= 40 else item[:40] + '…'


def render() -> str:
    table = collect()
    cross = sorted(n for n, axes in table.items() if len(axes) > 1)
    lines = [
        f'共 {len(table)} 个变量。其中 **{len(cross)} 个被多条轴同时读取**（见上一节）：'
        + '、'.join(f'`{n}`' for n in cross)
        + '。',
        '',
    ]
    lines += ['| 变量 | 轴 | 写入的键 | 默认值 | 文件数 |', '| --- | --- | --- | --- | --- |']
    for name in sorted(table):
        for axis in AXES:
            if axis not in table[name]:
                continue
            c = table[name][axis]
            defaults = {_default(name, d) for d in c['defaults']}
            lines.append(f'| `{name}` | {axis} | {_cell(c["keys"])} | {_cell(defaults, 4)} | {len(c["files"])} |')
    return '\n'.join(lines)


def _replace(text: str, body: str) -> str:
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f'{head}{START}\n{body}\n{END}{tail}'


def main(argv: list[str]) -> int:
    body = render()
    if '--write' in argv or '--check' in argv:
        # Explicit encoding: the table and the README around it are Chinese, and read_text()
        # otherwise follows the locale, which is ASCII in a container with LANG unset.
        text = README.read_text(encoding='utf-8')
        if START not in text or END not in text:
            print(f'README.md 里缺少标记 {START} / {END}', file=sys.stderr)
            return 2
        new = _replace(text, body)
        if '--check' in argv:
            if new != text:
                print('README.md 的环境变量表已过期，运行 `python env_table.py --write` 更新', file=sys.stderr)
                return 1
            print('环境变量表是最新的')
            return 0
        README.write_text(new, encoding='utf-8')
        rows = sum(1 for line in body.splitlines() if line.startswith('| `'))
        print(f'已更新 README.md 的环境变量表（{rows} 行）')
        return 0
    print(body)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
