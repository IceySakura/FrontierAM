"""Board summary shared by the README board and the dashboard: per-model averages and the two ranking charts."""
from __future__ import annotations
import html

SETTINGS = ('单轮', '迭代')
CHART_FILES = {'单轮': 'board-single.svg', '迭代': 'board-iterative.svg'}


def cell_score(results: list[dict], model: str, setting: str):
    """The score of one board cell: the latest attempt that has a score, as the cell lists them."""
    hits = sorted((r for r in results if r['model'] == model and r.get('setting', '单轮') == setting),
                  key=lambda r: r.get('attempt', 1))
    scores = [r['score'] for r in hits if r.get('score') is not None]
    return scores[-1] if scores else None


def averages(groups: list[dict], models: list[str]) -> dict:
    """{(model, setting): (mean, tasks)}: the plain mean over the tasks whose cell has a score."""
    out = {}
    for model in models:
        for setting in SETTINGS:
            values = [v for g in groups if (v := cell_score(g['results'], model, setting)) is not None]
            out[(model, setting)] = (sum(values) / len(values), len(values)) if values else (None, 0)
    return out


def bar_svg(title: str, rows: list[tuple[str, float | None]], width: int = 560) -> str:
    """Horizontal bars (0–100) sorted by score, with the reference line at 50."""
    rows = sorted((r for r in rows if r[1] is not None), key=lambda r: -r[1])
    left, right, top, bar, gap = 130, 50, 36, 22, 8
    plot = width - left - right
    base = top + len(rows) * (bar + gap)
    height = base + 26

    def x(value: float) -> float:
        return left + plot * value / 100

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             'font-family="Helvetica, Arial, sans-serif" font-size="13">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="{left}" y="22" font-size="15" font-weight="bold">{html.escape(title)}</text>']
    for i, (label, score) in enumerate(rows):
        y = top + i * (bar + gap)
        parts += [f'<text x="{left - 8}" y="{y + bar * 0.72:.1f}" text-anchor="end">{html.escape(label)}</text>',
                  f'<rect x="{left}" y="{y}" width="{max(x(score) - left, 1):.1f}" height="{bar}" '
                  f'fill="{"#2f7ed8" if score >= 50 else "#8fb8e8"}"/>',
                  f'<text x="{x(score) + 6:.1f}" y="{y + bar * 0.72:.1f}">{score:.1f}</text>']
    parts += [f'<line x1="{x(50):.1f}" y1="{top - 6}" x2="{x(50):.1f}" y2="{base}" stroke="#d9534f" stroke-dasharray="4 3"/>',
              f'<text x="{x(50):.1f}" y="{base + 16}" text-anchor="middle" fill="#d9534f" font-size="11">参照 = 50</text>']
    parts += [f'<text x="{x(v):.1f}" y="{base + 16}" text-anchor="middle" fill="#666" font-size="11">{v}</text>' for v in (0, 100)]
    return '\n'.join(parts + ['</svg>']) + '\n'


def chart_svgs(groups: list[dict], models: list[str], names: dict[str, str]) -> dict[str, str]:
    """{file name: svg} for the single and iterative rankings of the per-model averages."""
    means = averages(groups, models)
    return {CHART_FILES[s]: bar_svg(f'{s}平均分', [(names.get(m, m), means[(m, s)][0]) for m in models]) for s in SETTINGS}
