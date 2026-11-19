"""Board summary shared by the README board and the dashboard: per-model averages and the two ranking charts."""
from __future__ import annotations
import html

SETTINGS = ('单轮', '迭代')
CHART_FILES = {'单轮': 'board-single.svg', '迭代': 'board-iterative.svg'}
# One colour per model, the same in both charts.
COLORS = {'gpt-6.1-sol': '#10a37f', 'claude-opus-5-5': '#d97757', 'hy4': '#3b6ff5', 'qwen3.8-max-0902': '#7c5cff',
          'intern-s2': '#0ea5b7'}
FONT = ("Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, "
        "'PingFang SC', 'Microsoft YaHei', sans-serif")


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


def rank_models(groups: list[dict], models: list[str]) -> list[str]:
    """Models ordered by their iterative average, then their single average (best first; no score last)."""
    means = averages(groups, models)

    def key(model: str):
        return tuple(-(means[(model, s)][0] if means[(model, s)][0] is not None else -1.0) for s in ('迭代', '单轮'))
    return sorted(models, key=key)


def column_svg(title: str, subtitle: str, rows: list[tuple[str, str, float | None]], width: int = 520, height: int = 330) -> str:
    """Vertical bars sorted by score on a 0-100 scale, value labels on top, faint grid and a reference line at 50.

    rows: [(model, label, score)]; a label's first space breaks it onto two lines under its bar."""
    rows = sorted((r for r in rows if r[2] is not None), key=lambda r: -r[2])
    left, right, top, bottom = 34, 16, 78, 52
    plot_w, plot_h = width - left - right, height - top - bottom

    def y(value: float) -> float:
        return top + plot_h * (1 - value / 100)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'font-family="{FONT}">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="{left - 14}" y="30" font-size="18" font-weight="700" fill="#111827">{html.escape(title)}</text>',
             f'<text x="{left - 14}" y="52" font-size="12" fill="#6b7280">{html.escape(subtitle)}</text>']
    for value in (0, 25, 50, 75, 100):
        parts += [f'<line x1="{left}" y1="{y(value):.1f}" x2="{width - right}" y2="{y(value):.1f}" '
                  f'stroke="{"#d1d5db" if value == 0 else "#f0f1f3"}" stroke-width="1"/>',
                  f'<text x="{left - 8}" y="{y(value) + 4:.1f}" font-size="11" fill="#9ca3af" text-anchor="end">{value}</text>']
    slot = plot_w / max(len(rows), 1)
    bar = min(56.0, slot * 0.58)

    def haloed(text: str, attrs: str, fill: str) -> list[str]:
        """The text over a thick white copy of itself, so a line passing behind it stays clear in any renderer."""
        return [f'<text {attrs} fill="#ffffff" stroke="#ffffff" stroke-width="5" stroke-linejoin="round">{text}</text>',
                f'<text {attrs} fill="{fill}">{text}</text>']
    labels = []
    for i, (model, label, score) in enumerate(rows):
        cx = left + slot * (i + 0.5)
        top_y = y(score)
        parts.append(f'<rect x="{cx - bar / 2:.1f}" y="{top_y:.1f}" width="{bar:.1f}" height="{max(y(0) - top_y, 1.5):.1f}" rx="5" '
                     f'fill="{COLORS.get(model, "#6b7280")}"/>')
        labels += haloed(f'{score:.1f}', f'x="{cx:.1f}" y="{top_y - 8:.1f}" font-size="14" font-weight="700" '
                                         'text-anchor="middle"', '#111827')
        first, _, rest = label.partition(' ')
        for line, text in enumerate(filter(None, (first, rest))):
            labels.append(f'<text x="{cx:.1f}" y="{y(0) + 18 + 14 * line:.1f}" font-size="12" fill="#374151" '
                          f'text-anchor="middle">{html.escape(text)}</text>')
    # the reference line passes over the bars but under the value labels
    parts += [f'<line x1="{left}" y1="{y(50):.1f}" x2="{width - right}" y2="{y(50):.1f}" stroke="#6b7280" '
              'stroke-width="1" stroke-dasharray="5 4"/>',
              *haloed('参照 50', f'x="{width - right}" y="{y(50) - 6:.1f}" font-size="11" text-anchor="end"', '#6b7280'),
              *labels]
    return '\n'.join(parts + ['</svg>']) + '\n'


def chart_svgs(groups: list[dict], models: list[str], names: dict[str, str]) -> dict[str, str]:
    """{file name: svg} for the single and iterative rankings of the per-model averages."""
    means = averages(groups, models)
    tasks = len(groups)
    titles = {'单轮': '单轮（1 小时）', '迭代': '迭代（4 小时，带 public 反馈）'}
    return {CHART_FILES[s]: column_svg(titles[s], f'{tasks} 道题的平均分 · 满分 100',
                                       [(m, names.get(m, m), means[(m, s)][0]) for m in models]) for s in SETTINGS}
