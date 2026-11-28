"""Board summary shared by the README board and the dashboard: per-model averages and the two ranking charts."""
from __future__ import annotations
import html

SETTINGS = ('单轮', '迭代')
CHART_FILES = {'单轮': 'board-single.svg', '迭代': 'board-iterative.svg'}
# One colour per model, the same in both charts.
COLORS = {'gpt-6.1-sol': '#10a37f', 'claude-opus-5-5': '#d97757', 'hy4': '#3b6ff5', 'qwen3.8-max-0902': '#7c5cff',
          'intern-s2': '#0ea5b7', 'gpt-6.2-sol': '#065f46', 'gpt-6-astra': '#be185d', 'qwen4-max': '#5b21b6',
          'minimax-m3.1-flash': '#dc2626', 'glm-5.4': '#ca8a04'}
# Board column order: the newer version directly before its predecessor (user instruction, 2026-11-19).
BEFORE = {'gpt-6.2-sol': 'gpt-6.1-sol'}
# Kept in meta.json, shown on no page (README board, task pages, dashboard) by default (user instruction, 2026-11-26).
HIDDEN = {'intern-s2'}
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
    """Models ordered by their iterative average, then their single average (best first; no score last). A row still
    missing tasks ranks by its average with the missing cells counted as 0, so it sits roughly where it will end up
    instead of jumping ahead on a few cells (user instruction, 2026-11-26: order roughly by ranking). A model in BEFORE
    then moves directly before the model it names, whatever the averages."""
    means = averages(groups, models)

    def ranked(model: str, setting: str) -> float:
        mean, count = means[(model, setting)]
        return -1.0 if mean is None else mean * count / len(groups)
    order = sorted(models, key=lambda m: tuple(-ranked(m, s) for s in ('迭代', '单轮')))
    for model, anchor in BEFORE.items():
        if model in order and anchor in order and order.index(model) > order.index(anchor):
            order.remove(model)
            order.insert(order.index(anchor), model)
    return order


def ranking(groups: list[dict], models: list[str], names: dict[str, str], setting: str) -> list[tuple[str, str, float]]:
    """[(model, label, average)] for one setting, best first; only models with a score on every task, since a partial
    average is not comparable."""
    means = averages(groups, models)
    rows = [(m, names.get(m, m), means[(m, setting)][0]) for m in models if means[(m, setting)][1] == len(groups)]
    return sorted(rows, key=lambda r: -r[2])


def hbar_svg(title: str, subtitle: str, rows: list[tuple[str, str, float]], width: int = 640) -> str:
    """Horizontal bars on a 0-100 track, one colour per model, the value after each bar and a reference line at 50."""
    left, right, top, row, bar = 132, 56, 64, 32, 16
    track = width - left - right
    base = top + len(rows) * row
    height = base + 30

    def x(value: float) -> float:
        return left + track * value / 100

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
             f'font-family="{FONT}">',
             f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
             f'<text x="16" y="28" font-size="16" font-weight="700" fill="#111827">{html.escape(title)}</text>',
             f'<text x="16" y="48" font-size="12" fill="#6b7280">{html.escape(subtitle)}</text>']
    for i, (model, label, score) in enumerate(rows):
        y = top + i * row + (row - bar) / 2
        parts += [f'<text x="{left - 12}" y="{y + bar - 3:.1f}" font-size="13" fill="#374151" text-anchor="end">{html.escape(label)}</text>',
                  f'<rect x="{left}" y="{y:.1f}" width="{track}" height="{bar}" rx="4" fill="#f3f4f6"/>',
                  f'<rect x="{left}" y="{y:.1f}" width="{max(x(score) - left, 3):.1f}" height="{bar}" rx="4" '
                  f'fill="{COLORS.get(model, "#6b7280")}"/>',
                  f'<text x="{left + track + 12}" y="{y + bar - 3:.1f}" font-size="13" font-weight="700" fill="#111827">{score:.1f}</text>']
    parts += [f'<line x1="{x(50):.1f}" y1="{top - 4}" x2="{x(50):.1f}" y2="{base + 4}" stroke="#9ca3af" stroke-dasharray="4 3"/>']
    parts += [f'<text x="{x(v):.1f}" y="{base + 20}" font-size="11" fill="{"#6b7280" if v == 50 else "#9ca3af"}" '
              f'text-anchor="middle">{"参照 50" if v == 50 else v}</text>' for v in (0, 50, 100)]
    return '\n'.join(parts + ['</svg>']) + '\n'


def chart_svgs(groups: list[dict], models: list[str], names: dict[str, str]) -> dict[str, str]:
    """{file name: svg} for the single and iterative rankings of the per-model averages."""
    return {CHART_FILES[s]: hbar_svg(f'平均分 · compared by {s}', f'{len(groups)} 道题的平均分 · 满分 100',
                                     ranking(groups, models, names, s)) for s in SETTINGS}


def ranking_html(groups: list[dict], models: list[str], names: dict[str, str], default: str = '迭代') -> str:
    """The dashboard's ranking card: horizontal bars with a "Compared by" switch between the two settings."""
    def body(setting: str) -> str:
        rows = ''.join(
            f'<div class="rank-row"><span class="rank-name">{html.escape(label)}</span>'
            f'<span class="rank-track"><span class="rank-fill" style="width:{score:.1f}%;background:{COLORS.get(model, "#6b7280")}"></span>'
            f'<span class="rank-ref"></span></span><span class="rank-val">{score:.1f}</span></div>'
            for model, label, score in ranking(groups, models, names, setting))
        hidden = '' if setting == default else ' hidden'
        return f'<div class="rank-body" data-k="{setting}"{hidden}>{rows}</div>'
    buttons = ''.join(f'<button type="button" data-k="{s}" class="{"on" if s == default else ""}">{s}</button>'
                      for s in (default, *(s for s in SETTINGS if s != default)))
    return (f'<div class="rank"><div class="rank-head"><div class="rank-title">平均分排行<small>{len(groups)} 道题 · 满分 100 · 参照 = 50</small></div>'
            f'<div class="seg"><span>Compared by</span>{buttons}</div></div>'
            + ''.join(body(s) for s in SETTINGS)
            + '<div class="rank-axis"><span></span><span class="rank-ticks"><i style="left:0">0</i><i class="ref" style="left:50%">参照 50</i>'
              '<i style="left:100%">100</i></span><span></span></div></div>'
            '<script>document.querySelectorAll(".rank").forEach(function(card){card.querySelectorAll(".seg button").forEach('
            'function(b){b.addEventListener("click",function(){card.querySelectorAll(".seg button").forEach(function(x){'
            'x.classList.toggle("on",x===b)});card.querySelectorAll(".rank-body").forEach(function(p){p.hidden=p.dataset.k!==b.dataset.k})})})})'
            '</script>')
