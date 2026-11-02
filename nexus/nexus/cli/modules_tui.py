"""Textual TUI for nexus modules — interactive module & dependency browser."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Label, Static
from textual.widgets._data_table import ColumnKey, Coordinate

from nexus.core.module_meta import ModuleMeta


# ---------------------------------------------------------------------------
# Category display order
# ---------------------------------------------------------------------------

CATEGORY_ORDER = [
    'Agent',
    'Task',
    'Project',
    'LLM Client',
    'Tool',
    'Executor',
    'Provider',
    'Processor',
    'Other',
]

CATEGORY_COLOR: dict[str, str] = {
    'Agent': 'dodger_blue1',
    'Task': 'green3',
    'Tool': 'dark_orange',
    'LLM Client': 'medium_purple1',
    'Executor': 'cyan3',
    'Provider': 'steel_blue1',
    'Processor': 'grey70',
    'Project': 'gold1',
    'Other': 'white',
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModuleRow:
    meta: ModuleMeta
    # Check results (3-category)
    installed_ok: bool | None = None
    missing_base: list[str] = field(default_factory=list)
    missing_host_extras: list[str] = field(default_factory=list)
    missing_runtime_extras: list[str] = field(default_factory=list)

    @property
    def missing_host(self) -> list[str]:
        return self.missing_base + self.missing_host_extras

    @property
    def missing_runtime(self) -> list[str]:
        return self.missing_base + self.missing_runtime_extras

    @property
    def check_icon(self) -> str:
        if self.installed_ok is None:
            return '  '
        return '✅' if self.installed_ok else '❌'

    def dep_summary(self) -> str:
        h_total = len(self.meta.base_deps) + len(self.meta.host_extras)
        r_total = len(self.meta.runtime_extras)
        if h_total == 0 and r_total == 0:
            return '—'

        checked = self.installed_ok is not None

        def _fmt(prefix: str, total: int, missing: int) -> str:
            if not checked:
                s = f'{prefix}{total}'
            else:
                installed = total - missing
                s = f'{prefix}{installed}/{total}'
                s = f'[red]{s}[/red]' if missing else f'[green]{s}[/green]'
            return s

        parts: list[str] = []
        h_missing = len(self.missing_base) + len(self.missing_host_extras)
        r_missing = len(self.missing_runtime_extras)

        if h_total and r_total:
            parts.append(_fmt('H:', h_total, h_missing))
            parts.append(_fmt('R:', r_total, r_missing))
        elif h_total:
            parts.append(_fmt('', h_total, h_missing))
        else:
            parts.append(_fmt('R:', r_total, r_missing))

        return ' '.join(parts)

    @property
    def obj_summary(self) -> str:
        objs = self.meta.objects
        if not objs:
            return '—'
        # Singular display names for object types
        type_label = {
            'agents': 'agent',
            'tasks': 'task',
            'tools': 'tool',
            'executors': 'executor',
            'openai_clients': 'client',
            'runtime_providers': 'provider',
            'span_processors': 'processor',
            'run_parts': 'run part',
            'compose_hooks': 'compose hook',
            'dataset_sources': 'dataset source',
            'commands': 'command',
        }
        parts = []
        for key, obj_map in objs.items():
            n = len(obj_map)
            label = type_label.get(key, key)
            if n > 1:
                label += 's'
            parts.append(f'{n} {label}')
        return ' + '.join(parts)


# ---------------------------------------------------------------------------
# Table row types
# ---------------------------------------------------------------------------

ROW_TYPE_GROUP = 'group'
ROW_TYPE_MODULE = 'module'


@dataclass
class TableRowInfo:
    row_type: str  # 'group' or 'module'
    category: str = ''
    module_idx: int = -1  # index into self.rows for module rows


# ---------------------------------------------------------------------------
# Detail panel
# ---------------------------------------------------------------------------


class DetailPanel(Static):
    """Right-side panel showing selected module details."""

    def compose(self) -> ComposeResult:
        yield Label('Select a module', id='detail-content')

    def update_detail(self, row: ModuleRow | None) -> None:
        label = self.query_one('#detail-content', Label)
        if row is None:
            label.update('Select a module')
            return

        m = row.meta
        lines: list[str] = []
        lines.append(f'[bold]{m.name}[/bold]  [dim]{m.category}[/dim]')
        lines.append(f'  {m.module_path}')

        # Tags for composite types
        if len(m.object_type_tags) > 1:
            tag_str = '  '.join(f'[on dark_green] {t} [/]' for t in m.object_type_tags)
            lines.append(f'  {tag_str}')

        lines.append('')

        # Base dependencies (shared)
        lines.append('[bold underline]Base Dependencies[/bold underline]')
        if m.base_deps:
            for dep in m.base_deps:
                if dep in row.missing_base:
                    lines.append(f'  [red]✗ {dep}[/red]')
                elif row.installed_ok is not None:
                    lines.append(f'  [green]✓ {dep}[/green]')
                else:
                    lines.append(f'  {dep}')
        else:
            lines.append('  [dim]none[/dim]')

        # Host extras
        lines.append('')
        lines.append('[bold underline]Host Extras[/bold underline]')
        if m.host_extras:
            for dep in m.host_extras:
                if dep in row.missing_host_extras:
                    lines.append(f'  [red]✗ {dep}[/red]')
                elif row.installed_ok is not None:
                    lines.append(f'  [green]✓ {dep}[/green]')
                else:
                    lines.append(f'  {dep}')
        else:
            lines.append('  [dim]none[/dim]')

        # Runtime extras
        lines.append('')
        lines.append('[bold underline]Runtime Extras[/bold underline]')
        if m.runtime_extras:
            for dep in m.runtime_extras:
                if dep in row.missing_runtime_extras:
                    lines.append(f'  [red]✗ {dep}[/red]')
                elif row.installed_ok is not None:
                    lines.append(f'  [green]✓ {dep}[/green]')
                else:
                    lines.append(f'  {dep}')
        else:
            lines.append('  [dim]none[/dim]')

        # Objects
        if m.objects:
            lines.append('')
            lines.append('[bold underline]Registered Objects[/bold underline]')
            for obj_type, obj_map in m.objects.items():
                lines.append(f'  [cyan]{obj_type}[/cyan]:')
                for obj_name in obj_map:
                    lines.append(f'    {obj_name}')

        label.update('\n'.join(lines))


# ---------------------------------------------------------------------------
# Summary bar
# ---------------------------------------------------------------------------


class SummaryBar(Static):
    total: reactive[int] = reactive(0)
    checked: reactive[int] = reactive(0)
    ok: reactive[int] = reactive(0)
    failed: reactive[int] = reactive(0)

    def render(self) -> str:
        parts = [f' Modules: {self.total}']
        if self.checked > 0:
            parts.append(f'Checked: {self.checked}/{self.total}')
            parts.append(f'[green]OK: {self.ok}[/green]')
            if self.failed:
                parts.append(f'[red]Missing: {self.failed}[/red]')
        return '  │  '.join(parts)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

CSS = """
#main-container {
    height: 1fr;
}

#table-container {
    width: 3fr;
    height: 100%;
}

#detail-container {
    width: 2fr;
    height: 100%;
    border-left: solid $accent;
    padding: 1 2;
    overflow-y: auto;
}

#detail-content {
    width: 100%;
}

#summary-bar {
    height: 1;
    dock: bottom;
    background: $surface;
    color: $text;
    padding: 0 1;
}

DataTable {
    height: 100%;
}
"""


class DepsTUI(App):
    """Interactive dependency browser for nexus modules."""

    TITLE = 'nexus modules'
    CSS = CSS
    BINDINGS = [
        Binding('q', 'quit', 'Quit'),
        Binding('c', 'check_all', 'Check installed'),
        Binding('i', 'install_selected', 'Install host'),
        Binding('I', 'install_all_selected', 'Install host+rt'),
        Binding('r', 'refresh_table', 'Refresh'),
        Binding('space', 'toggle_group', 'Toggle group', show=False),
        Binding('enter', 'toggle_group', 'Toggle group', show=False),
    ]

    rows: list[ModuleRow] = []
    _col_keys: dict[str, ColumnKey] = {}
    _row_info: dict[str, TableRowInfo] = {}  # row_key → info
    _collapsed: set[str] = set()  # collapsed category names
    _grouped: dict[str, list[int]] = {}  # category → [module indices]

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id='main-container'):
            with Vertical(id='table-container'):
                yield DataTable(id='modules-table')
            with Vertical(id='detail-container'):
                yield DetailPanel()
        yield SummaryBar(id='summary-bar')
        yield Footer()

    def on_mount(self) -> None:
        self._load_modules()
        self.action_check_all()

    def _load_modules(self) -> None:
        from nexus.core.module_meta import discover_all_modules

        modules = discover_all_modules()
        self.rows = [ModuleRow(meta=m) for m in modules]

        # Group by category
        grouped: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(self.rows):
            grouped[row.meta.category].append(i)
        self._grouped = dict(grouped)
        self._collapsed = set()

        self._populate_table()
        summary = self.query_one('#summary-bar', SummaryBar)
        summary.total = len(self.rows)

    def _populate_table(self) -> None:
        table = self.query_one('#modules-table', DataTable)
        table.clear(columns=True)
        col_keys = table.add_columns('Module', 'Deps', 'Objects', 'Chk')
        labels = ('Module', 'Deps', 'Objects', 'Chk')
        self._col_keys = dict(zip(labels, col_keys, strict=False))
        table.cursor_type = 'row'
        self._row_info = {}

        for cat in CATEGORY_ORDER:
            if cat not in self._grouped:
                continue
            indices = self._grouped[cat]
            collapsed = cat in self._collapsed
            arrow = '▶' if collapsed else '▼'
            count = len(indices)

            # Group header row
            group_key = f'g:{cat}'
            color = CATEGORY_COLOR.get(cat, 'white')
            table.add_row(
                f'{arrow} [bold {color}]{cat}[/bold {color}] [dim]({count})[/dim]',
                '',
                '',
                '',
                key=group_key,
            )
            self._row_info[group_key] = TableRowInfo(row_type=ROW_TYPE_GROUP, category=cat)

            if not collapsed:
                for idx in indices:
                    row = self.rows[idx]
                    row_key = f'm:{idx}'
                    table.add_row(
                        f'  {row.meta.name}',
                        row.dep_summary(),
                        row.obj_summary,
                        row.check_icon,
                        key=row_key,
                    )
                    self._row_info[row_key] = TableRowInfo(row_type=ROW_TYPE_MODULE, category=cat, module_idx=idx)

    def _rebuild_table(self) -> None:
        """Rebuild table preserving cursor position as best we can."""
        table = self.query_one('#modules-table', DataTable)
        # Remember current row key
        cur_key: str | None = None
        if table.cursor_row is not None:
            try:
                cur_row_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key
                cur_key = str(cur_row_key.value) if cur_row_key.value is not None else None
            except Exception:
                pass

        self._populate_table()

        # Try to restore cursor
        if cur_key is not None:
            try:
                row_idx = table.get_row_index(cur_key)
                table.move_cursor(row=row_idx)
            except Exception:
                pass

    def _update_module_row(self, idx: int) -> None:
        """Update a single module row's dynamic cells."""
        row = self.rows[idx]
        row_key = f'm:{idx}'
        table = self.query_one('#modules-table', DataTable)
        # Only update if the row exists (not collapsed)
        try:
            table.get_row_index(row_key)
        except Exception:
            return
        chk_key = self._col_keys.get('Chk')
        deps_key = self._col_keys.get('Deps')
        if chk_key is not None:
            table.update_cell(row_key, chk_key, row.check_icon)
        if deps_key is not None:
            table.update_cell(row_key, deps_key, row.dep_summary(), update_width=True)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.row_key is None or event.row_key.value is None:
            return
        key = str(event.row_key.value)
        info = self._row_info.get(key)
        if info is None:
            return
        detail = self.query_one(DetailPanel)
        if info.row_type == ROW_TYPE_MODULE and 0 <= info.module_idx < len(self.rows):
            detail.update_detail(self.rows[info.module_idx])
        else:
            detail.update_detail(None)

    # ------------------------------------------------------------------
    # Toggle group collapse
    # ------------------------------------------------------------------

    def action_toggle_group(self) -> None:
        table = self.query_one('#modules-table', DataTable)
        if table.cursor_row is None:
            return
        try:
            cur_row_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key
            key = str(cur_row_key.value) if cur_row_key.value is not None else None
        except Exception:
            return
        if key is None:
            return
        info = self._row_info.get(key)
        if info is None or info.row_type != ROW_TYPE_GROUP:
            return

        cat = info.category
        if cat in self._collapsed:
            self._collapsed.discard(cat)
        else:
            self._collapsed.add(cat)
        self._rebuild_table()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    @work(thread=True)
    def action_check_all(self) -> None:
        from nexus.core.dep_checker import check_module

        summary = self.query_one('#summary-bar', SummaryBar)

        for i, row in enumerate(self.rows):
            status = check_module(row.meta)
            row.installed_ok = status.ok
            row.missing_base = status.missing_base
            row.missing_host_extras = status.missing_host_extras
            row.missing_runtime_extras = status.missing_runtime_extras
            self.call_from_thread(self._update_module_row, i)
            summary.checked = i + 1
            summary.ok = sum(1 for r in self.rows[: i + 1] if r.installed_ok)
            summary.failed = sum(1 for r in self.rows[: i + 1] if r.installed_ok is False)

        self.call_from_thread(self._refresh_current_detail)

    # ------------------------------------------------------------------
    # Install
    # ------------------------------------------------------------------

    def _get_selected_row(self) -> ModuleRow | None:
        table = self.query_one('#modules-table', DataTable)
        if table.cursor_row is None:
            return None
        try:
            cur_row_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key
            key = str(cur_row_key.value) if cur_row_key.value is not None else None
        except Exception:
            return None
        if key is None:
            return None
        info = self._row_info.get(key)
        if info is not None and info.row_type == ROW_TYPE_MODULE and 0 <= info.module_idx < len(self.rows):
            return self.rows[info.module_idx]
        return None

    def action_install_selected(self) -> None:
        """Install missing host deps (base + host extras) for the selected module."""
        row = self._get_selected_row()
        if row is None:
            self.notify('No module selected', severity='warning')
            return
        if row.installed_ok is None:
            self.notify('Run check first (c)', severity='warning')
            return
        packages = row.missing_host
        if not packages:
            self.notify(f'{row.meta.name}: no missing host deps', severity='information')
            return
        self._do_install(list(packages), row.meta.name)

    def action_install_all_selected(self) -> None:
        """Install missing runtime deps (base + runtime extras) for the selected module."""
        row = self._get_selected_row()
        if row is None:
            self.notify('No module selected', severity='warning')
            return
        if row.installed_ok is None:
            self.notify('Run check first (c)', severity='warning')
            return
        packages = row.missing_runtime
        if not packages:
            self.notify(f'{row.meta.name}: no missing runtime deps', severity='information')
            return
        self._do_install(list(packages), row.meta.name)

    @work(thread=True)
    def _do_install(self, packages: list[str], label: str) -> None:
        from nexus.core.dep_checker import install_packages

        self.call_from_thread(self.notify, f'Installing {label}: {" ".join(packages)}...')
        ok, msg = install_packages(packages)
        if ok:
            self.call_from_thread(self.notify, f'{label}: installed successfully', severity='information')
            self.action_check_all()
        else:
            self.call_from_thread(self.notify, f'Install failed: {msg[:200]}', severity='error')

    def _refresh_current_detail(self) -> None:
        table = self.query_one('#modules-table', DataTable)
        if table.cursor_row is None:
            return
        try:
            cur_row_key = table.coordinate_to_cell_key(Coordinate(table.cursor_row, 0)).row_key
            key = str(cur_row_key.value) if cur_row_key.value is not None else None
        except Exception:
            return
        if key is None:
            return
        info = self._row_info.get(key)
        if info is not None and info.row_type == ROW_TYPE_MODULE:
            detail = self.query_one(DetailPanel)
            detail.update_detail(self.rows[info.module_idx])

    def action_refresh_table(self) -> None:
        self._load_modules()
        self.action_check_all()


def run_tui() -> None:
    """Launch the modules TUI."""
    app = DepsTUI()
    app.run()
