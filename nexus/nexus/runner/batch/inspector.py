"""Batch inspector TUI application."""

from __future__ import annotations

import asyncio
import json
import re

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rich.highlighter import Highlighter
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.reactive import reactive
from textual.widgets import DataTable, Footer, Header, Static, TabbedContent, Tree
from textual.widgets._tabbed_content import TabPane

from nexus.runner.batch.database import BatchDatabase
from nexus.trace.processors import SpanProcessor
from nexus.trace.tracer import replay_span_events
from nexus.trace.types import Span


class LogHighlighter(Highlighter):
    """Custom highlighter for log lines with format: YYYY-MM-DD HH:MM:SS | LEVEL | module | message"""

    def highlight(self, text: Text) -> None:
        """Apply highlighting to log text."""
        # Pattern: YYYY-MM-DD HH:MM:SS | LEVEL | module | message
        log_pattern = re.compile(
            r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(DEBUG|INFO|WARNING|ERROR|CRITICAL)\s*\|\s*([^\|]+)\s*\|\s*(.*)$'
        )

        # Work with plain text for regex matching
        plain_text = text.plain
        current_pos = 0

        for line in plain_text.split('\n'):
            if not line:
                current_pos += 1  # Account for newline
                continue

            match = log_pattern.match(line)
            if match:
                timestamp, level, module, _ = match.groups()

                # Calculate positions relative to current line
                line_start = current_pos

                # Highlight timestamp (dim cyan)
                timestamp_start = line_start
                timestamp_end = timestamp_start + len(timestamp)
                text.stylize('dim cyan', timestamp_start, timestamp_end)

                # Highlight level with different colors
                level_start = line_start + len(timestamp) + 3  # Skip ' | '
                level_end = level_start + len(level)
                level_style = {
                    'DEBUG': 'dim blue',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red bold',
                    'CRITICAL': 'red bold reverse',
                }.get(level, 'white')
                text.stylize(level_style, level_start, level_end)

                # Highlight module name (magenta)
                module_start = level_end + 3  # Skip ' | '
                module_end = module_start + len(module)
                text.stylize('magenta', module_start, module_end)

                # Message stays default color - don't apply blanket highlighting
                # Only highlight ERROR/WARNING level messages, not based on message content
                # This prevents false positives like "completed" or "error" appearing in normal messages

            # Move position to next line
            current_pos += len(line) + 1  # +1 for newline


@dataclass
class RunSummary:
    """Summary statistics for batch runs."""

    total: int
    pending: int
    running: int
    success: int
    failed: int
    timeout: int
    cancelled: int
    score_correct: int  # success with score=1
    score_incorrect: int  # success with score=0
    score_unknown: int  # success with no score or other score values


class ProgressBar(Static):
    """Custom progress bar widget showing run status and score."""

    summary = reactive(RunSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = 'Progress'
        self.start_time = datetime.now()
        self.last_completed = 0
        self.last_update_time = datetime.now()
        self.rate = 0.0  # items per second

    def render(self) -> Text:
        """Render the progress bar in tqdm style."""
        s = self.summary

        # Avoid division by zero
        total = max(s.total, 1)
        completed = s.success + s.failed + s.timeout + s.cancelled

        # Calculate completion percentage
        pct = (completed / total) * 100

        # Update rate calculation
        now = datetime.now()
        time_delta = (now - self.last_update_time).total_seconds()
        if time_delta > 0:
            completed_delta = completed - self.last_completed
            # Smooth rate with exponential moving average
            instant_rate = completed_delta / time_delta
            self.rate = 0.7 * self.rate + 0.3 * instant_rate if self.rate > 0 else instant_rate
            self.last_completed = completed
            self.last_update_time = now

        # Calculate elapsed and remaining time
        elapsed = (now - self.start_time).total_seconds()
        elapsed_str = self._format_time(elapsed)

        remaining_str = '??:??'
        if self.rate > 0 and completed < total:
            remaining_items = total - completed
            remaining_sec = remaining_items / self.rate
            remaining_str = self._format_time(remaining_sec)

        # Build prefix and suffix with minimal width (no extra padding)
        # Prefix: "XX%|"
        prefix = f'{int(pct)}%|'

        # Suffix: "| completed/total [elapsed<remaining, rate]"
        suffix = f'| {completed}/{total} [{elapsed_str}<{remaining_str}, {self.rate:.2f}it/s]'

        # Calculate bar width based on available space
        reserved_width = len(prefix) + len(suffix)
        bar_width = max(self.size.width - reserved_width - 2, 10)  # -2 for padding, minimum 10

        # Calculate bar segments
        success_ratio = s.success / total
        failed_ratio = (s.failed + s.timeout + s.cancelled) / total
        running_ratio = s.running / total

        success_width = round(success_ratio * bar_width)
        failed_width = round(failed_ratio * bar_width)
        running_width = round(running_ratio * bar_width)
        pending_width = bar_width - success_width - failed_width - running_width

        # Adjust for rounding errors
        total_width = success_width + failed_width + running_width + pending_width
        if total_width != bar_width:
            diff = bar_width - total_width
            if success_width >= max(failed_width, running_width, pending_width):
                success_width += diff
            elif failed_width >= max(running_width, pending_width):
                failed_width += diff
            elif running_width >= pending_width:
                running_width += diff
            else:
                pending_width += diff

        # Line 1: tqdm-style progress bar (dynamic width, no padding)
        text = Text()
        text.append(prefix, style='bold cyan')
        if success_width > 0:
            text.append('█' * success_width, style='green')
        if failed_width > 0:
            text.append('█' * failed_width, style='red')
        if running_width > 0:
            text.append('▒' * running_width, style='yellow')
        if pending_width > 0:
            text.append(' ' * pending_width)
        # Build suffix with different styles (no padding)
        text.append('| ', style='')
        text.append(f'{completed}/{total}', style='bold')
        text.append(f' [{elapsed_str}', style='dim')
        text.append('<', style='dim')
        text.append(f'{remaining_str}', style='dim')
        text.append(', ', style='dim')
        text.append(f'{self.rate:.2f}it/s', style='cyan')
        text.append(']\n', style='dim')

        # Line 2: Status breakdown
        text.append('Status: ', style='bold')
        text.append(f'success: {s.success}', style='green')
        text.append(', ', style='dim')
        text.append(f'failed: {s.failed}', style='red')
        if s.timeout > 0:
            text.append(', ', style='dim')
            text.append(f'timeout: {s.timeout}', style='red')
        if s.cancelled > 0:
            text.append(', ', style='dim')
            text.append(f'cancelled: {s.cancelled}', style='red')
        text.append(' | ', style='dim')
        text.append(f'running: {s.running}', style='yellow')
        text.append(' | ', style='dim')
        text.append(f'pending: {s.pending}\n', style='blue')

        # Line 3: Score
        if s.success > 0:
            score_total = s.score_correct + s.score_incorrect + s.score_unknown
            text.append('Score: ', style='bold')
            text.append(f'correct: {s.score_correct}', style='green')
            text.append(' / ', style='dim')
            text.append(f'incorrect: {s.score_incorrect}', style='red')
            if s.score_unknown > 0:
                text.append(' / ', style='dim')
                text.append(f'unknown: {s.score_unknown}', style='yellow')

            if score_total > 0:
                score_pct = (s.score_correct / score_total) * 100
                text.append(f' = {score_pct:.1f}%', style='bold cyan')

            # Estimate final score range
            remaining = s.pending + s.running
            if remaining > 0:
                best_correct = s.score_correct + remaining
                best_total = completed + remaining
                best_pct = (best_correct / best_total) * 100

                worst_correct = s.score_correct
                worst_total = completed + remaining
                worst_pct = (worst_correct / worst_total) * 100

                text.append(' | estimated: ', style='dim')
                text.append(f'{worst_pct:.1f}%-{best_pct:.1f}%', style='magenta')

        return text

    def _format_time(self, seconds: float) -> str:
        """Format seconds into MM:SS or HH:MM:SS."""
        if seconds < 0:
            return '??:??'
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f'{hours:02d}:{minutes:02d}:{secs:02d}'
        return f'{minutes:02d}:{secs:02d}'


class RunList(DataTable):
    """List of all runs in the batch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cursor_type = 'row'
        self.zebra_stripes = True
        self.border_title = 'Runs'
        self.sort_column: str | None = None
        self.sort_reverse: bool = False

    def on_mount(self) -> None:
        """Setup table columns."""
        self.add_columns('Run ID', 'Status', 'Start Time', 'Duration', 'Score', 'Error')


class RunDetail(Container):
    """Detailed view of a single run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = 'Run Details'
        self.current_run_id: str | None = None
        self.db_path: str | None = None
        self.current_run_data: dict | None = None  # Store for 'p' key viewing

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        with TabbedContent():
            with TabPane('Basic Info', id='info-pane'):
                with VerticalScroll(id='info-scroll'):
                    yield Static('Select a run to view details', id='info-content')
            with TabPane('Logs', id='logs-pane'):
                with VerticalScroll(id='logs-scroll'):
                    yield Static('Select a run to view logs', id='logs-content')
            with TabPane('Trace Events', id='trace-pane'):
                trace_tree = Tree('Trace Events', id='trace-content')
                trace_tree.show_root = False
                yield trace_tree

    def show_run(self, db_path: str, run_id: str, run_data: dict) -> None:
        """Display details for a run."""
        self.current_run_id = run_id
        self.db_path = db_path
        self.current_run_data = run_data  # Store for full viewing

        # Update basic info
        info_text = self._render_basic_info(run_data)
        info_content = self.query_one('#info-content', Static)
        info_content.update(info_text)

        # Load logs immediately
        logs_content = self.query_one('#logs-content', Static)
        logs_content.update('[dim]Loading logs...[/dim]')
        asyncio.create_task(self._load_logs())

        # Load trace events immediately
        trace_tree = self.query_one('#trace-content', Tree)
        trace_tree.clear()
        trace_tree.root.label = 'Loading trace events...'
        asyncio.create_task(self._load_trace_events())

    def _truncate_string(self, s: str, max_len: int = 500) -> str:
        """Truncate a string showing head and tail."""
        if len(s) <= max_len:
            return s
        head_len = max_len // 2 - 10
        tail_len = max_len // 2 - 10
        return f'{s[:head_len]}\n... ({len(s) - max_len} chars truncated) ...\n{s[-tail_len:]}'

    def _truncate_json_values(self, obj: Any, max_str_len: int = 500) -> Any:
        """Recursively truncate long string values in JSON objects."""
        if isinstance(obj, dict):
            return {k: self._truncate_json_values(v, max_str_len) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._truncate_json_values(item, max_str_len) for item in obj]
        if isinstance(obj, str):
            return self._truncate_string(obj, max_str_len)
        return obj

    def _render_basic_info(self, run_data: dict) -> Text:
        """Render basic run information with smart truncation."""
        text = Text()

        # Status and times
        text.append(f'Run ID: {run_data["run_id"]}\n', style='bold cyan')
        text.append(f'Status: {run_data["status"]}\n', style='bold')
        text.append(f'Start Time: {run_data.get("start_time", "N/A")}\n', style='dim')
        text.append(f'End Time: {run_data.get("end_time", "N/A")}\n', style='dim')

        if run_data.get('error'):
            text.append('\nError:\n', style='bold red')
            error_msg = self._truncate_string(run_data['error'], 4000)
            text.append(f'{error_msg}\n', style='red')

        # Helper to render JSON with smart truncation
        def append_json_section(title: str, data: dict, style: str) -> None:
            text.append(f'\n--- {title} ---\n', style=f'bold {style}')
            # Truncate long string values in the JSON
            truncated_data = self._truncate_json_values(data)
            json_str = json.dumps(truncated_data, indent=2, ensure_ascii=False)
            text.append(json_str + '\n', style='dim')

        # Task Output
        if run_data.get('task_output'):
            append_json_section('Task Output', run_data['task_output'], 'yellow')

        # Agent Output
        if run_data.get('agent_output'):
            append_json_section('Agent Output', run_data['agent_output'], 'green')

        if run_data.get('judge_output'):
            append_json_section('Judge Output', run_data['judge_output'], 'cyan')

        # Processor Results
        if run_data.get('processor_results'):
            append_json_section('Processor Results', run_data['processor_results'], 'magenta')

        return text

    async def _load_logs(self) -> None:
        """Load and display logs from database (limited by character count, not lines)."""
        if not self.current_run_id or not self.db_path:
            return

        try:
            async with BatchDatabase(self.db_path) as db:
                log_chunks = await db.get_log_chunks(self.current_run_id)

            logs_content = self.query_one('#logs-content', Static)

            if not log_chunks:
                # Use Text() for plain text output
                logs_content.update(Text('No logs available', style='dim'))
                return

            # Concatenate all log chunks
            all_logs = []
            for chunk in sorted(log_chunks, key=lambda c: c['chunk_index']):
                all_logs.append(chunk['content'])

            full_log = ''.join(all_logs)
            total_size = len(full_log)

            # Use character limit instead of line limit to avoid splitlines() overhead
            # Limit to last 500KB of logs (adjust as needed)
            max_chars = 500_000  # 500KB
            warning_text = ''
            if total_size > max_chars:
                warning_text = f'Warning: Log is {total_size:,} chars, showing last {max_chars:,} chars only\n\n'
                full_log = full_log[-max_chars:]

            # Create Text object with no_wrap=True to prevent automatic wrapping
            final_text = Text(no_wrap=True)
            if warning_text:
                final_text.append(warning_text, style='yellow')
            final_text.append(full_log)

            # Apply custom log highlighting
            highlighter = LogHighlighter()
            highlighter.highlight(final_text)

            logs_content.update(final_text)

            # Scroll to bottom after loading logs
            logs_scroll = self.query_one('#logs-scroll', VerticalScroll)
            logs_scroll.scroll_end(animate=False)

        except Exception as e:
            logs_content = self.query_one('#logs-content', Static)
            error_text = Text()
            error_text.append(f'Error loading logs: {e}', style='red')
            logs_content.update(error_text)

    async def _load_trace_events(self) -> None:
        """Load and display trace events as a tree."""
        if not self.current_run_id or not self.db_path:
            return

        try:
            async with BatchDatabase(self.db_path) as db:
                events = await db.get_trace_events(self.current_run_id)

            trace_tree = self.query_one('#trace-content', Tree)

            # Handle None or empty events
            if events is None or not events:
                trace_tree.clear()
                trace_tree.root.label = 'No trace events available'
                return

            # Limit trace events to avoid performance issues
            total_events = len(events)
            if total_events > 1000:
                events = events[:1000]

            # Build span tree
            spans = self._aggregate_spans_from_events(events)
            if not spans:
                trace_tree.clear()
                trace_tree.root.label = 'No valid spans found'
                return

            # Build tree widget
            self._build_span_tree(trace_tree, spans)

            # Add warning if truncated
            if total_events > 1000:
                trace_tree.root.label = f'Trace Events (showing first 1000 of {total_events})'

        except Exception as e:
            import traceback

            trace_tree = self.query_one('#trace-content', Tree)
            trace_tree.clear()
            trace_tree.root.label = f'Error loading trace events: {e}\n{traceback.format_exc()}'

    def _aggregate_spans_from_events(self, events: list[dict]) -> dict[str, Span]:
        """Aggregate events into spans using replay_span_events."""

        # Simple span collector processor
        class SpanCollector(SpanProcessor):
            def __init__(self):
                self.spans: dict[str, Span] = {}

            def on_start(self, span: Span, event) -> None:  # noqa: ARG002
                self.spans[span.span_id] = span

            def on_update(self, span: Span, event) -> None:  # noqa: ARG002
                if span.span_id in self.spans:
                    self.spans[span.span_id] = span

            def on_end(self, span: Span, event) -> None:  # noqa: ARG002
                if span.span_id in self.spans:
                    self.spans[span.span_id] = span

        # Create collector and replay events
        collector = SpanCollector()
        try:
            replay_span_events(events, processors=[collector])
        except Exception:
            # If replay fails, return empty dict
            return {}

        return collector.spans

    def _build_span_tree(self, tree_widget: Tree, spans: dict[str, Span]) -> None:
        """Build a Tree widget from span objects."""
        tree_widget.clear()
        tree_widget.root.label = 'Trace Events'

        # Build parent-child relationships
        root_spans = [s for s in spans.values() if not s.parent_span_id or s.parent_span_id == '']
        span_children: dict[str, list[Span]] = {}
        for span in spans.values():
            parent_id = span.parent_span_id
            if parent_id and parent_id != '':
                if parent_id not in span_children:
                    span_children[parent_id] = []
                span_children[parent_id].append(span)

        # Recursively add spans to tree
        def add_span_node(parent_node, span: Span) -> None:
            # Calculate duration
            duration_ms = 0
            if span.start_time_unix_nano and span.end_time_unix_nano:
                duration_ns = span.end_time_unix_nano - span.start_time_unix_nano
                duration_ms = duration_ns / 1_000_000

            # Format label with status color
            status_code = span.status.code if span.status else 'unknown'
            status_style = 'green' if status_code == 'ok' else 'red' if status_code == 'error' else 'yellow'

            label = Text()
            label.append(f'{span.name}', style='cyan bold')
            label.append(f' ({duration_ms:.2f}ms)', style='dim')
            label.append(f' [{status_code}]', style=status_style)

            # Add node to tree (collapsed by default to avoid clutter)
            node = parent_node.add(label, expand=False)

            # Add details as sub-nodes
            # Attributes
            if span.attributes:
                attrs_node = node.add('[bold yellow]Attributes[/bold yellow]')
                for key, value in sorted(span.attributes.items()):
                    # Truncate long values
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + '...'
                    attrs_node.add_leaf(f'[dim]{key}:[/dim] {value_str}')

            # Events
            if span.events:
                events_node = node.add(f'[bold magenta]Events ({len(span.events)})[/bold magenta]')
                for event in span.events[:10]:  # Limit to first 10 events
                    event_name = getattr(event, 'name', 'Unknown')
                    events_node.add_leaf(f'[dim]{event_name}[/dim]')

            # Status details
            if span.status and span.status.message:
                status_node = node.add('[bold]Status Details[/bold]')
                status_node.add_leaf(f'[dim]Message:[/dim] {span.status.message}')

            # Add child spans
            children = span_children.get(span.span_id, [])
            for child in sorted(children, key=lambda s: s.start_time_unix_nano or 0):
                add_span_node(node, child)

        # Add root spans
        for span in sorted(root_spans, key=lambda s: s.start_time_unix_nano or 0):
            add_span_node(tree_widget.root, span)


class BatchInspectorApp(App):
    """Main batch inspector application."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #main-container {
        layout: vertical;
        height: 100%;
    }

    #browser-container {
        height: 1fr;
        layout: horizontal;
    }

    #run-list {
        width: 40%;
    }

    #run-detail {
        width: 60%;
    }

    #progress-bar {
        height: auto;
        max-height: 7;
    }

    ProgressBar {
        padding: 0 1;
        height: auto;
    }

    RunList {
        border: solid cyan;
    }

    RunDetail {
        border: solid magenta;
    }

    #info-pane {
        height: 1fr;
    }

    #info-scroll {
        height: 100%;
        width: 100%;
    }

    #info-content {
        height: auto;
        width: 100%;
    }

    #logs-pane {
        height: 1fr;
    }

    #logs-scroll {
        height: 100%;
        width: 100%;
        overflow-x: auto;
        overflow-y: auto;
    }

    #logs-content {
        height: auto;
        width: auto;
    }
    """

    BINDINGS = [
        ('q', 'quit', 'Quit'),
        ('r', 'refresh', 'Refresh'),
        ('c', 'clear_sort', 'Clear Sort'),
    ]

    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.runs_data: list[dict] = []
        self.db: BatchDatabase | None = None
        # Column name mapping for sorting
        self.column_map = {
            'Run ID': 'run_id',
            'Status': 'status',
            'Start Time': 'start_time',
            'Duration': 'duration_sec',
            'Score': 'score_val',
            'Error': 'error',
        }

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()

        with Container(id='main-container'):
            with Horizontal(id='browser-container'):
                yield RunList(id='run-list')
                yield RunDetail(id='run-detail')
            yield ProgressBar(id='progress-bar')

        yield Footer()

    async def on_mount(self) -> None:
        """Load initial data when app mounts."""
        self.title = 'Nexus Batch Inspector'
        self.sub_title = f'Database: {self.db_path}'

        # Initialize persistent database connection
        self.db = BatchDatabase(self.db_path)
        await self.db.connect()

        # Initial load of all data
        await self.refresh_all()

        # Set up periodic refresh for progress bar only (1 second)
        self.set_interval(1, self.refresh_progress)

    async def refresh_progress(self) -> None:
        """Refresh only the progress bar (fast, every 1s)."""
        if not self.db:
            return

        try:
            # Get all statistics including score breakdown from database
            summary_dict = await self.db.get_summary()

            # Update progress bar with stats from database
            summary = RunSummary(
                total=summary_dict['total'],
                pending=summary_dict['pending'],
                running=summary_dict['running'],
                success=summary_dict['success'],
                failed=summary_dict['failed'],
                timeout=summary_dict['timeout'],
                cancelled=summary_dict['cancelled'],
                score_correct=summary_dict['score_correct'],
                score_incorrect=summary_dict['score_incorrect'],
                score_unknown=summary_dict['score_unknown'],
            )

            progress_bar = self.query_one('#progress-bar', ProgressBar)
            progress_bar.summary = summary

        except Exception:
            # Silently ignore errors in progress refresh to avoid spam
            pass

    async def refresh_runs(self) -> None:
        """Refresh the runs list (manual refresh only)."""
        if not self.db:
            return

        try:
            # Reload runs data from database
            self.runs_data = await self.db.get_all_runs()

            # Update run list
            run_list = self.query_one('#run-list', RunList)

            # Save cursor position before clearing
            cursor_row = run_list.cursor_row if run_list.cursor_row is not None else 0

            run_list.clear()

            # Prepare rows with sortable data
            rows_data = []
            for run in self.runs_data:
                # Calculate duration
                duration = 'N/A'
                duration_sec = 0.0
                if run.get('start_time') and run.get('end_time'):
                    try:
                        start = datetime.fromisoformat(run['start_time'])
                        end = datetime.fromisoformat(run['end_time'])
                        duration_sec = (end - start).total_seconds()
                        duration = f'{duration_sec:.1f}s'
                    except Exception:
                        pass

                # Get score
                score = 'N/A'
                score_val = -1  # For sorting: -1 = N/A, 0 = incorrect, 1 = correct
                if run['status'] == 'success' and run.get('task_output'):
                    task_output = run['task_output']
                    score_val_raw = task_output.get('score')
                    if score_val_raw is not None:
                        score_val = score_val_raw
                        score = '✓' if score_val == 1 else '✗' if score_val == 0 else str(score_val)

                # Get error (truncated)
                error = ''
                if run.get('error'):
                    error = run['error'][:50] + '...' if len(run['error']) > 50 else run['error']

                # Safe get start_time
                start_time_str = run.get('start_time') or 'N/A'
                if start_time_str != 'N/A':
                    start_time_str = start_time_str[:19]

                rows_data.append({
                    'run_id': run['run_id'],
                    'status': run['status'],
                    'start_time': start_time_str,
                    'duration': duration,
                    'duration_sec': duration_sec,
                    'score': score,
                    'score_val': score_val,
                    'error': error,
                })

            # Apply sorting if a column is selected
            if run_list.sort_column and run_list.sort_column in self.column_map:
                sort_key = self.column_map[run_list.sort_column]
                rows_data.sort(key=lambda x: x[sort_key], reverse=run_list.sort_reverse)

            # Add rows to table
            for row in rows_data:
                run_list.add_row(
                    row['run_id'],
                    row['status'],
                    row['start_time'],
                    row['duration'],
                    row['score'],
                    row['error'],
                )

            # Restore cursor position after updating the table
            if run_list.row_count > 0:
                # Ensure cursor_row is within valid range
                cursor_row = min(cursor_row, run_list.row_count - 1)
                run_list.move_cursor(row=cursor_row)

        except Exception as e:
            self.notify(f'Error refreshing runs: {e}', severity='error')

    async def refresh_all(self) -> None:
        """Refresh both progress bar and runs list (used on mount and manual refresh)."""
        await self.refresh_runs()
        await self.refresh_progress()
        await self._update_detail_for_current_row()

    async def _update_detail_for_current_row(self) -> None:
        """Update the right detail panel based on current cursor position."""
        run_list = self.query_one('#run-list', RunList)
        if run_list.row_count == 0:
            return

        # Get current cursor row
        cursor_row = run_list.cursor_row
        if cursor_row is None or cursor_row < 0 or cursor_row >= run_list.row_count:
            cursor_row = 0

        # Get row key at cursor position
        try:
            row_key = run_list.get_row_at(cursor_row)
            if row_key is None:
                return
            # Get run ID from first column
            run_id = str(row_key[0])

            # Find run data
            run_data = next((r for r in self.runs_data if r['run_id'] == run_id), None)
            if run_data:
                run_detail = self.query_one('#run-detail', RunDetail)
                run_detail.show_run(self.db_path, run_id, run_data)
        except Exception:
            # Silently ignore errors to avoid disrupting UX
            pass

    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        """Handle column header click to sort."""
        run_list = self.query_one('#run-list', RunList)
        column_label = event.label.plain

        # Toggle sort order if clicking the same column, otherwise reset to ascending
        if run_list.sort_column == column_label:
            run_list.sort_reverse = not run_list.sort_reverse
        else:
            run_list.sort_column = column_label
            run_list.sort_reverse = False

        # Update border title to show sort indicator
        sort_indicator = '↓' if run_list.sort_reverse else '↑'
        run_list.border_title = f'Runs (sorted by {column_label} {sort_indicator})'

        # Refresh to apply sort
        asyncio.create_task(self.refresh_runs())

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle run selection."""
        run_list = self.query_one('#run-list', RunList)
        row_key = event.row_key
        row_data = run_list.get_row(row_key)

        # Get run ID from first column
        run_id = str(row_data[0])

        # Find run data
        run_data = next((r for r in self.runs_data if r['run_id'] == run_id), None)
        if run_data:
            run_detail = self.query_one('#run-detail', RunDetail)
            run_detail.show_run(self.db_path, run_id, run_data)

    def action_refresh(self) -> None:
        """Manual refresh action."""
        asyncio.create_task(self.refresh_all())
        self.notify('Data refreshed')

    def action_clear_sort(self) -> None:
        """Clear sorting and restore default order."""
        run_list = self.query_one('#run-list', RunList)
        run_list.sort_column = None
        run_list.sort_reverse = False
        run_list.border_title = 'Runs'
        asyncio.create_task(self.refresh_runs())
        self.notify('Sort cleared')

    async def action_quit(self) -> None:
        """Quit the application."""
        # Clean up database connection
        if self.db:
            await self.db.close()
        self.exit()


if __name__ == '__main__':
    app = BatchInspectorApp('./workspace/swe/results.db')
    app.run()


__all__ = ['BatchInspectorApp']
