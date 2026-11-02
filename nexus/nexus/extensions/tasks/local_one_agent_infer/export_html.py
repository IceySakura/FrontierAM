"""将 YB OneAgentBench 评测结果转换为自包含的 HTML 对话记录文件。

每条评测结果生成一个独立的 HTML 文件，包含完整的对话界面。
用法: python3 export_html.py [results_jsonl] [output_dir]
"""

import base64
import html
import json
import os
import re
import sys


def escape(s) -> str:
    if not isinstance(s, str):
        s = str(s)
    return html.escape(s)


def render_markdown_safe(text: str) -> str:
    """简单的 Markdown → HTML 转换（不依赖外部库）。"""

    # 代码块
    def replace_code_block(m):
        lang = m.group(1) or ''
        code = escape(m.group(2))
        return f'<pre><code class="language-{escape(lang)}">{code}</code></pre>'

    text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)

    # 行内代码
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)

    # 粗体
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)

    # 斜体
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # 标题
    text = re.sub(r'^### (.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)

    # 段落
    paragraphs = text.split('\n\n')
    rendered = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if p.startswith('<pre>') or p.startswith('<h'):
            rendered.append(p)
        else:
            rendered.append(f'<p>{p}</p>')

    return '\n'.join(rendered)


def render_tool_call(tc: dict) -> str:
    # Support both flat format {"name": ..., "arguments": ...}
    # and nested format {"function": {"name": ..., "arguments": ...}}
    func = tc.get('function', {})
    name = escape(func.get('name', '') or tc.get('name', ''))
    args_raw = func.get('arguments', '') or tc.get('arguments', '{}')
    try:
        args = json.loads(args_raw)
        # 对于 bash 命令，特殊高亮
        if name == 'bash' and 'command' in args:
            cmd = escape(args['command'])
            return f"""<div class="tool-card">
                <div class="tool-header">
                    <span class="tool-icon">⚡</span>
                    <span class="tool-name">{name}</span>
                </div>
                <div class="tool-body"><pre><code>{cmd}</code></pre></div>
            </div>"""
        formatted = escape(json.dumps(args, indent=2, ensure_ascii=False))
        return f"""<div class="tool-card">
                <div class="tool-header">
                    <span class="tool-icon">🔧</span>
                    <span class="tool-name">{name}</span>
                </div>
                <div class="tool-body"><pre><code>{formatted}</code></pre></div>
            </div>"""
    except json.JSONDecodeError:
        return f"""<div class="tool-card">
            <div class="tool-header">
                <span class="tool-icon">🔧</span>
                <span class="tool-name">{name}</span>
            </div>
            <div class="tool-body"><pre><code>{escape(args_raw)}</code></pre></div>
        </div>"""


def render_tool_output(content: str) -> str:
    truncated = content if len(content) <= 2000 else content[:2000] + '\n... (truncated)'
    return f"""<div class="tool-output">
        <div class="tool-output-header">📋 Tool Output</div>
        <pre><code>{escape(truncated)}</code></pre>
    </div>"""


def _mime_type(filename: str) -> str:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return {
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'pdf': 'application/pdf',
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'csv': 'text/csv',
        'txt': 'text/plain',
        'html': 'text/html',
    }.get(ext, 'application/octet-stream')


def _file_icon(filename: str) -> str:
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return {
        'docx': '📄',
        'doc': '📄',
        'xlsx': '📊',
        'xls': '📊',
        'csv': '📊',
        'pptx': '📽',
        'ppt': '📽',
        'pdf': '📕',
        'png': '🖼',
        'jpg': '🖼',
        'jpeg': '🖼',
        'html': '🌐',
    }.get(ext, '📎')


def _file_size_str(b64_content: str) -> str:
    size = len(base64.b64decode(b64_content))
    if size < 1024:
        return f'{size} B'
    if size < 1024 * 1024:
        return f'{size / 1024:.1f} KB'
    return f'{size / 1024 / 1024:.1f} MB'


def build_html(
    task_id: str,
    query: str,
    trajectory: list,
    metadata: dict,
    downloaded_files: dict[str, str] | None = None,
    skip_user_message: bool = False,
) -> str:
    messages_html = []

    # User message (only add if trajectory doesn't already contain it)
    if not skip_user_message and query:
        messages_html.append(f"""
    <div class="message user-message">
        <div class="message-header"><span class="role-badge user">👤 User</span></div>
        <div class="message-body">{render_markdown_safe(escape(query))}</div>
    </div>""")

    # Trajectory messages
    for msg in trajectory:
        role = msg.get('role', '')
        content = msg.get('content') or ''
        tool_calls = msg.get('tool_calls') or []

        if role == 'user':
            if content:
                messages_html.append(f"""
            <div class="message user-message">
                <div class="message-header"><span class="role-badge user">👤 User</span></div>
                <div class="message-body">{render_markdown_safe(escape(content))}</div>
            </div>""")

        elif role == 'assistant':
            parts = []
            if content:
                parts.append(f'<div class="message-body">{render_markdown_safe(escape(content))}</div>')
            for tc in tool_calls:
                parts.append(render_tool_call(tc))

            messages_html.append(f"""
            <div class="message assistant-message">
                <div class="message-header"><span class="role-badge assistant">🤖 Assistant</span></div>
                {''.join(parts)}
            </div>""")

        elif role == 'tool':
            messages_html.append(f"""
            <div class="message tool-message">
                {render_tool_output(content)}
            </div>""")

    conversation = '\n'.join(messages_html)

    l1 = escape(metadata.get('L1', ''))
    l2 = escape(metadata.get('L2', ''))
    difficulty = escape(metadata.get('difficulty', ''))
    expected = escape(metadata.get('expected_output', ''))

    # Build output files download section
    files_html = ''
    if downloaded_files:
        file_cards = []
        for fname, b64content in downloaded_files.items():
            mime = _mime_type(fname)
            icon = _file_icon(fname)
            size_str = _file_size_str(b64content)
            bare_name = fname.split('/')[-1]
            data_uri = f'data:{mime};base64,{b64content}'
            file_cards.append(f'''
            <a class="file-card" href="{data_uri}" download="{escape(bare_name)}">
                <span class="file-icon">{icon}</span>
                <div class="file-info">
                    <span class="file-name">{escape(bare_name)}</span>
                    <span class="file-size">{size_str}</span>
                </div>
                <span class="file-download-icon">⬇</span>
            </a>''')

        files_html = f"""
<div class="output-files-section">
    <div class="output-files-header">📦 Agent 产出文件</div>
    <div class="output-files-grid">
        {''.join(file_cards)}
    </div>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(task_id)} - One Agent Evaluation</title>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
:root {{
  --bg: #ffffff; --surface: #f7f8fa; --border: #e5e7eb;
  --accent: #5b6af0; --accent-light: #eef0fd;
  --text: #111827; --text-muted: #6b7280; --text-light: #9ca3af;
  --green: #16a34a; --green-bg: #f0fdf4;
  --user-bubble: #f0f1fd; --tool-bg: #f9fafb; --code-bg: #f3f4f6;
  --radius: 10px;
  --font-mono: 'Cascadia Code', 'JetBrains Mono', 'Fira Code', Consolas, monospace;
  --shadow-sm: 0 1px 3px rgba(0,0,0,.06);
}}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Inter', sans-serif;
  background: var(--bg); color: var(--text); font-size: 14px;
  -webkit-font-smoothing: antialiased; line-height: 1.6;
}}
header {{
  display: flex; align-items: center; justify-content: space-between;
  padding: 0 24px; height: 54px; border-bottom: 1px solid var(--border);
  background: var(--bg); position: sticky; top: 0; z-index: 100;
}}
.header-left {{ display: flex; align-items: center; gap: 10px; }}
.header-left h1 {{ font-size: 15px; font-weight: 600; }}
.logo {{
  width: 28px; height: 28px; background: var(--accent); border-radius: 7px;
  display: flex; align-items: center; justify-content: center; color: #fff; font-size: 14px;
}}
.meta-bar {{
  display: flex; gap: 16px; padding: 10px 24px;
  background: var(--surface); border-bottom: 1px solid var(--border);
  font-size: 12px; color: var(--text-muted); flex-wrap: wrap;
}}
.meta-item {{ display: flex; gap: 4px; }}
.meta-label {{ font-weight: 600; color: var(--text); }}
.chat-area {{
  max-width: 900px; margin: 0 auto; padding: 24px;
}}
.message {{ margin-bottom: 20px; }}
.message-header {{ margin-bottom: 6px; }}
.role-badge {{
  display: inline-flex; align-items: center; gap: 4px;
  padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: 600;
}}
.role-badge.user {{ background: var(--user-bubble); color: var(--accent); }}
.role-badge.assistant {{ background: var(--green-bg); color: var(--green); }}
.message-body {{
  padding: 12px 16px; border-radius: var(--radius);
  line-height: 1.7;
}}
.user-message .message-body {{
  background: var(--user-bubble); border: 1px solid #e0e3f8;
}}
.assistant-message .message-body {{
  background: var(--bg); border: 1px solid var(--border);
}}
.message-body p {{ margin: 8px 0; }}
.message-body p:first-child {{ margin-top: 0; }}
.message-body p:last-child {{ margin-bottom: 0; }}
.message-body h2, .message-body h3, .message-body h4 {{ margin: 12px 0 6px; }}
.message-body pre {{
  background: var(--code-bg); padding: 12px; border-radius: 8px;
  overflow-x: auto; font-size: 13px; margin: 8px 0;
  font-family: var(--font-mono);
}}
.message-body code {{
  font-family: var(--font-mono); font-size: 13px;
}}
.message-body pre code {{ background: none; padding: 0; }}
.message-body code:not(pre code) {{
  background: var(--code-bg); padding: 1px 5px; border-radius: 4px;
}}
.tool-card {{
  margin: 8px 0; border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden;
}}
.tool-header {{
  display: flex; align-items: center; gap: 6px;
  padding: 8px 12px; background: var(--tool-bg);
  font-size: 12px; font-weight: 600; color: var(--text-muted);
  border-bottom: 1px solid var(--border);
}}
.tool-icon {{ font-size: 14px; }}
.tool-name {{ color: var(--accent); }}
.tool-body {{
  padding: 0; max-height: 400px; overflow: auto;
}}
.tool-body pre {{
  margin: 0; padding: 10px 12px; background: var(--bg);
  font-size: 12px; border-radius: 0; white-space: pre-wrap; word-break: break-all;
}}
.tool-message {{ margin-bottom: 12px; }}
.tool-output {{
  border: 1px solid var(--border); border-radius: var(--radius);
  overflow: hidden;
}}
.tool-output-header {{
  padding: 6px 12px; background: var(--surface); font-size: 12px;
  font-weight: 600; color: var(--text-muted); border-bottom: 1px solid var(--border);
}}
.tool-output pre {{
  margin: 0; padding: 10px 12px; background: var(--bg);
  font-size: 12px; max-height: 300px; overflow: auto;
  white-space: pre-wrap; word-break: break-all;
  font-family: var(--font-mono);
}}
strong {{ font-weight: 600; }}
.output-files-section {{
  max-width: 900px; margin: 0 auto 40px; padding: 0 24px;
}}
.output-files-header {{
  font-size: 15px; font-weight: 600; color: var(--text);
  margin-bottom: 12px; padding-bottom: 8px;
  border-bottom: 2px solid var(--accent);
}}
.output-files-grid {{
  display: flex; flex-wrap: wrap; gap: 12px;
}}
.file-card {{
  display: flex; align-items: center; gap: 10px;
  padding: 12px 16px; border: 1px solid var(--border);
  border-radius: var(--radius); background: var(--surface);
  text-decoration: none; color: var(--text);
  transition: all 0.15s; cursor: pointer; min-width: 220px;
}}
.file-card:hover {{
  border-color: var(--accent); background: var(--accent-light);
  box-shadow: var(--shadow-sm);
}}
.file-icon {{ font-size: 24px; flex-shrink: 0; }}
.file-info {{ display: flex; flex-direction: column; flex: 1; min-width: 0; }}
.file-name {{
  font-size: 13px; font-weight: 600; color: var(--text);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.file-size {{ font-size: 11px; color: var(--text-muted); }}
.file-download-icon {{
  font-size: 16px; color: var(--accent); flex-shrink: 0;
}}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <div class="logo">🤖</div>
    <h1>One Agent · {escape(task_id)}</h1>
  </div>
</header>

<div class="meta-bar">
  <div class="meta-item"><span class="meta-label">Task ID:</span> {escape(task_id)}</div>
  <div class="meta-item"><span class="meta-label">L1:</span> {l1}</div>
  <div class="meta-item"><span class="meta-label">L2:</span> {l2}</div>
  <div class="meta-item"><span class="meta-label">Difficulty:</span> {difficulty}</div>
  <div class="meta-item"><span class="meta-label">Expected Output:</span> {expected}</div>
</div>

<div class="chat-area">
{conversation}
</div>

{files_html}

</body>
</html>"""


def _traj_is_complete(msgs: list) -> bool:
    """轨迹是否以最终回答收尾：末条为 assistant 且不带 tool_calls。"""
    if not msgs:
        return False
    last = msgs[-1]
    return isinstance(last, dict) and last.get('role') == 'assistant' and not last.get('tool_calls')


def _select_complete_trajectory(chat_trajs: list) -> list:
    """从 chat_tracker 的多条 root→leaf 轨迹里稳健地挑一条完整的。

    ``chat_tracker.dump_trajectories()`` 按 DFS 返回每条路径一条轨迹，其中可能包含
    断在中间工具调用处的**截断分支**。既不能盲取 ``[0]``（旧导出 bug，写出截断轨迹），
    也不能盲取 ``[-1]``（DFS 顺序不保证完整分支在末尾）。这里优先选**以最终回答结尾
    （末条 assistant 且无 tool_calls）的最长那条**；若无完整者，退回最长的一条。
    """
    lists = [t for t in chat_trajs if isinstance(t, list) and t]
    if not lists:
        return []
    complete = [t for t in lists if _traj_is_complete(t)]
    pool = complete or lists
    return max(pool, key=len)  # type: ignore[invalid-return-type]


def _system_msg_from_chat(chat_trajs: list) -> dict | None:
    """从 chat 轨迹里取带 system 的那条的首条 system 消息（落盘轨迹需在头部保留 system）。"""
    for t in chat_trajs:
        if isinstance(t, list) and t and isinstance(t[0], dict) and t[0].get('role') == 'system':
            return t[0]
    return None


def _turn_from_task_id(task_id: str) -> int:
    """从 task_id 末尾的 ``_tN`` 推断轮次；无后缀记为第 1 轮。"""
    m = re.search(r'_t(\d+)$', task_id or '')
    return int(m.group(1)) if m else 1


def regenerate_trajectories(results_path: str, bench_map: dict, traj_dir: str) -> int:
    """从 ``results.jsonl`` 权威重建 ``trajectory/*.jsonl``，修复截断不一致。

    轨迹体优先取 ``task_output.trajectory``（DB 里每轮完整的累积轨迹，与 results.db 一致），
    仅当其为空时才退回 ``_select_complete_trajectory`` 兜底；头部补 system（取自本记录的
    ``processor_results.chat`` 轨迹），并附 ``tools`` / ``metadata`` / ``source_metadata``，
    与运行时 ``_persist_turn_output`` 的落盘格式一致。task_id 已含 ``_tN``，天然按轮拆好。

    Returns: 写出的文件数。
    """
    os.makedirs(traj_dir, exist_ok=True)
    total = 0
    with open(results_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            to = rec.get('task_output', {}) or {}
            meta = dict(rec.get('metadata', {}) or {})
            task_id = to.get('task_id') or meta.get('task_id', 'unknown')

            pr = rec.get('processor_results', {}) or {}
            chat_pr = pr.get('chat', {}) if isinstance(pr, dict) else {}
            chat_trajs = chat_pr.get('trajectories', []) if isinstance(chat_pr, dict) else []
            tools = chat_pr.get('tools', []) if isinstance(chat_pr, dict) else []

            # 轨迹体：优先 agent_output.main_trajectory（完整、含 system，权威不截断），
            # 空则退回 task_output.trajectory，再退回完整叶子。system 在下方统一处理。
            body: list = []
            sys_from_main: dict | None = None
            ao = rec.get('agent_output', {}) or {}
            main_traj = ao.get('main_trajectory') if isinstance(ao, dict) else None
            if isinstance(main_traj, str):
                try:
                    main_traj = json.loads(main_traj)
                except json.JSONDecodeError:
                    main_traj = None
            if isinstance(main_traj, list) and main_traj:
                for m in main_traj:
                    if isinstance(m, dict) and m.get('role') == 'system' and sys_from_main is None:
                        sys_from_main = m
                    elif isinstance(m, dict):
                        body.append(m)
            if not body:
                traj_raw = to.get('trajectory', '')
                if traj_raw:
                    try:
                        body = json.loads(traj_raw) if isinstance(traj_raw, str) else list(traj_raw)
                    except json.JSONDecodeError:
                        body = []
            if not body:
                body = [m for m in _select_complete_trajectory(chat_trajs) if m.get('role') != 'system']

            # 头部补 system：优先 main_trajectory 自带的 system，否则取 chat 轨迹里的。
            full_traj = list(body)
            sys_msg = sys_from_main or _system_msg_from_chat(chat_trajs)
            if sys_msg and (not full_traj or full_traj[0].get('role') != 'system'):
                full_traj = [sys_msg] + full_traj

            bench_rec = bench_map.get(task_id, {})
            meta.setdefault('task_id', task_id)
            meta.setdefault('turn', _turn_from_task_id(task_id))
            traj_data: dict = {'trajectory': full_traj, 'tools': tools, 'metadata': meta}
            if bench_rec.get('source_metadata'):
                traj_data['source_metadata'] = bench_rec['source_metadata']

            out_path = os.path.join(traj_dir, f'{task_id}.jsonl')
            with open(out_path, 'w', encoding='utf-8') as out:
                out.write(json.dumps(traj_data, ensure_ascii=False) + '\n')
            total += 1
            print(f'  [TRAJ] {out_path} ({len(full_traj)} msgs)')
    return total


def main():
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
    results_path = (
        sys.argv[1] if len(sys.argv) > 1 else os.path.join(_project_root, 'outputs', 'one_agent_infer_results.jsonl')
    )
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_project_root, 'outputs', 'yb_bench_html')
    bench_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(
            _project_root,
            'yb-agent-openbench',
            'benchmarks',
            'releases',
            'YB_OneAgentBench_V0.1',
            'YB_OneAgentBench_V0.1_benchmark.jsonl',
        )
    )

    os.makedirs(output_dir, exist_ok=True)

    # Load benchmark data for query lookup
    bench_map: dict[str, dict] = {}
    if os.path.exists(bench_path):
        with open(bench_path, encoding='utf-8') as bf:
            for line in bf:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    bench_map[rec['task_id']] = rec

    with open(results_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            to = rec.get('task_output', {})
            meta = rec.get('metadata', {})
            task_id = to.get('task_id') or meta.get('task_id', 'unknown')

            # Parse trajectory — prefer agent_output.main_trajectory (the authoritative,
            # complete conversation captured by the agent; never truncated). Fall back to
            # processor_results.chat.trajectories, then task_output.trajectory.
            trajectory = []
            ao = rec.get('agent_output', {}) or {}
            main_traj = ao.get('main_trajectory') if isinstance(ao, dict) else None
            if isinstance(main_traj, str):
                try:
                    main_traj = json.loads(main_traj)
                except json.JSONDecodeError:
                    main_traj = None
            if isinstance(main_traj, list) and main_traj:
                trajectory = [m for m in main_traj if isinstance(m, dict) and m.get('role') != 'system']
            else:
                pr = rec.get('processor_results', {})
                chat_pr = pr.get('chat', {}) if isinstance(pr, dict) else {}
                chat_trajs = chat_pr.get('trajectories', []) if isinstance(chat_pr, dict) else []
                if chat_trajs and isinstance(chat_trajs[0], list) and len(chat_trajs[0]) > 0:
                    # Pick a COMPLETE trajectory (ends on a final answer), never a truncated
                    # chat_tracker branch — see _select_complete_trajectory.
                    trajectory = [m for m in _select_complete_trajectory(chat_trajs) if m.get('role') != 'system']
                else:
                    # Fallback to task_output.trajectory
                    traj_raw = to.get('trajectory', '')
                    try:
                        trajectory = json.loads(traj_raw) if traj_raw else []
                    except json.JSONDecodeError:
                        trajectory = []

            # Get query — for display in meta bar; if trajectory has user msg, don't duplicate
            bench_rec = bench_map.get(task_id, {})
            query = bench_rec.get('query', '')

            # Check if trajectory already contains the user query (from chat_tracker)
            has_user_in_traj = any(m.get('role') == 'user' for m in trajectory)

            metadata = {
                'L1': meta.get('L1', bench_rec.get('L1', '')),
                'L2': meta.get('L2', bench_rec.get('L2', '')),
                'difficulty': meta.get('difficulty', bench_rec.get('difficulty', '')),
                'expected_output': to.get('expected_output', bench_rec.get('expected_output', '')),
            }

            # Get downloaded output files
            downloaded_files = to.get('downloaded_files', {})

            html_content = build_html(
                task_id, query, trajectory, metadata, downloaded_files, skip_user_message=has_user_in_traj
            )
            out_path = os.path.join(output_dir, f'{task_id}.html')
            with open(out_path, 'w', encoding='utf-8') as out:
                out.write(html_content)

            n_msgs = len(trajectory)
            print(f'  {out_path} ({n_msgs} messages, {len(html_content):,} bytes)')

    print(f'\nDone. HTML files saved to {output_dir}/')

    # Regenerate trajectory/*.jsonl authoritatively (fixes truncated exports).
    # trajectory/ 与 results.jsonl 同级（run 输出目录布局）；可用 argv[4] 覆盖。
    traj_dir = (
        sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(results_path)), 'trajectory')
    )
    print(f'\n── Regenerating trajectory/ from {os.path.basename(results_path)} ──')
    n_traj = regenerate_trajectories(results_path, bench_map, traj_dir)
    print(f'\nDone. {n_traj} trajectory files written to {traj_dir}/')


if __name__ == '__main__':
    main()
