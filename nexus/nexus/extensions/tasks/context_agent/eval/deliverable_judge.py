"""Agentic rubric judge for file-producing Context Agent tasks.

GDPval commonly asks for workbooks, presentations, PDFs, or documents. A
single text completion cannot verify those deliverables, so this evaluator
gives the Judge a private copy of the Runtime ``workspace/`` plus read-only
inspection tools (no network access). The Judge opens the actual files and
returns one evidence-backed verdict per rubric.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import uuid

from pathlib import PurePosixPath
from typing import Any

from nexus.core import ToolContext
from nexus.core.tool import tool_registry
from nexus.extensions.agents.context_agent import general_path_privacy as path_privacy
from nexus.extensions.agents.context_agent.loop import (
    append_tool_results,
    dump_assistant_message,
    safe_parse_json,
    tool_call_arguments,
    tool_call_name,
)
from nexus.extensions.agents.context_agent.tools.general import build_tool_schemas
from nexus.extensions.agents.context_agent.tools.general.executor import tool_response_to_text
from nexus.extensions.agents.context_agent.tools.general.runtime_tools import registry_id

from .judge import judge as text_judge


logger = logging.getLogger(__name__)

JUDGE_TOOLS: tuple[str, ...] = (
    'bash',
    'python',
    'file_search_msearch',
    'file_search_mclick',
    'grep',
    'read_lines',
)
MAX_JUDGE_ROUNDS = 40
MAX_PARSE_RETRIES = 3
MAX_INVENTORY_ENTRIES = 200
NOISE_DIRS = {'__pycache__', '.git', '.ipynb_checkpoints'}
HARNESS_ARTIFACTS = {
    'conversation.json',
    'history.json',
    'final_answer.md',
    'file_manifest.md',
    'result.json',
    'summary.json',
}

_SYSTEM_PROMPT = """\
You are grading an AI agent's work on a professional-domain task. You are not \
solving the task — do not attempt it, and do not reward an answer for being one \
you would have given.

# What you are looking at

Your working directory is `{cwd}`: the final state of the directory the agent \
worked in, copied so that nothing you do can alter the evidence. It holds \
whatever the agent downloaded, converted, or authored. The agent was told to \
write its files into `./workspace/`, but its shell ran one level up, so a \
deliverable may sit at either level.

**The agent's deliverables are the files it created or modified here.** The \
inventory below marks every entry matching a name from the task's own CONTEXT \
FILES: those were provided to the agent, so finding one proves nothing. \
Everything else it produced itself — which means genuine deliverables, but also \
ordinary working residue: text it extracted from a PDF, a scratch script, a \
file it downloaded only in order to read it. Residue is not a deliverable, and \
its presence violates nothing unless a criterion constrains what the directory \
may contain.

# Grading source of truth — read before assigning any verdict

- **Grade the deliverable the task actually asked for.**
  - File deliverable: open the file and inspect its actual contents.
  - Text deliverable: where the task asked for the answer itself as prose with \
no file expected, the agent's FINAL MESSAGE (given below) is the deliverable.
  - Many tasks ask for both. Grade each criterion against the deliverable it \
concerns.

- **The final message is evidence ONLY where it is the requested text \
deliverable.** Never accept "I have created…", "I updated…", "the workbook \
contains…" as evidence for what a file holds, that a file was produced, or that \
an edit was made. Such claims are not evidence and must never justify a pass.

- **Cite concrete evidence for every verdict**: a filename plus the specific \
content you found in it (a sheet name, a cell reference and its value, a slide \
number and its text), or the specific passage of the final message. If you \
cannot locate the required content in the actual deliverable, the criterion \
FAILS — an unverifiable claim is a fail, not a pass.

- **Verify produced OUTPUT, not mechanism.** A formula, function or script that \
*should* yield the required result is not the result: read the inputs it refers \
to and compute the value yourself, then confirm it matches. Never infer that a \
chart, table or document is correct from the code that generated it.

- **Distinguish provided input from authored deliverable.** A file that is \
unchanged from the input the agent was given has not been worked on; do not \
credit any required change to it.

- **A missing, empty, stub or placeholder file fails every criterion that \
depends on its intended contents.** Do not extend credit for a plausible \
filename.

- **Apply each criterion to exactly what it asks** — no more, no less. Do not \
fail a criterion over a flaw it does not mention, and do not pass it by \
accepting a weaker or partial result than it specifies. Judge each criterion \
independently: one missing file does not fail a criterion about something else.

- **Be precise on facts, lenient on wording.** Check that values, names, dates, \
numbers and entities are what the criterion requires; do not demand verbatim \
phrasing unless the criterion says so. Allow 1-2% tolerance on currency and \
measurements, and compare strings case-insensitively, unless told otherwise.

# Reading the deliverables

Office files are binary — `read_lines`, `grep` and `file_search` refuse them. \
Open them in `python`; `openpyxl`, `python-docx`, `python-pptx`, `pypdf` and \
`pdfplumber` are installed.

```python
from openpyxl import load_workbook
wb = load_workbook('workspace/model.xlsx')          # formulas, as written
print(wb.sheetnames)
ws = wb['Market Sizing']
print(ws.max_row, ws.max_column, ws['B7'].value)
```

**A spreadsheet written by a library has no cached values.** \
`load_workbook(..., data_only=True)` returns `None` for every formula cell \
unless the file has been opened and saved by Excel itself. `None` there means \
"never opened in Excel", NOT "the formula is broken" — never fail a criterion on \
that alone. To check that a formula computes what it should, read the cells it \
references and do the arithmetic yourself.

For a presentation, walk shapes rather than trusting a slide count alone:

```python
from pptx import Presentation
prs = Presentation('workspace/ic_deck.pptx')
for i, slide in enumerate(prs.slides, 1):
    text = ' | '.join(s.text_frame.text.replace(chr(10), ' ')
                      for s in slide.shapes if s.has_text_frame)
    print(i, text[:300])
```

Print statistics and targeted extracts, not whole documents — a full dump \
spends the context you need for the remaining criteria. Images cannot be read: \
there is no vision tool here, so judge an image only by what surrounds it, and \
say so in the evidence.

# Your verdicts
"""

_OUTPUT_CONTRACT = """
**Decide first which criteria concern a file at all.** Where every criterion \
reads on the content of the response itself, the task asked only for prose: the \
final message is the whole deliverable, there is nothing on disk to check, and \
you should grade it and answer without exploring. Where even one criterion names \
a file, open that file — for that criterion a claim about the file is never a \
substitute, however plainly the rest of the task is text.

Then work through the criteria in order, using as many tool calls as you need. \
When every criterion has been checked against the actual deliverable, reply with
ONE JSON object and nothing else — no prose before it, no code fence around it:

{
  "criteria": [
    {"id": "c001", "verdict": "pass", "evidence": "workspace/model.xlsx sheet 'rNPV' B14 = =B12*B13, B12=0.62 B13=410 -> 254.2, matches the stated peak revenue"},
    {"id": "c002", "verdict": "fail", "evidence": "no .pptx anywhere in the directory; the final message claims one was produced"}
  ],
  "summary": "one or two sentences on what was delivered and what was missing"
}

One entry per criterion id, in the order given. `verdict` is exactly "pass" or
"fail". `evidence` names where you looked and what you found there — a filename
plus the specific content — never a restatement of the criterion.
"""


def _prepare_snapshot(
    source_dir: str,
    destination_dir: str,
    max_files: int,
    max_bytes: int,
    harness_artifacts: list[str],
    noise_dirs: list[str],
) -> dict[str, Any]:
    """Runtime-side copy of regular files, with inventory and size limits."""
    import shutil

    source_dir = os.path.abspath(source_dir)
    destination_dir = os.path.abspath(destination_dir)
    shutil.rmtree(destination_dir, ignore_errors=True)
    os.makedirs(destination_dir, exist_ok=True)

    files: list[dict[str, Any]] = []
    total_bytes = 0
    if not os.path.isdir(source_dir):
        return {'files': files, 'file_count': 0, 'total_bytes': 0}

    excluded_files = set(harness_artifacts)
    excluded_dirs = set(noise_dirs) | {'tool_calls'}
    for root, dirs, names in os.walk(source_dir, followlinks=False):
        dirs[:] = sorted(
            name
            for name in dirs
            if name not in excluded_dirs
            and not name.startswith('.nexus_deliverable_judge_')
            and not os.path.islink(os.path.join(root, name))
        )
        for name in sorted(names):
            source = os.path.join(root, name)
            if os.path.islink(source) or not os.path.isfile(source):
                continue
            relative = os.path.relpath(source, source_dir).replace(os.sep, '/')
            is_top_level = '/' not in relative
            if relative == 'workspace/.keep':
                continue
            if is_top_level and (name in excluded_files or name.endswith('.log')):
                continue
            if name == '.nexus_workspace_artifact.tar.gz':
                continue
            size = os.path.getsize(source)
            if len(files) >= max_files:
                raise ValueError(f'workspace contains more than {max_files} regular files')
            if total_bytes + size > max_bytes:
                raise ValueError(f'workspace exceeds the {max_bytes}-byte judge limit')
            target = os.path.join(destination_dir, relative)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            shutil.copy2(source, target)
            files.append({'path': relative, 'size_bytes': size})
            total_bytes += size
    return {'files': files, 'file_count': len(files), 'total_bytes': total_bytes}


def _remove_snapshot(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _format_size(num_bytes: int) -> str:
    size = float(max(0, int(num_bytes)))
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size < 1024 or unit == 'GB':
            return f'{int(size)} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024
    return f'{size:.1f} GB'


def _context_names(task: dict, staged_files: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for item in task.get('context_files') or []:
        if isinstance(item, dict):
            candidates = [
                item.get('name'),
                item.get('uri'),
                item.get('url'),
                item.get('path'),
            ]
        else:
            candidates = [item]
        for raw in candidates:
            basename = str(raw or '').split('?', 1)[0].rstrip('/').rsplit('/', 1)[-1].strip()
            if basename:
                names.add(basename.lower())
    for item in staged_files:
        if isinstance(item, dict) and item.get('name'):
            names.add(str(item['name']).lower())
    return names


def _inventory(files: list[dict[str, Any]], context_names: set[str]) -> tuple[str, int]:
    if not files:
        return '(the workspace is empty — the agent left no files)', 0
    authored = sum(PurePosixPath(str(entry.get('path') or '')).name.lower() not in context_names for entry in files)
    lines: list[str] = []
    for entry in files[:MAX_INVENTORY_ENTRIES]:
        path = str(entry.get('path') or '')
        is_context = PurePosixPath(path).name.lower() in context_names
        lines.append(
            f'- {"[context]" if is_context else "[agent]  "} {path} ({_format_size(int(entry.get("size_bytes") or 0))})'
        )
    if len(files) > MAX_INVENTORY_ENTRIES:
        lines.append(f'- … and {len(files) - MAX_INVENTORY_ENTRIES} more')
    return '\n'.join(lines), authored


def _criteria_block(rubrics: list[str]) -> tuple[str, list[str]]:
    ids = [f'c{i:03d}' for i in range(1, len(rubrics) + 1)]
    text = '\n'.join(f'{cid}. {str(rubric).strip()}' for cid, rubric in zip(ids, rubrics, strict=True))
    return text, ids


def _user_prompt(
    *,
    question: str,
    context_names: set[str],
    inventory: str,
    model_output: str,
    criteria: str,
) -> str:
    context_block = '\n'.join(f'- {name}' for name in sorted(context_names)) or '(none — the task provided no files)'
    final_message = model_output.strip() or '(the agent produced no final message)'
    return (
        '<original_prompt>\n'
        'The task the agent was given, for context only — do not solve it.\n\n'
        f'{question}\n'
        '</original_prompt>\n\n'
        '<context_files>\n'
        'Names the task provided to the agent. A file on disk matching one of '
        'these is provided input, not a deliverable.\n\n'
        f'{context_block}\n'
        '</context_files>\n\n'
        '<workspace_inventory>\n'
        f'{inventory}\n'
        '</workspace_inventory>\n\n'
        '<final_message>\n'
        "The agent's closing message. It is the deliverable only where the task "
        'asked for prose; it is never evidence about a file.\n\n'
        f'{final_message}\n'
        '</final_message>\n\n'
        '<criteria>\n'
        f'{criteria}\n'
        '</criteria>\n'
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    fenced = re.search(r'```(?:json)?\s*(.+?)\s*```', candidate, re.S)
    if fenced:
        candidate = fenced.group(1).strip()
    start, end = candidate.find('{'), candidate.rfind('}')
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _collect_verdicts(parsed: dict[str, Any], ids: list[str]) -> list[dict[str, str]] | None:
    raw = parsed.get('criteria')
    if not isinstance(raw, list):
        return None
    by_id: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        criterion_id = str(item.get('id') or '').strip()
        if criterion_id not in ids and len(raw) == len(ids):
            criterion_id = ids[index]
        if criterion_id in ids:
            by_id[criterion_id] = item
    if len(by_id) != len(ids):
        return None
    verdicts: list[dict[str, str]] = []
    for criterion_id in ids:
        item = by_id[criterion_id]
        verdict = str(item.get('verdict') or '').strip().lower()
        if verdict not in {'pass', 'fail'}:
            return None
        verdicts.append({
            'id': criterion_id,
            'verdict': verdict,
            'evidence': str(item.get('evidence') or '').strip(),
        })
    return verdicts


class _RuntimeJudgeTools:
    def __init__(
        self,
        ctx: Any,
        *,
        session_dir: str,
        evidence_dir: str,
        judge_id: str,
    ) -> None:
        self.ctx = ctx
        self.judge_id = judge_id
        self.tool_state = {
            'working_dir': session_dir,
            'workspace_root': session_dir,
            'cwd': evidence_dir,
            'docs_dir': evidence_dir,
            'declared_files': [],
            'knobs': {
                'file_search_top_k': 5,
                'file_search_chunk_lines': 50,
                'python_timeout': 120,
                'curl_timeout': 1,
                'curl_max_bytes': 1,
            },
        }

    def _tool_context(self) -> ToolContext:
        return ToolContext(
            run_id=self.judge_id,
            tool_state=self.tool_state,
            trace_id=self.ctx.trace_id,
        )

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        entry = tool_registry.get_entry(registry_id(name))
        if entry.import_path is None:
            raise RuntimeError(f'deliverable judge tool has no import path: {name}')
        runtime = await self.ctx.get_runtime('main')
        response = await runtime.call_tool(entry.import_path, arguments, self._tool_context())
        return tool_response_to_text(response)

    async def close(self) -> None:
        entry = tool_registry.get_entry(registry_id('_cleanup'))
        if entry.import_path is None:
            return
        runtime = await self.ctx.get_runtime('main')
        await runtime.call_tool(entry.import_path, {}, self._tool_context())


async def _grade(
    llm: Any,
    *,
    system_prompt: str,
    user_prompt: str,
    ids: list[str],
    tools: _RuntimeJudgeTools,
) -> tuple[list[dict[str, str]] | None, str, int]:
    conversation: list[dict[str, Any]] = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]
    schemas = build_tool_schemas(list(JUDGE_TOOLS))
    retries_left = MAX_PARSE_RETRIES

    for round_num in range(MAX_JUDGE_ROUNDS):
        allowed = schemas if round_num < MAX_JUDGE_ROUNDS - 1 else None
        response = await llm.create_completion(messages=conversation, tools=allowed)
        if not response.choices:
            return None, 'judge returned no choices', round_num + 1
        message = response.choices[0].message
        tool_calls = list(message.tool_calls or [])
        content = (message.content or '').strip()

        if tool_calls and allowed:
            conversation.append(dump_assistant_message(message))
            outputs: list[str] = []
            for call in tool_calls:
                name = tool_call_name(call)
                if name not in JUDGE_TOOLS:
                    outputs.append(f'[Error] Tool {name!r} is not available to the judge.')
                    continue
                arguments = safe_parse_json(tool_call_arguments(call))
                if arguments is None:
                    outputs.append('[Error] Tool arguments were not valid JSON.')
                    continue
                try:
                    output = await tools.execute(name, path_privacy.restore_obj(arguments))
                except Exception as exc:  # noqa: BLE001
                    output = f'[Error] Tool {name} raised {type(exc).__name__}: {exc}'
                outputs.append(path_privacy.scrub(output))
            append_tool_results(
                conversation,
                tool_calls=tool_calls,
                outputs=outputs,
                round_num=round_num,
            )
            continue

        conversation.append(dump_assistant_message(message))
        parsed = _extract_json(content) if content else None
        verdicts = _collect_verdicts(parsed, ids) if parsed else None
        if verdicts is not None and parsed is not None:
            return verdicts, str(parsed.get('summary') or '').strip(), round_num + 1
        if retries_left <= 0:
            return None, content[:400], round_num + 1
        retries_left -= 1
        conversation.append({
            'role': 'user',
            'content': (
                'That is not the reply this grading run needs. Send ONE JSON '
                f'object and nothing else, holding a verdict for each of the '
                f'{len(ids)} criteria ({ids[0]}…{ids[-1]}), each with an '
                '"id", a "verdict" of exactly "pass" or "fail", and an '
                '"evidence" string naming the file and content you checked.'
            ),
        })

    return None, 'judge exhausted its rounds without returning verdicts', MAX_JUDGE_ROUNDS


def _result(
    *,
    score: float | int | None,
    rationale: str,
    verdicts: list[dict[str, str]] | None = None,
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        'score': score,
        'rationale': rationale,
        'requirement_status': ['yes' if verdict['verdict'] == 'pass' else 'no' for verdict in verdicts or []],
        'judge_prompt': None,
    }
    if verdicts is not None:
        result['criteria'] = verdicts
    result.update(extra)
    return result


def _rubric_accuracy(passed: int, total: int) -> float:
    """Return partial-credit rubric accuracy for a deliverable judgment."""
    return passed / total if total > 0 else 0.0


async def _fall_back_to_text(model_output: str, rubrics: list[str], llm: Any, why: str) -> dict[str, Any]:
    logger.warning('deliverable judge: %s — grading final message only', why)
    result = await text_judge(model_output, rubrics, llm)
    result['graded_on'] = 'final_message'
    result['rationale'] = f'[no deliverables available: {why}] ' + str(result.get('rationale') or '')
    return result


async def evaluate(
    model_output: str,
    task: dict,
    llm: Any,
    *,
    runtime_context: Any = None,
    working_dir: str | None = None,
    staged_files: list[dict[str, Any]] | None = None,
    max_files: int = 10_000,
    max_bytes: int = 1024 * 1024 * 1024,
) -> dict[str, Any]:
    """Grade GDPval rubrics against files in the rollout Runtime workspace."""
    rubrics = list(task.get('rubrics') or [])
    if not rubrics:
        return _result(score=None, rationale='the row carries no rubrics to grade')
    if runtime_context is None or not working_dir:
        return await _fall_back_to_text(model_output, rubrics, llm, 'no Runtime workspace was passed')

    runtime = await runtime_context.get_runtime('main')
    source_dir = working_dir.rstrip('/')
    judge_root = f'/tmp/context_agent_deliverable_judge/{uuid.uuid4().hex}'
    judge_dir = f'{judge_root}/judge'
    evidence_dir = f'{judge_root}/deliverables'
    try:
        snapshot = await runtime.invoke(
            _prepare_snapshot,
            source_dir,
            evidence_dir,
            int(max_files),
            int(max_bytes),
            sorted(HARNESS_ARTIFACTS),
            sorted(NOISE_DIRS),
        )
    except Exception as exc:  # noqa: BLE001
        return _result(score=None, rationale=f'could not snapshot Runtime workspace: {exc}')

    tools = _RuntimeJudgeTools(
        runtime_context,
        session_dir=judge_dir,
        evidence_dir=evidence_dir,
        judge_id=f'{runtime_context.run_id or "run"}-deliverable-judge',
    )
    try:
        files = list(snapshot.get('files') or []) if isinstance(snapshot, dict) else []
        context_names = _context_names(task, list(staged_files or []))
        inventory, authored = _inventory(files, context_names)
        criteria, ids = _criteria_block(rubrics)
        verdicts, summary, rounds = await _grade(
            llm,
            system_prompt=path_privacy.scrub(_SYSTEM_PROMPT.format(cwd=evidence_dir) + _OUTPUT_CONTRACT),
            user_prompt=path_privacy.scrub(
                _user_prompt(
                    question=str(task.get('context_question') or ''),
                    context_names=context_names,
                    inventory=inventory,
                    model_output=model_output,
                    criteria=criteria,
                )
            ),
            ids=ids,
            tools=tools,
        )
    finally:
        try:
            await tools.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug('deliverable judge cleanup ignored error: %s', exc)
        try:
            await runtime.invoke(_remove_snapshot, judge_root)
        except Exception as exc:  # noqa: BLE001
            logger.debug('deliverable judge snapshot cleanup ignored error: %s', exc)

    if verdicts is None:
        return _result(
            score=None,
            rationale=f'judge returned no usable verdicts after {rounds} rounds: {summary}',
            graded_on='deliverables',
        )

    passed = sum(verdict['verdict'] == 'pass' for verdict in verdicts)
    failed = [verdict for verdict in verdicts if verdict['verdict'] == 'fail']
    detail = '; '.join(f'{verdict["id"]}: {verdict["evidence"]}' for verdict in failed[:3])
    return _result(
        score=_rubric_accuracy(passed, len(verdicts)),
        rationale=(
            f'{passed}/{len(verdicts)} criteria met over {authored} agent-authored file(s). '
            f'{summary}' + (f' Failed — {detail}' if detail else '')
        ),
        verdicts=verdicts,
        graded_on='deliverables',
        criteria_passed=passed,
        criteria_total=len(verdicts),
        judge_rounds=rounds,
    )


__all__ = ['evaluate']
