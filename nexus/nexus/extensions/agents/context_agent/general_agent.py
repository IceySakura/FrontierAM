"""Context Agent (general) — a path-native rollout ported from ``context_agent_harness``.

The rollout is handed a URI and nothing else. There is no preview, no content
injection and no pre-loaded document set: the agent decides whether a file is
worth fetching, how to convert it if it is binary, and which parts to read,
using :mod:`nexus.extensions.agents.context_agent.tools.general`'s seven path-native tools — ``curl``,
``bash``, ``python``, ``file_search_msearch``, ``file_search_mclick``,
``grep``, ``read_lines``.

Two consequences are worth knowing when reading a trajectory here:

* Early rounds go on acquisition (``curl``) and conversion (``python`` + pypdf)
  rather than retrieval. A task can fail because the extraction code was wrong,
  not because the model could not find the answer.
* Nothing is enforced mechanically. "Check the metadata before reading" is
  system-prompt discipline; the tool layer will happily let the model read a
  file it never inspected.

No frozen tool list is sent: schemas are generated from
:mod:`nexus.extensions.agents.context_agent.tools.general.schemas` at call time, so that module is the
contract. The prompt is data (``system_prompts/general_{zh,en}.md``) and can be
overridden with ``system_prompt_path``.

Host paths never reach the model: :mod:`.path_privacy` swaps sensitive
directory prefixes for ``/home/agent/dN`` in the system prompt, the manifest and
every tool result, and swaps them back on every tool call's arguments before
dispatch. That matters more here than in the harness this was ported from —
this package lives under the cluster path itself, so a ``python`` traceback
quoting ``repl_server.py`` would otherwise carry the org / project / user in
every request body. ``CTXAGENT_SCRUB_PATHS=0`` disables it.

Deliberate divergences from the harness rollout (``contextagent/methods/
context_agent_general.py``), all of them forced by Nexus rather than chosen:

* **Assistant messages are appended verbatim.** The harness rebuilds them with
  ``_assistant_tool_call_dict`` (minting tool-call ids). Under HyTuner RL
  ``ctx.openai_client`` is the policy being trained and enforces an APPEND-ONLY
  history, and the HY chat template renders one ``<tool_response>`` per
  assistant tool_call *by position* — so we append
  ``message.model_dump(exclude_none=True)`` and answer EVERY tool_call in
  order.
* **``max_parallel_calls`` defaults to uncapped** (as in the harness). When a
  cap is configured, calls past the cap receive synthetic error observations
  so the append-only history remains structurally complete.
* **``max_tool_rounds`` defaults to 20** for HyTuner RL safety. The legacy
  ``max_iterations`` field remains supported; benchmarks that need a larger
  budget (for example EnterpriseBench) set ``max_tool_rounds`` explicitly.
* **The task owns the working directory.** Like SWE, the task passes an explicit
  ``working_dir`` into the agent. That path is inside the configured Nexus
  runtime; all downloaded, converted, and generated files remain there.
* ``read_image`` is absent in the harness too — there is no second model handle
  for a native vision call. The frozen v1 prompt treats images as unreadable;
  experimental prompt versions may instead direct the policy to use installed
  local rendering, OCR, and pixel-analysis packages through bash/Python.

Language: ``CONTEXTAGENT_GENERAL_SYSTEM_LANG`` (zh|en|auto) pins the
system-prompt AND file-manifest language for all samples. ``auto`` resolves per
sample from the ``language`` field / question text. Tool schemas are
English-only regardless.
"""

import copy
import logging
import os
import re

from typing import Any

from pydantic import Field

from nexus.core import (
    AgentRunInput,
    WithUserPrompt,
    WithWorkingDir,
    agent_registry,
    require_context,
)
from nexus.core.tool.types import ToolSearchConfig
from nexus.extensions.agents.base.schema import TerminationReason
from nexus.extensions.agents.context_agent.tools.general import (
    GENERAL_TOOL_ORDER,
    Knobs,
    Workspace,
    parse_declared_files,
)
from nexus.extensions.agents.context_agent.tools.general.executor import ContextGeneralToolExecutor
from nexus.extensions.agents.context_agent.tools.general.staging import (
    cleanup_runtime_session,
    knobs_payload,
    stage_local_files,
)

from . import general_path_privacy as path_privacy
from .general_lang import resolve_lang
from .general_prompts import build_file_manifest, build_general_system_prompt
from .loop import (
    dump_assistant_message as _dump_assistant_message,
)
from .loop import (
    run_tool_call_loop,
)
from .loop import (
    safe_parse_json as _safe_parse_json,
)
from .loop import (
    tool_call_arguments as _tool_call_arguments,
)
from .loop import (
    tool_call_name as _tool_call_name,
)
from .models import ContextAgentRunOutput
from .profiles import GENERAL_PROFILE


logger = logging.getLogger(__name__)
PROFILE = GENERAL_PROFILE

if tuple(GENERAL_TOOL_ORDER) != PROFILE.tool_names:
    raise RuntimeError('context_agent_general tool schema order does not match its serving profile')

# Same joiner the sibling chains use, so a task-side system prompt sits where
# the model saw it during training.
TASK_SP_MARK = '\n\n# 任务特定设定（来自原始对话）\n\n'
_ATTACHMENT_RE = re.compile(r'\[Attachment:\s*`(?P<name>[^`]+)`\s+at\s+`(?P<path>[^`]+)`\]')

_FORCE_ANSWER_EN = (
    'You have used all available tool rounds. Based on the information '
    'gathered from your tool calls above, provide your best answer now. '
    'Do not request any more tools.'
)
_FORCE_ANSWER_ZH = (
    '你已用完所有可用的工具轮次。请基于以上工具调用所收集到的信息，现在直接给出你最好的答案，不要再请求任何工具。'
)


# ---------------------------------------------------------------------------
# Env knobs (launch scripts expose these; baked defaults make zero-config work)
# ---------------------------------------------------------------------------


def _env_str(name: str) -> str | None:
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _env_int(name: str) -> int | None:
    val = _env_str(name)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        logger.warning('context_agent_general: ignoring non-int env %s=%r', name, val)
        return None


def _resolve_lang_mode(explicit_field: str | None, sample_language: str | None, question: str) -> str:
    """Decide the prompt/manifest language ("zh" / "en").

    Precedence: ``CONTEXTAGENT_GENERAL_SYSTEM_LANG`` env > agent_config field >
    per-sample resolution. ``auto`` forces per-sample resolution.
    """
    mode = _env_str('CONTEXTAGENT_GENERAL_SYSTEM_LANG') or explicit_field or 'auto'
    mode = mode.strip().lower()
    if mode == 'auto':
        return resolve_lang(sample_language, question)
    resolved = resolve_lang(mode, question)
    if mode not in ('zh', 'cn', 'zh-cn', 'zh_cn', 'chinese', '中文', 'en', 'eng', 'english'):
        logger.warning(
            'context_agent_general: unknown system language %r; resolved to %r',
            mode,
            resolved,
        )
    return resolved


def _resolve_enabled_tools(requested: list[str] | None) -> list[str]:
    """Intersect a requested whitelist with the real tool set, in canonical order.

    Unset / empty / no-overlap all fall back to every tool. Unlike the
    ``context_agent`` chain there is no legacy-name rescue here: this chain is
    new, so a whitelist that misses is a config error rather than a stale
    dataset, and it is logged as one.
    """
    if not requested:
        return list(GENERAL_TOOL_ORDER)
    requested_set = {str(name) for name in requested}
    kept = [name for name in GENERAL_TOOL_ORDER if name in requested_set]
    if not kept:
        logger.warning(
            'context_agent_general: enabled_tools=%s has no overlap with %s; using all tools',
            requested,
            list(GENERAL_TOOL_ORDER),
        )
        return list(GENERAL_TOOL_ORDER)
    unknown = sorted(requested_set - set(GENERAL_TOOL_ORDER))
    if unknown:
        logger.warning('context_agent_general: ignoring unknown tool name(s) %s', unknown)
    return kept


def _task_system_text(system_messages: Any) -> str:
    """Join the task's own system persona, when the row carries one."""
    if isinstance(system_messages, str):
        return system_messages.strip()
    if not isinstance(system_messages, list):
        return ''
    return '\n\n'.join(
        str(msg.get('content') or '').strip()
        for msg in system_messages
        if isinstance(msg, dict) and msg.get('role') == 'system' and str(msg.get('content') or '').strip()
    )


def _normalize_dialogue(dialogue: Any, declared_files: list[Any]) -> list[dict[str, str]]:
    """Normalize a seeded user/assistant history and rewrite attachment paths.

    CLB-family rows carry the document-introduction point in user turns as
    ``[Attachment: `name` at `old/path`]``. Local and TaskFiles staging changes
    those paths, so rewrite each marker by declared filename while preserving
    every other byte of the historical turn. The complete file manifest is
    separately prepended to the first user turn, matching the original
    ``context_agent_general`` harness.
    """
    if not isinstance(dialogue, list):
        return []

    uri_by_name: dict[str, str] = {}
    for declared in declared_files:
        name = str(getattr(declared, 'name', '') or '').strip()
        uri = str(getattr(declared, 'uri', '') or '').strip()
        if name and uri:
            uri_by_name.setdefault(name, uri)

    def rewrite_attachment(match: re.Match[str]) -> str:
        name = match.group('name')
        uri = uri_by_name.get(name)
        if uri is None:
            logger.warning(
                'context_agent_general: dialogue attachment %r has no matching context_files entry; '
                'keeping its authored path',
                name,
            )
            return match.group(0)
        return f'[Attachment: `{name}` at `{uri}`]'

    normalized: list[dict[str, str]] = []
    for message in dialogue:
        if not isinstance(message, dict) or message.get('role') not in ('user', 'assistant'):
            continue
        content = str(message.get('content') or '').strip()
        if not content:
            continue
        normalized.append({
            'role': str(message['role']),
            'content': path_privacy.scrub(_ATTACHMENT_RE.sub(rewrite_attachment, content)),
        })
    return normalized


# ---------------------------------------------------------------------------
# Input / Output definitions
# ---------------------------------------------------------------------------


class ContextAgentGeneralInput(AgentRunInput, WithUserPrompt, WithWorkingDir):
    """Input for the path-native general context agent.

    Attributes:
        user_prompt: The question to answer.
        context_files: The task's file list — ``{"name", "uri"}`` items where
            ``uri`` is an http(s) URL, a local path, or a ``task-files://`` path
            into ``RunInput.task_files``. Bare strings and the ``url`` / ``path``
            spellings also parse. Nothing is fetched or read before the loop
            starts.
        context_dialogue: Optional seeded multi-turn user/assistant history.
            When present, its final user turn is the active question and
            ``user_prompt`` is not appended again. Attachment paths in the
            history are rewritten to the corresponding staged
            ``./workspace/<name>`` paths.
        system_messages: Optional task-side system persona, appended to (not
            replacing) the agent's own system prompt.
        language: Optional pre-tagged sample language; used unless the language
            mode pins one.
        files_base: Root that relative local ``uri`` values expand against
            (default: the Nexus repo root).
        working_dir: Workspace selected by the Task inside the configured
            Docker/Gongfeng/local Nexus runtime.
        max_iterations: Legacy round-budget field name. Used unless
            ``max_tool_rounds`` is set.
        max_tool_rounds: Tool-round budget (overrides ``max_iterations``).
        max_parallel_calls: Cap on tool calls executed per assistant turn.
            0 (the default) means uncapped, which is the only value compatible
            with the HyTuner append-only contract; a positive value is honoured
            but warned about.
        file_search_top_k / file_search_chunk_lines: BM25 file_search knobs.
        python_timeout: Per-call timeout for the python REPL tool (seconds).
        curl_timeout / curl_max_bytes: Download limits for the ``curl`` tool.
        system_prompt_lang: Optional per-config language pin (zh|en|auto).
        system_prompt_path: Optional prompt file. ``{lang}`` resolves to
            ``en`` or ``zh``. When unset, the original prompt is used.
        enabled_tools: Tool whitelist; intersected with the real tool set.
        mcp_servers: Optional application-tool servers loaded into this rollout.
        tool_search: Optional deferred-tool search configuration.
        A Nexus runtime provider is required. The Task passes the same explicit
            ``working_dir`` pattern used by SWE.
    """

    context_files: list[Any] = Field(default_factory=list)
    context_dialogue: list[dict[str, Any]] | None = None
    system_messages: list[dict] | str | None = None
    language: str | None = None
    files_base: str | None = None
    max_iterations: int = 20
    max_tool_rounds: int | None = None
    max_parallel_calls: int = 0
    file_search_top_k: int = 5
    file_search_chunk_lines: int = 50
    python_timeout: int = 120
    curl_timeout: int = 120
    curl_max_bytes: int = 200 * 1024 * 1024
    system_prompt_lang: str | None = None
    system_prompt_path: str | None = None
    enabled_tools: list[str] | None = None
    mcp_servers: list[dict[str, Any]] | None = None
    tool_search: ToolSearchConfig | None = None


class ContextAgentGeneralOutput(ContextAgentRunOutput):
    """Output from the general context agent.

    Attributes:
        main_trajectory: Full conversation (system + user + assistant + tool),
            in append-only order.
        final_answer: The extracted final answer text, or None on failure.
        language: The resolved rollout language ("zh" / "en").
        num_files: How many files the task declared.
        staged_files: Files copied into the rollout workspace.
        tool_schemas: Snapshot of tool definitions supplied to the client for normal
            rollout turns, before provider conversion. RL finalization retains the same schema.
        tool_rounds_used: Assistant turns that requested at least one tool.
        tool_calls_executed / tool_calls_skipped: Tool calls run, and skipped
            by ``max_parallel_calls``. Skipped calls still receive error
            observations to preserve the append-only conversation contract.
        hit_max_turn_limit: The round budget ran out before an answer.
        forced_finalization: A final-answer fallback turn was issued with the same tool schema for RL.
        response_budget_exhausted: The policy exceeded its input context or output token budget.
    """

    tool_name_resolution_error_count: int = 0
    language: str = 'en'
    num_files: int = 0
    staged_files: list[dict] = Field(default_factory=list)
    tool_schemas: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Main agent implementation
# ---------------------------------------------------------------------------


@agent_registry.register('context_agent_general')
async def run_context_agent_general(agent_input: ContextAgentGeneralInput) -> ContextAgentGeneralOutput:
    """Run a path-native general context-agent rollout.

    1. Resolve the prompt language and the knobs (env > agent_config > default).
    2. Parse ``context_files`` into a manifest and stage local files into the
       Task-provided workspace.
    3. Either open with the URI manifest + question, or preserve a supplied
       multi-turn dialogue after rewriting its attachment paths.
    4. Drive the policy with the seven path-native tools, appending every
       assistant message verbatim and every tool result in order.
    5. Return the final answer plus the full trajectory.
    """
    ctx = require_context()
    client = ctx.openai_client

    question = agent_input.user_prompt if isinstance(agent_input.user_prompt, str) else str(agent_input.user_prompt)

    # ---- Resolve knobs (env > field > baked default) ----
    lang = _resolve_lang_mode(agent_input.system_prompt_lang, agent_input.language, question)
    system_prompt_path = _env_str('CONTEXTAGENT_GENERAL_SYSTEM_PROMPT_PATH') or agent_input.system_prompt_path
    files_base = _env_str('CONTEXTAGENT_GENERAL_FILES_BASE') or agent_input.files_base

    max_tool_rounds = _env_int('CONTEXTAGENT_GENERAL_MAX_TOOL_ROUNDS')
    if max_tool_rounds is None:
        max_tool_rounds = int(
            agent_input.max_tool_rounds if agent_input.max_tool_rounds is not None else agent_input.max_iterations
        )
    max_tool_rounds = max(1, max_tool_rounds)

    cap = _env_int('CONTEXTAGENT_GENERAL_MAX_PARALLEL_CALLS')
    if cap is None:
        cap = int(agent_input.max_parallel_calls or 0)
    max_parallel_calls: int | None = cap if cap > 0 else None
    if max_parallel_calls is not None:
        # Calls past the cap are not run, but the shared loop still appends an
        # error response for each one so the assistant/tool history stays
        # positionally aligned under HyTuner.
        logger.warning(
            'context_agent_general: max_parallel_calls=%d skips execution past the cap; '
            'synthetic tool errors will preserve append-only history',
            max_parallel_calls,
        )

    enabled_tool_names = _resolve_enabled_tools(agent_input.enabled_tools)
    if ctx.runtime_provider is None:
        raise RuntimeError('context_agent_general requires a runtime provider')

    conversation: list[dict[str, Any]] = []
    final_answer: str | None = None
    staged: list[dict[str, str]] = []
    try:
        if agent_input.mcp_servers:
            await ctx.load_mcp_tools(extra_servers=agent_input.mcp_servers)
        allowed_mcp_tool_names = sorted(getattr(ctx, 'mcp_tools', {}))

        # Relative local uris expand here (see parse_declared_files) so every
        # path is unambiguous before anything touches it.
        declared_files = parse_declared_files(agent_input.context_files, files_base)

        knobs = Knobs(
            file_search_top_k=int(agent_input.file_search_top_k),
            file_search_chunk_lines=int(agent_input.file_search_chunk_lines),
            python_timeout=int(agent_input.python_timeout),
            curl_timeout=int(agent_input.curl_timeout),
            curl_max_bytes=int(agent_input.curl_max_bytes),
        )
        declared_files, staged, sandbox_root = await stage_local_files(
            ctx,
            declared_files,
            knobs_payload(
                file_search_top_k=knobs.file_search_top_k,
                file_search_chunk_lines=knobs.file_search_chunk_lines,
                python_timeout=knobs.python_timeout,
                curl_timeout=knobs.curl_timeout,
                curl_max_bytes=knobs.curl_max_bytes,
            ),
            working_dir=agent_input.working_dir,
        )
        # The system-prompt template concatenates ``{workspace}`` directly with
        # ``<name>`` and concrete filenames. Match ``Workspace.path_for('')`` and
        # keep the trailing slash, otherwise it renders broken examples such as
        # ``./workspace<name>`` and ``./workspacedoc1.pdf``.
        prompt_workspace = f'./{Workspace.DOCS_SUBDIR}/'
        prompt_cwd = sandbox_root
        if staged:
            logger.debug(
                'context_agent_general: staged %d file(s) into runtime %s',
                len(staged),
                sandbox_root,
            )

        tool_executor = ContextGeneralToolExecutor(
            enabled_tool_names=enabled_tool_names,
            allowed_mcp_tool_names=allowed_mcp_tool_names,
            tool_search_config=agent_input.tool_search,
        )
        tool_schemas = tool_executor.get_tool_schemas()
        # Keep the recorded definitions independent of any client-side mutation.
        tool_schemas_snapshot = copy.deepcopy(tool_schemas)
        tool_search_reminder = tool_executor.build_tool_search_reminder()
        system_prompt = path_privacy.scrub(
            build_general_system_prompt(
                language=lang,
                max_rounds=max_tool_rounds,
                workspace=prompt_workspace,
                cwd=prompt_cwd,
                system_prompt_path=system_prompt_path,
            )
        )
        task_system = _task_system_text(agent_input.system_messages)
        if task_system:
            # Scrubbed too: a row's persona is authored data and can quote a
            # path from wherever the dataset was generated.
            system_prompt = system_prompt + TASK_SP_MARK + path_privacy.scrub(task_system)

        dialogue = _normalize_dialogue(agent_input.context_dialogue, declared_files)
        # Match the original context_agent_general harness: a replay is valid
        # only when it starts and ends with a user turn. Invalid authored
        # history falls back to the standalone context_question instead of
        # failing the rollout.
        if dialogue and not (dialogue[0]['role'] == 'user' and dialogue[-1]['role'] == 'user'):
            logger.warning(
                'context_agent_general: ignoring dialogue: expected it to start and end with a user turn, got %s',
                [turn['role'] for turn in dialogue],
            )
            dialogue = []

        conversation.append({'role': 'system', 'content': system_prompt})
        if dialogue:
            # The original harness puts the complete file manifest in the
            # first user turn, where the benchmark originally introduced its
            # attachments, then replays the remaining turns unchanged.
            first_user_content = path_privacy.scrub(build_file_manifest(declared_files, dialogue[0]['content'], lang))
            if tool_search_reminder:
                first_user_content = f'{first_user_content}\n\n{tool_search_reminder}'
            conversation.append({'role': 'user', 'content': first_user_content})
            conversation.extend(dialogue[1:])
        else:
            # Staged local files advertise ./workspace/<name>; URLs remain
            # unchanged. No file content is injected.
            first_user_content = path_privacy.scrub(build_file_manifest(declared_files, question, lang))
            if tool_search_reminder:
                first_user_content = f'{first_user_content}\n\n{tool_search_reminder}'
            conversation.append({'role': 'user', 'content': first_user_content})
        logger.debug(
            'context_agent_general: lang=%s, system_prompt=%s, input_mode=%s, files=%d (%s), tools=%s, '
            'max_tool_rounds=%d, question="%s..."',
            lang,
            f'path:{system_prompt_path}' if system_prompt_path else 'default',
            'dialogue' if dialogue else 'manifest_question',
            len(declared_files),
            ', '.join(f.uri for f in declared_files) or 'none',
            enabled_tool_names,
            max_tool_rounds,
            question[:100],
        )

        async def execute_tool(tool_call: Any, round_num: int, call_idx: int) -> str:
            return await _execute_tool(
                tool_call=tool_call,
                fallback_id=f'call_{round_num}_{call_idx}',
                executor=tool_executor,
            )

        loop_result = await run_tool_call_loop(
            client=client,
            conversation=conversation,
            tool_schemas=tool_schemas,
            max_tool_rounds=max_tool_rounds,
            execute_tool=execute_tool,
            max_parallel_calls=max_parallel_calls,
            log_label='context_agent_general',
        )
        final_answer = loop_result.final_answer
        exit_status = loop_result.exit_status
        forced_finalization = False
        finalization_failed = False
        finalization_tool_calls = 0

        # ---- Budget exhausted → force a final plain-text answer ----
        if final_answer is None and exit_status == TerminationReason.MAX_ITERATIONS:
            logger.warning(
                'context_agent_general: exhausted %d rounds, forcing final answer',
                max_tool_rounds,
            )
            conversation.append({'role': 'user', 'content': _FORCE_ANSWER_ZH if lang == 'zh' else _FORCE_ANSWER_EN})
            forced_finalization = True
            try:
                # Keep the same schema to preserve HyTuner's append-only
                # rendering prefix. The cue asks for text; if the model still
                # calls a tool, retain that exact message and score it as a
                # failed finalization rather than mutating history.
                fallback = await client.create_completion(messages=conversation, tools=tool_schemas)
                if fallback.choices:
                    fb_message = fallback.choices[0].message
                    conversation.append(_dump_assistant_message(fb_message))
                    fallback_calls = list(fb_message.tool_calls or [])
                    finalization_tool_calls = len(fallback_calls)
                    if fallback_calls:
                        finalization_failed = True
                        logger.warning(
                            'context_agent_general: finalization requested %d tool call(s); scoring as 0',
                            finalization_tool_calls,
                        )
                    else:
                        final_answer = (fb_message.content or '').strip() or None
                        if not final_answer:
                            finalization_failed = True
                else:
                    finalization_failed = True
            except Exception as exc:
                logger.error('context_agent_general: fallback LLM call failed: %s', exc)
                finalization_failed = True

        return ContextAgentGeneralOutput(
            main_trajectory=conversation,
            final_answer=final_answer,
            language=lang,
            num_files=len(declared_files),
            staged_files=list(staged),
            tool_schemas=tool_schemas_snapshot,
            exit_status=exit_status,
            tool_rounds_used=loop_result.tool_rounds_used,
            tool_calls_requested=loop_result.tool_calls_requested + finalization_tool_calls,
            tool_calls_executed=loop_result.tool_calls_executed,
            tool_calls_skipped=loop_result.tool_calls_skipped + finalization_tool_calls,
            hit_max_turn_limit=loop_result.exit_status == TerminationReason.MAX_ITERATIONS,
            forced_finalization=forced_finalization,
            finalization_failed=finalization_failed,
            response_budget_exhausted=loop_result.response_budget_exhausted,
            tool_name_resolution_error_count=tool_executor.tool_name_resolution_error_count,
        )
    finally:
        await cleanup_runtime_session(ctx)


# ---------------------------------------------------------------------------
# Per-tool-call execution helper
# ---------------------------------------------------------------------------


async def _execute_tool(
    *,
    tool_call: Any,
    fallback_id: str,
    executor: ContextGeneralToolExecutor,
) -> str:
    """Execute a single tool call and return its string result."""
    name = _tool_call_name(tool_call)
    raw_arguments = _tool_call_arguments(tool_call)

    arguments = _safe_parse_json(raw_arguments)
    if arguments is None:
        return (
            '[Error] Failed to parse tool arguments as JSON. '
            f'Raw arguments: {str(raw_arguments)[:200]}\n'
            'Please ensure your tool call arguments are valid JSON with properly escaped strings.'
        )
    # The model only ever saw scrubbed paths, so a path it hands back is a fake
    # one — map it to the real filesystem before any tool tries to open it.
    arguments = path_privacy.restore_obj(arguments)

    try:
        return path_privacy.scrub(await executor.execute(name, arguments))
    except Exception as exc:
        # Exception text quotes paths too — a FileNotFoundError names the file,
        # and a REPL traceback names this package's own modules.
        logger.exception('context_agent_general: tool %s failed (call %s)', name, fallback_id)
        return path_privacy.scrub(f'[Error] Tool {name} raised {type(exc).__name__}: {exc}')
