import base64
import logging
import posixpath
import shlex
import uuid

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.extensions.tasks.mm_agent.utils import TaskArtifact, upload_artifacts
from nexus.runtime.executors import FileItem
from nexus.trace import get_current_span_id, update_span
from nexus.utils.reward_score import ReasoningReward

from . import judge
from .codex_ppt12_mcp import build_ppt12_mcp_server
from .ppt_pdf_layout import judge_long_edge_px
from .ppt_tools import CODE_WORKDIR_STATE_KEY, IMAGE_STATE_KEY, PPT12_ALL_REGISTRY_NAMES, expand_agent_tools
from .scoring import ELEMENT_IOU_FACTORS, FIDELITY_METRIC_NAMES, LAYOUT_METRIC_NAMES, score_pptx


logger = logging.getLogger(__name__)


_HOST_TIMEOUT_MARGIN_SECONDS = 180.0


_BUILTIN_AGENT_NAMES = frozenset({'main_agent', 'main', 'mm_agent'})
_INSTALLED_HARNESS_NAMES = frozenset({'claude_code', 'codex'})


_RUNTIME_IMAGE_DIR = '/tmp'


class PPTTaskInput(TaskRunInput, WithAgentRunner):
    user_prompt: Any = ''

    system_prompt: str = ''
    """Optional system prompt passed through to the agent. The built-in loop
    receives it as the agent's ``system_prompt``; on the codex harness it is
    staged into the runtime and selected through codex's root config key
    ``model_instructions_file`` (the same channel ``codex exec -c
    model_instructions_file=`` uses), so it replaces the CLI's built-in
    instructions instead of being prepended to the user task. Passed through
    the agent input's dedicated ``model_instructions_file`` field so it cannot
    clobber a configured ``extra_config_toml``."""

    images: list[str] = Field(default_factory=list)

    artifacts: list[TaskArtifact] = Field(default_factory=list)

    codex_image_inject_mode: Literal['attach', 'path'] = 'attach'

    working_dir: str = '/workspace'

    deck_filename: str = 'slide.pptx'

    runtime_name: str = 'main'

    source_size: list | None = None

    rubric_config: dict[str, Any] = Field(default_factory=dict)

    ref_pptx_path: str = ''

    render_timeout: float = 300.0

    agent_tools: list[str] = Field(default_factory=list)
    """Tools handed to the agent. ``["ppt12"]`` is the one switch both arms read:
    the built-in loop expands it to the twelve sandbox-executing slide tools, and
    an installed codex harness mounts the same twelve over MCP; other names pass
    through to the loop only."""

    media_prompt_template: str | None = None

    gold_text_chars: int | None = None

    question: str = ''

    instance_id: str | None = None

    reward_func: str = 'ppt_topic'

    answer: str = ''

    reward_system_router_id: str = ''

    ref_png_path: str = ''

    judge_timeout: float = 600.0

    max_iterations: int | None = None
    """Per-row iteration cap for the built-in loop; unset keeps
    ``agent_config.max_iterations``. No effect on installed harnesses."""

    diagnose_failed_generator: bool = False

    @field_validator('rubric_config')
    @classmethod
    def _validate_rubric_config(cls, value: dict[str, Any]) -> dict[str, Any]:

        from .ppt_metrics import PPTRubricConfig  # noqa: PLC0415

        weights = (value or {}).get('weights')
        if weights:
            PPTRubricConfig.from_mapping(weights)
        return value


def _selected_agent_name() -> str:

    return (require_context().agent_name or '').strip()


def _is_installed_harness(agent_name: str) -> bool:

    return agent_name in _INSTALLED_HARNESS_NAMES


def _is_local_file(image_source: str) -> bool:
    return not image_source.startswith('data:image') and Path(image_source).is_file()


_DEFAULT_MEDIA_PROMPT_TEMPLATE = """{question}

The slide(s) to reproduce are provided as local files:
{image_paths}
Open them with your image-reading tool before you start.

Write the finished presentation to {deck_path} (one slide)."""

_TEXT_ONLY_PROMPT_TEMPLATE = """{question}

Write the finished presentation to {deck_path} (one slide)."""


def _resolve_media_template(override: str | None, fallback: str) -> str:

    if not override or not override.strip():
        return fallback
    try:
        override.format(question='', image_paths='', deck_path='')
    except (KeyError, IndexError, ValueError) as err:
        logger.warning('[ppt_rubric] invalid media_prompt_template (%s); using the default', err)
        return fallback
    return override


def build_installed_harness_prompt(
    question: str,
    image_paths: list[str],
    deck_path: str,
    *,
    media_template: str | None = None,
) -> str:

    fallback = _DEFAULT_MEDIA_PROMPT_TEMPLATE if image_paths else _TEXT_ONLY_PROMPT_TEMPLATE
    template = _resolve_media_template(media_template, fallback)
    return template.format(
        question=question.strip(),
        image_paths='\n'.join(f'  - {path}' for path in image_paths),
        deck_path=deck_path,
    )


class PPTTaskOutput(TaskRunOutput, WithScore):
    completed: bool = False
    objective: float | None = None
    fidelity: float | None = None
    subjective: float | None = None
    unrenderable: bool = False
    error: str | None = None
    termination_reason: str | None = None
    extra_fields: dict[str, Any] = Field(default_factory=dict)


def _metric_columns(result: dict[str, Any]) -> dict[str, Any]:

    metrics = result.get('metrics') or {}
    columns: dict[str, float] = {}
    for name in LAYOUT_METRIC_NAMES + FIDELITY_METRIC_NAMES:
        value = metrics.get(name)
        columns[f'ppt_{name}'] = judge.NOT_MEASURED if value is None else float(value)

    relative = result.get('relative_metrics') or {}
    for name in LAYOUT_METRIC_NAMES:
        value = relative.get(name)
        columns[f'ppt_rel_{name}'] = judge.NOT_MEASURED if value is None else float(value)

    objective = result.get('objective')
    objective_relative = result.get('objective_relative')
    fidelity = result.get('fidelity')
    columns['ppt_objective'] = judge.NOT_MEASURED if objective is None else float(objective)
    columns['ppt_objective_relative'] = judge.NOT_MEASURED if objective_relative is None else float(objective_relative)
    columns['ppt_fidelity'] = judge.NOT_MEASURED if fidelity is None else float(fidelity)
    columns['ppt_metrics_used'] = float(result.get('metrics_used') or 0)
    diag_for_relative = result.get('diagnostics') or {}
    gold_objective = diag_for_relative.get('gold_objective')
    columns['ppt_gold_objective'] = judge.NOT_MEASURED if gold_objective is None else float(gold_objective)
    columns['ppt_relative_metrics_used'] = float(diag_for_relative.get('relative_metrics_used') or 0)
    inkless = diag_for_relative.get('decorative_inkless')
    columns['ppt_decorative_inkless'] = judge.NOT_MEASURED if inkless is None else float(inkless)
    for key in ELEMENT_IOU_FACTORS:
        factor = diag_for_relative.get(f'element_{key}')
        columns[f'ppt_element_{key}'] = judge.NOT_MEASURED if factor is None else float(factor)
    columns['ppt_fidelity_used'] = float(result.get('fidelity_used') or 0)
    columns['ppt_render_ok'] = 0.0 if result.get('unrenderable') else 1.0
    columns['ppt_unrenderable'] = 1.0 if result.get('unrenderable') else 0.0

    columns['ppt_text_stripped'] = 1.0 if result.get('text_stripped') else 0.0

    columns.update(judge.text_fidelity_columns(result))

    diagnostics = result.get('diagnostics') or {}
    columns['ppt_pdf_measured'] = 1.0 if diagnostics.get('pdf_measured') else 0.0
    columns['ppt_n_text_measured'] = float(diagnostics.get('n_text_measured') or 0)
    baked = result.get('baked_text_ratio')
    columns['ppt_baked_text_ratio'] = judge.NOT_MEASURED if baked is None else float(baked)
    return columns


def _failed_output(
    reason: str, *, unrenderable: bool, evaluator_failed: bool = False, columns: dict[str, Any] | None = None
) -> PPTTaskOutput:
    empty: dict[str, Any] = {
        'metrics': {},
        'objective': None,
        'fidelity': None,
        'unrenderable': unrenderable,
    }
    return PPTTaskOutput(
        score=float(ReasoningReward.UNJUDGED.value) if evaluator_failed else 0.0,
        completed=False,
        unrenderable=unrenderable,
        error=reason[:500],
        termination_reason='REWARD_ERROR' if evaluator_failed else None,
        extra_fields={
            **judge.stable_columns(),
            **_metric_columns(empty),
            **(columns or {}),
            'ppt_scoring_failed': 1.0,
            'ppt_scoring_error': reason[:200],
            'ppt_infra_missing': float(evaluator_failed),
        },
    )


async def _upload_reference_deck(runtime: Any, ref_pptx_path: str) -> str:
    if not ref_pptx_path or not _is_local_file(ref_pptx_path):
        return ''

    sandbox_path = f'{_RUNTIME_IMAGE_DIR}/ppt_ref_{uuid.uuid4().hex[:8]}/{Path(ref_pptx_path).name}'
    try:
        response = await runtime.upload_file(
            [
                FileItem(
                    path=sandbox_path,
                    content=base64.b64encode(Path(ref_pptx_path).read_bytes()).decode('utf-8'),
                    encoding='base64',
                )
            ],
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to no reference, do not fail
        logger.warning('[ppt_rubric] could not upload the reference deck: %s', type(exc).__name__)
        return ''
    results = getattr(response, 'results', None) or []
    if not results or not all(getattr(r, 'success', False) for r in results):
        logger.warning(
            '[ppt_rubric] reference deck upload reported failure: %s',
            [getattr(r, 'message', '') for r in results],
        )
        return ''
    logger.info('[ppt_rubric] uploaded the reference deck to %s', sandbox_path)
    return sandbox_path


async def _stage_codex_instructions(runtime: Any, system_prompt: str) -> str:
    """Stage the row's system prompt in the runtime for ``model_instructions_file``.

    ``model_instructions_file`` is a codex root config key: codex reads the file
    and uses its content as the model's base instructions, replacing its
    built-in prompt -- the same channel ``codex exec -c model_instructions_file=``
    exposes on the CLI. Staging through ``artifacts`` instead would trip the
    prompt's mention check, so the task uploads the file itself, exactly like
    the reference deck.
    """
    if not system_prompt:
        return ''
    sandbox_path = f'{_RUNTIME_IMAGE_DIR}/ppt_sysp_{uuid.uuid4().hex}.txt'
    try:
        response = await runtime.upload_file(
            [
                FileItem(
                    path=sandbox_path,
                    content=base64.b64encode(system_prompt.encode('utf-8')).decode('utf-8'),
                    encoding='base64',
                )
            ],
            overwrite=True,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to built-in instructions, do not fail
        logger.warning('[ppt_rubric] could not stage the codex system prompt: %s', type(exc).__name__)
        return ''
    results = getattr(response, 'results', None) or []
    if not results or not all(getattr(r, 'success', False) for r in results):
        logger.warning(
            '[ppt_rubric] codex system prompt upload reported failure: %s',
            [getattr(r, 'message', '') for r in results],
        )
        return ''
    logger.info('[ppt_rubric] staged the codex system prompt at %s', sandbox_path)
    return sandbox_path


async def _find_generator(runtime: Any, working_dir: str, deck_path: str) -> str:
    newer_filter = (
        f'-not -newer {shlex.quote(deck_path)}' if deck_path and await _sandbox_has_file(runtime, deck_path) else ''
    )
    probe = (
        f'find {shlex.quote(working_dir)} -maxdepth 3 -type f '
        f'\\( -name "*.js" -o -name "*.mjs" -o -name "*.cjs" -o -name "*.ts" -o -name "*.py" \\) '
        f'-not -path "*/node_modules/*" {newer_filter} -print0 2>/dev/null '
        f'| xargs -0 -r grep -lE "pptxgenjs|PptxGenJS|from pptx import|Presentation\\\\(" '
        f'| xargs -r ls -t | head -1'
    )
    try:
        done = await runtime.run_command(command=probe, timeout=60)
    except Exception as exc:  # noqa: BLE001 - a probe must not fail the rollout
        logger.warning('[ppt_rubric] generator lookup failed: %s', type(exc).__name__)
        return ''
    found = (getattr(done, 'stdout', '') or '').strip().splitlines()
    return found[0].strip() if found else ''


async def _sandbox_has_file(runtime: Any, path: str) -> bool:

    try:
        done = await runtime.run_command(command=f'test -f {shlex.quote(path)}', timeout=30)
    except Exception:  # noqa: BLE001
        return False
    return getattr(done, 'exit_code', 1) == 0


async def _list_working_dir(runtime: Any, working_dir: str, limit: int = 40) -> str:
    probe = (
        f'ls -la {shlex.quote(working_dir)} 2>&1 | grep -v "^total" | '
        f"awk '{{print $NF, $5}}' | grep -vE '^\\.{{1,2}} ' | head -{int(limit)} | tr '\\n' '|'"
    )
    try:
        done = await runtime.run_command(command=probe, timeout=60)
    except Exception as exc:  # noqa: BLE001 - a probe must not fail the rollout
        return f'<listing failed: {type(exc).__name__}>'
    listing = (getattr(done, 'stdout', '') or '').strip()
    return listing or '<empty>'


async def _diagnose_failed_generator(runtime: Any, code_path: str) -> str:
    interpreter = 'node' if code_path.endswith(('.js', '.mjs', '.cjs')) else 'python3'
    name = posixpath.basename(code_path)
    if any(ch in name for ch in '\'"$` '):
        return f'<unsafe generator name {name!r}>'

    probe = (
        f'scratch=$(mktemp -d) && cp -r "$(dirname {shlex.quote(code_path)})"/. "$scratch"/ && '
        f'cd "$scratch" && {interpreter} "$scratch/{name}" 2>&1 | tail -3 | tr "\\n" " "'
    )
    try:
        done = await runtime.run_command(command=probe, timeout=120)
    except Exception as exc:  # noqa: BLE001 - a probe must not fail the rollout
        return f'<rerun failed: {type(exc).__name__}>'
    tail = (getattr(done, 'stdout', '') or '').strip()
    exit_code = getattr(done, 'exit_code', None)

    produced = ' produced_deck' if exit_code == 0 else ''
    return f'exit={exit_code}{produced} {tail[:300]}' if tail or exit_code is not None else '<no output>'


def _seed_slide_tools(
    ctx: Any,
    task_input: PPTTaskInput,
    artifacts: list[TaskArtifact],
    tools: list[str],
) -> None:
    """Publish the source slide (PIL image) and the code workdir into
    ``tool_state`` -- the contracts the 12 slide tools were distilled against.
    Seeded only when the row names a slide tool; never raises."""
    if not set(tools) & set(PPT12_ALL_REGISTRY_NAMES):
        return
    source = next((artifact.source for artifact in artifacts if artifact.is_image), '')
    # task_input.images is dead here by construction: both branches of the
    # caller already rejected rows that set images, so the loop's real image
    # channel is user_prompt content parts. No artifacts row therefore means
    # the 12 tools will all answer "no reference image" -- fail loud instead
    # of letting a mis-configured rollout score silently.
    if not source:
        raise ValueError(
            'ppt_topic: slide tools are requested (agent_tools includes "ppt12") but no '
            'image artifact was found; the built-in loop publishes the source slide from '
            'task_config.artifacts ({source, target}, is_image). Without it every tool '
            'would return "no reference image".'
        )
    try:
        from PIL import Image  # noqa: PLC0415 - the host needs pillow only for this

        with Image.open(source) as image:
            image.load()
            ctx.tool_state[IMAGE_STATE_KEY] = image
    except Exception as exc:  # noqa: BLE001 - degrade, do not fail the rollout
        logger.warning('[ppt_topic] could not publish the source slide to tool_state: %s', type(exc).__name__)
    ctx.tool_state[CODE_WORKDIR_STATE_KEY] = task_input.working_dir
    logger.info('[ppt_topic] seeded tool_state for %d slide tool(s)', len(tools))


@task_registry.register('ppt_topic')
async def run_ppt_topic(task_input: PPTTaskInput) -> PPTTaskOutput:

    if task_input.runtime_name != 'main':
        raise ValueError(
            f'ppt_topic: runtime_name must be "main" (got {task_input.runtime_name!r}); '
            'the agent and artifacts always run on main, scoring elsewhere would read a different sandbox.'
        )

    ctx = require_context()
    runtime = await ctx.get_runtime(task_input.runtime_name)

    deck_path = task_input.deck_filename
    if not posixpath.isabs(deck_path):
        deck_path = posixpath.join(task_input.working_dir, deck_path)

    agent_name = _selected_agent_name()
    harness = agent_name.lower()
    installed = _is_installed_harness(harness)
    logger.info(
        '[ppt_rubric] agent_name=%s (%s), deck=%s, instance_id=%s',
        agent_name or 'unknown',
        'installed harness' if installed else 'built-in loop',
        deck_path,
        task_input.instance_id or 'N/A',
    )

    agent_kwargs: dict[str, Any] = {'working_dir': task_input.working_dir}

    # Stage input files on both harness paths (the artifacts field's own
    # contract); the installed-only branch used to leave the built-in loop
    # with a prompt naming a sandbox path nothing had uploaded.
    artifacts = task_input.artifacts or []
    if artifacts:
        await upload_artifacts(artifacts)

    if installed:
        if task_input.system_prompt and harness != 'codex':
            raise ValueError(
                f'ppt_topic: system_prompt is set but the {agent_name!r} harness has no system-prompt '
                'input, so it would be silently dropped. Clear task_config.system_prompt, or run the '
                'codex harness, which stages it as codex model_instructions_file.'
            )
        if task_input.images:
            raise ValueError(
                'ppt_topic installed harnesses take task_config.artifacts ({source, target}), not '
                'images; inline image content cannot reach a CLI harness.'
            )

        question = task_input.user_prompt if isinstance(task_input.user_prompt, str) else ''
        question = question or task_input.question

        can_attach = task_input.codex_image_inject_mode == 'attach' and harness == 'codex'
        attached = [artifact.target for artifact in artifacts if artifact.is_image] if can_attach else []
        by_path = [artifact.target for artifact in artifacts if artifact.target not in attached]

        agent_kwargs['user_prompt'] = build_installed_harness_prompt(
            question,
            by_path,
            deck_path,
            media_template=task_input.media_prompt_template,
        )
        unreferenced = [t for t in by_path if t not in agent_kwargs['user_prompt']]
        if unreferenced:
            raise ValueError(f'ppt_topic artifacts are not mentioned in the prompt and cannot be found: {unreferenced}')
        if attached:
            agent_kwargs['images'] = attached
        if task_input.system_prompt and harness == 'codex':
            sysp_path = await _stage_codex_instructions(runtime, task_input.system_prompt)
            if sysp_path:
                # The dedicated input field, not an extra_config_toml prefix:
                # run_agent merges {**run_input.agent_config, **task_kwargs},
                # so a task-side extra_config_toml would silently replace
                # whatever the yaml configured there (osworld's
                # model_reasoning_summary, swe_cosmos's [features]); the
                # field is rendered as a root key that coexists with it.
                agent_kwargs['model_instructions_file'] = sysp_path
            else:
                logger.warning(
                    '[ppt_rubric] codex system_prompt set but staging failed; '
                    'codex will run with its built-in instructions'
                )
        # The codex counterpart of the loop's tool expansion: the same
        # agent_tools=["ppt12"] switch mounts the twelve tools over MCP.
        # build_ppt12_mcp_server raises for non-codex harnesses and when the
        # worker venv the MCP server needs is missing.
        if 'ppt12' in (task_input.agent_tools or []):
            agent_kwargs['mcp_servers'] = [
                build_ppt12_mcp_server(
                    agent_name=agent_name,
                    task_input=task_input,
                    artifacts=artifacts,
                    runtime=runtime,
                )
            ]
    else:
        if task_input.images:
            raise ValueError(
                'ppt_topic rejects images on every harness: inline image content cannot reach a CLI, '
                'and the built-in loop reads the source from user_prompt content parts.'
            )
        if harness and harness not in _BUILTIN_AGENT_NAMES:
            raise ValueError(
                f'ppt_topic: unsupported nexus.agent_name={agent_name!r}. Expected empty / '
                f'{sorted(_BUILTIN_AGENT_NAMES)} (built-in loop) or {sorted(_INSTALLED_HARNESS_NAMES)}.'
            )
        agent_kwargs['user_prompt'] = task_input.user_prompt
        if task_input.system_prompt:
            agent_kwargs['system_prompt'] = task_input.system_prompt
        tools = expand_agent_tools(task_input.agent_tools) if task_input.agent_tools else []
        if tools:
            agent_kwargs['tools'] = tools
        if task_input.max_iterations is not None:
            agent_kwargs['max_iterations'] = task_input.max_iterations
        _seed_slide_tools(ctx, task_input, artifacts, tools)

    await task_input.run_agent(**agent_kwargs)

    config = dict(task_input.rubric_config)
    config.setdefault('pdf_timeout', task_input.render_timeout)
    config.setdefault('raster_timeout', task_input.render_timeout)

    code_path = await _find_generator(runtime, task_input.working_dir, deck_path)

    listing = await _list_working_dir(runtime, task_input.working_dir)
    logger.info(
        '[ppt_rubric] working_dir=%s generator=%s contents=%s',
        task_input.working_dir,
        code_path or '<none>',
        listing,
    )

    sandbox_ref_pptx = await _upload_reference_deck(runtime, task_input.ref_pptx_path)
    if task_input.ref_pptx_path and not sandbox_ref_pptx:
        return _finalize(
            _failed_output('reference deck upload failed', unrenderable=False, evaluator_failed=True),
            task_input.reward_func,
        )

    try:
        result = await runtime.invoke(
            score_pptx,
            deck_path,
            task_input.source_size,
            config,
            sandbox_ref_pptx,
            want_judge_png=True,
            gold_text_chars=task_input.gold_text_chars,
            code_path=code_path,
            request_timeout=task_input.render_timeout + _HOST_TIMEOUT_MARGIN_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - a scoring outage must not kill the rollout
        logger.exception('[ppt_rubric] sandbox scoring invoke failed for %s', deck_path)
        output = _failed_output(
            f'scoring invoke failed: {type(exc).__name__}: {exc}', unrenderable=False, evaluator_failed=True
        )
        return _finalize(output, task_input.reward_func)

    if not isinstance(result, dict):
        return _finalize(
            _failed_output(
                f'scoring returned {type(result).__name__}, expected dict', unrenderable=False, evaluator_failed=True
            ),
            task_input.reward_func,
        )

    columns = _metric_columns(result)
    columns['ppt_generator_found'] = 1.0 if code_path else 0.0

    if not result.get('completed'):
        reason = str(result.get('error') or 'scoring did not complete')
        logger.warning('[ppt_rubric] scoring failed for %s: %s', deck_path, reason)

        if task_input.diagnose_failed_generator and code_path and reason.startswith('pptx not found'):
            logger.warning(
                '[ppt_rubric] generator produced no deck; rerun says: %s',
                await _diagnose_failed_generator(runtime, code_path),
            )
        output = _failed_output(
            reason,
            unrenderable=bool(result.get('unrenderable')),
            evaluator_failed=bool(result.get('evaluator_failed')) or not result.get('unrenderable'),
            columns=columns,
        )

        output.extra_fields['ppt_scoring_diagnostics'] = result.get('diagnostics') or {}
        return _finalize(output, task_input.reward_func)

    objective = result.get('objective')
    fidelity = result.get('fidelity')
    diagnostics = result.get('diagnostics') or {}
    if diagnostics.get('objective_mode_fallback'):
        logger.warning(
            '[ppt_rubric] instance %s: gold_relative objective unavailable (%s), ppt_objective falls back to the '
            'absolute calibre; a row without a reference deck should set rubric_config.weights.objective_mode '
            'to "absolute"',
            task_input.instance_id or 'N/A',
            diagnostics['objective_mode_fallback'],
        )
    columns.update(judge.stable_columns() | columns)
    columns['ppt_scoring_diagnostics'] = diagnostics
    if (result.get('text_detail') or {}).get('truncated'):
        return _finalize(
            _failed_output(
                'text comparison exceeded its limit', unrenderable=False, evaluator_failed=True, columns=columns
            ),
            task_input.reward_func,
        )

    paste_columns = judge.paste_gate_columns(diagnostics, task_input.answer or '')
    host_stripped = bool(paste_columns.pop('stripped', False))
    columns.update(paste_columns)
    if float(result.get('text_stripped') or 0.0) or host_stripped:
        reason = columns.get('ppt_text_stripped_reason') or diagnostics.get('paste_reason') or 'paste gate fired'
        logger.warning('[ppt_rubric] %s', reason)
        columns['ppt_text_stripped'] = 1.0
        columns['ppt_text_stripped_reason'] = str(reason)[:200]
        output = PPTTaskOutput(
            score=0.0,
            completed=True,
            objective=None if objective is None else float(objective),
            fidelity=None if fidelity is None else float(fidelity),
            unrenderable=False,
            extra_fields=columns,
        )
        return _finalize(output, task_input.reward_func)

    render_png = judge.decode_judge_png(result.get('judge_png_b64'))

    subjective, subj_detail = await judge.subjective_score(
        prompt_str=(task_input.user_prompt if isinstance(task_input.user_prompt, str) else '') or task_input.question,
        response='',
        router_id=task_input.reward_system_router_id,
        ref_texts=result.get('ref_texts') or [],
        ref_png_path=task_input.ref_png_path,
        render_png=render_png,
        judge_long_edge_px=judge_long_edge_px(task_input.source_size),
        generated_listing=result.get('cand_structure_listing') or '',
        reference_listing=result.get('ref_structure_listing') or '',
        require_source=True,
    )
    for key, value in subj_detail.items():
        columns[judge.detail_column(key)] = value
    columns['ppt_subjective'] = judge.NOT_MEASURED if subjective is None else float(subjective)
    columns['ppt_subjective_ok'] = 0.0 if subjective is None else 1.0
    if task_input.reward_system_router_id:
        required_dimensions = [subj_detail.get(f'subj_{name}') for name in ('accuracy', 'quality', 'aesthetics')]
        complete = (
            subjective is not None
            and subj_detail.get('judge_source_seen') == 1.0
            and all(isinstance(value, (int, float)) and 0.0 <= value <= 1.0 for value in required_dimensions)
        )
        if not complete:
            reason = (
                subj_detail.get('subjective_error') or 'judge requires source_seen=1 and all three valid dimensions'
            )
            return _finalize(
                _failed_output(str(reason), unrenderable=False, evaluator_failed=True, columns=columns),
                task_input.reward_func,
            )

    fused: dict[str, Any] = dict(columns)
    canvas = result.get('metrics', {}).get('canvas_fidelity')
    text_fidelity = result.get('text_fidelity')
    try:
        fused_score = judge.fuse(
            objective=None if objective is None else float(objective),
            fidelity=None if fidelity is None else float(fidelity),
            subjective=subjective,
            canvas=None if canvas is None else float(canvas),
            cfg=_fusion_config(config),
            out=fused,
            text=None if text_fidelity is None else float(text_fidelity),
            baked=result.get('baked_text_ratio'),
        )
    except (TypeError, ValueError) as exc:
        return _finalize(
            _failed_output(str(exc), unrenderable=False, evaluator_failed=True, columns=columns),
            task_input.reward_func,
        )

    output = PPTTaskOutput(
        score=fused_score,
        completed=True,
        objective=None if objective is None else float(objective),
        fidelity=None if fidelity is None else float(fidelity),
        subjective=subjective,
        unrenderable=False,
        extra_fields=fused,
    )
    return _finalize(output, task_input.reward_func)


def _fusion_config(config: dict[str, Any]) -> Any:

    from .ppt_metrics import PPTRubricConfig  # noqa: PLC0415

    return PPTRubricConfig.from_mapping((config or {}).get('weights') or {})


def _finalize(output: PPTTaskOutput, reward_func: str) -> PPTTaskOutput:

    judge.mirror_into_reward_extra_info(output.extra_fields)
    if output.termination_reason:
        span_id = get_current_span_id()
        if span_id:
            update_span(span_id, attributes={'hytuner.termination_reason': output.termination_reason})
    _publish_task_span(output, reward_func)
    return output


def _publish_task_span(output: PPTTaskOutput, reward_func: str) -> None:

    span_id = get_current_span_id()
    if not span_id:
        return
    update_span(
        span_id,
        attributes={
            'hytuner.task.reward_func': reward_func,
            'hytuner.task.extra.ppt_objective': (-1.0 if output.objective is None else float(output.objective)),
            'hytuner.task.extra.ppt_fidelity': (-1.0 if output.fidelity is None else float(output.fidelity)),
        },
    )


__all__ = [
    'PPTTaskInput',
    'PPTTaskOutput',
    'run_ppt_topic',
]
