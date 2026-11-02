"""CodeQA-file task: run an agent that writes its answer to a file, read it back
host-side, and score it with an LLM judge. Returns a scalar ``score`` so hytuner's
generic ``task_output.score -> reward`` path trains it (no hytuner change needed).

Reward semantics:
  - success                              -> score in [0, 1] from the rubric judge
  - empty / oversize / non-regular file  -> real 0.0 (agent-controllable; stays in GRPO group)
  - judge responded but unparseable      -> real 0.0 (agent-controllable)
  - judge infra failure (5xx/timeout...) -> REWARD_ERROR span + 0.0 (masked out by trainer)
"""

import asyncio
import base64
import logging
import os
import shlex
import uuid

from typing import Any

from nexus.core import (
    OpenAIConfig,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    openai_client_registry,
    require_context,
    task_registry,
)
from nexus.extensions.agents.installed_common import InstalledAgentOutput  # for isinstance dispatch
from nexus.runtime.executors.models.file import FileItem  # path == swe_cosmos:22
from nexus.trace import get_current_span_id, update_span

from .rubric_judge import build_rubric_messages, parse_rubric_score_strict


logger = logging.getLogger(__name__)

# Agent prompt: aligned VERBATIM with the reference hytuner codeqa task.py SYSTEM_INSTRUCTION_FILE
# (nick's), for behavioral parity across implementations. Only the {answer_path} VALUE differs:
# it is filled per-run-unique here (see run_id below) to avoid GRPO cross-rollout residue, vs
# nick's fixed /testbed/final_answer.txt -- the prompt TEXT is byte-identical.
SYSTEM_INSTRUCTION_FILE = """I've uploaded a code repository in the directory {working_dir}.
{question}
help me answer this question about the codebase and write your COMPLETE final answer to the file {answer_path}.
"""

# Fallback judge spec when task_config carries no judge_client. Users override per-row.
DEFAULT_JUDGE_SPEC: dict[str, Any] = {
    'name': 'litellm',
    'client_args': {'base_url': 'http://llm-api.model-eval.woa.com/v1', 'max_retries': 3},
    'request_args': {'model': 'api_deepseek_deepseek-v4-flash', 'temperature': 0},
}

_JUDGE_SEMAPHORE = asyncio.Semaphore(int(os.getenv('CODEQA_JUDGE_MAX_CONCURRENCY', '16')))
_MAX_ANSWER_BYTES = int(os.getenv('CODEQA_MAX_ANSWER_BYTES', str(32 * 1024)))  # 32 KiB
_ANSWER_READ_TIMEOUT = 20

# swe_agent ExitStatus classification (values in swe_agent/agent.py):
#   normal-finish -> submitted/finished ; infra(maskout) -> error ;
#   agent-controllable(real 0.0, keep in group) -> tool_error/max_iterations/truncation/timeout
_SWE_OK = {'submitted', 'finished'}
_SWE_INFRA = {'error'}
# hycode exit codes that indicate an infra kill (OOM/segv/term), not agent's fault -> maskout.
_HYCODE_INFRA_EXIT = {137, 139, 143}


class CodeQAFileTaskInput(TaskRunInput, WithAgentRunner):
    instance_id: str
    question: str
    working_dir: str = '/testbed'
    reference_answer: str | dict | None = None  # optional: this data has no standalone ref; rubric self-contains it
    critic_rule: str  # required: has [QUESTION]/[REFERENCE]/[ANSWER] + Step-3 transform
    score_key: str = '归一化最终得分（0-100）'  # rubric JSON score key (carried by data, not hardcoded)
    score_system_prompt: str | None = None  # judge system prompt (carried by data; keeps reward contract)
    question_variant: str = 'original'
    judge_client: dict[str, Any] | None = None  # judge client spec; None -> DEFAULT_JUDGE_SPEC
    reward_contract_version: str | None = None
    # traj_id must be a DECLARED field, else _build_input(single.py) drops it. hytuner writes
    # the per-rollout id under task_config['traj_id'] (runner/hytuner.py), so alias it here.
    traj_id: str | None = None


class CodeQAFileTaskOutput(TaskRunOutput, WithScore):  # WithScore => score: float required
    instance_id: str
    prediction: str
    error_message: str | None = None


def _resolve_judge(ctx, judge_client_spec: dict[str, Any] | None):
    """Resolve the judge client. NEVER fall back to the policy 'main' client -- letting the
    model under training grade its own answers is a reward-hacking channel.
    Order: (1) task_config.judge_client -> build; (2) an explicitly-registered 'judge' client;
    (3) DEFAULT_JUDGE_SPEC. 'main' is intentionally excluded."""
    if judge_client_spec:
        cls = openai_client_registry.get(judge_client_spec.get('name', 'litellm'))
        return cls(
            OpenAIConfig(
                client_args=judge_client_spec.get('client_args', {}),
                request_args=judge_client_spec.get('request_args', {}),
            )
        )
    try:
        return ctx.get_openai_client('judge')  # only a dedicated judge, never 'main'
    except KeyError:
        pass
    cls = openai_client_registry.get(DEFAULT_JUDGE_SPEC['name'])
    return cls(
        OpenAIConfig(client_args=DEFAULT_JUDGE_SPEC['client_args'], request_args=DEFAULT_JUDGE_SPEC['request_args'])
    )


@task_registry.register('codeqa_file')
async def codeqa_file_task(task_input: CodeQAFileTaskInput) -> CodeQAFileTaskOutput:
    # Top-level safety net: no unexpected exception may leak reward_score=None
    # (else handle_nexus_none_reward_score maps it to TASK_FAILED=-300 into the gradient).
    try:
        return await _codeqa_file_impl(task_input)
    except Exception as e:
        logger.error('codeqa_file unexpected error: %r', e, exc_info=True)
        _mark_reward_error(f'unexpected task error: {type(e).__name__}')
        return CodeQAFileTaskOutput(instance_id=task_input.instance_id, prediction='', score=0.0)


async def _codeqa_file_impl(task_input: CodeQAFileTaskInput) -> CodeQAFileTaskOutput:
    ctx = require_context()
    # Per-run unique answer path. traj_id is the real per-sample id (declared field, aliased
    # from task_config['traj_id']). uuid4 is the last resort so the GRPO group's samples of the
    # same instance NEVER collide on one path (instance_id alone is identical across the group).
    run_id = (
        task_input.traj_id or (ctx.get_runtime_id() if hasattr(ctx, 'get_runtime_id') else None) or uuid.uuid4().hex
    )
    answer_path = f'{task_input.working_dir.rstrip("/")}/.nexus_codeqa/final_answer_{_safe(run_id)}.txt'

    runtime = await ctx.get_runtime()
    # Record the runtime id onto the task span so hytuner reports nexus-<id> instead of 'unknown'
    # (parity with swe/task.py:341 & terminal/task.py:112). hytuner reads span attr
    # 'hytuner.runtime_id' (hytuner.py:604) -> self.runtime_id; codeqa never set it, hence 'unknown'.
    _rt_span_id = get_current_span_id()
    if _rt_span_id:
        update_span(_rt_span_id, attributes={'hytuner.runtime_id': ctx.get_runtime_id(runtime) or 'unknown'})
    await runtime.run_command(
        f'mkdir -p {shlex.quote(os.path.dirname(answer_path))} && rm -f {shlex.quote(answer_path)}', timeout=15
    )

    prompt = SYSTEM_INSTRUCTION_FILE.format(
        question=task_input.question, working_dir=task_input.working_dir, answer_path=answer_path
    )

    # 1) Run the agent (agent_name in nexus_config picks swe_agent / hycode).
    agent_output = await task_input.run_agent(user_prompt=prompt, working_dir=task_input.working_dir)

    # 2) Did the agent finish cleanly? A file written but the agent then timed out / crashed
    #    should NOT be scored (else it learns "write answer then stall"). Split agent-controllable
    #    (real 0.0, stays in group) vs infra (REWARD_ERROR -> maskout).
    ok, is_infra = _agent_completed_ok(agent_output)
    if not ok:
        if is_infra:
            _mark_reward_error('agent infra failure (runtime/oom/timeout)')
        return CodeQAFileTaskOutput(instance_id=task_input.instance_id, prediction='', score=0.0)

    # 3) Read the answer file back to host. Returns (text, kind): kind 'ok'/'missing'/'infra'.
    answer, read_kind = await _read_answer_file(ctx, runtime, answer_path, max_bytes=_MAX_ANSWER_BYTES)
    # A runtime/RPC failure while reading is INFRA -> maskout, not a fake 0.0.
    if read_kind == 'infra':
        _mark_reward_error('answer-file read infra failure')
        return CodeQAFileTaskOutput(instance_id=task_input.instance_id, prediction='', score=0.0)
    # empty / missing / oversize / non-regular file -> agent-controllable real 0.0 (no maskout).
    if not answer:
        return CodeQAFileTaskOutput(instance_id=task_input.instance_id, prediction='', score=0.0)

    # 4) LLM-as-judge on host side (never in sandbox). system_prompt carried by data.
    reference = _coerce_reference(task_input.reference_answer)  # None-safe: empty when no standalone ref
    messages = build_rubric_messages(
        question=task_input.question,
        reference=reference,
        answer=answer,
        critic_rule=task_input.critic_rule,
    )
    judge = _resolve_judge(ctx, task_input.judge_client)
    score, failure_kind = await _judge_score(judge, messages, score_key=task_input.score_key)

    if score is not None:
        return CodeQAFileTaskOutput(instance_id=task_input.instance_id, prediction=answer, score=float(score))
    if failure_kind == 'infra':
        _mark_reward_error('judge infra failure after retries')
        return CodeQAFileTaskOutput(instance_id=task_input.instance_id, prediction=answer, score=0.0)
    return CodeQAFileTaskOutput(instance_id=task_input.instance_id, prediction=answer, score=0.0)


def _mark_reward_error(reason: str) -> None:
    # Log when no span is available so a lost maskout is not silent.
    span_id = get_current_span_id()
    if span_id:
        update_span(span_id, attributes={'hytuner.termination_reason': 'REWARD_ERROR'})
    else:
        logger.error('codeqa_file: no active span to mark REWARD_ERROR (%s); sample may NOT be masked out', reason)


def _safe(s) -> str:
    import re

    return re.sub(r'[^A-Za-z0-9._-]', '_', str(s))[:80]


def _coerce_reference(ref) -> str:
    """Normalize reference_answer to a plain string; '' when absent (rubric self-contains ref)."""
    if ref is None:
        return ''
    if isinstance(ref, dict):
        for k in ('answer', 'text', 'reference'):
            v = ref.get(k)
            if isinstance(v, str) and v.strip():
                return v
        import json

        return json.dumps(ref, ensure_ascii=False)
    return str(ref)


async def _read_answer_file(ctx, runtime, answer_path: str, max_bytes: int) -> tuple[str, str]:
    """Fetch final_answer from sandbox to host. Returns (text, kind), kind in {'ok','missing','infra'}.

    Uses the runtime's native ``download_file`` (server-side, aiofiles-based) -- same as
    swe_cosmos. This does NOT require python3 (or any interpreter) inside the sandbox: the
    codeqa per-repo images are arbitrary-language repos (C#/PHP/Java/C++/...) that often have
    no python3, so an in-sandbox ``python3 -c`` reader fails with rc=127 across whole images.

    Host-side size cap: download_file pulls the whole file, then we reject if it exceeds
    max_bytes (answers are small; acceptable for eval). download_file's own missing/non-file
    guard returns via ``resp.errors``.
    """
    try:
        resp = await runtime.download_file([FileItem(path=answer_path, encoding='base64')])
    except Exception as e:
        logger.warning('codeqa_file answer read RPC failed: %r', e)
        return '', 'infra'
    if not resp.files or resp.files[0].content is None:
        # download_file records missing/not-a-file in resp.errors -> agent-controllable empty
        return '', 'missing'
    try:
        raw = base64.b64decode(resp.files[0].content)
    except Exception:
        return '', 'missing'
    if len(raw) > max_bytes:  # oversize: agent-controllable real 0.0
        return '', 'missing'
    return raw.decode('utf-8', errors='replace').strip(), 'ok'


def _agent_completed_ok(agent_output) -> tuple[bool, bool]:
    """(ok, is_infra). Did the agent finish cleanly?
      - (True,  False) -> finished, proceed to read+judge
      - (False, False) -> agent-controllable failure (timeout / nonzero / max-iter) -> real 0.0
      - (False, True)  -> infra failure (runtime/oom) -> REWARD_ERROR maskout
    Dispatch by isinstance(InstalledAgentOutput), not hasattr('exit_code')."""
    if isinstance(agent_output, InstalledAgentOutput):  # hycode & other installed CLIs
        exit_code = getattr(agent_output, 'exit_code', 1)
        if bool(getattr(agent_output, 'success', False)) and exit_code == 0:
            return True, False
        # OOM/segv/term exit codes are infra, not agent's fault -> maskout.
        return False, (exit_code in _HYCODE_INFRA_EXIT)
    # swe_agent: SWEAgentOutput.exit_status: ExitStatus (str enum)
    es = getattr(agent_output, 'exit_status', None)
    es = str(getattr(es, 'value', es) or '').lower()
    if es in _SWE_OK:
        return True, False
    return False, (es in _SWE_INFRA)


def _is_infra_judge_error(exc: Exception) -> bool:
    """Classify judge exceptions by STRUCTURED status code, not substring scan.
    4xx (bad request: oversize answer+rubric over context, malformed) -> NOT infra (agent 0.0).
    5xx / timeout / connection / rate-limit -> infra (maskout)."""
    code = getattr(exc, 'status_code', None) or getattr(getattr(exc, 'response', None), 'status_code', None)
    if isinstance(code, int):
        return not (400 <= code < 500)  # 4xx = agent-controllable; else infra
    # No structured code: transport-level errors (timeout/connection) are infra by default.
    return True


async def _judge_score(judge, messages, attempts: int = 2, score_key: str = '归一化最终得分（0-100）'):
    """Return (score|None, kind). kind in {'ok','unparseable','infra'}:
      - unparseable: judge responded but its output has no parseable score_key (incl. 4xx from
        an oversize agent answer) -> caller gives real 0.0 (agent-controllable).
      - infra: connection/timeout/5xx/rate-limit exhausted, judge never scored -> maskout.
    Semaphore wraps only the network call; released during backoff. Judge client is passed in
    by _resolve_judge (never the policy 'main').
    NOTE: create_completion is @cacheable; identical messages re-hit cache, so a parse-retry
    with the SAME messages cannot recover. Retries here only help transient transport errors.
    A genuine unparseable response is terminal -> real 0.0 (correct: an agent that provokes an
    unparseable verdict should not escape via maskout)."""
    saw_response = False
    for i in range(attempts):
        try:
            async with _JUDGE_SEMAPHORE:
                resp = await judge.create_completion(messages=messages)
            saw_response = True
            score = parse_rubric_score_strict(resp.choices[0].message.content or '', score_key=score_key)
            if score is not None:
                return score, 'ok'
            # responded but unparseable -> terminal, do not burn a retry on a cached identical call
            return None, 'unparseable'
        except Exception as e:
            if not _is_infra_judge_error(e):  # 4xx from agent-controlled payload
                logger.warning('codeqa_file judge 4xx (agent-controllable): %r', e)
                return None, 'unparseable'
            logger.warning('codeqa_file judge infra error (attempt %d): %r', i + 1, e)
        if i + 1 < attempts:
            await asyncio.sleep(2**i)
    return None, ('unparseable' if saw_response else 'infra')
