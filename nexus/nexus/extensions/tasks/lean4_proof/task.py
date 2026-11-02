"""Lean 4 proof generation task.

Pipeline:
  1. Ask the configured agent for a proof candidate, using ``task_input.prompt``
     verbatim. That field is required (and validated non-empty); the task never
     synthesizes a prompt, so a rollout is always scored against exactly the task
     the dataset defines.
  2. Extract the Lean source from the agent reply's
     ``<final_proof>...</final_proof>`` block (stripping any nested code fence).
     That block is the only accepted answer channel. A lightweight format gate
     checks that, after comments, it begins like ordinary top-level Lean source
     (for example an ``import``, declaration, namespace, option, or command).
     This deliberately is not a parser. It only rejects obvious non-code such as
     prose, dangling Markdown/protocol text, and bare tactic snippets before
     consuming a Lean 4 server slot.
  3. Verify that source with ``verify_proof`` (see ``verify_proof.py`` next to
     this module). A candidate with its own imports is treated as a
     self-contained file; otherwise it is appended after the trusted ``header``.
     There is no source-text security filtering. Trust comes from the check
     itself, which judges the proof at the Lean *semantic* layer:

       * the target's elaborated type must equal the reference statement's
         elaborated type (catches injected hypotheses, ``notation`` shadowing of
         names like ``Real.exp``, ``OfNat`` literal overloads, weakened goals);
       * the target's kernel axiom closure must be clean -- no ``sorryAx``
         (which also covers ``apply?`` / ``hint`` / ``exact?`` partial proofs
         that leave no literal ``sorry`` in the source), no custom ``axiom``
         (including ones injected dynamically via ``run_cmd``/``addDecl``, where
         the keyword never appears), and no ``native_decide`` (it trusts
         compiled output rather than the kernel);
       * prelude definitions the statement depends on must not be redefined;
       * the audit is fail-closed: ``#exit``, a shadowed harness, or a crash
         yields no verdict and therefore no reward.

     ``verify_proof`` elaborates the trusted ``header + statement`` itself to
     obtain the reference facts, so a candidate that rewrites the declaration,
     injects hypotheses, shadows names, or redefines a prelude definition fails
     on a type/axiom/dependency mismatch instead of having to be stripped out of
     the source beforehand.

     ``score = 1.0`` iff ``verify_proof`` returns ``is_ok=True``.

Why not score on ``check_whole``'s ``is_ok`` alone: that check is text- and
message-based, and both directions were observed to be wrong on real data. It
passed ``apply?`` proofs (no literal ``sorry``, ``errors=0``, ``sorries=0``, yet
``sorryAx`` in the axiom closure) and its ``axiom`` filter is a regex that
``run_cmd``/``addDecl`` bypasses; conversely its statement-containment check
rejected ~15% of *valid* proofs that merely reformatted whitespace.

``server_response`` preserves the full lean4-server reply, including
``repl_res`` diagnostics, so downstream tooling can inspect the exact Lean
errors/warnings. ``verify_verdict`` carries the structured verdict.

NOTE: do NOT add ``from __future__ import annotations`` here. The task
registry validates the registered task's signature via ``inspect.signature``
and requires ``param.annotation`` to be the real ``TaskRunInput`` subclass
at registration time (not a deferred string under PEP 563), otherwise
registration fails with "Task input parameter must be a subclass of
TaskRunInput, got Lean4ProofTaskInput".
"""

import logging
import os
import re

from typing import Any

from nexus.core import (
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)


logger = logging.getLogger(__name__)

# Sentinel score reserved for verification-service failures (e.g. server
# exceptions), distinct from a legitimately failed proof (0.0) or success (1.0).
SERVICE_FAILURE_SCORE = -300.0

# Absolute path of the nexus package dir, derived from THIS task module's
# location (``.../nexus/extensions/tasks/lean4_proof/task.py`` -> ``.../nexus``).
# Relative ``mcp_servers`` script paths in the config are resolved against this
# by default, so the dataset can store a portable relative path (e.g.
# ``extensions/tools/lean4/mcp_server.py``) that becomes a machine-specific
# absolute path at run time regardless of where the checkout lives.
_NEXUS_PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def _resolve_mcp_servers(servers: list[dict[str, Any]], base_dir: str) -> list[dict[str, Any]]:
    """Return a copy of ``servers`` with relative ``*.py`` script paths in each
    server's ``args`` resolved to absolute paths under ``base_dir``.

    Only relative paths ending in ``.py`` are rewritten; absolute paths and
    non-path args (flags, ``-m`` module names, etc.) are left untouched. This
    lets the MCP server stay fully configurable in the dataset/config while its
    machine-specific absolute location is inferred here at load time.
    """
    resolved: list[dict[str, Any]] = []
    for server in servers:
        if not isinstance(server, dict):
            resolved.append(server)
            continue
        new_server = dict(server)
        args = new_server.get('args')
        if isinstance(args, list):
            new_server['args'] = [
                os.path.abspath(os.path.join(base_dir, arg))
                if isinstance(arg, str) and arg.endswith('.py') and not os.path.isabs(arg)
                else arg
                for arg in args
            ]
        resolved.append(new_server)
    return resolved


# ``<final_proof> ... </final_proof>`` is the contract the kimina /
# miniF2F-style user prompts ask the model to follow.
_FINAL_PROOF_OPEN_RE = re.compile(r'<final_proof>\s*', re.IGNORECASE)
_FINAL_PROOF_CLOSE_RE = re.compile(r'\s*</final_proof>', re.IGNORECASE)

# Strip a code fence the model may nest inside ``<final_proof>``. The info
# string is accepted in any form (```lean4, ```lean, ```Lean, or a bare ```)
# because the fence is decoration around the answer, not part of it -- but it
# must not contain a backtick, so an inline ``code`` span is left alone.
_INNER_FENCE_OPEN_RE = re.compile(r'^\s*```[^\n`]*\n')
_INNER_FENCE_CLOSE_RE = re.compile(r'\n[ \t]*```\s*$')


def _content_to_text(content: Any) -> str:
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get('text') or part.get('content')
                if isinstance(text, str):
                    parts.append(text)
            else:
                parts.append(str(part))
        return '\n'.join(p for p in parts if p)
    return str(content)


def _extract_agent_text(agent_output: object) -> str:
    """Pull a plain-text reply out of the various agent output shapes."""
    success = getattr(agent_output, 'success', True)
    if success is False:
        exit_code = getattr(agent_output, 'exit_code', None)
        stderr = (getattr(agent_output, 'stderr', '') or '').strip()
        if len(stderr) > 1000:
            stderr = stderr[:1000] + '...'
        detail = f'exit_code={exit_code}'
        if stderr:
            detail = f'{detail}, stderr={stderr}'
        raise RuntimeError(f'Agent execution failed: {detail}')

    if hasattr(agent_output, 'final_answer'):
        text = _content_to_text(agent_output.final_answer)
        if text:
            return text
    if hasattr(agent_output, 'output'):
        text = _content_to_text(agent_output.output)
        if text:
            return text
    # The agent finished (success is not False) but emitted no extractable text
    # -- e.g. the last turn was a dangling tool_call, or generation hit the token
    # budget right after a tool call. Return empty instead of raising so the task
    # still completes and scores 0 (via the empty-final_proof guard) rather than
    # failing the whole trajectory (which would become ReasoningReward.TASK_FAILED).
    logger.warning(
        'No extractable text in agent output (%s); treating as empty final proof (score 0).',
        type(agent_output).__name__,
    )
    return ''


def _strip_inner_fence(body: str) -> str:
    body = _INNER_FENCE_OPEN_RE.sub('', body, count=1)
    body = _INNER_FENCE_CLOSE_RE.sub('', body, count=1)
    return body.strip()


def _extract_lean_code(text: str) -> str:
    """Pull the Lean 4 source out of the ``<final_proof>`` block, or return ''.

    The dataset prompt states the ``<final_proof>...</final_proof>`` contract, so
    that block is the ONLY accepted answer channel: the last closing tag paired
    with its nearest preceding opening tag. An optional nested ```lean fence is
    stripped.

    Anything else -- a fenced block with no tags, a dangling ``<final_proof>``
    with no close, or bare prose -- returns '' and is scored as a format error by
    the caller. There is deliberately no fenced-block or raw-text fallback: a
    reply that trails off mid-tool-call still contains prose, so a raw-text
    fallback would submit that prose as the candidate proof. Lean cannot parse it,
    and a service-side exception on the garbage then maps to
    SERVICE_FAILURE_SCORE -- turning a format error into a fake retryable
    infra failure while burning a ``timeout``-by-``retry_num`` service slot.
    """
    close_matches = list(_FINAL_PROOF_CLOSE_RE.finditer(text))
    if not close_matches:
        return ''
    close_match = close_matches[-1]
    open_matches = list(_FINAL_PROOF_OPEN_RE.finditer(text, 0, close_match.start()))
    if not open_matches:
        return ''
    open_match = open_matches[-1]
    return _strip_inner_fence(text[open_match.end() : close_match.start()])


# This is intentionally only a cheap "looks like Lean" gate. The semantic
# verifier supports either a full file with imports or a declaration snippet
# appended after the trusted header, so requiring `import` here would reject
# valid proofs before Lean ever sees them.
_LEAN_COMMAND_PREFIX = (
    r'(?:'
    r'@\[[^\]]*\]\s*|'
    r'(?:private|protected|noncomputable|partial|unsafe|nonrec|scoped|local|public)\s+'
    r')*'
)
_LEAN_TOP_LEVEL_COMMAND = (
    r'(?:'
    r'import|prelude|'
    r'set_option|open|export|namespace|section|end|'
    r'universe|variable|include|omit|attribute|'
    r'theorem|lemma|def|abbrev|axiom|opaque|example|'
    r'structure|class|inductive|instance|mutual|'
    r'notation|infix|infixl|infixr|prefix|postfix|'
    r'syntax|macro|macro_rules|elab|elab_rules|declare_syntax_cat|'
    r'initialize|builtin_initialize'
    r')'
)
_LEAN_SOURCE_START_RE = re.compile(
    rf'^(?:{_LEAN_COMMAND_PREFIX}{_LEAN_TOP_LEVEL_COMMAND}\b|#[A-Za-z_][A-Za-z0-9_\']*\b)'
)
_LEAN_IMPORT_LINE_RE = re.compile(r'^[ \t]*import\b', re.MULTILINE)
_LEAN_VALID_IMPORT_PREFIX_RE = re.compile(r'^(?:prelude\b|import\b)')


def _looks_like_lean4_code(proof: str) -> bool:
    """Whether ``proof`` superficially resembles top-level Lean 4 source.

    An ``import`` is not required: ``verify_proof`` also accepts a declaration
    snippet and prepends the trusted header. This check is deliberately not a
    parser. It only filters obvious non-code without consuming a Lean 4 server
    slot; real syntax and semantic judgement stay in ``verify_proof``.

    Leading comments are allowed -- ``-- header``, ``/- ... -/``, and nested
    block comments are all legal. They are blanked out via ``mask_lean_noise``
    (the same masker ``verify_proof`` uses) rather than stripped by an ad-hoc
    regex. Everything after the recognizable first command is left for Lean to
    validate.
    """
    from .verify_proof import mask_lean_noise

    masked = mask_lean_noise(proof).strip()
    if not _LEAN_SOURCE_START_RE.match(masked):
        return False

    # A declaration/header snippet with no imports is valid because
    # `verify_proof` prepends the trusted header. If the snippet does contain an
    # import, however, Lean requires imports at the beginning of the file
    # (`prelude` is the sole legal command before them). Rejecting a misplaced
    # import here avoids an unnecessary server request while retaining both
    # supported candidate forms.
    return not (_LEAN_IMPORT_LINE_RE.search(masked) and not _LEAN_VALID_IMPORT_PREFIX_RE.match(masked))


class Lean4ProofTaskInput(TaskRunInput, WithAgentRunner):
    """Input for the Lean 4 proof generation task."""

    statement: str
    """The Lean 4 theorem statement (the ``:= by sorry`` line). Together with
    ``header`` it is the trusted reference: ``verify_proof`` elaborates
    ``header + statement`` to obtain the type the candidate must match."""

    header: str = 'import Mathlib\n'
    """Imports / preamble (joined ``deps``) belonging to the trusted statement.
    Used to elaborate the reference and prepended when the model returns a
    declaration snippet without its own imports."""

    prompt: str | None = None
    """Verbatim user prompt handed to the agent, from the dataset (e.g.
    ``messages[0].content``). Required and validated non-empty: there is no
    locally synthesized fallback, since a divergent prompt would mean scoring
    rollouts against a different task than the dataset defines."""

    verify_with_server: bool = True
    """Re-run ``async_check_whole`` on the host to compute the score."""

    # Lean 4 server config (mirrors AsyncLean4Client.__init__)
    polaris_url: str = 'http://lean4-server-test-v2.production.polaris:8080'
    repo_path: str = '/miniF2F-lean4'
    timeout: float = 600.0
    retry_num: int = 5

    permitted_axioms: list[str] | None = None
    """Kernel axioms the proof may depend on. ``None`` uses
    ``verify_proof.DEFAULT_PERMITTED_AXIOMS`` (``Classical.choice``, ``propext``,
    ``Quot.sound``) -- i.e. Mathlib's own foundations. Anything else (a custom
    ``axiom``, or ``native_decide``'s generated ``*._native.native_decide.ax_N``)
    is rejected as an unproved assumption."""

    # MCP servers for the agent, configured in the dataset/config file. Their
    # ``args`` may use a path RELATIVE to ``mcp_server_base_dir`` (default: the
    # nexus package dir); the task resolves it to an absolute path at run time
    # and forwards the spec to ``run_agent`` (overriding ``agent_config``), so
    # nothing machine-specific is baked into the parquet. Leave empty/None to
    # keep whatever the agent_config already declares.
    mcp_servers: list[dict[str, Any]] | None = None
    mcp_server_base_dir: str = ''
    """Base directory for resolving relative ``*.py`` paths in
    ``mcp_servers[*].args``. Empty => the nexus package directory."""


class Lean4ProofTaskOutput(TaskRunOutput, WithScore):
    """Output for the Lean 4 proof generation task."""

    statement: str
    """The trusted theorem statement this rollout was scored against."""

    header: str
    """The trusted header (imports / options / prelude definitions) that goes
    with ``statement``. Echoed here so a rollout record is self-sufficient:
    re-scoring or auditing it later needs ``header + statement`` to rebuild the
    same reference, and recovering the header from the prompt afterwards is
    lossy."""

    final_proof: str
    """Lean source extracted from ``<final_proof>...</final_proof>``. It may be
    a self-contained file or a declaration snippet. Empty when the reply had no
    such block."""

    proof: str
    """The model-produced Lean 4 source that was verified. Same as
    ``final_proof``; ``verify_proof`` decides whether to use it as a full file
    or append it after the trusted header."""

    is_ok: bool
    """Whether the Lean 4 server accepted the snippet (== accuracy)."""

    server_response: dict[str, Any] | None = None
    """Raw response from ``async_check_whole``. ``repl_res.messages``
    inside this dict carries the warnings / errors the Lean 4 engine
    produced and is what downstream tooling should look at."""

    verify_verdict: str | None = None
    """Human-readable verdict from ``verify_proof``: either a confirmation or
    the specific violation(s) (``TYPE_MISMATCH``, ``SORRY_DEPENDENCY``,
    ``UNPERMITTED_AXIOM``, ``DEPENDENCY_VALUE_CHANGED``, ``AUDIT_MISSING``, ...).
    Useful for auditing why a rollout scored 0."""

    fail_reason: str | None = None
    """Why the run failed outside of a normal proof verdict. One of:

    * ``"empty_final_proof"`` -- no usable ``<final_proof>`` block in the agent
      reply (missing or unclosed tag, or an empty block). Format error, scored 0.
    * ``"final_proof_not_self_contained"`` -- legacy label for a block that
      does not superficially begin like top-level Lean source (leading comments
      are allowed). Format error, scored 0.
    * ``"verification_service_error: <cause>"`` -- the score is
      ``SERVICE_FAILURE_SCORE``. The concrete cause is appended so these
      retryable rollouts can be triaged from the record alone (Lean service
      timeout vs. a reference statement that does not compile).
    """


async def _verify_with_server(
    model_proof: str,
    *,
    statement: str,
    header: str,
    polaris_url: str,
    repo_path: str,
    timeout: float,
    retry_num: int,
    permitted_axioms: list[str] | None = None,
) -> dict[str, Any]:
    """Semantically verify the model's Lean source with ``verify_proof``.

    ``model_proof`` may be a self-contained Lean file or a declaration snippet.
    ``verify_proof`` appends the latter after ``header`` and derives the
    reference facts by elaborating ``header + statement`` on its own. Rewriting
    the declaration, injecting hypotheses, shadowing names, or redefining
    prelude definitions therefore surface as a type/axiom/dependency mismatch
    rather than needing to be filtered out of the source text first.

    Uses the native async API (``aiohttp``), so many rollouts verify
    concurrently on the event loop without thread hand-offs.
    """
    from .verify_proof import DEFAULT_PERMITTED_AXIOMS, async_verify_proof

    try:
        verdict = await async_verify_proof(
            header,
            statement,
            model_proof,
            server_url=polaris_url,
            permitted_axioms=permitted_axioms or list(DEFAULT_PERMITTED_AXIOMS),
            repo_path=repo_path,
            timeout=int(timeout),
            retry_num=retry_num,
            # RL samples the same statement many times, so caching the
            # reference elaboration removes one Lean round-trip per rollout;
            # in-flight dedup means concurrent rollouts share a single call.
            use_ground_truth_cache=True,
        )
    except Exception as exc:
        logger.exception('Lean 4 verification raised an exception')
        return {'is_ok': False, 'exception': f'{type(exc).__name__}: {exc}'}

    if not isinstance(verdict, dict):
        return {
            'is_ok': False,
            'exception': f'unexpected verdict type: {type(verdict).__name__}',
        }

    # Normalize into the shape the rest of the task (and downstream tooling)
    # already expects: is_ok / repl_res / exception, plus the structured verdict.
    normalized: dict[str, Any] = {
        'is_ok': bool(verdict.get('is_ok')),
        'repl_res': verdict.get('repl_res') or {},
        'verify_message': verdict.get('other_message') or '',
    }
    if verdict.get('infra_error'):
        # Not evidence against the proof: service unreachable, or the reference
        # statement itself could not be elaborated. Surfaced as `exception` so it
        # maps to SERVICE_FAILURE_SCORE and can be retried instead of training on
        # a bogus 0.
        normalized['exception'] = verdict.get('other_message') or 'verification infrastructure error'
    return normalized


# Cap for the exception detail appended to ``fail_reason``. The field is a short
# triage label, not a log: `verify_proof` can put a whole multi-line verdict in
# `exception` (the GROUND_TRUTH path echoes Lean's errors), and the untruncated
# text belongs in ``server_response`` / ``verify_verdict``.
MAX_FAIL_REASON_DETAIL_CHARS = 300


def _service_failure_reason(server_response: dict[str, Any]) -> str:
    """Build a ``fail_reason`` that names the concrete verification failure.

    Always prefixed with ``verification_service_error`` so existing consumers
    that match on that label keep working, then the specific cause.
    """
    detail = str(server_response.get('exception') or '').strip()
    if not detail:
        return 'verification_service_error'
    detail = ' | '.join(line.strip() for line in detail.splitlines() if line.strip())
    if len(detail) > MAX_FAIL_REASON_DETAIL_CHARS:
        detail = detail[:MAX_FAIL_REASON_DETAIL_CHARS] + '...(truncated)'
    return f'verification_service_error: {detail}'


def _score_server_response(server_response: dict[str, Any]) -> float:
    """Score a Lean verification response, reserving SERVICE_FAILURE_SCORE for service failures."""
    if server_response.get('exception'):
        return SERVICE_FAILURE_SCORE
    return 1.0 if bool(server_response.get('is_ok')) else 0.0


@task_registry.register('lean4_proof')
async def lean4_proof_task(task_input: Lean4ProofTaskInput) -> Lean4ProofTaskOutput:
    """Generate and verify a Lean 4 proof for the given statement."""
    ctx = require_context()
    # In-process (no-sandbox) mode: the lean4 MCP tool runs as a local stdio
    # subprocess and host-side verification reaches the Lean 4 server over HTTP,
    # so a sandbox runtime is optional. Only acquire one when a provider exists.
    try:
        runtime = await ctx.get_runtime('main')
        logger.debug('Runtime initialized: %s', runtime)
    except RuntimeError:
        logger.debug('No runtime provider configured; running lean4_proof without a sandbox runtime.')

    # The prompt must come from the dataset. There is no synthesized fallback:
    # a locally-built prompt would silently diverge from the one the dataset was
    # validated against (different tool instructions, different `<final_proof>`
    # contract), so rollouts would be scored against a different task than
    # intended. Fail loudly instead.
    if not task_input.prompt or not task_input.prompt.strip():
        raise ValueError(
            'task_input.prompt is required and must be non-empty the dataset must supply the verbatim user prompt)'
        )
    agent_kwargs: dict[str, Any] = {'user_prompt': task_input.prompt}
    if task_input.mcp_servers:
        # Resolve relative script paths in the configured mcp_servers to
        # absolute (portable dataset) and override agent_config.mcp_servers.
        base_dir = task_input.mcp_server_base_dir or _NEXUS_PACKAGE_DIR
        resolved_servers = _resolve_mcp_servers(task_input.mcp_servers, base_dir)
        agent_kwargs['mcp_servers'] = resolved_servers
        logger.debug('Resolved mcp_servers args: %s', [srv.get('args') for srv in resolved_servers])
    agent_output = await task_input.run_agent(**agent_kwargs)
    agent_text = _extract_agent_text(agent_output)
    final_proof_raw = _extract_lean_code(agent_text)

    server_response: dict[str, Any] | None = None
    is_ok = False
    fail_reason: str | None = None

    # Format gate: score 0 and skip the Lean 4 server entirely. These replies are
    # not evidence about any proof, and submitting one anyway risks a service-side
    # exception on unparseable text, which would be mis-scored as
    # SERVICE_FAILURE_SCORE (a fake retryable infra failure) while burning a
    # `timeout`-by-`retry_num` service slot.
    format_error: str | None = None
    if not final_proof_raw.strip():
        format_error = 'empty_final_proof'
        logger.warning(
            'No final_proof extracted (missing/unclosed <final_proof> tag, or empty block); '
            'scoring 0 as a format error without calling the Lean 4 server.'
        )
    elif not _looks_like_lean4_code(final_proof_raw):
        format_error = 'final_proof_not_looks_like_lean4'
        logger.warning(
            'final_proof does not begin like top-level Lean 4 source; '
            'scoring 0 as a format error without calling the Lean 4 server. First line: %r',
            final_proof_raw.strip().splitlines()[0][:200],
        )

    if format_error is not None:
        return Lean4ProofTaskOutput(
            score=0.0,
            statement=task_input.statement,
            header=task_input.header,
            final_proof=final_proof_raw,
            proof='',
            is_ok=False,
            server_response=None,
            fail_reason=format_error,
        )

    if task_input.verify_with_server:
        logger.debug('Verifying proof semantically via verify_proof (%s)', task_input.polaris_url)
        server_response = await _verify_with_server(
            model_proof=final_proof_raw,
            statement=task_input.statement,
            header=task_input.header,
            polaris_url=task_input.polaris_url,
            repo_path=task_input.repo_path,
            timeout=task_input.timeout,
            retry_num=task_input.retry_num,
            permitted_axioms=task_input.permitted_axioms,
        )
        score = _score_server_response(server_response)
        is_ok = score == 1.0
        if score == SERVICE_FAILURE_SCORE:
            # Carry the concrete cause, not just the category: these rollouts are
            # meant to be retried, and triaging them from the record alone needs
            # to distinguish "Lean service timed out" from "the reference
            # statement does not compile" (a data defect that retrying won't fix).
            fail_reason = _service_failure_reason(server_response)
        logger.info(
            'Lean 4 verification result: is_ok=%s exception=%s verdict=%s',
            is_ok,
            server_response.get('exception'),
            (server_response.get('verify_message') or '').replace('\n', ' | ')[:400],
        )
    else:
        logger.debug('Skipping server-side verification (verify_with_server=False)')
        score = 0.0

    return Lean4ProofTaskOutput(
        score=score,
        statement=task_input.statement,
        header=task_input.header,
        final_proof=final_proof_raw,
        proof=final_proof_raw,
        is_ok=is_ok,
        server_response=server_response,
        verify_verdict=(server_response or {}).get('verify_message') or None,
        fail_reason=fail_reason,
    )
