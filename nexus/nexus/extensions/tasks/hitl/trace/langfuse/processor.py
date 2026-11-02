"""Langfuse upload for a run with several agents in it.

One run uploads one Langfuse trace. Directly under it sits one span per distinct LLM
conversation, named by its :mod:`~nexus.extensions.tasks.hitl.trace.labels` label: the
worker's main loop, plus one per subagent it ran. They sit side by side, not nested --
attribution is by content prefix, which carries no parent-child link to hang a hierarchy
on. A compaction rewrites the history past recognition, so the worker's loop resumes as a
fresh sibling span rather than one continuous line; these really are separate trajectories.
A completion whose span carries no label is recorded under
:data:`~nexus.extensions.tasks.hitl.trace.labels.UNATTRIBUTED`.

Which record a call belongs to is :func:`.attribution.record_for`'s answer. A call that
issues no tool -- a title, a summary -- is not a conversation; the label's calls of that
kind share one ``aux`` bucket rather than each opening a record. The inherited subagent
detection is switched off -- it tracks one subagent at a time, while Claude Code runs
several at once.
"""

import contextlib
import logging

from typing import Any

from nexus.extensions.tasks.hitl.trace.labels import UNATTRIBUTED
from nexus.extensions.tasks.hitl.trace.langfuse.attribution import LangfuseAgentRecord, has_tool_call, record_for
from nexus.extensions.tasks.hitl.trace.langfuse.upload_dedup import DeduplicatingSpan, MessageLedger
from nexus.extensions.tasks.hitl.trace.usage import LABEL_ATTRIBUTE
from nexus.trace.events import SpanEndEvent
from nexus.trace.processors.base import span_processor_registry
from nexus.trace.processors.langfuse_tracker import (
    LangfuseTrackerProcessor,
    _AgentContext,
    _extract_numeric_score,
    _extract_text_content,
    _suppressed_otel_detach_log,
)
from nexus.trace.types import Span


logger = logging.getLogger(__name__)


COMPLETION_SPAN = 'openai_completion'

# The inherited attributes swapped per agent. The rest of the instance -- the client,
# the trace, the credentials, the truncation limits, the task output -- is shared.
SWAPPED_ATTRS = (
    '_agent_span',
    '_ctx_stack',
    '_ctx',
    '_subagent_tool_span',
    '_subagent_wrapper_span',
    '_model_name',
    '_model_parameters',
    '_tools_summary',
    '_total_usage',
    '_global_step',
)

MAX_USER_MESSAGE_CHARS = 2000
USER_MESSAGE_LABEL_CHARS = 80


@span_processor_registry.register('multiagent_langfuse')
class MultiAgentLangfuseProcessor(LangfuseTrackerProcessor):
    """Upload one Langfuse trace per run, with one span per trace label.

    Configured from a recipe generator::

        processors:
          langfuse:
            name: multiagent_langfuse
            args:
              session_id: <instance_id>
              trace_name: <instance_id>
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._records_by_label: dict[str, list[LangfuseAgentRecord]] = {}
        self._aux_by_label: dict[str, LangfuseAgentRecord] = {}
        self._active: LangfuseAgentRecord | None = None
        self._ledger = MessageLedger()
        self._adopted_init_span = False
        self._init_span: Any = None
        self._init_ctx: _AgentContext | None = None

    # -- per-agent state ---------------------------------------------------

    def _open_record(self, label: str, name: str) -> LangfuseAgentRecord:
        """Create a record for a conversation seen for the first time, and its span.

        The run's first conversation reuses the ``Agent`` span ``_ensure_init`` opened,
        renamed to its label. Every later one -- a subagent, or the worker's own loop
        resuming after a compaction rewrote its history out of recognition -- opens its
        own span directly under the trace, beside the others. There is no main/subagent
        nesting: attribution is by content prefix, which carries no parent-child link,
        and a post-compaction continuation is indistinguishable from a subagent anyway.
        """
        records = self._records_by_label[label]

        if not self._adopted_init_span:
            self._adopted_init_span = True
            # The init span, snapshotted before any record was activated -- self._agent_span
            # is by now whichever record (an aux bucket, say) was activated last.
            assert self._init_ctx is not None
            span = DeduplicatingSpan(self._init_span, self._ledger, scope=name)
            span.update(name=label, metadata={'label': label, 'source': 'multiagent_langfuse'})
            context = self._init_ctx
            context.parent_span = span
        else:
            span = DeduplicatingSpan(
                self._trace.start_observation(
                    name=name,
                    as_type='span',
                    metadata={'label': label, 'source': 'multiagent_langfuse'},
                ),
                self._ledger,
                scope=name,
            )
            context = _AgentContext(parent_span=span)

        record = LangfuseAgentRecord(name=name, state=self._fresh_state(span, context))
        records.append(record)
        return record

    def _aux_record(self, label: str) -> LangfuseAgentRecord:
        """The label's one bucket for calls that issue no tool -- a title, a summary.

        These are not agents: they answer in prose once and are never resumed. Holding
        them in one shared span keeps them out of the agent records, so they neither take
        the main name nor show up as subagents. It never adopts the init span.
        """
        existing = self._aux_by_label.get(label)
        if existing is not None:
            return existing
        name = f'{label} · aux'
        span = DeduplicatingSpan(
            self._trace.start_observation(
                name=name, as_type='span', metadata={'label': label, 'aux': True, 'source': 'multiagent_langfuse'}
            ),
            self._ledger,
            scope=name,
        )
        record = LangfuseAgentRecord(name=name, state=self._fresh_state(span, _AgentContext(parent_span=span)))
        self._aux_by_label[label] = record
        return record

    @staticmethod
    def _fresh_state(span: Any, context: _AgentContext) -> dict[str, Any]:
        """The per-agent state a new record starts with, wrapped around its span."""
        return {
            '_agent_span': span,
            '_ctx_stack': [context],
            '_ctx': context,
            '_subagent_tool_span': None,
            '_subagent_wrapper_span': None,
            '_model_name': '',
            '_model_parameters': {},
            '_tools_summary': [],
            '_total_usage': {'input': 0, 'output': 0, 'total': 0, 'cached_tokens': 0},
            '_global_step': 0,
        }

    def _labelled_records(self) -> Any:
        """Every record with its label: the agent records, then the aux buckets."""
        for label, records in self._records_by_label.items():
            for record in records:
                yield label, record
        for label, record in self._aux_by_label.items():
            yield label, record

    def _activate(self, record: LangfuseAgentRecord) -> None:
        """Make ``record`` the agent whose state the inherited methods read and write."""
        if self._active is record:
            return
        self._save_active()
        for name, value in record.state.items():
            setattr(self, name, value)
        self._active = record

    def _save_active(self) -> None:
        """Copy the live attributes back into the active record's state.

        The inherited code rebinds some of them rather than mutating in place, so the
        stored state is stale until this runs.
        """
        if self._active is None:
            return
        self._active.state.update({name: getattr(self, name) for name in SWAPPED_ATTRS})

    # -- recording ---------------------------------------------------------

    def on_end(self, span: Span, event: SpanEndEvent) -> None:
        """Attribute a completion to its agent, then record it as usual.

        Anything that is not a completion -- the task span the score is read off -- is
        passed straight through.
        """
        if span.name != COMPLETION_SPAN:
            super().on_end(span, event)
            return
        if not self._ensure_init():
            return
        if self._init_span is None:
            # Before any record is activated, the inherited fields still point at the
            # span _ensure_init opened; _open_record's first agent adopts it.
            self._init_span = self._agent_span
            self._init_ctx = self._ctx_stack[0]

        label = span.attributes.get(LABEL_ATTRIBUTE)
        label = label if isinstance(label, str) and label else UNATTRIBUTED
        messages = (span.attributes.get('inputs') or {}).get('messages') or []

        records = self._records_by_label.setdefault(label, [])
        record = record_for(messages, records)
        if record is None:
            if has_tool_call(messages, span.attributes.get('outputs')):
                record = self._open_record(label, self._name_for(label, records))
            else:
                record = self._aux_record(label)
        self._activate(record)
        record.extend_identity(messages)

        # From the second call on: the first call's messages are the opening prompt,
        # which _create_system_user_prompts already renders.
        if messages and self._ctx is not None and self._ctx.prev_message_count:
            self._add_user_message_spans(messages[self._ctx.prev_message_count :])

        super().on_end(span, event)

    @staticmethod
    def _name_for(label: str, records: list[LangfuseAgentRecord]) -> str:
        """The span name for the next record under ``label``: the label, then numbered siblings."""
        return label if not records else f'{label} · {len(records)}'

    def _enter_subagent(self, messages: list[dict[str, Any]], sys_norm: str) -> None:
        """Off: a subagent is its own record, located by attribution, not a stack entry.

        The inherited ``on_end`` still calls this when a call's system prompt differs
        from the baseline, which now only means one agent's prompt drifted mid-run.
        """

    def _exit_subagent(self, messages: list[dict[str, Any]]) -> None:
        """Off: see :meth:`_enter_subagent`."""

    def _add_user_message_spans(self, messages: list[dict[str, Any]]) -> None:
        """Give each user message injected since the last call a span of its own.

        Skips ``<system-reminder>`` blocks.
        """
        if self._ctx is None:
            return
        for message in messages:
            if message.get('role') != 'user':
                continue
            text = _extract_text_content(message.get('content', ''))
            if not text:
                continue
            stripped = text.strip()
            if stripped.startswith('<system-reminder>') and stripped.endswith('</system-reminder>'):
                continue

            shown = text[:MAX_USER_MESSAGE_CHARS]
            summary = shown[:USER_MESSAGE_LABEL_CHARS].replace('\n', ' ').strip()
            if len(shown) > USER_MESSAGE_LABEL_CHARS:
                summary += '...'
            span = self._ctx.parent_span.start_observation(
                name=f'User Message: {summary}',
                as_type='span',
                input={'content': shown},
                metadata={'length': len(text), 'source': 'multiagent_langfuse'},
            )
            span.end()

    # -- closing -----------------------------------------------------------

    def finalize(self) -> None:
        """Close every agent's span, then the trace, and upload the score once."""
        if not self._langfuse or not self._trace:
            return
        if not self._records_by_label and not self._aux_by_label:
            super().finalize()
            return

        try:
            steps: dict[str, int] = {}
            totals = {'input': 0, 'output': 0, 'total': 0, 'cached_tokens': 0}
            for label, record in self._labelled_records():
                self._activate(record)
                steps[label] = steps.get(label, 0) + self._close_agent_span()
                for key in totals:
                    totals[key] += self._total_usage.get(key, 0)

            self._close_trace(steps, totals)
        except Exception:
            logger.warning('multiagent Langfuse finalize failed', exc_info=True)
        finally:
            # The context is process-global, and one left attached names every span
            # created after it.
            self._detach_propagated_context()

    def _detach_propagated_context(self) -> None:
        """Give back the OpenTelemetry context ``_ensure_init`` attached.

        The inherited ``finalize`` ends by doing this; this class replaces that method
        rather than extending it.

        ``__exit__`` detaches a context attached in a different asyncio Task.
        OpenTelemetry catches the resulting ValueError itself and logs it before
        returning normally, so ``contextlib.suppress`` cannot silence it --
        ``_suppressed_otel_detach_log`` drops that one record instead.
        """
        if not self._propagate_ctx:
            return
        with contextlib.suppress(Exception), _suppressed_otel_detach_log():
            self._propagate_ctx.__exit__(None, None, None)
        self._propagate_ctx = None

    def _close_agent_span(self) -> int:
        """Close the active agent's span and any tools still waiting on a result.

        Returns the number of steps it recorded.
        """
        context = self._ctx_stack[0]
        if context.prev_generation and context.prev_tool_calls:
            self._attach_tool_results_for(context, [])

        if context.last_output:
            final = self._agent_span.start_observation(
                name='Final Response',
                as_type='generation',
                model=self._model_name,
                model_parameters=self._model_parameters or None,
                input={'note': 'Final assistant response'},
                output={'role': 'assistant', 'content': self._trunc_content(context.last_output)},
                metadata={
                    **self._turn_metadata_fields(context.step, self._global_step),
                    'is_final': True,
                    'total_steps': context.step,
                    'source': 'multiagent_langfuse',
                },
            )
            final.end()

        self._agent_span.update(
            output={'total_steps': context.step, 'model': self._model_name},
            metadata={
                'total_steps': context.step,
                'model': self._model_name,
                'model_parameters': self._model_parameters,
                'total_usage': self._total_usage,
                'tools': self._tools_summary,
                'source': 'multiagent_langfuse',
            },
        )
        self._agent_span.end()
        return context.step

    def _close_trace(self, steps: dict[str, int], totals: dict[str, int]) -> None:
        """Close the shared trace and upload the score, once for the run."""
        task_metadata = {'task_result': self._task_output} if self._task_output else {}
        score = _extract_numeric_score(self._task_output)
        if score is not None:
            task_metadata['score'] = score

        labels = sorted(self._records_by_label.keys() | self._aux_by_label.keys())
        self._trace.update(
            output={'steps_by_label': steps, 'total_usage': totals},
            metadata={
                **self._trace_metadata,
                'labels': labels,
                'steps_by_label': steps,
                'total_usage': totals,
                'message_upload': self._ledger.summary(),
                **task_metadata,
            },
        )
        self._trace.end()

        if score is not None:
            try:
                self._langfuse.create_score(
                    trace_id=self._trace_id,
                    name='episode_reward',
                    value=score,
                    comment=self._score_comment(),
                )
            except Exception:
                logger.warning('Failed to upload Langfuse score', exc_info=True)

        self._langfuse.flush()
        logger.info(
            'multiagent Langfuse finalized: trace_id=%s, labels=%s, tokens=%s',
            self._trace_id,
            labels,
            totals,
        )

    # -- reading -----------------------------------------------------------

    def get_result(self) -> dict[str, Any]:
        """The trace, plus each label's steps, model and token totals.

        A label's figures sum its records -- the main loop, every sibling conversation, and
        the aux bucket. ``by_label`` has the shape ``HITLUsageProcessor`` reports. Callable
        mid-run: the active agent's live attributes are copied back first.
        """
        self._save_active()
        by_label: dict[str, dict[str, Any]] = {}
        for label, record in self._labelled_records():
            entry = by_label.setdefault(
                label,
                {'steps': 0, 'model': '', 'total_usage': {'input': 0, 'output': 0, 'total': 0, 'cached_tokens': 0}},
            )
            stack = record.state.get('_ctx_stack') or []
            entry['steps'] += stack[0].step if stack else 0
            entry['model'] = entry['model'] or record.state.get('_model_name', '')
            for key in entry['total_usage']:
                entry['total_usage'][key] += record.state.get('_total_usage', {}).get(key, 0)
        return {
            'trace_id': self._trace_id,
            'trace_url': f'{self._host}/trace/{self._trace_id}' if self._trace_id else None,
            'by_label': by_label,
        }


__all__ = ['SWAPPED_ATTRS', 'MultiAgentLangfuseProcessor']
