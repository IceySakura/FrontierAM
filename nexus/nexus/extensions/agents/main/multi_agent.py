"""Bounded in-process subagent control for main-agent runs."""

from __future__ import annotations

import asyncio
import json
import math
import time

from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, Field


MULTI_AGENT_CONTROLLER_STATE_KEY = '_main_agent_subagent_controller'
MULTI_AGENT_SUMMARY_STATE_KEY = '_main_agent_subagent_summary'
SPAWN_AGENT_TOOL_NAME = 'multi_agent/spawn_agent'
WAIT_AGENT_TOOL_NAME = 'multi_agent/wait_agent'
SUCCESSFUL_CHILD_EXIT_STATUSES = frozenset({'finished', 'refused'})


def _validate_finite_positive_seconds(value: Any) -> int | float:
    """Reject coercive or unbounded duration values before Pydantic conversion."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('duration must be a finite positive int or float')
    if not math.isfinite(value) or value <= 0:
        raise ValueError('duration must be finite and greater than zero')
    return value


FinitePositiveSeconds = Annotated[float, BeforeValidator(_validate_finite_positive_seconds)]


class AgentStatus(str, Enum):
    RUNNING = 'running'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'


class SubagentConfig(BaseModel):
    """Define static configuration shared by every child of one root run."""

    system_prompt: str
    tools: list[str] = Field(min_length=1)
    max_iterations: int = Field(default=4, ge=1)
    parallel_tool_calls: bool = False


class MultiAgentConfig(BaseModel):
    """Define the bounded control-plane configuration for a main-agent run."""

    enabled: bool = True
    max_agents: int = Field(default=3, ge=1)
    max_active_agents: int = Field(default=2, ge=1)
    child_timeout_seconds: FinitePositiveSeconds | None = None
    subagent: SubagentConfig


class AgentExecutionIdentity(BaseModel):
    """Describe a declared main or first-level subagent chain."""

    agent_id: str
    parent_agent_id: str | None
    chain_id: str
    parent_chain_id: str | None
    kind: Literal['main', 'subagent']
    depth: int = Field(ge=0)
    spawn_order: int | None = Field(default=None, ge=0)

    @classmethod
    def main(cls) -> AgentExecutionIdentity:
        """Build the canonical identity for the root main agent."""
        return cls(
            agent_id='main',
            parent_agent_id=None,
            chain_id='main',
            parent_chain_id=None,
            kind='main',
            depth=0,
            spawn_order=0,
        )


class SpawnAgentResult(BaseModel):
    agent_id: str
    name: str | None
    status: AgentStatus


class AgentResult(BaseModel):
    agent_id: str
    name: str | None
    status: AgentStatus
    final_answer: str | None = None
    exit_status: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class WaitAgentResult(BaseModel):
    completed: list[AgentResult] = Field(default_factory=list)
    failed: list[AgentResult] = Field(default_factory=list)
    pending: list[str] = Field(default_factory=list)
    timed_out: bool = False


class SubagentRunFailed(RuntimeError):  # noqa: N818
    """Report terminal child failures from a normal root completion barrier."""

    def __init__(self, result: WaitAgentResult) -> None:
        super().__init__('one or more subagents did not complete successfully')
        self.result = result


@dataclass
class AgentRecord:
    identity: AgentExecutionIdentity
    name: str | None
    prompt: str
    spawn_order: int
    status: AgentStatus
    created_at: float
    task: asyncio.Task[Any] | None = None
    output: Any = None
    error: BaseException | None = None
    completed_at: float | None = None


ChildRunner = Callable[[AgentExecutionIdentity, str], Awaitable[Any]]


class SubagentController:
    """Own child tasks and the root terminal barrier."""

    def __init__(self, config: MultiAgentConfig, *, run_child: ChildRunner) -> None:
        self.config = config
        self._run_child = run_child
        self._records: dict[str, AgentRecord] = {}
        self._events: list[dict[str, Any]] = []
        self._observed_agent_ids: set[str] = set()
        self._lock = asyncio.Lock()
        self._closing = False
        self._spawn_count = 0
        self._max_active_agents = 0

    async def spawn(self, prompt: str, name: str | None = None) -> SpawnAgentResult:
        """Start one first-level subagent without waiting for its completion."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError('subagent prompt must be non-empty')
        async with self._lock:
            if self._closing:
                raise RuntimeError('subagent controller is closing')
            if len(self._records) >= self.config.max_agents:
                raise RuntimeError(f'max_agents limit reached ({self.config.max_agents})')
            active_count = sum(record.status == AgentStatus.RUNNING for record in self._records.values())
            if active_count >= self.config.max_active_agents:
                raise RuntimeError(f'max_active_agents limit reached ({self.config.max_active_agents})')

            self._spawn_count += 1
            agent_id = f'agent-{self._spawn_count}'
            record = AgentRecord(
                identity=AgentExecutionIdentity(
                    agent_id=agent_id,
                    parent_agent_id='main',
                    chain_id=agent_id,
                    parent_chain_id='main',
                    kind='subagent',
                    depth=1,
                    spawn_order=self._spawn_count,
                ),
                name=name,
                prompt=prompt,
                spawn_order=self._spawn_count,
                status=AgentStatus.RUNNING,
                created_at=time.monotonic(),
            )
            self._records[agent_id] = record
            self._events.append({
                'event': 'spawn',
                'agent_id': agent_id,
                'name': name,
                'spawn_order': record.spawn_order,
            })
            record.task = asyncio.create_task(self._run(record), name=f'nexus-subagent-{agent_id}')
            self._max_active_agents = max(self._max_active_agents, active_count + 1)
        return SpawnAgentResult(agent_id=agent_id, name=name, status=AgentStatus.RUNNING)

    async def wait(
        self,
        agent_ids: list[str],
        *,
        return_when: Literal['all', 'any'] = 'all',
        timeout_seconds: float | None = None,
    ) -> WaitAgentResult:
        """Wait for requested children without cancelling an unfinished child."""
        if not agent_ids:
            raise ValueError('agent_ids must be non-empty')
        if len(set(agent_ids)) != len(agent_ids):
            raise ValueError('agent_ids must not contain duplicates')
        if return_when not in {'all', 'any'}:
            raise ValueError("return_when must be 'all' or 'any'")
        async with self._lock:
            records = [self.get_record(agent_id) for agent_id in agent_ids]
            tasks = [record.task for record in records if record.task is not None]
            self._events.append({'event': 'wait', 'agent_ids': list(agent_ids), 'return_when': return_when})
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout_seconds,
            return_when=asyncio.ALL_COMPLETED if return_when == 'all' else asyncio.FIRST_COMPLETED,
        )
        timed_out = timeout_seconds is not None and (bool(pending) if return_when == 'all' else not done)
        result = self._wait_result(records, timed_out=timed_out)
        self._observed_agent_ids.update(item.agent_id for item in [*result.completed, *result.failed])
        return result

    def has_unobserved_agents(self) -> bool:
        """Return whether the root still needs a terminal wait observation for any child."""
        return any(agent_id not in self._observed_agent_ids for agent_id in self._records)

    def is_unobserved_capacity_full(self) -> bool:
        """Return whether wait observations are required before another spawn is allowed."""
        unobserved_count = sum(agent_id not in self._observed_agent_ids for agent_id in self._records)
        return unobserved_count >= self.config.max_active_agents

    async def finish(self) -> WaitAgentResult:
        """Stop accepting children, await them, and reject non-successful exits."""
        async with self._lock:
            self._closing = True
            records = list(self._records.values())
            tasks = [record.task for record in records if record.task is not None]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        result = self._wait_result(records, timed_out=False)
        if result.failed or result.pending:
            raise SubagentRunFailed(result)
        return result

    async def abort(self) -> None:
        """Cancel unfinished children after a root exception or cancellation."""
        async with self._lock:
            self._closing = True
            records = list(self._records.values())
            tasks = [record.task for record in records if record.task is not None]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run(self, record: AgentRecord) -> None:
        try:
            child = self._run_child(record.identity, record.prompt)
            record.output = (
                await child
                if self.config.child_timeout_seconds is None
                else await asyncio.wait_for(child, timeout=self.config.child_timeout_seconds)
            )
        except asyncio.CancelledError as exc:
            await self._set_terminal_state(record, AgentStatus.CANCELLED, error=exc)
            raise
        except Exception as exc:
            await self._set_terminal_state(record, AgentStatus.FAILED, error=exc)
        else:
            if self._is_successful_output(record.output):
                await self._set_terminal_state(record, AgentStatus.COMPLETED)
            else:
                exit_status = getattr(record.output, 'exit_status', None)
                await self._set_terminal_state(
                    record,
                    AgentStatus.FAILED,
                    error=RuntimeError(f'child returned non-successful exit_status={exit_status!r}'),
                )

    async def _set_terminal_state(
        self, record: AgentRecord, status: AgentStatus, *, error: BaseException | None = None
    ) -> None:
        async with self._lock:
            record.status = status
            record.error = error
            record.completed_at = time.monotonic()
            event: dict[str, Any] = {'event': status.value, 'agent_id': record.identity.agent_id}
            if error is not None and status == AgentStatus.FAILED:
                event['error_type'] = type(error).__name__
            self._events.append(event)

    def get_record(self, agent_id: str) -> AgentRecord:
        """Return the record for an existing child."""
        try:
            return self._records[agent_id]
        except KeyError as exc:
            raise KeyError(f'unknown agent_id: {agent_id}') from exc

    def _wait_result(self, records: list[AgentRecord], *, timed_out: bool) -> WaitAgentResult:
        completed: list[AgentResult] = []
        failed: list[AgentResult] = []
        pending: list[str] = []
        for record in records:
            if record.status == AgentStatus.COMPLETED:
                completed.append(self._record_result(record))
            elif record.status in {AgentStatus.FAILED, AgentStatus.CANCELLED}:
                failed.append(self._record_result(record))
            else:
                pending.append(record.identity.agent_id)
        return WaitAgentResult(completed=completed, failed=failed, pending=pending, timed_out=timed_out)

    @staticmethod
    def _is_successful_output(output: Any) -> bool:
        return getattr(output, 'exit_status', None) in SUCCESSFUL_CHILD_EXIT_STATUSES

    @staticmethod
    def _record_result(record: AgentRecord) -> AgentResult:
        error = record.error
        return AgentResult(
            agent_id=record.identity.agent_id,
            name=record.name,
            status=record.status,
            final_answer=getattr(record.output, 'final_answer', None),
            exit_status=getattr(record.output, 'exit_status', None),
            error_type=type(error).__name__ if error is not None else None,
            error_message=str(error) if error is not None else None,
        )

    @staticmethod
    def _tool_call_audit(output: Any) -> list[dict[str, Any]]:
        audit: list[dict[str, Any]] = []
        for message in getattr(output, 'main_trajectory', None) or []:
            if not isinstance(message, dict):
                continue
            for tool_call in message.get('tool_calls') or []:
                function = tool_call.get('function') if isinstance(tool_call, dict) else None
                if not isinstance(function, dict):
                    continue
                arguments = function.get('arguments')
                if isinstance(arguments, str):
                    with suppress(json.JSONDecodeError):
                        arguments = json.loads(arguments)
                audit.append({'name': function.get('name'), 'arguments': arguments})
        return audit

    def summary(self) -> dict[str, Any]:
        """Return a serializable snapshot of child lifecycle state."""
        records = sorted(self._records.values(), key=lambda record: record.spawn_order)
        return {
            'spawned_agents': len(records),
            'completed_agents': sum(record.status == AgentStatus.COMPLETED for record in records),
            'failed_agents': sum(record.status == AgentStatus.FAILED for record in records),
            'cancelled_agents': sum(record.status == AgentStatus.CANCELLED for record in records),
            'pending_agents': sum(record.status == AgentStatus.RUNNING for record in records),
            'max_active_agents': self._max_active_agents,
            'events': [dict(event) for event in self._events],
            'agents': [
                {
                    'agent_id': record.identity.agent_id,
                    'name': record.name,
                    'prompt': record.prompt,
                    'spawn_order': record.spawn_order,
                    'status': record.status.value,
                    'final_answer': getattr(record.output, 'final_answer', None),
                    'exit_status': getattr(record.output, 'exit_status', None),
                    'tool_calls': self._tool_call_audit(record.output),
                }
                for record in records
            ],
        }


__all__ = [
    'AgentExecutionIdentity',
    'AgentResult',
    'AgentStatus',
    'MULTI_AGENT_CONTROLLER_STATE_KEY',
    'MULTI_AGENT_SUMMARY_STATE_KEY',
    'MultiAgentConfig',
    'SPAWN_AGENT_TOOL_NAME',
    'SpawnAgentResult',
    'SubagentConfig',
    'SubagentController',
    'SubagentRunFailed',
    'WAIT_AGENT_TOOL_NAME',
    'WaitAgentResult',
]
