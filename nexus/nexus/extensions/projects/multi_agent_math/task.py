"""Score multi-agent arithmetic runs with dependency-valid scheduling."""

from __future__ import annotations

import json
import math
import re

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from nexus.core import TaskRunInput, TaskRunOutput, WithAgentRunner, WithScore, WithUserPrompt, task_registry
from nexus.core.agent import WithMainTrajectory
from nexus.extensions.projects.calculator.task import extract_boxed_answer
from nexus.gateway.session_manager import session_manager
from nexus.gateway.trajectory import MAIN_CHAIN_ID, Trajectory


Operator = Literal['$', '@']
Category = Literal['serial', 'parallel', 'async']
TOOL_NAMES: dict[Operator, str] = {'$': 'compute_dollar', '@': 'compute_at'}
ATOMIC_OPERATION_PATTERN = re.compile(r'(?<!\d)(-?\d+)\s*([$@])\s*(-?\d+)(?!\d)')


class ExpectedOperation(BaseModel):
    """Describe one concrete custom operation and its dependency nodes."""

    node_id: str
    operator: Operator
    a: int
    b: int
    dependencies: list[str] = Field(default_factory=list)

    @field_validator('node_id')
    @classmethod
    def validate_node_id(cls, node_id: str) -> str:
        del cls
        return _validate_operation_id(node_id, field_name='node_id')

    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, dependencies: list[str]) -> list[str]:
        del cls
        return [_validate_operation_id(dependency, field_name='dependencies') for dependency in dependencies]


class RawExecutionChecks(BaseModel):
    """Expose hard acceptance components plus observational overlap."""

    answer_correct: bool
    topology_correct: bool
    tool_calls_correct: bool
    no_leaked_agents: bool
    observed_overlap: bool

    @property
    def passed(self) -> bool:
        """Return whether every hard acceptance component passed."""
        return all((
            self.answer_correct,
            self.topology_correct,
            self.tool_calls_correct,
            self.no_leaked_agents,
        ))

    @property
    def task_score(self) -> float:
        """Return the binary task reward without gating on physical overlap."""
        return float(self.passed)


class MultiAgentMathTaskInput(TaskRunInput, WithAgentRunner, WithUserPrompt):
    """Configure one deterministic expression-solving task."""

    ground_truth: str
    category: Category
    expected_dag: list[ExpectedOperation] = Field(min_length=1)
    expected_subagent_count: int = Field(ge=1)
    traj_id: str | None = None


class MultiAgentMathTaskOutput(TaskRunOutput, WithScore):
    """Return the answer, hard score components, and observed child overlap."""

    answer: str | None = None
    answer_correct: bool = False
    topology_correct: bool = False
    tool_calls_correct: bool = False
    no_leaked_agents: bool = False
    observed_overlap: bool = False
    extra_fields: dict[str, Any] = Field(default_factory=dict)


def _validate_operation_id(operation_id: str, *, field_name: str) -> str:
    if not isinstance(operation_id, str) or not operation_id.strip():
        raise ValueError(f'{field_name} must contain at least one non-whitespace character')
    return operation_id


def validate_expected_contract(
    expected_dag: list[ExpectedOperation],
    *,
    category: Category,
    expected_subagent_count: int,
) -> None:
    """Reject child-count mismatches and invalid category DAG shapes."""
    if expected_subagent_count != len(expected_dag) or expected_subagent_count <= 0:
        raise ValueError('multi-agent math contract requires child count to match expected_dag length')
    node_ids = [_validate_operation_id(operation.node_id, field_name='node_id') for operation in expected_dag]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError('multi-agent math contract requires unique operation node IDs')
    operation_signatures = {_operation_signature(operation) for operation in expected_dag}
    if len(operation_signatures) != len(expected_dag):
        raise ValueError('multi-agent math contract requires unique operation signatures')
    operation_results: dict[str, int] = {}
    for operation in expected_dag:
        if len(operation.dependencies) > 2 or len(set(operation.dependencies)) != len(operation.dependencies):
            raise ValueError('multi-agent math contract allows at most two unique dependencies per operation')
        for dependency in operation.dependencies:
            _validate_operation_id(dependency, field_name='dependencies')
            if dependency not in node_ids:
                raise ValueError(f'multi-agent math contract references unknown dependency {dependency!r}')
            if dependency not in operation_results:
                raise ValueError(
                    f'multi-agent math dependency {dependency!r} must reference an earlier operation in expected_dag'
                )
        if operation.dependencies and operation.a != operation_results[operation.dependencies[0]]:
            raise ValueError(
                f'multi-agent math operation {operation.node_id!r} left operand must equal its first dependency result'
            )
        if len(operation.dependencies) == 2 and operation.b != operation_results[operation.dependencies[1]]:
            raise ValueError(
                f'multi-agent math operation {operation.node_id!r} right operand must equal its second dependency result'
            )
        operation_results[operation.node_id] = _expected_result(operation)

    dependencies = [operation.dependencies for operation in expected_dag]
    if category == 'serial':
        expected_dependencies = [[]] + [[node_ids[index - 1]] for index in range(1, len(node_ids))]
        if dependencies != expected_dependencies:
            raise ValueError('multi-agent math serial contract requires one linear dependency chain')
    elif category == 'async':
        if any(dependencies):
            raise ValueError('multi-agent math async contract requires independent operations')
    else:
        root_node_ids = {operation.node_id for operation in expected_dag if not operation.dependencies}
        if len(root_node_ids) < 2:
            raise ValueError('multi-agent math parallel contract requires at least two independent root operations')
        referenced_node_ids = {dependency for operation in expected_dag for dependency in operation.dependencies}
        sink_node_ids = set(node_ids) - referenced_node_ids
        if sink_node_ids != {node_ids[-1]}:
            raise ValueError('multi-agent math parallel contract requires a single final sink containing every node')


def _operation_signature(operation: ExpectedOperation) -> tuple[str, int, int]:
    return TOOL_NAMES[operation.operator], operation.a, operation.b


def _expected_result(operation: ExpectedOperation) -> int:
    if operation.operator == '$':
        return (operation.a + 1) * (operation.b - 1)
    return 3 * operation.a - 2 * operation.b


def _tool_call_name(tool_call: Any) -> str | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get('function')
    if not isinstance(function, dict):
        return None
    name = function.get('name')
    return name if isinstance(name, str) else None


def _tool_call_arguments(tool_call: Any) -> dict[str, Any] | None:
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get('function')
    if not isinstance(function, dict):
        return None
    arguments = function.get('arguments')
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    return arguments if isinstance(arguments, dict) else None


def _raw_tool_calls(
    trajectory: Trajectory,
    chain_id: str,
) -> list[tuple[int, str | None, str | None, dict[str, Any] | None]]:
    calls: list[tuple[int, str | None, str | None, dict[str, Any] | None]] = []
    for index, node in enumerate(trajectory.iter_nodes()):
        message = node.original_message
        if trajectory.node_chain(node.node_id) != chain_id or message.get('role') != 'assistant':
            continue
        for tool_call in message.get('tool_calls') or []:
            call_id = tool_call.get('id') if isinstance(tool_call, dict) else None
            calls.append((
                index,
                call_id if isinstance(call_id, str) else None,
                _tool_call_name(tool_call),
                _tool_call_arguments(tool_call),
            ))
    return calls


def _get_unique_tool_observation(
    trajectory: Trajectory,
    chain_id: str,
    call_id: str,
) -> tuple[int, str] | None:
    observations: list[tuple[int, str]] = []
    for index, node in enumerate(trajectory.iter_nodes()):
        message = node.original_message
        content = message.get('content')
        if (
            trajectory.node_chain(node.node_id) == chain_id
            and message.get('role') == 'tool'
            and message.get('tool_call_id') == call_id
            and isinstance(content, str)
        ):
            observations.append((index, content))
    return observations[0] if len(observations) == 1 else None


def _parse_json_object(content: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _child_terminal_index(trajectory: Trajectory, chain_id: str) -> int | None:
    terminal_indices = [
        index
        for index, node in enumerate(trajectory.iter_nodes())
        if trajectory.node_chain(node.node_id) == chain_id
        and node.generated_by == 'llm'
        and node.original_message.get('role') == 'assistant'
        and not node.original_message.get('tool_calls')
        and not node.children
    ]
    return terminal_indices[0] if len(terminal_indices) == 1 else None


def _declared_subagent_ids(trajectory: Trajectory) -> list[str]:
    subagent_ids = [chain_id for chain_id in trajectory.chains if chain_id != MAIN_CHAIN_ID]
    if any(
        trajectory.chains[chain_id].kind != 'subagent' or trajectory.chains[chain_id].parent_chain_id != MAIN_CHAIN_ID
        for chain_id in subagent_ids
    ):
        return []
    return subagent_ids


def _spawned_agent_ids(trajectory: Trajectory) -> dict[str, str]:
    return {
        chain.spawn.tool_use_id: chain_id
        for chain_id, chain in trajectory.chains.items()
        if chain_id != MAIN_CHAIN_ID and chain.spawn is not None and chain.spawn.tool_use_id is not None
    }


def _joined_by(trajectory: Trajectory, agent_id: str) -> str | None:
    chain = trajectory.chains.get(agent_id)
    return chain.join.tool_use_id if chain is not None and chain.join is not None else None


def _audit_children(
    trajectory: Trajectory,
    expected_dag: list[ExpectedOperation],
    subagent_ids: list[str],
) -> tuple[bool, dict[str, str], dict[str, int]]:
    expected_by_signature = {_operation_signature(operation): operation for operation in expected_dag}
    if len(expected_by_signature) != len(expected_dag):
        return False, {}, {}
    operation_agents: dict[str, str] = {}
    terminal_indices: dict[str, int] = {}
    for agent_id in subagent_ids:
        calls = _raw_tool_calls(trajectory, agent_id)
        if len(calls) != 1:
            return False, {}, {}
        call_index, call_id, name, arguments = calls[0]
        if call_id is None or arguments is None or set(arguments) != {'a', 'b'}:
            return False, {}, {}
        a = arguments.get('a')
        b = arguments.get('b')
        if type(a) is not int or type(b) is not int:
            return False, {}, {}
        operation = expected_by_signature.get((name, a, b))
        if operation is None or operation.node_id in operation_agents:
            return False, {}, {}
        observation = _get_unique_tool_observation(trajectory, agent_id, call_id)
        terminal_index = _child_terminal_index(trajectory, agent_id)
        expected_result = _expected_result(operation)
        expected_observation = f'{operation.a} {operation.operator} {operation.b} = {expected_result}'
        if (
            observation is None
            or terminal_index is None
            or not call_index < observation[0] < terminal_index
            or observation[1] != expected_observation
        ):
            return False, {}, {}
        terminal_content = trajectory.nodes[
            next(
                node.node_id
                for node in trajectory.iter_nodes()
                if trajectory.node_chain(node.node_id) == agent_id
                and node.generated_by == 'llm'
                and node.original_message.get('role') == 'assistant'
                and not node.original_message.get('tool_calls')
                and not node.children
            )
        ].original_message.get('content')
        if not isinstance(terminal_content, str) or terminal_content.strip() != str(expected_result):
            return False, {}, {}
        operation_agents[operation.node_id] = agent_id
        terminal_indices[operation.node_id] = terminal_index
    return len(operation_agents) == len(expected_dag), operation_agents, terminal_indices


def _spawn_operation(
    arguments: dict[str, Any] | None, expected_dag: list[ExpectedOperation]
) -> ExpectedOperation | None:
    if arguments is None or set(arguments) not in ({'prompt'}, {'prompt', 'name'}):
        return None
    if 'name' in arguments and arguments['name'] is not None and not isinstance(arguments['name'], str):
        return None
    prompt = arguments.get('prompt')
    if not isinstance(prompt, str):
        return None
    matches = ATOMIC_OPERATION_PATTERN.findall(prompt.replace('(', ' ').replace(')', ' '))
    if len(matches) != 1:
        return None
    prompt_a, operator, prompt_b = matches[0]
    signature = TOOL_NAMES[operator], int(prompt_a), int(prompt_b)
    return next((operation for operation in expected_dag if _operation_signature(operation) == signature), None)


def _controls_follow_dependencies(
    controls: list[tuple[str, str | tuple[str, ...], int, int]],
    expected_dag: list[ExpectedOperation],
    operation_agents: dict[str, str],
) -> bool:
    expected_node_ids = {operation.node_id for operation in expected_dag}
    operation_by_id = {operation.node_id: operation for operation in expected_dag}
    operation_by_agent = {agent_id: node_id for node_id, agent_id in operation_agents.items()}
    spawned_node_ids: set[str] = set()
    waited_node_ids: set[str] = set()
    wait_observation_indices: dict[str, int] = {}
    for control_kind, control_value, call_index, observation_index in controls:
        if control_kind == 'spawn':
            if not isinstance(control_value, str) or control_value in spawned_node_ids:
                return False
            operation = operation_by_id.get(control_value)
            if operation is None or any(
                dependency not in waited_node_ids or wait_observation_indices[dependency] >= call_index
                for dependency in operation.dependencies
            ):
                return False
            spawned_node_ids.add(control_value)
            continue
        if control_kind != 'wait' or not isinstance(control_value, tuple):
            return False
        waited_operation_ids = [operation_by_agent.get(agent_id) for agent_id in control_value]
        if any(
            node_id is None or node_id not in spawned_node_ids or node_id in waited_node_ids
            for node_id in waited_operation_ids
        ):
            return False
        waited_node_ids.update(node_id for node_id in waited_operation_ids if node_id is not None)
        wait_observation_indices.update({
            node_id: observation_index for node_id in waited_operation_ids if node_id is not None
        })
    return spawned_node_ids == expected_node_ids and waited_node_ids == expected_node_ids


def _valid_wait_arguments(
    arguments: dict[str, Any] | None,
) -> tuple[list[str], Literal['all', 'any']] | None:
    if (
        arguments is None
        or not {'agent_ids'} <= set(arguments)
        or not set(arguments)
        <= {
            'agent_ids',
            'return_when',
            'timeout_seconds',
        }
    ):
        return None
    agent_ids = arguments.get('agent_ids')
    if (
        not isinstance(agent_ids, list)
        or not agent_ids
        or any(not isinstance(agent_id, str) or not agent_id for agent_id in agent_ids)
        or len(set(agent_ids)) != len(agent_ids)
    ):
        return None
    return_when = arguments.get('return_when', 'all')
    if return_when not in {'all', 'any'}:
        return None
    timeout_seconds = arguments.get('timeout_seconds')
    if timeout_seconds is not None and (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        return None
    return agent_ids, return_when


def _completed_wait_agent_ids(
    content: str,
    agent_ids: list[str],
    return_when: Literal['all', 'any'],
    operation_agents: dict[str, str],
    expected_dag: list[ExpectedOperation],
) -> tuple[str, ...] | None:
    payload = _parse_json_object(content)
    if payload is None or payload.get('failed') != [] or payload.get('timed_out') is not False:
        return None
    completed = payload.get('completed')
    pending = payload.get('pending')
    if not isinstance(completed, list) or not isinstance(pending, list):
        return None
    if any(not isinstance(agent_id, str) or not agent_id for agent_id in pending) or len(set(pending)) != len(pending):
        return None
    expected_results = {
        agent_id: str(_expected_result(operation))
        for operation in expected_dag
        for agent_id in [operation_agents.get(operation.node_id)]
        if agent_id is not None
    }
    seen: set[str] = set()
    for item in completed:
        if not isinstance(item, dict):
            return None
        agent_id = item.get('agent_id')
        if (
            agent_id not in agent_ids
            or agent_id in seen
            or item.get('status') != 'completed'
            or item.get('exit_status') not in {'finished', 'refused'}
            or item.get('final_answer') != expected_results.get(agent_id)
        ):
            return None
        seen.add(agent_id)
    if (
        seen & set(pending)
        or seen | set(pending) != set(agent_ids)
        or (return_when == 'all' and pending)
        or (return_when == 'any' and not seen)
    ):
        return None
    return tuple(item['agent_id'] for item in completed)


def _audit_root_control(
    trajectory: Trajectory,
    expected_dag: list[ExpectedOperation],
    operation_agents: dict[str, str],
    *,
    terminal_indices: dict[str, int],
) -> tuple[bool, bool, bool, int | None]:
    expected_node_ids = {operation.node_id for operation in expected_dag}
    if set(operation_agents) != expected_node_ids:
        return False, False, False, None
    operation_by_agent = {agent_id: node_id for node_id, agent_id in operation_agents.items()}
    spawned_agent_ids = _spawned_agent_ids(trajectory)
    calls = _raw_tool_calls(trajectory, MAIN_CHAIN_ID)
    if any(name not in {'spawn_agent', 'wait_agent'} for _, _, name, _ in calls):
        return False, False, False, None

    controls: list[tuple[str, str | tuple[str, ...], int, int]] = []
    spawn_ack_indices: dict[str, int] = {}
    root_calls_correct = True
    for call_index, call_id, name, arguments in calls:
        if call_id is None:
            root_calls_correct = False
            continue
        observation = _get_unique_tool_observation(trajectory, MAIN_CHAIN_ID, call_id)
        if observation is None or observation[0] <= call_index:
            root_calls_correct = False
            continue
        if name == 'spawn_agent':
            operation = _spawn_operation(arguments, expected_dag)
            agent_id = spawned_agent_ids.get(call_id)
            if (
                operation is None
                or agent_id is None
                or operation_agents.get(operation.node_id) != agent_id
                or operation.node_id in spawn_ack_indices
            ):
                root_calls_correct = False
                continue
            controls.append(('spawn', operation.node_id, call_index, observation[0]))
            spawn_ack_indices[operation.node_id] = observation[0]
        elif name == 'wait_agent':
            wait_arguments = _valid_wait_arguments(arguments)
            if wait_arguments is None:
                root_calls_correct = False
                continue
            agent_ids, return_when = wait_arguments
            requested_node_ids = [operation_by_agent.get(agent_id) for agent_id in agent_ids]
            if any(
                node_id is None or node_id not in spawn_ack_indices or spawn_ack_indices[node_id] >= call_index
                for node_id in requested_node_ids
            ):
                root_calls_correct = False
                continue
            completed_agent_ids = _completed_wait_agent_ids(
                observation[1], agent_ids, return_when, operation_agents, expected_dag
            )
            if completed_agent_ids is None or any(
                _joined_by(trajectory, agent_id) != call_id for agent_id in completed_agent_ids
            ):
                root_calls_correct = False
                continue
            controls.append(('wait', completed_agent_ids, call_index, observation[0]))
        else:
            root_calls_correct = False

    topology_correct = _controls_follow_dependencies(controls, expected_dag, operation_agents)
    spawned_node_ids = [
        control_value
        for control_kind, control_value, _, _ in controls
        if control_kind == 'spawn' and isinstance(control_value, str)
    ]
    observed_overlap = any(
        spawn_ack_indices[later_node_id] < terminal_indices[earlier_node_id]
        for earlier_index, earlier_node_id in enumerate(spawned_node_ids)
        for later_node_id in spawned_node_ids[earlier_index + 1 :]
    )
    last_wait_observation_index = max(
        (observation_index for control_kind, _, _, observation_index in controls if control_kind == 'wait'),
        default=None,
    )
    return root_calls_correct, topology_correct, observed_overlap, last_wait_observation_index


def _answer_correct(trajectory: Trajectory, ground_truth: str, *, after_index: int | None) -> bool:
    if after_index is None:
        return False
    final_contents: list[str] = []
    for index, node in enumerate(trajectory.iter_nodes()):
        content = node.original_message.get('content')
        if (
            index > after_index
            and trajectory.node_chain(node.node_id) == MAIN_CHAIN_ID
            and node.generated_by == 'llm'
            and node.original_message.get('role') == 'assistant'
            and not node.original_message.get('tool_calls')
            and isinstance(content, str)
        ):
            final_contents.append(content)
    answer = extract_boxed_answer(final_contents[-1]) if final_contents else None
    return answer is not None and answer.strip() == ground_truth.strip()


def validate_raw_multi_agent_execution(
    trajectory: Trajectory,
    expected_dag: list[ExpectedOperation],
    *,
    category: Category,
    expected_subagent_count: int,
    ground_truth: str,
    require_completed: bool = True,
) -> RawExecutionChecks:
    """Audit one canonical trajectory without requiring a specific legal schedule."""
    validate_expected_contract(
        expected_dag,
        category=category,
        expected_subagent_count=expected_subagent_count,
    )
    subagent_ids = _declared_subagent_ids(trajectory)
    children_correct, operation_agents, terminal_indices = _audit_children(
        trajectory,
        expected_dag,
        subagent_ids,
    )
    root_calls_correct, topology_correct, observed_overlap, last_wait_observation_index = _audit_root_control(
        trajectory,
        expected_dag,
        operation_agents,
        terminal_indices=terminal_indices,
    )
    main_chain = trajectory.chains.get(MAIN_CHAIN_ID)
    no_leaked_agents = (
        main_chain is not None
        and main_chain.kind == 'main'
        and len(subagent_ids) == expected_subagent_count
        and set(trajectory.chains) == {MAIN_CHAIN_ID, *subagent_ids}
        and len(terminal_indices) == expected_subagent_count
        and (not require_completed or trajectory.state == 'completed')
        and (not require_completed or trajectory.extra.get('hytuner', {}).get('success') is True)
    )
    return RawExecutionChecks(
        answer_correct=_answer_correct(trajectory, ground_truth, after_index=last_wait_observation_index),
        topology_correct=root_calls_correct and topology_correct,
        tool_calls_correct=children_correct and root_calls_correct,
        no_leaked_agents=no_leaked_agents,
        observed_overlap=observed_overlap,
    )


def _summary_cleanup_correct(summary: dict[str, Any], expected_subagent_count: int) -> bool:
    agents = summary.get('agents')
    return (
        isinstance(agents, list)
        and len(agents) == expected_subagent_count
        and summary.get('spawned_agents') == expected_subagent_count
        and summary.get('completed_agents') == expected_subagent_count
        and summary.get('failed_agents') == 0
        and summary.get('cancelled_agents') == 0
        and summary.get('pending_agents') == 0
        and all(isinstance(agent, dict) and agent.get('status') == 'completed' for agent in agents)
    )


@task_registry.register('multi_agent_math_task')
async def run_multi_agent_math_task(task_input: MultiAgentMathTaskInput) -> MultiAgentMathTaskOutput:
    """Run and score one expression-only multi-agent math task."""
    validate_expected_contract(
        task_input.expected_dag,
        category=task_input.category,
        expected_subagent_count=task_input.expected_subagent_count,
    )
    agent_output = await task_input.run_agent(user_prompt=task_input.user_prompt)
    if not isinstance(agent_output, WithMainTrajectory):
        raise RuntimeError('multi_agent_math_task requires an agent output with main_trajectory')
    if task_input.traj_id is None:
        raise ValueError('multi_agent_math_task requires Gateway traj_id for raw trajectory audit')
    session = session_manager.get_session(task_input.traj_id)
    if session is None:
        raise RuntimeError(
            f'multi_agent_math_task requires an active Gateway session for traj_id={task_input.traj_id!r}'
        )
    raw_checks = validate_raw_multi_agent_execution(
        session.get_trajectory(),
        task_input.expected_dag,
        category=task_input.category,
        expected_subagent_count=task_input.expected_subagent_count,
        ground_truth=task_input.ground_truth,
        require_completed=False,
    )
    final_content = next(
        (
            message.get('content')
            for message in reversed(agent_output.main_trajectory)
            if message.get('role') == 'assistant' and message.get('content') and not message.get('tool_calls')
        ),
        None,
    )
    answer = extract_boxed_answer(final_content) if isinstance(final_content, str) else None
    summary = getattr(agent_output, 'multi_agent_summary', None)
    summary = summary if isinstance(summary, dict) else {}
    components = {
        'answer_correct': raw_checks.answer_correct and answer == task_input.ground_truth,
        'topology_correct': raw_checks.topology_correct,
        'tool_calls_correct': raw_checks.tool_calls_correct,
        'no_leaked_agents': raw_checks.no_leaked_agents
        and _summary_cleanup_correct(summary, task_input.expected_subagent_count),
    }
    return MultiAgentMathTaskOutput(
        score=float(all(components.values())),
        answer=answer,
        observed_overlap=raw_checks.observed_overlap,
        **components,
        extra_fields={
            'reward_extra_info': {
                **{name: float(value) for name, value in components.items()},
                'observed_overlap': float(raw_checks.observed_overlap),
                'category': task_input.category,
            },
            'multi_agent_summary': summary,
        },
    )
