"""Behavioral tests for expression-only multi-agent math acceptance."""

from __future__ import annotations

import json

from typing import Any, Literal

import pytest

import nexus.extensions.projects.multi_agent_math  # noqa: F401

from nexus.core import EntryNotFoundError, tool_registry
from nexus.extensions.projects.multi_agent_math.task import (
    Category,
    ExpectedOperation,
    validate_expected_contract,
    validate_raw_multi_agent_execution,
)
from nexus.gateway.trajectory import Chain, JoinLink, SpawnLink, Trajectory, TrajectoryNode


def _tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {'id': call_id, 'type': 'function', 'function': {'name': name, 'arguments': arguments}}


def _expectations(category: Category) -> tuple[list[ExpectedOperation], str]:
    if category == 'async':
        operations = [
            ExpectedOperation(node_id='n1', operator='@', a=3, b=2),
            ExpectedOperation(node_id='n2', operator='$', a=4, b=3),
            ExpectedOperation(node_id='n3', operator='@', a=8, b=1),
            ExpectedOperation(node_id='n4', operator='$', a=6, b=2),
        ]
        return operations, '44'
    if category == 'parallel':
        operations = [
            ExpectedOperation(node_id='n1', operator='@', a=3, b=2),
            ExpectedOperation(node_id='n2', operator='$', a=4, b=3),
            ExpectedOperation(node_id='n3', operator='@', a=8, b=1),
            ExpectedOperation(node_id='n4', operator='@', a=5, b=10, dependencies=['n1', 'n2']),
            ExpectedOperation(node_id='n5', operator='$', a=-5, b=22, dependencies=['n4', 'n3']),
            ExpectedOperation(node_id='n6', operator='@', a=-84, b=4, dependencies=['n5']),
        ]
        return operations, '-260'
    operations = [
        ExpectedOperation(node_id='n1', operator='@', a=3, b=2),
        ExpectedOperation(node_id='n2', operator='$', a=5, b=4, dependencies=['n1']),
        ExpectedOperation(node_id='n3', operator='@', a=18, b=7, dependencies=['n2']),
        ExpectedOperation(node_id='n4', operator='$', a=40, b=3, dependencies=['n3']),
        ExpectedOperation(node_id='n5', operator='@', a=82, b=10, dependencies=['n4']),
    ]
    return operations, '226'


def _raw_fixture(
    category: Category,
    *,
    serialize_ready_operations: bool = False,
    batch_dependency_wait_and_spawn: bool = False,
    partial_any_wait: bool = False,
    early_final_before_last_wait: bool = False,
    pending_before_spawn: bool = False,
) -> Trajectory:
    operations, ground_truth = _expectations(category)
    nodes: dict[int, TrajectoryNode] = {}
    chains: dict[str, Chain] = {'main': Chain(chain_id='main', kind='main', roots=[])}
    last_node: dict[str, int] = {}

    def add(
        chain_id: str,
        message: dict[str, Any],
        generated_by: Literal['llm', 'observation', 'client'],
    ) -> int:
        node_id = len(nodes)
        parent = last_node.get(chain_id)
        nodes[node_id] = TrajectoryNode(
            node_id=node_id,
            message_id=node_id,
            parent=parent,
            generated_by=generated_by,
            children=[],
            original_message=message,
        )
        if parent is None:
            chains[chain_id].roots.append(node_id)
        else:
            nodes[parent].children.append(node_id)
        last_node[chain_id] = node_id
        return node_id

    def root_call(
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        result: dict[str, Any],
        *,
        content: str | None = None,
    ) -> tuple[int, int]:
        message = {'role': 'assistant', 'tool_calls': [_tool_call(call_id, name, arguments)]}
        if content is not None:
            message['content'] = content
        call_node = add('main', message, 'llm')
        observation_node = add(
            'main',
            {'role': 'tool', 'tool_call_id': call_id, 'content': json.dumps(result)},
            'observation',
        )
        return call_node, observation_node

    def declare_child(agent_id: str, spawn_node: int, spawn_call_id: str) -> None:
        chains[agent_id] = Chain(
            chain_id=agent_id,
            kind='subagent',
            parent_chain_id='main',
            roots=[],
            spawn=SpawnLink(node=spawn_node, tool_use_id=spawn_call_id),
            evidence={'attributes': {'parent_chain_id': 'main'}},
        )

    def spawn(index: int) -> str:
        operation = operations[index - 1]
        agent_id = f'agent-{index}'
        prompt = f'Evaluate ({operation.a} {operation.operator} {operation.b}).'
        call_node, _ = root_call(
            f'spawn-{index}',
            'spawn_agent',
            {'prompt': prompt},
            {'agent_id': agent_id, 'name': None, 'status': 'running'},
        )
        declare_child(agent_id, call_node, f'spawn-{index}')
        return agent_id

    def compute(index: int) -> None:
        operation = operations[index - 1]
        agent_id = f'agent-{index}'
        result = (
            (operation.a + 1) * (operation.b - 1) if operation.operator == '$' else 3 * operation.a - 2 * operation.b
        )
        tool_name = 'compute_dollar' if operation.operator == '$' else 'compute_at'
        add(
            agent_id,
            {
                'role': 'assistant',
                'tool_calls': [_tool_call(f'compute-{index}', tool_name, {'a': operation.a, 'b': operation.b})],
            },
            'llm',
        )
        add(
            agent_id,
            {
                'role': 'tool',
                'tool_call_id': f'compute-{index}',
                'content': f'{operation.a} {operation.operator} {operation.b} = {result}',
            },
            'observation',
        )
        add(agent_id, {'role': 'assistant', 'content': str(result)}, 'llm')

    def wait(
        indices: list[int],
        call_index: int,
        *,
        return_when: Literal['all', 'any'] = 'all',
        completed_indices: list[int] | None = None,
        content: str | None = None,
    ) -> None:
        completed_indices = indices if completed_indices is None else completed_indices
        completed = []
        for index in completed_indices:
            operation = operations[index - 1]
            result = (
                (operation.a + 1) * (operation.b - 1)
                if operation.operator == '$'
                else 3 * operation.a - 2 * operation.b
            )
            completed.append({
                'agent_id': f'agent-{index}',
                'name': None,
                'status': 'completed',
                'final_answer': str(result),
                'exit_status': 'finished',
                'error_type': None,
                'error_message': None,
            })
        arguments: dict[str, Any] = {'agent_ids': [f'agent-{index}' for index in indices]}
        if return_when != 'all':
            arguments['return_when'] = return_when
        _, observation_node = root_call(
            f'wait-{call_index}',
            'wait_agent',
            arguments,
            {
                'completed': completed,
                'failed': [],
                'pending': [f'agent-{index}' for index in indices if index not in completed_indices],
                'timed_out': False,
            },
            content=content,
        )
        for index in completed_indices:
            if f'agent-{index}' in chains:
                chains[f'agent-{index}'].join = JoinLink(
                    node=observation_node, tool_use_id=f'wait-{call_index}', kind='poll'
                )

    if batch_dependency_wait_and_spawn:
        spawn(1)
        compute(1)
        first_operation = operations[0]
        second_operation = operations[1]
        first_result = 3 * first_operation.a - 2 * first_operation.b
        batch_node = add(
            'main',
            {
                'role': 'assistant',
                'tool_calls': [
                    _tool_call('wait-1', 'wait_agent', {'agent_ids': ['agent-1']}),
                    _tool_call(
                        'spawn-2',
                        'spawn_agent',
                        {'prompt': f'Evaluate ({second_operation.a} {second_operation.operator} {second_operation.b}).'},
                    ),
                ],
            },
            'llm',
        )
        wait_observation_node = add(
            'main',
            {
                'role': 'tool',
                'tool_call_id': 'wait-1',
                'content': json.dumps({
                    'completed': [
                        {
                            'agent_id': 'agent-1',
                            'name': None,
                            'status': 'completed',
                            'final_answer': str(first_result),
                            'exit_status': 'finished',
                            'error_type': None,
                            'error_message': None,
                        }
                    ],
                    'failed': [],
                    'pending': [],
                    'timed_out': False,
                }),
            },
            'observation',
        )
        add(
            'main',
            {
                'role': 'tool',
                'tool_call_id': 'spawn-2',
                'content': json.dumps({'agent_id': 'agent-2', 'name': None, 'status': 'running'}),
            },
            'observation',
        )
        chains['agent-1'].join = JoinLink(node=wait_observation_node, tool_use_id='wait-1', kind='poll')
        declare_child('agent-2', batch_node, 'spawn-2')
        for index in range(2, len(operations) + 1):
            if index > 2:
                spawn(index)
            compute(index)
            wait([index], index)
        add('main', {'role': 'assistant', 'content': f'The answer is \\boxed{{{ground_truth}}}.'}, 'llm')
        return Trajectory(
            session_id=f'fixture-{category}',
            nodes=nodes,
            roots=[root for chain in chains.values() for root in chain.roots],
            chains=chains,
            extra={'hytuner': {'success': True}},
        )

    if serialize_ready_operations:
        for index in range(1, len(operations) + 1):
            spawn(index)
            compute(index)
            wait([index], index)
    elif category == 'parallel':
        node_layers: dict[str, int] = {}
        operation_layers: list[list[int]] = []
        for index, operation in enumerate(operations, start=1):
            layer_index = (
                0
                if not operation.dependencies
                else 1 + max(node_layers[dependency] for dependency in operation.dependencies)
            )
            while len(operation_layers) <= layer_index:
                operation_layers.append([])
            operation_layers[layer_index].append(index)
            node_layers[operation.node_id] = layer_index
        for layer_index, operation_indices in enumerate(operation_layers, start=1):
            for index in operation_indices:
                spawn(index)
            if partial_any_wait and layer_index == 1:
                compute(operation_indices[0])
                wait(
                    operation_indices,
                    layer_index,
                    return_when='any',
                    completed_indices=operation_indices[:1],
                )
                for index in operation_indices[1:]:
                    compute(index)
                wait(operation_indices[1:], 100 + layer_index)
            else:
                for index in operation_indices:
                    compute(index)
                wait(operation_indices, layer_index)
    elif category == 'async':
        for index in range(1, len(operations) + 1):
            spawn(index)
        for index in range(1, len(operations) + 1):
            compute(index)
        wait(list(range(1, len(operations) + 1)), 1)
    else:
        start_index = 1
        if pending_before_spawn:
            spawn(1)
            compute(1)
            wait([1, 2], 100, return_when='any', completed_indices=[1])
            start_index = 2
        for index in range(start_index, len(operations) + 1):
            spawn(index)
            compute(index)
            if early_final_before_last_wait and index == len(operations):
                add('main', {'role': 'assistant', 'content': f'The answer is \\boxed{{{ground_truth}}}.'}, 'llm')
                add('main', {'role': 'user', 'content': 'Continue waiting for the remaining child.'}, 'client')
            wait([index], index)

    if not early_final_before_last_wait:
        add('main', {'role': 'assistant', 'content': f'The answer is \\boxed{{{ground_truth}}}.'}, 'llm')
    return Trajectory(
        session_id=f'fixture-{category}',
        nodes=nodes,
        roots=[root for chain in chains.values() for root in chain.roots],
        chains=chains,
        extra={'hytuner': {'success': True}},
    )


def _audit(trajectory: Trajectory, category: Category) -> Any:
    operations, ground_truth = _expectations(category)
    return validate_raw_multi_agent_execution(
        trajectory,
        operations,
        category=category,
        expected_subagent_count=len(operations),
        ground_truth=ground_truth,
    )


def _find_tool_call(trajectory: Trajectory, name: str, *, chain_id: str = 'main') -> dict[str, Any]:
    return next(
        call
        for node in trajectory.iter_nodes()
        if trajectory.node_chain(node.node_id) == chain_id
        for call in node.original_message.get('tool_calls') or []
        if call['function']['name'] == name
    )


def test_project_registers_only_spawn_wait_and_arithmetic_tools() -> None:
    assert tool_registry.get('multi_agent/spawn_agent').__tool_meta__.name == 'spawn_agent'
    assert tool_registry.get('multi_agent/wait_agent').__tool_meta__.name == 'wait_agent'
    assert tool_registry.get('multi_agent_math/compute_at').__tool_meta__.name == 'compute_at'
    assert tool_registry.get('multi_agent_math/compute_dollar').__tool_meta__.name == 'compute_dollar'
    for removed_tool in (
        'multi_agent/send_message',
        'multi_agent_math/release_mailbox',
        'multi_agent_math/wait_for_mailbox',
    ):
        with pytest.raises(EntryNotFoundError):
            tool_registry.get(removed_tool)


@pytest.mark.parametrize('category', ['serial', 'parallel', 'async'])
def test_raw_audit_accepts_each_variable_size_contract(category: Category) -> None:
    checks = _audit(_raw_fixture(category), category)

    assert checks.passed is True
    assert checks.task_score == 1.0


def test_parallel_audit_accepts_a_serialized_dependency_legal_schedule() -> None:
    checks = _audit(_raw_fixture('parallel', serialize_ready_operations=True), 'parallel')

    assert checks.passed is True
    assert checks.task_score == 1.0


def test_async_audit_does_not_require_a_model_visible_main_work_marker() -> None:
    checks = _audit(_raw_fixture('async'), 'async')

    assert checks.passed is True
    assert checks.task_score == 1.0


def test_async_score_does_not_require_concurrent_children() -> None:
    checks = _audit(_raw_fixture('async', serialize_ready_operations=True), 'async')

    assert checks.observed_overlap is False
    assert checks.task_score == 1.0


def test_async_audit_reports_concurrent_children_without_scoring_them() -> None:
    checks = _audit(_raw_fixture('async'), 'async')

    assert checks.observed_overlap is True
    assert checks.task_score == 1.0


def test_audit_rejects_a_dependent_spawn_batched_with_its_dependency_wait() -> None:
    checks = _audit(_raw_fixture('serial', batch_dependency_wait_and_spawn=True), 'serial')

    assert checks.topology_correct is False
    assert checks.task_score == 0.0


def test_single_agent_wait_accepts_return_when_any() -> None:
    trajectory = _raw_fixture('serial')
    for node in trajectory.iter_nodes():
        for tool_call in node.original_message.get('tool_calls') or []:
            if tool_call['function']['name'] == 'wait_agent':
                tool_call['function']['arguments']['return_when'] = 'any'

    assert _audit(trajectory, 'serial').task_score == 1.0


def test_parallel_audit_accepts_multi_agent_any_wait_with_pending_follow_up() -> None:
    checks = _audit(_raw_fixture('parallel', partial_any_wait=True), 'parallel')

    assert checks.passed is True
    assert checks.task_score == 1.0


def test_audit_rejects_pending_agent_id_requested_before_spawn() -> None:
    checks = _audit(_raw_fixture('serial', pending_before_spawn=True), 'serial')

    assert checks.topology_correct is False
    assert checks.task_score == 0.0


def test_audit_rejects_root_answer_before_final_wait_observation() -> None:
    checks = _audit(_raw_fixture('serial', early_final_before_last_wait=True), 'serial')

    assert checks.answer_correct is False
    assert checks.task_score == 0.0


def test_audit_accepts_root_answer_after_final_wait_observation() -> None:
    checks = _audit(_raw_fixture('serial'), 'serial')

    assert checks.answer_correct is True
    assert checks.task_score == 1.0


def test_raw_audit_rejects_extra_root_control_call() -> None:
    trajectory = _raw_fixture('serial')
    final_node = list(trajectory.iter_nodes())[-1]
    final_node.original_message['tool_calls'] = [
        _tool_call('spawn-extra', 'spawn_agent', {'prompt': 'Evaluate (1 @ 1).'})
    ]

    assert _audit(trajectory, 'serial').tool_calls_correct is False


def test_raw_audit_rejects_composite_child_prompt() -> None:
    trajectory = _raw_fixture('serial')
    spawn_call = _find_tool_call(trajectory, 'spawn_agent')
    spawn_call['function']['arguments']['prompt'] = 'Evaluate (3 @ 2) and then (4 $ 3).'

    assert _audit(trajectory, 'serial').tool_calls_correct is False


def test_raw_audit_rejects_changed_child_arguments() -> None:
    trajectory = _raw_fixture('serial')
    compute_call = _find_tool_call(trajectory, 'compute_at', chain_id='agent-1')
    compute_call['function']['arguments']['a'] = 4

    assert _audit(trajectory, 'serial').tool_calls_correct is False


def test_raw_audit_rejects_changed_arithmetic_observation() -> None:
    trajectory = _raw_fixture('serial')
    observation = next(
        node
        for node in trajectory.iter_nodes()
        if trajectory.node_chain(node.node_id) == 'agent-1' and node.original_message.get('role') == 'tool'
    )
    observation.original_message['content'] = '3 @ 2 = 6'

    assert _audit(trajectory, 'serial').tool_calls_correct is False


def test_raw_audit_rejects_unexpected_declared_child() -> None:
    trajectory = _raw_fixture('serial')
    trajectory.chains['agent-6'] = Chain(
        chain_id='agent-6',
        kind='subagent',
        parent_chain_id='main',
        evidence={'attributes': {'parent_chain_id': 'main'}},
    )
    trajectory._invalidate_caches()

    assert _audit(trajectory, 'serial').no_leaked_agents is False


def test_raw_audit_rejects_spawn_call_without_a_spawn_link() -> None:
    trajectory = _raw_fixture('serial')
    trajectory.chains['agent-2'].spawn = None

    checks = _audit(trajectory, 'serial')

    assert checks.tool_calls_correct is False
    assert checks.task_score == 0.0


def test_raw_audit_rejects_wait_result_whose_join_link_points_elsewhere() -> None:
    trajectory = _raw_fixture('serial')
    trajectory.chains['agent-2'].join = JoinLink(tool_use_id='wait-3', kind='poll')

    checks = _audit(trajectory, 'serial')

    assert checks.tool_calls_correct is False
    assert checks.task_score == 0.0


def test_raw_audit_does_not_depend_on_chain_declaration_order() -> None:
    trajectory = _raw_fixture('parallel')
    trajectory.chains = {
        'main': trajectory.chains['main'],
        **dict(reversed([(chain_id, chain) for chain_id, chain in trajectory.chains.items() if chain_id != 'main'])),
    }
    trajectory._invalidate_caches()

    assert _audit(trajectory, 'parallel').task_score == 1.0


def test_raw_audit_rejects_wrong_final_answer() -> None:
    trajectory = _raw_fixture('serial')
    list(trajectory.iter_nodes())[-1].original_message['content'] = 'The answer is \\boxed{41}.'

    assert _audit(trajectory, 'serial').answer_correct is False


@pytest.mark.parametrize('invalid_node_id', ['', ' ', '\t'])
def test_expected_contract_rejects_blank_operation_id(invalid_node_id: str) -> None:
    operations, _ = _expectations('serial')
    operations[0] = operations[0].model_copy(update={'node_id': invalid_node_id})

    with pytest.raises(ValueError, match='node_id'):
        validate_expected_contract(operations, category='serial', expected_subagent_count=len(operations))


def test_expected_contract_rejects_category_dependency_mismatch() -> None:
    operations, _ = _expectations('serial')

    with pytest.raises(ValueError, match='parallel'):
        validate_expected_contract(operations, category='parallel', expected_subagent_count=len(operations))


def test_expected_contract_accepts_a_three_operation_parallel_dag() -> None:
    operations = [
        ExpectedOperation(node_id='n1', operator='@', a=3, b=2),
        ExpectedOperation(node_id='n2', operator='$', a=4, b=3),
        ExpectedOperation(node_id='n3', operator='@', a=5, b=10, dependencies=['n1', 'n2']),
    ]

    validate_expected_contract(operations, category='parallel', expected_subagent_count=3)


def test_expected_contract_rejects_subagent_count_mismatch() -> None:
    operations, _ = _expectations('serial')

    with pytest.raises(ValueError, match='child count'):
        validate_expected_contract(operations, category='serial', expected_subagent_count=len(operations) - 1)


def test_parallel_contract_rejects_a_node_outside_the_final_dag() -> None:
    operations, _ = _expectations('parallel')
    operations[-1] = operations[-1].model_copy(update={'a': 22, 'dependencies': ['n3']})

    with pytest.raises(ValueError, match='single final sink'):
        validate_expected_contract(operations, category='parallel', expected_subagent_count=len(operations))


def test_contract_rejects_a_dependency_that_appears_later() -> None:
    operations, _ = _expectations('serial')
    operations[1] = operations[1].model_copy(update={'dependencies': ['n3']})

    with pytest.raises(ValueError, match='earlier operation'):
        validate_expected_contract(operations, category='serial', expected_subagent_count=len(operations))
