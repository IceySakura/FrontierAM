import pytest

from nexus.extensions.projects.r2_stem.generator import _build_config
from nexus.runner.single import RunInput


IMAGE = 'registry.example/verifier@sha256:' + 'b' * 64


def _base_config() -> RunInput:
    return RunInput.model_validate(
        {
            'task_name': 'science_agent_task',
            'task_config': {'question': '', 'evaluation_method': 'unit_cases'},
            'agent_name': 'terminus2',
            'runtime_provider_name': 'gongfeng',
            'runtime_configs': {'main': {'image': 'registry.example/main:fixed'}},
        }
    )


def test_generator_keeps_existing_unit_cases_shape() -> None:
    config = _build_config(
        _base_config(),
        {
            'question': 'Solve it.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'unit_cases',
            'task_dir': '/host/task',
        },
        'sample-1',
    )

    assert config.task_config['evaluation_method'] == 'unit_cases'
    assert config.runtime_configs == {'main': {'image': 'registry.example/main:fixed'}}


def test_generator_builds_named_verifier_runtime_from_flat_row() -> None:
    config = _build_config(
        _base_config(),
        {
            'question': 'Create a patch.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'verifier_image',
            'task_dir': '/host/task',
            'working_dir': '/app',
            'verifier_image': IMAGE,
            'verifier_dind': False,
            'verifier_mounts': ['/app/output.patch'],
            'verifier_resource_spec': {'cpu': 4, 'memory': 16},
            'verifier_runtime_config': {'working_dir': '/verifier-work'},
        },
        'sample-1',
    )

    assert config.task_config['evaluation_method'] == 'verifier_image'
    assert config.task_config['verifier_mounts'] == ['/app/output.patch']
    # Direct execution reads the tests off the rootfs, so it needs neither dind nor egress.
    assert config.runtime_configs['verifier'] == {
        'image': IMAGE,
        'enable_dind': False,
        'working_dir': '/verifier-work',
        'network_policy': {'default_action': 'deny', 'egress': []},
        'resource_spec': {'cpu': 4, 'memory': 16},
    }


def test_generator_enables_dind_and_registry_egress_from_the_row() -> None:
    config = _build_config(
        _base_config(),
        {
            'question': 'Create a patch.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'verifier_image',
            'verifier_image': IMAGE,
            'verifier_dind': True,
        },
        'sample-1',
    )

    assert config.task_config['verifier_dind'] is True
    assert config.runtime_configs['verifier']['enable_dind'] is True
    assert config.runtime_configs['verifier']['network_policy'] == {
        'default_action': 'deny',
        'egress': [{'action': 'allow', 'target': 'registry.example:443'}],
    }


def test_generator_passes_per_row_budget_and_cutoff_switches_through() -> None:
    config = _build_config(
        _base_config(),
        {
            'question': 'Create a patch.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'verifier_image',
            'verifier_image': IMAGE,
            'verifier_score_truncated': True,
            'stop_at_trajectory_budget': True,
            'agent_wall_time_reserve_seconds': 300.0,
        },
        'sample-1',
    )

    assert config.task_config['verifier_score_truncated'] is True
    assert config.task_config['stop_at_trajectory_budget'] is True
    assert config.task_config['agent_wall_time_reserve_seconds'] == 300.0


def test_generator_ignores_retired_verifier_direct_row_field() -> None:
    config = _build_config(
        _base_config(),
        {
            'question': 'Create a patch.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'verifier_image',
            'verifier_image': IMAGE,
            # Existing datasets still carry this; it must not reach task_config.
            'verifier_direct': False,
        },
        'sample-1',
    )

    assert 'verifier_direct' not in config.task_config


def test_generator_mirrors_runner_agent_into_task_config() -> None:
    config = _build_config(
        _base_config(),
        {
            'question': 'Solve it.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'unit_cases',
        },
        'sample-1',
    )

    assert config.task_config['agent_name'] == 'terminus2'


def test_generator_rejects_task_config_agent_contradicting_runner() -> None:
    base_config = _base_config()
    base_config.task_config['agent_name'] = 'mini_swe_agent_stem'

    with pytest.raises(ValueError, match='does not match agent_name'):
        _build_config(
            base_config,
            {
                'question': 'Solve it.',
                'image': 'registry.example/main@sha256:' + 'a' * 64,
                'evaluation_method': 'unit_cases',
            },
            'sample-1',
        )


def test_generator_requires_an_agent_name() -> None:
    base_config = _base_config()
    base_config.agent_name = None

    with pytest.raises(KeyError, match='agent_name'):
        _build_config(
            base_config,
            {
                'question': 'Solve it.',
                'image': 'registry.example/main@sha256:' + 'a' * 64,
                'evaluation_method': 'unit_cases',
            },
            'sample-1',
        )


def test_generator_requires_verifier_image_for_new_mode() -> None:
    with pytest.raises(KeyError, match='verifier_image'):
        _build_config(
            _base_config(),
            {
                'question': 'Create a patch.',
                'image': 'registry.example/main@sha256:' + 'a' * 64,
                'evaluation_method': 'verifier_image',
            },
            'sample-1',
        )


def test_generator_raises_request_timeout_above_the_agent_budget() -> None:
    base_config = _base_config()
    base_config.agent_config = {'timeout': 43200}
    base_config.runtime_provider_config = {'request_timeout': 14400.0}

    config = _build_config(
        base_config,
        {
            'question': 'Solve it.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'unit_cases',
            'agent_timeout': 43200,
        },
        'sample-1',
    )

    assert config.runtime_provider_config['request_timeout'] == 43800.0


def test_generator_keeps_a_request_timeout_already_above_the_agent_budget() -> None:
    base_config = _base_config()
    base_config.runtime_provider_config = {'request_timeout': 86400.0}

    config = _build_config(
        base_config,
        {
            'question': 'Solve it.',
            'image': 'registry.example/main@sha256:' + 'a' * 64,
            'evaluation_method': 'unit_cases',
            'agent_timeout': 43200,
        },
        'sample-1',
    )

    assert config.runtime_provider_config['request_timeout'] == 86400.0


def test_generator_rejects_compose_as_post_agent_verifier_mode() -> None:
    with pytest.raises(NotImplementedError, match='Compose lifecycle'):
        _build_config(
            _base_config(),
            {
                'question': 'Repair all services.',
                'image': 'registry.example/main@sha256:' + 'a' * 64,
                'evaluation_method': 'verifier_image',
                'verifier_image': IMAGE,
                'compose_enabled': True,
            },
            'sample-1',
        )
