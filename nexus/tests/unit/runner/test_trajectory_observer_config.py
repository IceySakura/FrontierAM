from nexus.runner.single import ModuleObjectConfig, RunInput


def test_run_input_parses_gateway_trajectory_observer_config():
    run_input = RunInput(
        task_name='unused',
        gateway_trajectory_observers={
            'langfuse': {
                'name': 'langfuse',
                'args': {'flush_on_final': False},
            }
        },
    )

    observer_config = run_input.gateway_trajectory_observers['langfuse']
    assert isinstance(observer_config, ModuleObjectConfig)
    assert observer_config.name == 'langfuse'
    assert observer_config.args == {'flush_on_final': False}
