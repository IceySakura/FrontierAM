from pathlib import Path

from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


_BASE_CONFIG = RunInput.from_file(Path(__file__).with_name('main-agent-single-gateway.yaml'))
_ITEM_COUNT = 5


def get_total_count() -> int:
    return _ITEM_COUNT


def bind_gateway(config: RunInput, run_id: str) -> RunInput:
    bound = config.model_copy(deep=True)
    bound.run_id = run_id
    bound.runner.use_gateway = True
    return bound


def generate_configs():
    for index in range(_ITEM_COUNT):
        run_id = f'online-main-agent-gateway-{index + 1:02d}'
        config = bind_gateway(_BASE_CONFIG, run_id)
        config.agent_config = {
            **config.agent_config,
            'user_prompt': f'Reply with exactly NEXUS_GATEWAY_BATCH_ITEM_{index + 1:02d}.',
        }
        yield BatchItem(run_id=run_id, config=config)
