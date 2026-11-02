from pathlib import Path

from nexus.runner.batch import BatchItem
from nexus.runner.single import RunInput


_BASE_CONFIG = RunInput.from_file(Path(__file__).with_name('main-agent-single.yaml'))
_ITEM_COUNT = 5


def get_total_count() -> int:
    return _ITEM_COUNT


def generate_configs():
    for index in range(_ITEM_COUNT):
        run_id = f'online-main-agent-{index + 1:02d}'
        config = _BASE_CONFIG.model_copy(
            deep=True,
            update={
                'run_id': run_id,
                'agent_config': {
                    **_BASE_CONFIG.agent_config,
                    'user_prompt': f'Reply with exactly NEXUS_BATCH_ITEM_{index + 1:02d}.',
                },
            },
        )
        yield BatchItem(run_id=run_id, config=config)
