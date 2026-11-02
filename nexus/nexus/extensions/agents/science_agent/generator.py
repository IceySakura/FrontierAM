# generator.py
"""
Science Agent batch job generator.

Recommended usage:
1. Create a config file under `workspace/research/` (e.g. `science_batch.yaml`).
2. Set environment variables:
   - export SCIENCE_CONFIG="/path/to/workspace/research/science_batch.yaml"
   - export SCIENCE_INPUT_PATH="/path/to/data.jsonl"
3. Run the batch job.

If no config file is provided, the generator falls back to a default config (requires env vars):
- SCIENCE_API_KEY: API key (required)
- SCIENCE_BASE_URL: API base URL
- SCIENCE_MODEL: model name
"""

import json
import os

from pathlib import Path
from typing import Any

from nexus import BatchItem


def generate_configs(input_path: str | None = None, base_config_path: str | Path | None = None):
    """Generate an iterator of BatchItems.

    Args:
        input_path: JSONL dataset path. Defaults to env var `SCIENCE_INPUT_PATH`.
        base_config_path: Base YAML config path (contains `openai_clients`, `tool_configs`, etc.).
            Lookup order:
            1. Explicit argument
            2. Env var `SCIENCE_CONFIG`
            3. `workspace/research/science_batch.yaml`
            4. Default config (requires env vars)

    Each BatchItem loads a config template from `base_config_path`, then overrides `question`/`answer` per record.
    """
    # Resolve dataset path
    if input_path is None:
        input_path = os.getenv(
            'SCIENCE_INPUT_PATH',
            '/apdcephfs_cq8/share_1324356/daisyyi/1_science_agent/distillation/data/testset/all_split_template_formatted_repeat_1/HLE_200.jsonl',
        )

    # Load dataset
    if not Path(input_path).exists():
        raise FileNotFoundError(f'Input data file not found: {input_path}')

    with open(input_path) as f:
        messages = [json.loads(line) for line in f]

    # Load base config
    base_config: dict[str, Any] | None = None
    if base_config_path is None:
        # Config lookup priority:
        # 1. env var `SCIENCE_CONFIG`
        # 2. `workspace/research/science_batch.yaml`
        config_from_env = os.getenv('SCIENCE_CONFIG')
        if config_from_env and Path(config_from_env).exists():
            base_config_path = Path(config_from_env)
        else:
            # Repo root (assumes this file lives under `nexus/extensions/agents/science_agent/`)
            package_root = Path(__file__).parent.parent.parent.parent
            workspace_config = package_root / 'workspace' / 'research' / 'science_batch.yaml'
            if workspace_config.exists():
                base_config_path = workspace_config

    if base_config_path and Path(base_config_path).exists():
        from omegaconf import OmegaConf

        base_config = OmegaConf.to_container(OmegaConf.load(base_config_path), resolve=True)  # type: ignore[assignment]

    # Generate one config per record
    for i, message in enumerate(messages):
        config: dict[str, Any]
        if base_config:
            # Use base config as template
            config = base_config.copy()

            # TODO(petersheng): migrate the math branch to the unified reward_system task (scorer=math)
            # and drop the legacy nexus.extensions.tasks.math package once offline batches move over.
            # Choose `task_config` schema based on `task_name`
            task_name = config.get('task_name', 'math')
            if task_name == 'hle':
                config['task_config'] = {
                    'task_id': f'hle_{i}',
                    'category': str(message.get('subject', 'general')),
                    'question': str(message['prompt']),
                    'ground_truth': str(message['ref_answer']),
                    'timeout': config.get('task_config', {}).get('timeout', 600),
                }
            else:
                # math task schema
                config['task_config'] = {
                    'question': str(message['prompt']),
                    'answer': str(message['ref_answer']),
                }
        else:
            # Default config (backwards-compatible)
            # Repo root
            package_root = Path(__file__).parent.parent.parent.parent
            data_dir = package_root / 'extensions' / 'tools' / 'search_stateful' / 'data'

            config = {
                'imports': [
                    'nexus.extensions',
                    'nexus.extensions.tasks.math',
                    'nexus.extensions.agents.science_agent.agent',
                    'nexus.extensions.tools.search_stateful',
                ],
                'task_name': 'math',
                'task_config': {
                    'question': str(message['prompt']),
                    'answer': str(message['ref_answer']),
                },
                'runtime_provider_name': 'docker',
                'runtime_provider_config': {},
                'runtime_configs': {},
                'agent_name': 'science_agent',
                'agent_config': {
                    'max_iterations': 1000,
                },
                'tool_configs': {
                    'search_stateful': {
                        'max_search_results': 10,
                        'url_block_list_file': str(data_dir / 'url_block_list.txt'),
                        'testset_name_to_filter_file': str(data_dir / 'testset_name_to_filter.txt'),
                    },
                },
                'openai_clients': {
                    'main': {
                        'name': 'litellm',
                        'config': {
                            'client_args': {
                                'api_key': os.getenv('SCIENCE_API_KEY', ''),
                                'base_url': os.getenv('SCIENCE_BASE_URL', 'http://api.model-eval.woa.com/v1'),
                                'max_retries': int(os.getenv('SCIENCE_MAX_RETRIES', '5')),
                            },
                            'request_args': {
                                'model': os.getenv('SCIENCE_MODEL', 'deepseek/deepseek-reasoner'),
                            },
                        },
                    },
                    'judge': {
                        'name': 'litellm',
                        'config': {
                            'client_args': {
                                'api_key': os.getenv('SCIENCE_API_KEY', ''),
                                'base_url': os.getenv('SCIENCE_BASE_URL', 'http://api.model-eval.woa.com/v1'),
                                'max_retries': int(os.getenv('SCIENCE_MAX_RETRIES', '5')),
                            },
                            'request_args': {
                                'model': os.getenv('SCIENCE_MODEL', 'deepseek/deepseek-reasoner'),
                            },
                        },
                    },
                },
            }

        yield BatchItem(
            config=config,  # type: ignore[arg-type]
            run_id=f'science_agent_{i}',
            metadata={
                'index': i,
                'message_length': len(str(message['prompt'])),
            },
        )


def get_total_count() -> int:
    """Return total item count (optional, for progress reporting)."""
    return 200


def get_batch_metadata() -> dict:
    """Return batch-level metadata (optional)."""
    return {
        'description': 'Science agent batch example',
        'experiment': 'test_stateful_v1',
    }
