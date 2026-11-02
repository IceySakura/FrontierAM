"""Fixed Claude bare mode; all configuration is assembled by generator_native."""

from functools import partial

from nexus.extensions.tasks.os_world.generators import generator_native as native


generate_configs = partial(native.generate_configs, model_family='claude', harness='bare')
get_total_count = partial(native.get_total_count, model_family='claude', harness='bare')
get_batch_metadata = partial(native.get_batch_metadata, model_family='claude', harness='bare')
