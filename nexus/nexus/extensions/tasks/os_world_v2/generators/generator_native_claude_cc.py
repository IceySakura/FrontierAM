from functools import partial

from nexus.extensions.tasks.os_world_v2.generators import generator_native as native


generate_configs = partial(native.generate_configs, model_family='claude', harness='cc')
get_total_count = partial(native.get_total_count, model_family='claude', harness='cc')
get_batch_metadata = partial(native.get_batch_metadata, model_family='claude', harness='cc')
