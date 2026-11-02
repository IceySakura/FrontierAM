"""Independent V2 generator and single-config entry."""

from nexus.extensions.tasks.os_world_v2.generators import config
from nexus.extensions.tasks.os_world_v2.generators.policy import SUPPORTED as SUPPORTED
from nexus.extensions.tasks.os_world_v2.generators.policy import selection


def generate_configs(*, model_family=None, harness=None):
    choice = selection(model_family=model_family, harness=harness)
    config.selected_release()
    for selected in config.tasks():
        yield config.build_item(selected, choice)


def get_total_count(*, model_family=None, harness=None):
    selection(model_family=model_family, harness=harness)
    config.selected_release()
    return len(config.tasks())


def get_batch_metadata(*, model_family=None, harness=None):
    choice = selection(model_family=model_family, harness=harness)
    config.selected_release()
    return {
        'task_name': 'os_world_v2',
        'benchmark_release': config.RELEASE,
        'source_commit': config.COMMIT,
        'model_family': choice.model_family,
        'harness': choice.harness,
    }


def single_config(*, task_id, model_family=None, harness=None):
    choice = selection(model_family=model_family, harness=harness)
    config.selected_release()
    (selected,) = config.tasks(task_id)
    item = config.build_item(selected, choice)
    return item.config.model_copy(update={'run_id': item.run_id})
