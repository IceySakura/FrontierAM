import json

from nexus.extensions.tasks.vita.data_model.tasks import Task
from nexus.extensions.tasks.vita.domains.delivery.data_model import DeliveryDB
from nexus.extensions.tasks.vita.domains.delivery.tools import DeliveryTools
from nexus.extensions.tasks.vita.environment.environment import Environment
from nexus.extensions.tasks.vita.vita_utils.utils import get_task_file_path


def get_environment(
    db: DeliveryDB | None = None,
    language: str | None = None,
) -> Environment:
    import nexus.extensions.tasks.vita.domains.delivery.tools_schema  # noqa: F401

    if language is not None:
        from nexus.extensions.tasks.vita.vita_utils.schema_utils import set_global_language

        set_global_language(language)

    if db is None:
        db = {}  # type: ignore[assignment]

    db_dict = {
        k: v
        for k, v in db.items()  # type: ignore[union-attr]
        if k in ['stores', 'orders', 'time', 'user_id', 'weather', 'location', 'user_historical_behaviors']
    }
    if 'stores' not in db_dict:
        db_dict['stores'] = {}
    if 'orders' not in db_dict:
        db_dict['orders'] = {}

    db = DeliveryDB.model_validate(db_dict)
    tools = DeliveryTools(db)
    from nexus.extensions.tasks.vita.environment.environment import get_agent_policy

    return Environment(
        domain_name='delivery',
        policy=get_agent_policy(language),
        tools=tools,
    )


def get_tasks(language: str | None = None, data_dir: str | None = None) -> list[Task]:
    task_path = get_task_file_path('delivery', language, data_dir=data_dir)
    with open(task_path) as fp:
        tasks = json.load(fp)
    return [Task.model_validate(task) for task in tasks]
