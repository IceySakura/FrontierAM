import json

from nexus.extensions.tasks.vita.data_model.tasks import Task
from nexus.extensions.tasks.vita.domains.instore.data_model import InStoreDB
from nexus.extensions.tasks.vita.domains.instore.tools import InStoreTools
from nexus.extensions.tasks.vita.environment.environment import Environment
from nexus.extensions.tasks.vita.vita_utils.utils import get_task_file_path


def get_environment(
    db: InStoreDB | None = None,
    language: str | None = None,
) -> Environment:
    import nexus.extensions.tasks.vita.domains.instore.tools_schema  # noqa: F401

    if language is not None:
        from nexus.extensions.tasks.vita.vita_utils.schema_utils import set_global_language

        set_global_language(language)

    if db is None:
        db = {}  # type: ignore[assignment]

    db_dict = {
        k: v
        for k, v in db.items()  # type: ignore[union-attr]
        if k
        in [
            'shops',
            'orders',
            'reservations',
            'books',
            'time',
            'user_id',
            'weather',
            'location',
            'user_historical_behaviors',
        ]
    }
    if 'shops' not in db_dict:
        db_dict['shops'] = {}
    if 'books' not in db_dict:
        db_dict['books'] = {}
    if 'reservations' not in db_dict:
        db_dict['reservations'] = {}
    if 'orders' not in db_dict:
        db_dict['orders'] = {}

    db = InStoreDB.model_validate(db_dict)
    tools = InStoreTools(db)
    from nexus.extensions.tasks.vita.environment.environment import get_agent_policy

    return Environment(
        domain_name='instore',
        policy=get_agent_policy(language),
        tools=tools,
    )


def get_tasks(language: str | None = None, data_dir: str | None = None) -> list[Task]:
    task_path = get_task_file_path('instore', language, data_dir=data_dir)
    with open(task_path) as fp:
        tasks = json.load(fp)
    return [Task.model_validate(task) for task in tasks]
