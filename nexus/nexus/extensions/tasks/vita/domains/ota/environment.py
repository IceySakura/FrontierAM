import json

from nexus.extensions.tasks.vita.data_model.tasks import Task
from nexus.extensions.tasks.vita.domains.ota.data_model import OTADB
from nexus.extensions.tasks.vita.domains.ota.tools import OTATools
from nexus.extensions.tasks.vita.environment.environment import Environment
from nexus.extensions.tasks.vita.vita_utils.utils import get_task_file_path


def get_environment(
    db: OTADB | None = None,
    language: str | None = None,
) -> Environment:
    import nexus.extensions.tasks.vita.domains.ota.tools_schema  # noqa: F401

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
            'orders',
            'hotels',
            'attractions',
            'flights',
            'trains',
            'time',
            'user_id',
            'weather',
            'location',
            'user_historical_behaviors',
        ]
    }
    if 'hotels' not in db_dict:
        db_dict['hotels'] = {}
    if 'attractions' not in db_dict:
        db_dict['attractions'] = {}
    if 'flights' not in db_dict:
        db_dict['flights'] = {}
    if 'trains' not in db_dict:
        db_dict['trains'] = {}
    if 'orders' not in db_dict:
        db_dict['orders'] = {}

    db = OTADB.model_validate(db_dict)
    tools = OTATools(db)
    from nexus.extensions.tasks.vita.environment.environment import get_agent_policy

    return Environment(
        domain_name='ota',
        policy=get_agent_policy(language),
        tools=tools,
    )


def get_tasks(language: str | None = None, data_dir: str | None = None) -> list[Task]:
    task_path = get_task_file_path('ota', language, data_dir=data_dir)
    with open(task_path) as fp:
        tasks = json.load(fp)
    return [Task.model_validate(task) for task in tasks]
