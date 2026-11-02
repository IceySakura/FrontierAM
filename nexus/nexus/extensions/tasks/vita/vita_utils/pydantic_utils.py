from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict

from .utils import get_hash


T = TypeVar('T', bound=BaseModel)


class BaseModelNoExtra(BaseModel):
    model_config = ConfigDict(extra='forbid')


def get_pydantic_hash(obj: BaseModel) -> str:
    hash_dict = obj.model_dump()
    return get_hash(hash_dict)


def update_pydantic_model_with_dict(
    model_instance: T,
    update_data: dict[str, Any],
) -> T:
    raw_data = model_instance.model_dump()
    _deep_update(raw_data, update_data)
    model_class = type(model_instance)
    return model_class.model_validate(raw_data)


def _deep_update(base: dict, update: dict) -> dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base
