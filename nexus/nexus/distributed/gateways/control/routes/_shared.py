from __future__ import annotations

import json

from typing import TypeVar

from aiohttp import web
from pydantic import BaseModel, ValidationError


ModelT = TypeVar('ModelT', bound=BaseModel)


async def read_model(request: web.Request, model: type[ModelT]) -> ModelT:
    try:
        data = await request.json()
    except json.JSONDecodeError as exc:
        raise web.HTTPBadRequest(text=f'invalid JSON body: {exc}') from exc
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise web.HTTPUnprocessableEntity(text=exc.json()) from exc


def is_truthy_query(value: str | None) -> bool:
    return value is not None and value.lower() in {'1', 'true', 'yes', 'on'}


def query_int(
    request: web.Request,
    name: str,
    default: int,
    *,
    min_value: int = 0,
    max_value: int | None = None,
) -> int:
    raw = request.query.get(name)
    if raw is None or raw == '':
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=f'invalid {name}') from exc
    value = max(min_value, value)
    if max_value is not None:
        value = min(max_value, value)
    return value


__all__ = ['is_truthy_query', 'query_int', 'read_model']
