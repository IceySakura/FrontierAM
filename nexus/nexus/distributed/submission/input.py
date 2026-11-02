from __future__ import annotations

import copy
import json

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

from nexus.runner.live.models import LiveConfig
from nexus.runner.single import RunInput


class InputNormalizationError(ValueError):
    def __init__(self, code: str, *, detail: str | None = None) -> None:
        self.code = code
        self.detail = detail
        super().__init__(code)


def normalize_run_input(value: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_model(
        RunInput,
        value,
        code='invalid_run_input',
        exclude={'compaction_manager'},
    )


def normalize_live_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return _normalize_model(LiveConfig, value, code='invalid_live_config')


def _normalize_model(
    model: type[BaseModel],
    value: Mapping[str, Any],
    *,
    code: str,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise InputNormalizationError(code, detail='expected an object')
    try:
        normalized = model.model_validate(copy.deepcopy(dict(value))).model_dump(
            mode='json',
            exclude=exclude,
        )
        json.dumps(normalized, allow_nan=False)
    except ValidationError as exc:
        raise InputNormalizationError(code, detail=_validation_detail(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise InputNormalizationError(code, detail='input must contain finite JSON values') from exc
    return normalized


def _validation_detail(error: ValidationError) -> str:
    first = error.errors(include_input=False)[0]
    location = '.'.join(str(part) for part in first.get('loc', ()))
    message = str(first.get('msg') or first.get('type') or 'invalid value')
    return f'{location}: {message}' if location else message


__all__ = [
    'InputNormalizationError',
    'normalize_live_config',
    'normalize_run_input',
]
