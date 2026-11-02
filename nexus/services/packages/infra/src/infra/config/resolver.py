"""将嵌套 dict 绑定到 pydantic model：按字段名（或 Key 覆盖）逐级下钻。

取值规则（对每个字段独立执行）：
1. 键名 = `Key` 标记覆盖值，否则字段名；
2. 嵌套 model 字段用该键名下钻一层，在子作用域内递归；
3. 标量字段在当前作用域按键名取值，键不存在则保留 model 默认值。

空值（含空字符串）视为**命中**，会覆盖默认值；仅键不存在或值为 None 才回落默认值。
"""

from __future__ import annotations

import logging

from typing import Any, TypeVar

from pydantic import BaseModel
from pydantic.fields import FieldInfo

from infra.config.keys import Key


logger = logging.getLogger(__name__)

ModelT = TypeVar('ModelT', bound=BaseModel)

_MISSING = object()


def _field_key(field_name: str, field: FieldInfo) -> str:
    """字段在配置源中的键名：Key 标记优先，否则用字段名。"""
    markers = [meta for meta in field.metadata if isinstance(meta, Key)]
    if not markers:
        return field_name
    if len(markers) > 1:
        names = ', '.join(marker.name for marker in markers)
        raise ValueError(f'field {field_name!r} declares multiple Key markers ({names}); only one is allowed')
    return markers[0].name


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)


def _lookup(scope: dict[str, Any], key: str) -> Any:
    """在当前作用域内按键名取值，大小写不敏感。"""
    if key in scope:
        return scope[key]

    lowered = key.lower()
    for candidate, value in scope.items():
        if isinstance(candidate, str) and candidate.lower() == lowered:
            return value
    return _MISSING


def bind(model_cls: type[ModelT], data: dict[str, Any]) -> ModelT:
    """用嵌套 dict 构建 model 实例，缺失字段保留默认值。"""
    return model_cls(**_collect(model_cls, data))


def _collect(model_cls: type[BaseModel], scope: dict[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}

    for field_name, field in model_cls.model_fields.items():
        key = _field_key(field_name, field)
        raw = _lookup(scope, key)

        if _is_model(field.annotation):
            # 嵌套 model：下钻一层，子作用域缺失时仍递归空 dict，
            # 以便下级字段各自回落到自己的默认值。
            nested_scope = raw if isinstance(raw, dict) else {}
            if raw is not _MISSING and not isinstance(raw, dict):
                logger.warning(
                    'config key %r for nested model %s is not a mapping, ignored',
                    key,
                    field.annotation.__name__,
                )
            nested = _collect(field.annotation, nested_scope)
            if nested:
                values[field_name] = nested
            continue

        # 空字符串算命中；仅键不存在或显式 None 才用默认值。
        if raw is _MISSING or raw is None:
            continue
        values[field_name] = raw

    return values
