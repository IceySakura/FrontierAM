import json
import os
import re

from pathlib import Path
from typing import Any

import yaml


def _substitute_env_vars(text: str) -> str:
    if not isinstance(text, str):
        return text
    pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'

    def replace_var(match):
        var_name = match.group(1)
        default_value = match.group(2) if match.group(2) is not None else ''
        return os.getenv(var_name, default_value)

    return re.sub(pattern, replace_var, text)


def _recursive_substitute(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: _recursive_substitute(value) for key, value in data.items()}
    if isinstance(data, list):
        return [_recursive_substitute(item) for item in data]
    if isinstance(data, str):
        return _substitute_env_vars(data)
    return data


def load_file(path: str | Path, **kwargs: Any) -> Any:
    path = Path(path)
    if path.suffix == '.json':
        with open(path) as fp:
            data = json.load(fp, **kwargs)
    elif path.suffix in ('.yaml', '.yml'):
        with open(path) as fp:
            data = yaml.load(fp, Loader=yaml.SafeLoader, **kwargs)
        data = _recursive_substitute(data)
    elif path.suffix in ('.txt', '.md'):
        encoding = kwargs.pop('encoding', None)
        if len(kwargs) > 0:
            raise ValueError(f'Unsupported keyword arguments: {kwargs}')
        with open(path, encoding=encoding) as fp:
            data = fp.read()
    else:
        raise ValueError(f'Unsupported file extension: {path}')
    return data


def dump_file(path: str | Path, data: dict[str, Any], **kwargs: Any) -> None:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    if path.suffix == '.json':
        with open(path, 'w') as fp:
            json.dump(data, fp, ensure_ascii=False, **kwargs)
    elif path.suffix in ('.yaml', '.yml'):
        with open(path, 'w') as fp:
            yaml.dump(data, fp, **kwargs)
    elif path.suffix in ('.txt', '.md'):
        encoding = kwargs.pop('encoding', None)
        if len(kwargs) > 0:
            raise ValueError(f'Unsupported keyword arguments: {kwargs}')
        with open(path, 'w', encoding=encoding) as fp:
            fp.write(str(data))
    else:
        raise ValueError(f'Unsupported file extension: {path}')
