"""Helpers for parsing Yuanbao search tool configuration values."""


def as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('1', 'true', 'yes', 'on')


def as_float(value: object, default: float) -> float:
    if value is None or value == '':
        return default
    if isinstance(value, (str, bytes, bytearray, int, float)):
        return float(value)
    raise TypeError(f'Expected a numeric configuration value, got {type(value).__name__}')


def as_int(value: object, default: int) -> int:
    if value is None or value == '':
        return default
    if isinstance(value, (str, bytes, bytearray, int, float)):
        return int(value)
    raise TypeError(f'Expected an integer configuration value, got {type(value).__name__}')
