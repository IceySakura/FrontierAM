"""Domain registry for VitaBench domains."""

from collections.abc import Callable

from nexus.extensions.tasks.vita.environment.environment import Environment as Environment


_domains: dict[str, Callable] = {}


def register_domain(name: str, get_environment: Callable) -> None:
    _domains[name] = get_environment


def get_env_constructor(name: str) -> Callable:
    if name not in _domains:
        raise ValueError(f'Domain {name!r} not registered. Available: {list(_domains.keys())}')
    return _domains[name]


def _register_defaults():
    from nexus.extensions.tasks.vita.domains.delivery.environment import get_environment as delivery_env
    from nexus.extensions.tasks.vita.domains.instore.environment import get_environment as instore_env
    from nexus.extensions.tasks.vita.domains.ota.environment import get_environment as ota_env

    register_domain('ota', ota_env)
    register_domain('delivery', delivery_env)
    register_domain('instore', instore_env)


_register_defaults()
