from __future__ import annotations


try:
    import asyncmy as _driver

    from asyncmy.cursors import SSCursor
    from asyncmy.errors import InterfaceError, OperationalError
    from asyncmy.errors import Warning as DriverWarning
except ImportError:  # Python >= 3.14: the distributed extra installs aiomysql instead
    import aiomysql as _driver

    from aiomysql import InterfaceError, OperationalError
    from aiomysql import Warning as DriverWarning
    from aiomysql.cursors import SSCursor


# Module-level connect() coroutine of the resolved driver.
connect = _driver.connect

# Name of the resolved driver ('asyncmy' or 'aiomysql'); handy for diagnostics.
driver_name = _driver.__name__


__all__ = [
    'DriverWarning',
    'SSCursor',
    'InterfaceError',
    'OperationalError',
    'connect',
    'driver_name',
]
