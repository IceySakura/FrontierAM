import logging
import sys


class _Logger:
    def __getattr__(self, name: str):
        caller = sys._getframe(1).f_globals.get('__name__', 'nexus.agents')
        return getattr(logging.getLogger(caller), name)


logger = _Logger()
