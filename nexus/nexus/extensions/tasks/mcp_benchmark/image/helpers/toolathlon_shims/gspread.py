from __future__ import annotations

import os

from typing import Any


# In API mode (NEXUS_TOOLATHLON_SERVER_MODE != 'mock'), delegate to the real
# gspread library installed in the venv.  The TOOLATHLON_SHIMS_ROOT directory
# sits first on PYTHONPATH so this shim file shadows the real package.
#
# Critical: by the time this module body executes, Python has already inserted
# this shim into sys.modules['gspread'].  If we call import_module('gspread')
# without first removing that entry, Python just returns the shim itself
# (cache hit) — we never reach the real package.  We must pop ourselves out,
# import the real gspread, then put ourselves back.
if os.getenv('NEXUS_TOOLATHLON_SERVER_MODE') != 'mock':
    import importlib as _importlib
    import sys as _sys

    # 1. Remove our shim directory from sys.path so the real package is found.
    _shim_dir = os.path.dirname(os.path.abspath(__file__))
    _original_path = _sys.path[:]
    _sys.path = [p for p in _sys.path if os.path.abspath(p) != _shim_dir]

    # 2. Temporarily remove this shim from sys.modules so import_module() does
    #    not short-circuit back to us.
    _shim_module = _sys.modules.pop('gspread', None)

    try:
        _real_gspread = _importlib.import_module('gspread')
    finally:
        # 3. Restore sys.path and put our shim back as the canonical 'gspread'
        #    so that any other `import gspread` in the process gets the shim
        #    (which we are about to populate with real attrs).
        _sys.path = _original_path
        if _shim_module is not None:
            _sys.modules['gspread'] = _shim_module

    # 4. Copy all public attributes from the real gspread onto this shim module.
    _this_module = _sys.modules[__name__]
    for _attr in dir(_real_gspread):
        if not _attr.startswith('_'):
            setattr(_this_module, _attr, getattr(_real_gspread, _attr))

    # 5. Also ensure submodules are accessible via attribute access.
    for _sub in ('exceptions', 'auth', 'utils', 'urls', 'client'):
        _val = getattr(_real_gspread, _sub, None)
        if _val is not None:
            setattr(_this_module, _sub, _val)

    # 6. gspread v6 removed the top-level gspread.authorize() function.
    #    Upstream Toolathlon eval scripts (e.g. vlm-history-completer) still
    #    call gspread.authorize(credentials) which worked in v5.  Provide a
    #    compat wrapper that delegates to gspread.Client(auth=credentials).
    if not hasattr(_this_module, 'authorize'):
        _Client = getattr(_real_gspread, 'Client', None)
        if _Client is not None:

            def authorize(credentials: Any, **kwargs: Any) -> Any:
                return _Client(auth=credentials, **kwargs)

            _this_module.authorize = authorize
else:
    from toolathlon_google_sheets_mock import (
        MockGspreadClient,
        SpreadsheetNotFound,
        WorksheetNotFound,
        authorize,
    )

    class _ExceptionsNamespace:
        SpreadsheetNotFound = SpreadsheetNotFound
        WorksheetNotFound = WorksheetNotFound

    Client = MockGspreadClient
    exceptions = _ExceptionsNamespace()

    __all__ = ['Client', 'authorize', 'exceptions', 'SpreadsheetNotFound', 'WorksheetNotFound']
