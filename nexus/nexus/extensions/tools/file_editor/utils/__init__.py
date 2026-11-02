"""Utility modules for file editor."""

from nexus.extensions.tools.file_editor.utils.config import SNIPPET_CONTEXT_WINDOW
from nexus.extensions.tools.file_editor.utils.constants import (
    BINARY_FILE_CONTENT_TRUNCATED_NOTICE,
    CONTENT_TRUNCATED_NOTICE,
    DIRECTORY_CONTENT_TRUNCATED_NOTICE,
    MAX_RESPONSE_LEN_CHAR,
    TEXT_FILE_CONTENT_TRUNCATED_NOTICE,
)
from nexus.extensions.tools.file_editor.utils.encoding import (
    EncodingManager,
    with_encoding,
)
from nexus.extensions.tools.file_editor.utils.file_cache import FileCache
from nexus.extensions.tools.file_editor.utils.history import FileHistoryManager
from nexus.extensions.tools.file_editor.utils.shell import run_shell_cmd
from nexus.extensions.tools.file_editor.utils.truncate import maybe_truncate


__all__ = [
    'SNIPPET_CONTEXT_WINDOW',
    'MAX_RESPONSE_LEN_CHAR',
    'CONTENT_TRUNCATED_NOTICE',
    'TEXT_FILE_CONTENT_TRUNCATED_NOTICE',
    'BINARY_FILE_CONTENT_TRUNCATED_NOTICE',
    'DIRECTORY_CONTENT_TRUNCATED_NOTICE',
    'EncodingManager',
    'with_encoding',
    'FileCache',
    'FileHistoryManager',
    'run_shell_cmd',
    'maybe_truncate',
]
