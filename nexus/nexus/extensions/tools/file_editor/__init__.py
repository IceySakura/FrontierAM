"""File editor module for nexus framework.

This module provides a FileEditor class for viewing, creating, and editing files
with str_replace operations.
"""

from nexus.extensions.tools.file_editor.editor import FileEditor
from nexus.extensions.tools.file_editor.exceptions import (
    EditorToolParameterInvalidError,
    EditorToolParameterMissingError,
    FileValidationError,
    ToolError,
)


__all__ = [
    'FileEditor',
    'ToolError',
    'EditorToolParameterMissingError',
    'EditorToolParameterInvalidError',
    'FileValidationError',
]
