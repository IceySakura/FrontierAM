"""Shared utilities for HITL-style multi-agent collaboration."""

from nexus.extensions.tasks.hitl.utils.git import (
    capture_git_patch,
    cumulative_git_patch,
    ensure_generated_gitignore,
    git_changed_files,
    git_diff,
    git_has_changes,
    git_status,
)
from nexus.extensions.tasks.hitl.utils.json import (
    JSONExtractionError,
    JsonArray,
    JsonObject,
    JsonValue,
    extract_json,
    extract_json_array,
    extract_json_object,
)
from nexus.extensions.tasks.hitl.utils.patch import (
    PatchFileDiff,
    PatchFileStat,
    PatchScaleStats,
    count_patch_lines,
    extract_patch_file_stats,
    extract_patch_files,
    extract_patch_scale_stats,
    format_current_patch_scale,
    format_patch_scale_reference,
    normalize_patch,
    split_patch_by_file,
)
from nexus.extensions.tasks.hitl.utils.text import section, tagged, text_parts, truncate


__all__ = [
    'JSONExtractionError',
    'JsonArray',
    'JsonObject',
    'JsonValue',
    'PatchFileDiff',
    'PatchFileStat',
    'PatchScaleStats',
    'capture_git_patch',
    'count_patch_lines',
    'cumulative_git_patch',
    'ensure_generated_gitignore',
    'extract_json',
    'extract_json_array',
    'extract_json_object',
    'extract_patch_file_stats',
    'extract_patch_files',
    'extract_patch_scale_stats',
    'format_current_patch_scale',
    'format_patch_scale_reference',
    'git_changed_files',
    'git_diff',
    'git_has_changes',
    'git_status',
    'normalize_patch',
    'section',
    'split_patch_by_file',
    'tagged',
    'text_parts',
    'truncate',
]
