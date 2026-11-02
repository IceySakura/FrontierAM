"""Compatibility re-exports for CodeMT's restricted-source detector.

The implementation sits beside this module in ``restricted_source.py``. It used to live
under ``harbor_bridge`` and be imported from here, but harbor_bridge's anti-hack now
carries only the tasks/swe answer-lookup tier, so the layer that has the only remaining
caller owns the code. CodeMT keeps this module path for older imports.
"""

from nexus.extensions.projects.code_mt.task.restricted_source import (
    AntiHackViolation,
    ancestor_commits,
    coerce_command_str,
    detect_local_command_hack,
    detect_local_output_hack,
    detect_source_download_hack,
    detect_source_download_hack_output,
    detect_unverified_commit_reads,
    extract_shell_command,
    observation_shows_failed_git_object_lookup,
    observation_shows_network_blocked,
    shell_command_events,
    should_exempt_blocked_network_violation,
    summarize_anti_hack_events,
)


__all__ = [
    'AntiHackViolation',
    'ancestor_commits',
    'coerce_command_str',
    'detect_local_command_hack',
    'detect_local_output_hack',
    'detect_source_download_hack',
    'detect_source_download_hack_output',
    'detect_unverified_commit_reads',
    'extract_shell_command',
    'observation_shows_failed_git_object_lookup',
    'observation_shows_network_blocked',
    'shell_command_events',
    'should_exempt_blocked_network_violation',
    'summarize_anti_hack_events',
]
