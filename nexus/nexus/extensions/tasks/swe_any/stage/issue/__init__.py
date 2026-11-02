"""SWE-Any Issue — write a non-leaking problem statement for a masked+oracled feature.

Agentic-only stage (documented framework deviation: no ``heuristics`` half). Its one job is a
single Claude Code pass over a **fully masked** working tree — the feature stubbed
(``feature_mask_patch``) and the native oracle tests deleted (``test_mask_patch``) — that, shown the
gold patches as reference (``feature_gold_patch`` + ``test_gold_patch``), writes a GitHub-issue-style
``problem_statement`` describing the desired behaviour WITHOUT reproducing the implementation or the
tests. See :mod:`.prompts` for the contract and :mod:`.schema` for the schema/join.
"""

from __future__ import annotations

from .prompts import build_issue_prompt
from .runner import run_issue_feature
from .schema import IssueTaskInput, IssueTaskOutput


__all__ = [
    'IssueTaskInput',
    'IssueTaskOutput',
    'build_issue_prompt',
    'run_issue_feature',
]
