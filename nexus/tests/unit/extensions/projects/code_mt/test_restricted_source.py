"""CodeMT restricted-source detector (CM-1): the network and local-history tiers.

These moved here with the implementation. harbor_bridge's anti-hack carries only the
tasks/swe answer-lookup tier now, so CM-1 lives beside its only caller and is tested
beside it -- see test_anti_hack_swe_parity.py for the harbor_bridge side.
"""

from __future__ import annotations


def test_code_mt_compatibility_imports_reexport_harbor_implementations() -> None:
    from nexus.extensions.harbor_bridge.anti_hack.text_parse import (
        normalize_text_parse_messages as harbor_normalize_text_parse_messages,
    )
    from nexus.extensions.projects.code_mt.task.anti_hack import (
        AntiHackViolation,
        coerce_command_str,
        detect_local_command_hack,
        detect_local_output_hack,
        detect_source_download_hack,
        detect_source_download_hack_output,
        extract_shell_command,
        summarize_anti_hack_events,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        AntiHackViolation as HarborAntiHackViolation,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        coerce_command_str as harbor_coerce_command_str,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        detect_local_command_hack as harbor_detect_local_command_hack,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        detect_local_output_hack as harbor_detect_local_output_hack,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        detect_source_download_hack as harbor_detect_source_download_hack,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        detect_source_download_hack_output as harbor_detect_source_download_hack_output,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        extract_shell_command as harbor_extract_shell_command,
    )
    from nexus.extensions.projects.code_mt.task.restricted_source import (
        summarize_anti_hack_events as harbor_summarize_anti_hack_events,
    )
    from nexus.extensions.projects.code_mt.task.reward_hacking.text_parse_bridge import (
        normalize_text_parse_messages,
    )

    assert AntiHackViolation is HarborAntiHackViolation
    assert coerce_command_str is harbor_coerce_command_str
    assert detect_local_command_hack is harbor_detect_local_command_hack
    assert detect_local_output_hack is harbor_detect_local_output_hack
    assert detect_source_download_hack is harbor_detect_source_download_hack
    assert detect_source_download_hack_output is harbor_detect_source_download_hack_output
    assert extract_shell_command is harbor_extract_shell_command
    assert summarize_anti_hack_events is harbor_summarize_anti_hack_events
    assert normalize_text_parse_messages is harbor_normalize_text_parse_messages


def test_gh_pr_only_flags_the_target_repo():
    """``gh pr`` must gate on the repo, like the ``gh repo clone`` arm beside it.

    The rule used to ``return`` unconditionally for any ``gh pr view|diff|checkout``,
    so reading an UNRELATED project's PR -- which leaks nothing about this instance's
    fix -- was reported as ``gh_pr_remote_source``. Its immediate neighbour
    ``gh_repo_clone_source`` already checked its repo argument.

    Absence of a repo stays a violation (fail closed): bare ``gh pr checkout 123``
    makes gh infer the repo from the cwd, which inside the checkout IS the target.
    That is the dominant real shape -- 75 of 84 ``gh pr`` occurrences in the a5b
    6053679 logs carry no ``--repo`` -- so exempting it would gut the rule.
    """
    from nexus.extensions.projects.code_mt.task.restricted_source import detect_source_download_hack

    def reason(command: str, repo: str) -> str | None:
        violation = detect_source_download_hack(
            command, repo=repo, working_dir='/testbed', instance_id=None, repo_anchored=True
        )
        return violation['reason'] if violation else None

    target = 'foo/bar'
    # Explicitly naming the TARGET -- via --repo, -R, --repo=, or a PR URL -- is a hack.
    assert reason('gh pr view 1 --repo foo/bar', target) == 'gh_pr_remote_source'
    assert reason('gh pr diff 1 -R foo/bar', target) == 'gh_pr_remote_source'
    assert reason('gh pr view 1 --repo=foo/bar', target) == 'gh_pr_remote_source'
    assert reason('gh pr checkout https://github.com/foo/bar/pull/123', target) == 'gh_pr_remote_source'
    # No repo named -> gh resolves from the cwd -> indistinguishable from the target.
    assert reason('gh pr checkout 123', target) == 'gh_pr_remote_source'
    assert reason('gh pr view 161', target) == 'gh_pr_remote_source'
    # An explicitly UNRELATED repo leaks nothing about this instance.
    assert reason('gh pr view 99 --repo totally/unrelated', target) is None
    assert reason('gh pr diff 99 -R acme/widgets', target) is None
    assert reason('gh pr checkout https://github.com/teaxyz/pantry/pull/123', target) is None
    # A real observed command, judged against its own target and an unrelated one.
    citylines = 'gh pr view 161 --repo citylines/citylines --json title,body'
    assert reason(citylines, 'citylines/citylines') == 'gh_pr_remote_source'
    assert reason(citylines, target) is None


def test_l4_rejects_non_head_ref_walks_and_patch_dumps():
    """The ancestor allowlist must only accept a plain HEAD-ancestry walk.

    Regression test for a real gap: an earlier revision of this layer let ANY
    `git log`/`rev-list` output launder SHAs into the exemption set, so
    `git log --all` (which prints commits the truncation meant to remove)
    followed by `git show <fix>` read as legitimate. It also missed
    `git log --all -p`, which dumps every ref's patch in one command and needs
    no second read at all.

    Ported from the t3-harbor-bridge implementation, whose L4 layer is newer
    than the one this branch previously carried.
    """
    import json as _json

    from nexus.extensions.projects.code_mt.task.restricted_source import (
        ancestor_commits,
        detect_unverified_commit_reads,
        shell_command_events,
    )

    base = 'a' * 40
    fix = 'b' * 40

    def reasons(pairs: list[tuple[str, str]]) -> list[str]:
        messages: list[dict] = []
        for index, (command, observation) in enumerate(pairs):
            messages.append({
                'role': 'assistant',
                'content': '',
                'tool_calls': [
                    {
                        'id': f'c{index}',
                        'type': 'function',
                        'function': {'name': 'bash', 'arguments': _json.dumps({'command': command})},
                    }
                ],
            })
            messages.append({'role': 'tool', 'tool_call_id': f'c{index}', 'content': observation})
        events = shell_command_events(messages)
        return [
            violation['reason']
            for violation in detect_unverified_commit_reads(
                events, base_commit=base, ancestors=ancestor_commits(events)
            )
        ]

    # One command that prints every ref's patch -- the fix leaks with no second read.
    for widening in ('--all', '--reflog', '--branches'):
        assert reasons([(f'git log {widening} -p', f'commit {fix}\ndiff --git a/x b/x')]) == [
            'git_unverified_ref_patch'
        ], widening

    # A widened walk must NOT launder the SHAs it printed into the allowlist.
    assert reasons([('git log --all', f'commit {fix}'), (f'git show {fix}', 'diff --git a/x b/x')]) == [
        'git_unverified_commit_read'
    ]
    assert reasons([('git rev-list --all', f'commit {fix}'), (f'git show {fix}', 'diff --git a/x b/x')]) == [
        'git_unverified_commit_read'
    ]

    # A plain HEAD walk is the one legitimate source of allowlist entries.
    assert reasons([('git log', f'commit {fix}'), (f'git show {fix}', 'diff --git a/x b/x')]) == []

    # Global git flags must not smuggle a read past the matcher.
    assert reasons([(f'git -C /t --no-pager show {fix}', 'diff --git a/x b/x')]) == ['git_unverified_commit_read']
    assert reasons([('git show HEAD', f'commit {base}')]) == []
