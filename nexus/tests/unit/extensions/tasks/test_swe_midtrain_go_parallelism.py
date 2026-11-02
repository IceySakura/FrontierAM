"""Tests for the T3Build_SWEBuilder_Midtrain eval-script post-processing.

Focused on the Go parallelism cap (see `_postprocess_eval_script` step 4): the
concurrent Go linker crash it works around only reproduces on large repos under a
cold build cache, so the behaviour that can be pinned down cheaply is *where* and
*whether* the export lands.
"""

from nexus.extensions.tasks.swe.t3build_swebuilder_midtrain import dataset_handler


_GO_CAP = 'export GOMAXPROCS=4'


def _instance(script: str, *, language: str = 'Go', **extra: object) -> dict:
    return {
        'instance_id': 'pingcap__tidb-53981',
        'repo': 'pingcap/tidb',
        'language': language,
        'eval_script': script,
        **extra,
    }


def _script(*body: str) -> str:
    return '\n'.join(('#!/bin/bash', 'set -xo pipefail', *body)) + '\n'


def test_cap_is_injected_before_any_build_step() -> None:
    """The export must precede every command that could invoke the linker."""
    script = _script('cd /testbed', 'git config --global --add safe.directory /testbed', 'go test ./...')

    out = dataset_handler.get_eval_script(_instance(script))

    lines = out.split('\n')
    assert _GO_CAP in lines
    assert lines.index(_GO_CAP) < lines.index('go test ./...')


def test_cap_precedes_repo_local_build_wrapper() -> None:
    """doltgresql builds its parser via `bash build.sh` before running any test.

    That wrapper shells out to `go` itself, so the cap has to be exported before it
    rather than merely before the `go test` line.
    """
    script = _script(
        'export PATH=/usr/local/go/bin:$PATH',
        'cd /testbed',
        'cd /testbed/postgres/parser && bash build.sh && cd /testbed',
        'go test -v ./testing/go/... -run TestInsert',
    )

    lines = dataset_handler.get_eval_script(_instance(script, instance_id='dolthub__doltgresql-1392')).split('\n')

    assert lines.index(_GO_CAP) < lines.index('cd /testbed/postgres/parser && bash build.sh && cd /testbed')


def test_cap_is_not_injected_for_non_go_instances() -> None:
    script = _script('cd /testbed', 'python -m pytest')

    out = dataset_handler.get_eval_script(_instance(script, language='Python'))

    assert 'GOMAXPROCS' not in out


def test_repo_language_is_honoured_when_language_is_absent() -> None:
    script = _script('cd /testbed', 'go test ./...')

    instance = {'instance_id': 'x__y-1', 'eval_script': script, 'repo_language': 'go'}

    assert _GO_CAP in dataset_handler.get_eval_script(instance)


def test_existing_gomaxprocs_is_left_alone() -> None:
    """A script that tunes GOMAXPROCS itself keeps its own value."""
    script = _script('cd /testbed', 'export GOMAXPROCS=2', 'go test ./...')

    out = dataset_handler.get_eval_script(_instance(script))

    assert 'export GOMAXPROCS=2' in out
    assert _GO_CAP not in out


def test_post_processing_is_idempotent() -> None:
    """This handler rewrites a script in place, so a second pass must not stack.

    Unlike swe_bench_pro's handler (MR !492), which rebuilds its env commands from
    scratch on every call, an unguarded insert here would accumulate duplicates.
    """
    script = _script('cd /testbed', 'go test ./...')

    once = dataset_handler.get_eval_script(_instance(script))
    twice = dataset_handler._postprocess_eval_script(once, _instance(script))

    assert twice == once
    assert twice.count(_GO_CAP) == 1
