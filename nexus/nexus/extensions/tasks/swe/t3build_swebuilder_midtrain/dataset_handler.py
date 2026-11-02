"""Dataset handler for T3Build_SWEBuilder_Midtrain dataset.

This dataset supports two modes:
1. Pre-built evaluation scripts (eval_script field) with OMNIGRIL_EXIT_CODE grading
2. Dynamic eval script generation for multilingual f2p data without eval_script,
   inferring test commands from FAIL_TO_PASS prefixes and language field.

Runtime initialization follows t3build_repolaunch_midtrain's pattern
(git reset --hard + clean + truncate).
"""

import json
import re

from pathlib import Path
from typing import Any, cast

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.runtime import Runtime


# Constants for log parsing - same as abaka
APPLY_PATCH_FAIL = '>>>>> Patch Apply Failed'
APPLY_PATCH_PASS = '>>>>> Patch Apply Passed'
EXIT_CODE_PATTERN = re.compile(r'(?:echo )?OMNIGRIL_EXIT_CODE=(\d+)')

# Language fallback defaults when FAIL_TO_PASS prefix detection fails
# Java build commands: prefer gradlew (project-local), fallback to mvn.
# Export common Maven locations to PATH before checking.
_JAVA_PATH_SETUP = 'export PATH="$PATH:/opt/maven/bin:/usr/share/maven/bin:/opt/apache-maven/bin"'
# Use `false` instead of `exit 1` to avoid terminating the entire eval script
# (which would skip OMNIGRIL_EXIT_CODE output). `false` returns exit code 1.
_JAVA_TEST_CMD = f'{_JAVA_PATH_SETUP}; if [ -f ./gradlew ]; then ./gradlew test 2>&1; elif command -v mvn >/dev/null 2>&1; then mvn test -B 2>&1; else echo "No build tool found"; false; fi'
_JAVA_BUILD_CMD = f'{_JAVA_PATH_SETUP}; if [ -f ./gradlew ]; then ./gradlew build 2>&1; elif command -v mvn >/dev/null 2>&1; then mvn compile -B 2>&1; else echo "No build tool found"; false; fi'

# C# build command: find .sln or .csproj since `dotnet test` without args
# fails when run from a directory that has no project/solution file.
_CSHARP_TEST_CMD = (
    'SLN=$(find . -maxdepth 3 -name "*.sln" | head -1);'
    ' if [ -n "$SLN" ]; then dotnet test "$SLN" 2>&1;'
    ' else PROJ=$(find . -maxdepth 4 -name "*[Tt]est*.csproj" | head -1);'
    ' if [ -n "$PROJ" ]; then dotnet test "$PROJ" 2>&1;'
    ' else dotnet test 2>&1; fi; fi'
)

# C++ test command: try cmake/ctest first (most common), then make test.
_CPP_TEST_CMD = (
    'if [ -f build/Makefile ] || [ -f build/CMakeCache.txt ]; then'
    ' cd build && ctest --output-on-failure 2>&1;'
    ' elif command -v ctest >/dev/null 2>&1 && [ -f CMakeCache.txt ]; then'
    ' ctest --output-on-failure 2>&1;'
    ' else make test 2>&1; fi'
)

_LANGUAGE_DEFAULT_TEST_CMDS: dict[str, list[str]] = {
    'python': ['python -m pytest -rA --tb=long'],
    'java': [_JAVA_TEST_CMD],
    'rust': ['cargo test 2>&1'],
    'c#': [_CSHARP_TEST_CMD],
    'c++': [_CPP_TEST_CMD],
    'typescript': ['npm test 2>&1'],
    'javascript': ['npm test 2>&1'],
}


def _parse_fail_to_pass(instance: SWEbenchInstance) -> list[str]:
    """Parse FAIL_TO_PASS field which can be a JSON string or list."""
    f2p = instance.get('FAIL_TO_PASS', [])
    if isinstance(f2p, str):
        try:
            f2p = json.loads(f2p)
        except (json.JSONDecodeError, ValueError):
            f2p = [f2p]
    return f2p


def _extract_pytest_test_files(instance: SWEbenchInstance) -> list[str]:
    """Extract unique test file paths from FAIL_TO_PASS for pytest targeting.

    Returns deduplicated file paths from entries like:
      'tests/functional/test_cache.py::test_foo' -> 'tests/functional/test_cache.py'
    """
    f2p = _parse_fail_to_pass(instance)
    seen: set[str] = set()
    files: list[str] = []
    for t in f2p:
        if '::' in t:
            path = t.split('::')[0]
            # Only include pytest-style paths (contain / or end with .py)
            if ('/' in path or path.endswith('.py')) and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def _language_text(value: Any) -> str:
    """Language fields are a string, or ``{'language': 'C', 'language_list': [...]}``."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        inner = value.get('language')
        if isinstance(inner, str):
            return inner.strip()
    return ''


def _instance_language(instance: SWEbenchInstance) -> str:
    return (_language_text(instance.get('repo_language')) or _language_text(instance.get('language'))).lower()


def _infer_test_cmds(instance: SWEbenchInstance) -> list[str]:
    """Infer test commands from FAIL_TO_PASS prefix patterns.

    Priority:
    1. Detect known prefixes in FAIL_TO_PASS entries (cargo::, junit::, etc.)
    2. Detect pytest-style paths (file.py::test_name) — targets specific test files
    3. Fall back to language field defaults — also targets specific test files for pytest
    """
    f2p = _parse_fail_to_pass(instance)

    prefixes: set[str] = set()
    for t in f2p:
        if '::' in t:
            prefixes.add(t.split('::')[0])

    # Detect by known prefix patterns
    if 'cargo' in prefixes:
        return ['cargo test 2>&1']
    if 'csharp' in prefixes:
        return [_CSHARP_TEST_CMD]
    if any(p in prefixes for p in ('junit', 'gradle_task')):
        return [_JAVA_TEST_CMD]
    if 'build' in prefixes:
        return [_JAVA_BUILD_CMD]
    if 'unittest' in prefixes:
        return ['npm test 2>&1']

    # Check for pytest-style paths (contains / or ends with .py)
    # Target only the specific test files to avoid unrelated collection errors
    test_files = _extract_pytest_test_files(instance)
    for p in prefixes:
        if '/' in p or p.endswith('.py'):
            file_args = ' '.join(test_files) if test_files else ''
            return [f'python -m pytest -rA --tb=long {file_args}'.rstrip()]

    # Fallback by language field — also target specific files for pytest
    lang = _instance_language(instance)
    if lang == 'python' or (not lang and not prefixes):
        file_args = ' '.join(test_files) if test_files else ''
        return [f'python -m pytest -rA --tb=long {file_args}'.rstrip()]
    return _LANGUAGE_DEFAULT_TEST_CMDS.get(lang, ['python -m pytest -rA --tb=long'])


def _instance_language(instance: SWEbenchInstance) -> str:
    """Return the instance's primary language, lowercased ('' if unknown).

    Tolerates the two shapes `repo_language` takes in the wild. In this dataset it is a
    dict — {'language': 'Go', 'language_list': [{'language': .., 'ratio': ..}, ...]} in
    150/150 v2 instances — but other producers store a plain string. Anything else
    (None, unexpected types) degrades to the top-level `language` field and finally to ''
    rather than raising, since this feeds script generation for every instance.
    """
    repo_language: Any = instance.get('repo_language')
    if isinstance(repo_language, dict):
        repo_language = repo_language.get('language')
    if not isinstance(repo_language, str) or not repo_language:
        repo_language = instance.get('language')
    if not isinstance(repo_language, str):
        return ''
    return repo_language.strip().lower()


def _postprocess_eval_script(eval_script: str, instance: SWEbenchInstance) -> str:
    """Post-process a pre-built eval_script to fix common issues.

    Fixes applied:
    1. Remove `set -u` (nounset) to avoid unbound variable errors when heredoc
       parsing leaves variables unset due to special characters in filenames.
    2. Inject GRADLE_OPTS proxy configuration so Gradle builds can reach
       external repositories through the container's HTTPS proxy.
    3. Disable HTTP/2 multiplexing for Cargo to avoid flaky downloads behind
       proxies (CARGO_HTTP_MULTIPLEXING=false).
    4. Cap GOMAXPROCS on Go instances so concurrent linking cannot crash the Go
       linker.
    """
    # 1. Remove -u from set flags (e.g. "set -uxo pipefail" -> "set -xo pipefail")
    eval_script = re.sub(r'^(set\s+-\w*)u(\w*)', r'\1\2', eval_script, flags=re.MULTILINE)
    # Clean up bare "set -" left when -u was the only flag (e.g. "set -u" -> "set -xo pipefail")
    eval_script = re.sub(r'^set\s+-\s*$', 'set -xo pipefail', eval_script, flags=re.MULTILINE)
    eval_script = re.sub(r'^set\s+-\s+', 'set ', eval_script, flags=re.MULTILINE)

    # 2. Inject GRADLE_OPTS proxy configuration before pre_install / eval commands
    if 'gradlew' in eval_script and 'GRADLE_OPTS' not in eval_script:
        gradle_proxy_snippet = (
            'if [ -n "$https_proxy" ]; then'
            " PROXY_HOST=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f1);"
            " PROXY_PORT=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f2);"
            ' export GRADLE_OPTS="-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'
            ' -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT";'
            ' fi\n'
        )
        eval_script = re.sub(
            r'^(cd\s+/testbed.*\n)',
            r'\1' + gradle_proxy_snippet,
            eval_script,
            count=1,
            flags=re.MULTILINE,
        )

    # 3. Inject CARGO_HTTP_MULTIPLEXING=false for Cargo builds
    if 'cargo' in eval_script and 'CARGO_HTTP_MULTIPLEXING' not in eval_script:
        eval_script = re.sub(
            r'^(cd\s+/testbed.*\n)',
            r'\1' + 'export CARGO_HTTP_MULTIPLEXING=false\n',
            eval_script,
            count=1,
            flags=re.MULTILINE,
        )

    # 4. Cap Go build/runtime parallelism on Go instances.
    #
    # `go build`/`go test` run up to -p=NumCPU build actions at once. With 16 CPUs
    # that is enough concurrent linker processes for the Go linker to die on itself:
    #
    #   unexpected fault address 0x...
    #   fatal error: fault
    #   [signal SIGSEGV: segmentation violation]
    #   cmd/internal/goobj.(*Reader).Relocs / (*FuncInfo).ReadFuncInfoLengths /
    #   (*InlTreeNode).Read / cmd/link/internal/ld.decodetypeStr
    #
    # i.e. the linker reads outside its own mmap of an object file it just wrote.
    # It is a linker/kernel concurrency bug, not resource exhaustion: measured in a
    # held sandbox, peak memory was 2.9 GB of a 32 GB limit and the disk had TBs
    # free, and moving GOCACHE to a tmpfs did not help. Only the concurrency level
    # matters -- cold-cache builds of the two largest Go repos in the RL set
    # (pingcap/tidb, dolthub/doltgresql) faulted 10/10 at -p=NumCPU and 0/4 serial.
    # Capping concurrency is therefore the only lever available from the harness.
    #
    # This mirrors the swe_bench_pro handler's fix (MR !492, "fix(swe_bench_pro):
    # cap GOMAXPROCS=4 on Go instances to avoid 1.24 linker SIGSEGV"), which hit the
    # same crash from the same cause and was validated across 5 rounds of the full
    # 731-instance Pro set (730/731). That fix only covered swe_bench_pro, which is
    # why these instances -- dataset_type T3Build_SWEBuilder_Midtrain, a different
    # handler -- were still failing.
    #
    # GOMAXPROCS rather than GOFLAGS=-p=N to stay identical to !492. Note that repos
    # deriving their own parallelism from it can scale it back up -- tidb's
    # tools/check/ut.go does `buildParallel = GOMAXPROCS * 2`, so 4 becomes 8 build
    # actions -- but 8 links cleanly (verified: 3/3 rounds, both instances) while the
    # uncapped 32 does not, and 4 keeps roughly 3x the throughput of full serial.
    #
    # Guarded on absence rather than appended unconditionally: unlike the Pro handler,
    # which rebuilds env_cmds from scratch, this one rewrites an existing script in
    # place, so an unguarded insert would stack a duplicate export on a second pass.
    # Skipping a script that sets GOMAXPROCS itself also respects hand-tuned values.
    is_go = _instance_language(instance) == 'go'
    if is_go and 'GOMAXPROCS' not in eval_script:
        eval_script = re.sub(
            r'^(cd\s+/testbed.*\n)',
            r'\1' + 'export GOMAXPROCS=4\n',
            eval_script,
            count=1,
            flags=re.MULTILINE,
        )

    # (conda/Nix activation is skipped — SWE Builder images use system python, not conda.)

    # Preserve the Go limit for images whose repository is outside /testbed.
    if _instance_language(instance) == 'go' and 'GOMAXPROCS' not in eval_script:
        export_line = 'export GOMAXPROCS=4\n'
        patched, count = re.subn(r'^(cd\s+\S+.*\n)', r'\1' + export_line, eval_script, count=1, flags=re.MULTILINE)
        if not count:
            patched, count = re.subn(
                r'^(set\s+-\S*.*\n)', r'\1' + export_line, eval_script, count=1, flags=re.MULTILINE
            )
        if count:
            eval_script = patched

    return eval_script


def get_eval_script(instance: SWEbenchInstance) -> str:
    """Generate the evaluation script for the instance.

    Supports two modes:
    1. Pre-built eval_script from instance (with post-processing fixes)
    2. Dynamic generation from test_patch + inferred test commands (f2p multilingual)
    """
    eval_script = instance.get('eval_script', '')
    if eval_script:
        return _postprocess_eval_script(eval_script, instance)

    # Dynamic generation for instances without eval_script (e.g. f2p multilingual data)
    # We build the eval script directly instead of using make_eval_script_list,
    # because make_eval_script_list's get_test_cmds reads from instance['spec_dict']
    # directly (ignoring our overrides) and would always fallback to pytest.
    from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.test_spec import get_modified_files

    base_commit = instance.get('base_commit', '')
    test_patch = instance.get('test_patch', '')
    working_dir = instance.get('workdir') or '/testbed'

    if not test_patch:
        raise ValueError(f'Instance {instance.get("instance_id")} has neither eval_script nor test_patch')

    test_commands = _infer_test_cmds(instance)
    test_files = get_modified_files(test_patch)
    heredoc_delimiter = 'EOF_114329324912'

    if test_files:
        # Quote each file path to handle filenames with special characters (spaces, apostrophes, etc.)
        quoted_files = ' '.join(f'"{f}"' for f in test_files)
        reset_tests_command = f'git checkout {base_commit} {quoted_files}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    # For projects that may use pytest (Python, C++), activate the right environment.
    # f2p Docker images use either conda or Nix to provide python/pytest.
    lang = _instance_language(instance)
    uses_pytest = lang in ('python', 'c++') or any('pytest' in cmd for cmd in test_commands)

    # Try conda first, then detect Nix environments where pytest lives in a Nix profile.
    # The order matters: conda activate may silently fail in Nix images, so we also
    # check for Nix and add its bin paths to ensure `python -m pytest` can find pytest.
    env_setup_lines: list[str] = []
    if uses_pytest:
        env_setup_lines.append(
            'source /opt/miniconda3/bin/activate 2>/dev/null && conda activate testbed 2>/dev/null || true'
        )
        # Nix environments: pytest is installed in a Nix profile, not in conda.
        # Detect Nix and prepend its bin paths so `python` and `pytest` resolve correctly.
        env_setup_lines.append(
            'if [ -d /nix ]; then'
            ' export PATH="/nix/runtime/nix-env/bin:$PATH";'
            ' NIX_PYTHON=$(find /nix -name "python3" -path "*/bin/*" 2>/dev/null | head -1);'
            ' if [ -n "$NIX_PYTHON" ]; then'
            '   NIX_BIN=$(dirname "$NIX_PYTHON");'
            '   export PATH="$NIX_BIN:$PATH";'
            ' fi;'
            ' fi'
        )

    uses_gradle = any('gradlew' in cmd for cmd in test_commands)
    gradle_proxy_lines = (
        [
            'if [ -n "$https_proxy" ]; then'
            " PROXY_HOST=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f1);"
            " PROXY_PORT=$(echo $https_proxy | sed 's|http://||;s|https://||' | cut -d: -f2);"
            ' export GRADLE_OPTS="-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'
            ' -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT";'
            ' fi',
        ]
        if uses_gradle
        else []
    )

    eval_script_list = [
        'export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt',
        f'cd {working_dir}',
        f'git config --global --add safe.directory {working_dir}',
        *gradle_proxy_lines,
        *env_setup_lines,
        reset_tests_command,
        apply_test_patch_command,
        *test_commands,
        'TEST_EXIT_CODE=$?',
        reset_tests_command,
        'echo OMNIGRIL_EXIT_CODE=$TEST_EXIT_CODE',
    ]

    # NOTE: do NOT use `set -u` (nounset).  Some test_patch files contain special
    # characters (e.g. apostrophes in snapshot filenames) that can break heredoc
    # parsing mid-script, leaving TEST_EXIT_CODE unset and crashing the final
    # `echo OMNIGRIL_EXIT_CODE=$TEST_EXIT_CODE` line.
    return '\n'.join(['#!/bin/bash', 'set -xo pipefail'] + eval_script_list) + '\n'


def get_eval_report_map(
    instance: SWEbenchInstance,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """Get the evaluation report map.

    Uses OMNIGRIL_EXIT_CODE to determine if tests passed.
    Falls back to t3build_repolaunch_midtrain log-parser-based grading
    when OMNIGRIL_EXIT_CODE is not found (e.g. for dynamically generated scripts).
    """
    instance_id = instance['instance_id']

    report_map: dict[str, Any] = {
        instance_id: {
            'patch_is_None': False,
            'patch_exists': False,
            # Tri-state: True / False / None(=UNKNOWN, could not be determined).
            # See the `else` branch below for why UNKNOWN exists.
            'patch_successfully_applied': False,
            'resolved': False,
        }
    }

    # Treat missing or whitespace-only patch text as no applicable change.
    model_patch = prediction.get('model_patch')
    if not model_patch or not model_patch.strip():
        report_map[instance_id]['patch_is_None'] = True
        return report_map
    report_map[instance_id]['patch_exists'] = True
    report_map[instance_id]['patch_successfully_applied'] = None

    test_log_content = ''
    test_log_file = Path(test_log_path)
    if test_log_file.exists():
        test_log_content = test_log_file.read_text(encoding='utf-8')

    # Check patch application status from run_instance logs
    run_instance_log_content = ''

    run_instance_after_apply_log = test_log_file.parent / 'run_instance_after_apply.log'
    if run_instance_after_apply_log.exists():
        run_instance_log_content = run_instance_after_apply_log.read_text(encoding='utf-8')
    else:
        run_instance_log_path = test_log_file.parent / 'run_instance.log'
        if run_instance_log_path.exists():
            run_instance_log_content = run_instance_log_path.read_text(encoding='utf-8')

    if run_instance_log_content:
        if APPLY_PATCH_PASS in run_instance_log_content:
            report_map[instance_id]['patch_successfully_applied'] = True
        elif APPLY_PATCH_FAIL in run_instance_log_content:
            report_map[instance_id]['patch_successfully_applied'] = False
    else:
        if APPLY_PATCH_PASS in test_log_content:
            report_map[instance_id]['patch_successfully_applied'] = True
        elif APPLY_PATCH_FAIL in test_log_content:
            report_map[instance_id]['patch_successfully_applied'] = False

    # Check exit code for resolution status
    if test_log_content:
        match = EXIT_CODE_PATTERN.search(test_log_content)
        if match:
            exit_code = int(match.group(1))
            if exit_code == 0:
                report_map[instance_id]['resolved'] = True
        else:
            # Fallback: try log-parser-based grading for dynamically generated scripts
            report_map = _fallback_log_parser_grading(
                instance, prediction, test_log_path, include_tests_status, report_map
            )

    return report_map


def _fallback_log_parser_grading(
    instance: SWEbenchInstance,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
    report_map: dict[str, Any],
) -> dict[str, Any]:
    """Fallback grading using t3build_repolaunch_midtrain's log parser when OMNIGRIL_EXIT_CODE is absent."""
    try:
        from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.constants import get_repo_ext
        from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.grading import get_eval_report
        from nexus.extensions.tasks.swe.t3build_repolaunch_midtrain.test_spec import TestSpec

        instance_id = instance['instance_id']

        def _from_json_or_obj(key: str) -> Any:
            instance_dict = cast(dict[str, Any], instance)
            if key not in instance_dict:
                return []
            if isinstance(instance_dict[key], str):
                return json.loads(instance_dict[key])
            return instance_dict[key]

        test_spec = TestSpec(
            instance=instance,
            instance_id=instance_id,
            repo=instance.get('repo', ''),
            version=instance.get('version', 'latest'),
            eval_script_list=[],
            FAIL_TO_PASS=_from_json_or_obj('FAIL_TO_PASS'),
            PASS_TO_PASS=_from_json_or_obj('PASS_TO_PASS'),
            language=get_repo_ext(instance),
        )

        fallback_report = get_eval_report(test_spec, prediction, test_log_path, include_tests_status)
        # Merge fallback results, preserving patch application status from original report_map
        if instance_id in fallback_report:
            patch_applied = report_map[instance_id]['patch_successfully_applied']
            report_map[instance_id].update(fallback_report[instance_id])
            report_map[instance_id]['patch_successfully_applied'] = patch_applied
    except Exception:
        pass  # If fallback fails, keep original report_map with resolved=False

    return report_map


def get_problem_statement(instance: SWEbenchInstance) -> str:
    """Get the problem statement for the instance."""
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    """Get the repository language from the instance's language field."""
    return instance.get('language') or None


def get_test_command(instance: SWEbenchInstance, timeout_seconds: int = 1800) -> str | None:
    """Get a quick test command for intermediate testing (multi-turn).

    Infers the test command from FAIL_TO_PASS prefixes and language,
    supporting multilingual repos (Python, Java, Rust, C#, C++, TypeScript).
    """
    f2p = _parse_fail_to_pass(instance)
    if not f2p:
        return None

    working_dir = instance.get('workdir') or '/testbed'
    test_cmds = _infer_test_cmds(instance)
    cmd = ' && '.join(test_cmds)

    # For pytest commands, also set up conda/Nix environment
    env_setup = (
        'source /opt/miniconda3/bin/activate 2>/dev/null && conda activate testbed 2>/dev/null || true;'
        ' if [ -d /nix ]; then'
        ' export PATH="/nix/runtime/nix-env/bin:$PATH";'
        ' NIX_PYTHON=$(find /nix -name "python3" -path "*/bin/*" 2>/dev/null | head -1);'
        ' if [ -n "$NIX_PYTHON" ]; then export PATH="$(dirname $NIX_PYTHON):$PATH"; fi;'
        ' fi;'
    )
    if any('pytest' in c for c in test_cmds):
        cmd = f'{env_setup} {cmd}'

    return f'timeout {timeout_seconds} bash -c "cd {working_dir} && {cmd}" 2>&1 | tail -50'


async def infer_runtime_init(
    instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
) -> None:
    """Initialize runtime for inference.

    Follows the same pattern as t3build_repolaunch_midtrain:
    1. Configure git safe.directory
    2. Reset tracked files to base_commit
    3. Clean untracked files
    4. Remove remote refs
    5. Truncate git history
    """
    from nexus.extensions.tasks.swe.infer import run_git_truncate

    working_dir = instance.get('workdir', '/testbed')
    base_commit = instance.get('base_commit', '')

    # 0. Configure git safe.directory to avoid ownership issues in container
    safe_dir_cmd = f'git config --global --add safe.directory {working_dir}'
    await runtime.call_executor('run_command', {'command': safe_dir_cmd, 'cwd': working_dir})

    # 1. Reset tracked files to base_commit state
    if base_commit:
        reset_cmd = f'git reset --hard {base_commit}'
        await runtime.call_executor('run_command', {'command': reset_cmd, 'cwd': working_dir})

    # 2. Clean untracked files and directories
    clean_cmd = 'git clean -fdx'
    await runtime.call_executor('run_command', {'command': clean_cmd, 'cwd': working_dir})

    # 3. Remove all remote refs before truncation
    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    remove_remotes_cmd = (
        'git remote remove origin 2>/dev/null || true && '
        'rm -rf .git/refs/remotes && '
        "sed -i '/refs\\/remotes\\//d' .git/packed-refs 2>/dev/null || true"
    )
    await runtime.call_executor('run_command', {'command': remove_remotes_cmd, 'cwd': working_dir})

    # 4. Truncate git history to prevent peeking at future changes
    if not skip_git_truncate:
        await run_git_truncate(runtime, base_commit, working_dir)


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Remove untracked residue while preserving ignored dependencies baked into the image."""
    import shlex

    working_dir = instance.get('workdir', '/testbed')
    await runtime.call_executor(
        'run_command',
        {'command': f'git config --global --add safe.directory {shlex.quote(working_dir)}', 'cwd': working_dir},
    )
    await runtime.call_executor('run_command', {'command': 'git clean -fd', 'cwd': working_dir})


def eval_write_test_output(instance: SWEbenchInstance, path: Path, stdout: str | None) -> None:
    """Write test output to file.

    Uses OMNIGRIL_EXIT_CODE in the output to determine resolution.
    """
    path.write_text(stdout or '', encoding='utf-8')
