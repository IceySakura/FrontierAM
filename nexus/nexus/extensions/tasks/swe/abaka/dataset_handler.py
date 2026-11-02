"""Dataset handler for abaka dataset.

Abaka dataset has pre-built evaluation scripts and images.
The evaluation uses exit code to determine resolution status.
"""

import re

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.runtime import Runtime


# Constants for log parsing - matching evaluation/evaluation.py
APPLY_PATCH_FAIL = '>>>>> Patch Apply Failed'
APPLY_PATCH_PASS = '>>>>> Patch Apply Passed'
# Match both formats: "echo OMNIGRIL_EXIT_CODE=0" and "OMNIGRIL_EXIT_CODE=0"
EXIT_CODE_PATTERN = re.compile(r'(?:echo )?OMNIGRIL_EXIT_CODE=(\d+)')

# Mapping of repo to language extension
# All repos in abaka dataset are Python projects
MAP_REPO_TO_LANGUAGE: dict[str, str] = {
    'aio-libs/aiohttp': 'python',
    'autoscrape-labs/pydoll': 'python',
    'bloomberg/memray': 'python',
    'canonical/microk8s': 'python',
    'cookiecutter/cookiecutter-django': 'python',
    'coursera-dl/coursera-dl': 'python',
    'dbcli/mycli': 'python',
    'dbcli/pgcli': 'python',
    'dbt-labs/dbt-core': 'python',
    'dgtlmoon/changedetection.io': 'python',
    'docling-project/docling': 'python',
    'encode/django-rest-framework': 'python',
    'fail2ban/fail2ban': 'python',
    'kivy/kivy': 'python',
    'magic-wormhole/magic-wormhole': 'python',
    'marimo-team/marimo': 'python',
    'mem0ai/mem0': 'python',
    'mindsdb/mindsdb': 'python',
    'mitmproxy/mitmproxy': 'python',
    'paperless-ngx/paperless-ngx': 'python',
    'piskvorky/gensim': 'python',
    'prowler-cloud/prowler': 'python',
    'pypa/pipenv': 'python',
    'pypa/pipx': 'python',
    'python-poetry/poetry': 'python',
    'python-telegram-bot/python-telegram-bot': 'python',
    'reflex-dev/reflex': 'python',
    'spotDL/spotify-downloader': 'python',
    'tqdm/tqdm': 'python',
}


def get_eval_script(instance: SWEbenchInstance) -> str:
    """Generate the evaluation script for the instance.

    Abaka dataset has the eval_script pre-embedded in the instance.
    """
    eval_script = instance.get('eval_script', '')
    if not eval_script:
        raise ValueError(f'Instance {instance.get("instance_id")} does not have eval_script')
    return eval_script


def get_eval_report_map(
    instance: SWEbenchInstance,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool,
) -> dict[str, Any]:
    """Get the evaluation report map for an abaka instance.

    Abaka uses OMNIGRIL_EXIT_CODE to determine if tests passed.
    Exit code 0 means all tests passed (resolved).

    This follows the same logic as evaluation/evaluation.py:get_pred_report.
    """
    instance_id = instance['instance_id']

    report_map: dict[str, Any] = {
        instance_id: {
            'patch_is_None': False,
            'patch_exists': False,
            'patch_successfully_applied': False,
            'resolved': False,
        }
    }

    # Check if model patch exists
    model_patch = prediction.get('model_patch')
    if model_patch is None:
        report_map[instance_id]['patch_is_None'] = True
        return report_map
    report_map[instance_id]['patch_exists'] = True

    # Read test log
    test_log_content = ''
    test_log_file = Path(test_log_path)
    if test_log_file.exists():
        test_log_content = test_log_file.read_text(encoding='utf-8')

    # Check patch application status
    # Following evaluation/evaluation.py logic:
    # 1. First check run_instance_after_apply.log (for fail2pass mode)
    # 2. Then check run_instance.log
    # 3. Finally check the test log itself
    run_instance_log_content = ''

    # Try run_instance_after_apply.log first (evaluation.py style)
    run_instance_after_apply_log = test_log_file.parent / 'run_instance_after_apply.log'
    if run_instance_after_apply_log.exists():
        run_instance_log_content = run_instance_after_apply_log.read_text(encoding='utf-8')
    else:
        # Fallback to run_instance.log
        run_instance_log_path = test_log_file.parent / 'run_instance.log'
        if run_instance_log_path.exists():
            run_instance_log_content = run_instance_log_path.read_text(encoding='utf-8')

    if run_instance_log_content:
        if APPLY_PATCH_PASS in run_instance_log_content:
            report_map[instance_id]['patch_successfully_applied'] = True
        elif APPLY_PATCH_FAIL in run_instance_log_content:
            report_map[instance_id]['patch_successfully_applied'] = False
    else:
        # If no run instance log, check test log for patch status
        if APPLY_PATCH_PASS in test_log_content:
            report_map[instance_id]['patch_successfully_applied'] = True
        elif APPLY_PATCH_FAIL in test_log_content:
            report_map[instance_id]['patch_successfully_applied'] = False
        else:
            # Assume patch applied if we have test output
            report_map[instance_id]['patch_successfully_applied'] = bool(test_log_content)

    # Check exit code for resolution status
    # Following evaluation/evaluation.py logic
    if test_log_content:
        match = EXIT_CODE_PATTERN.search(test_log_content)
        if match:
            exit_code = int(match.group(1))
            if exit_code == 0:
                report_map[instance_id]['resolved'] = True

    return report_map


def get_problem_statement(instance: SWEbenchInstance) -> str:
    """Get the problem statement for the instance."""
    return instance.get('problem_statement') or instance.get('problem') or ''


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    """Get the repository language.

    Returns the language based on repo mapping or defaults to python.
    """
    repo = instance.get('repo', '')
    return MAP_REPO_TO_LANGUAGE.get(repo, 'python')


async def infer_runtime_init(
    instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
) -> None:
    """Initialize runtime for abaka inference.

    Reset repository to base_commit and clean up untracked files to ensure
    git diff only captures agent's changes.

    The images may have:
    1. Modified tracked files
    2. Extra untracked files (e.g., pip install artifacts like "=1.2.0")

    Both would pollute the model_patch if not handled.
    """
    from nexus.extensions.tasks.swe.infer import run_git_truncate

    base_commit = instance.get('base_commit', '')
    working_dir = instance.get('workdir', '/testbed')

    # 1. Reset tracked files to base_commit state
    if base_commit:
        reset_cmd = f'git reset --hard {base_commit}'
        await runtime.call_executor('run_command', {'command': reset_cmd, 'cwd': working_dir})

    # 2. Clean untracked files and directories to ensure clean git diff
    # -f: force, -d: directories, -x: also remove ignored files
    clean_cmd = 'git clean -fdx'
    await runtime.call_executor('run_command', {'command': clean_cmd, 'cwd': working_dir})

    # 3. Remove .git or truncate git history
    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    # Remove all remote refs (including symbolic refs like origin/HEAD) before truncation
    # This prevents "fatal: bad object refs/remotes/origin/HEAD" error during git gc
    # when the underlying ref it points to gets deleted by git truncate
    # Also remove remote config to prevent git pull/fetch from re-downloading remote info
    remove_remotes_cmd = (
        'git remote remove origin 2>/dev/null || true && '
        'rm -rf .git/refs/remotes && '
        "sed -i '/refs\\/remotes\\//d' .git/packed-refs 2>/dev/null || true"
    )
    await runtime.call_executor('run_command', {'command': remove_remotes_cmd, 'cwd': working_dir})

    # 4. Truncate git history to prevent peeking at future changes
    if not skip_git_truncate:
        await run_git_truncate(runtime, base_commit, working_dir)


def eval_write_test_output(instance: SWEbenchInstance, path: Path, stdout: str | None) -> None:
    """Write test output to file.

    This follows the default behavior - write stdout directly to the output file.
    Abaka dataset uses OMNIGRIL_EXIT_CODE in the output to determine resolution.
    """
    path.write_text(stdout or '', encoding='utf-8')
