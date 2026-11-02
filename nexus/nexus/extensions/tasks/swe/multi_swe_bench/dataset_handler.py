from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.multi_swe_bench.image import Config
from nexus.extensions.tasks.swe.multi_swe_bench.instance import Instance
from nexus.extensions.tasks.swe.multi_swe_bench.pull_request import PullRequest
from nexus.extensions.tasks.swe.multi_swe_bench.test_result import TestStatus
from nexus.runtime import Runtime


def _write_patch_file(path: str, content: str) -> None:
    """Write patch content to a file (executed in runtime)."""
    Path(path).write_text(content, encoding='utf-8')


def _write_patch_file_if_not_exists(path: str, content: str) -> None:
    """Write patch content to a file only if it doesn't exist (executed in runtime)."""
    p = Path(path)
    if not p.exists():
        p.write_text(content, encoding='utf-8')


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    return instance['language']


def get_eval_script(instance: SWEbenchInstance) -> str:
    """
    从 SWEbenchInstance 生成 test-run.sh 脚本
    Args:
        instance: 测试实例信息
    Returns:
        test-run.sh 脚本的内容
    """
    # 优先从最外层读取 scripts 字段
    if 'scripts' in instance:
        test_run_script = instance['scripts'].get('test_run')  # type: ignore[typeddict-item]
        if test_run_script:
            return test_run_script

    # Fallback: 使用旧的 Instance 实现
    pull_request = PullRequest.from_dict(instance)  # type: ignore[arg-type]

    config = Config(
        need_clone=False,
        global_env=None,
        clear_env=False,
    )
    instance_obj = Instance.create(pull_request, config)
    image = instance_obj.dependency()
    files = image.files()
    for file in files:
        if file.name == 'test-run.sh':
            return file.content

    raise ValueError(f'test-run.sh not found for instance {instance.get("instance_id")}')


def get_eval_report_map(
    instance: SWEbenchInstance | dict,
    prediction: dict[str, str],
    test_log_path: str,
    include_tests_status: bool = False,
) -> dict[str, Any]:
    """
    generate eval report based on test-patch-run.log

    Args:
        instance: SWE-bench instance, include: f2p_tests, p2p_tests
        prediction: include: 'instance_id', 'model_patch', 'model_name_or_path'
        test_log_path: test log path
        include_tests_status: whether to include tests status in the report

    Returns:
        report_map:
        {
            'instance_id': {
                'patch_is_None': bool,
                'patch_exists': bool,
                'patch_successfully_applied': bool,
                'resolved': bool,
                'tests_status': {
                    'FAIL_TO_PASS': {'success': [...], 'failure': [...]},
                    'PASS_TO_PASS': {'success': [...], 'failure': [...]},
                    ...
                }
            }
        }
    """
    report_map = {}

    instance_id = prediction['instance_id']

    report_map[instance_id] = {
        'patch_is_None': False,
        'patch_exists': False,
        'patch_successfully_applied': False,
        'resolved': False,
    }

    model_patch = prediction['model_patch']
    if model_patch is None or model_patch == '':
        report_map[instance_id]['patch_is_None'] = True
        return report_map

    report_map[instance_id]['patch_exists'] = True

    test_log_path_obj = Path(test_log_path)
    if not test_log_path_obj.exists():
        return report_map

    with open(test_log_path_obj, encoding='utf-8', errors='replace') as f:
        test_log = f.read()
    if not test_log or len(test_log.strip()) == 0:
        return report_map

    report_map[instance_id]['patch_successfully_applied'] = True

    instance_obj = get_instance_from_dict(instance)  # type: ignore[arg-type]
    fix_result = instance_obj.parse_log(test_log)

    eval_status_map = {}
    for test_name in fix_result.passed_tests:
        eval_status_map[test_name] = TestStatus.PASSED.value
    for test_name in fix_result.failed_tests:
        eval_status_map[test_name] = TestStatus.FAILED.value

    f2p_tests = instance.get('f2p_tests', {})
    p2p_tests = instance.get('p2p_tests', {})
    s2p_tests = instance.get('s2p_tests', {})
    n2p_tests = instance.get('n2p_tests', {})

    f2p_test_names = list(f2p_tests.keys()) if isinstance(f2p_tests, dict) else []
    p2p_test_names = list(p2p_tests.keys()) if isinstance(p2p_tests, dict) else []
    s2p_test_names = list(s2p_tests.keys()) if isinstance(s2p_tests, dict) else []
    n2p_test_names = list(n2p_tests.keys()) if isinstance(n2p_tests, dict) else []

    def is_test_passed(test_name: str, status_map: dict[str, str]) -> bool:
        return status_map.get(test_name) == TestStatus.PASSED.value

    # 新的resolved判断逻辑：只要eval通过的测试包含fix_patch_result中通过的测试即可
    resolved = True
    fix_patch_result = instance.get('fix_patch_result', {})
    if fix_patch_result and 'passed_tests' in fix_patch_result:
        # 获取fix_patch_result中通过的测试列表
        fix_passed_tests = fix_patch_result['passed_tests']

        # 检查fix_patch_result中通过的所有测试是否在eval中也通过了
        for test_name in fix_passed_tests:
            if not is_test_passed(test_name, eval_status_map):
                resolved = False
                break
    else:
        # 如果没有fix_patch_result，使用旧逻辑（所有f2p/p2p/s2p/n2p测试都要通过）
        for test_name in f2p_test_names:
            if not is_test_passed(test_name, eval_status_map):
                resolved = False
                break
        if resolved:
            for test_name in p2p_test_names:
                if not is_test_passed(test_name, eval_status_map):
                    resolved = False
                    break
        if resolved:
            for test_name in s2p_test_names:
                if not is_test_passed(test_name, eval_status_map):
                    resolved = False
                    break
        if resolved:
            for test_name in n2p_test_names:
                if not is_test_passed(test_name, eval_status_map):
                    resolved = False
                    break

    report_map[instance_id]['resolved'] = resolved

    if include_tests_status:
        tests_status = _build_tests_status(
            eval_status_map,
            f2p_test_names,
            p2p_test_names,
            s2p_test_names,
            n2p_test_names,
        )
        report_map[instance_id]['tests_status'] = tests_status

    return report_map


def _build_tests_status(
    eval_status_map: dict[str, str],
    f2p_tests: list[str],
    p2p_tests: list[str],
    s2p_tests: list[str],
    n2p_tests: list[str],
) -> dict[str, dict[str, list[str]]]:
    """
    build tests status

    Args:
        eval_status_map: name -> status mapping
        f2p_tests: Fail to Pass test list
        p2p_tests: Pass to Pass test list
        s2p_tests: Skip to Pass test list
        n2p_tests: None to Pass test list

    Returns:
        {
            'FAIL_TO_PASS': {'success': [...], 'failure': [...]},
            'PASS_TO_PASS': {'success': [...], 'failure': [...]},
            'SKIP_TO_PASS': {'success': [...], 'failure': [...]},
            'NONE_TO_PASS': {'success': [...], 'failure': [...]},
        }
    """

    def categorize_tests(test_names: list[str]) -> tuple[list[str], list[str]]:
        success = []
        failure = []
        for test_name in test_names:
            status = eval_status_map.get(test_name, TestStatus.NONE.value)
            if status == TestStatus.PASSED.value:
                success.append(test_name)
            else:
                failure.append(test_name)
        return success, failure

    f2p_success, f2p_failure = categorize_tests(f2p_tests)
    p2p_success, p2p_failure = categorize_tests(p2p_tests)
    s2p_success, s2p_failure = categorize_tests(s2p_tests)
    n2p_success, n2p_failure = categorize_tests(n2p_tests)

    return {
        'FAIL_TO_PASS': {
            'success': f2p_success,
            'failure': f2p_failure,
        },
        'PASS_TO_PASS': {
            'success': p2p_success,
            'failure': p2p_failure,
        },
        'SKIP_TO_PASS': {
            'success': s2p_success,
            'failure': s2p_failure,
        },
        'NONE_TO_PASS': {
            'success': n2p_success,
            'failure': n2p_failure,
        },
    }


def get_instance_from_dict(instance_dict: dict[str, Any]) -> Instance:
    """从字典创建 Instance 对象，优先使用最外层的配置字段"""
    from nexus.extensions.tasks.swe.multi_swe_bench.generic_instance import GenericInstance

    pr = PullRequest.from_dict(instance_dict)
    config = Config(
        need_clone=False,
        global_env=None,
        clear_env=False,
    )

    # 如果有 dataset_type 字段，说明是新格式，使用 GenericInstance
    if 'dataset_type' in instance_dict:
        return GenericInstance(pr, config, instance_dict)

    # 否则使用旧的注册机制
    return Instance.create(pr, config)


def get_runtime_init_cmds(instance: SWEbenchInstance) -> list[str] | None:  # noqa: ARG001
    """SWE-bench images contain buggy code, no initialization commands needed."""
    return None


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    """Initialize runtime for multi-swe-bench evaluation.

    Reset repository to base_commit before applying model patch to ensure
    the repo state matches the expected base. Also:
    1. Create test.patch file if it doesn't exist (from instance data)
    2. Run prepare.sh for dependencies
    """
    base_commit = instance.get('base_commit', '')
    if not base_commit:
        base_commit = instance.get('base', {}).get('sha', '')

    working_dir = instance.get('workdir', '/testbed')

    # Reset to base commit and clean untracked files
    # Use 'git clean -fd' instead of 'git clean -fdx' to preserve .gitignore'd files
    # like vendor/ (PHP), node_modules/ (Node.js), which contain installed dependencies
    if base_commit:
        reset_cmd = f'cd {working_dir} && git reset --hard {base_commit} && git clean -fd'
        await runtime.call_executor('run_command', {'command': reset_cmd, 'cwd': working_dir})

    # Special handling for checkstyle repo
    repo = instance.get('repo', '')
    if repo == 'checkstyle':
        # 1. Clean target directory (Maven has issues deleting it)
        clean_target_cmd = f'rm -rf {working_dir}/target 2>/dev/null || true'
        await runtime.call_executor('run_command', {'command': clean_target_cmd, 'cwd': working_dir})

        # 2. Modify test scripts to exclude testExistingTargetFileButWithoutReadAccess
        # This test fails in Docker because root can read any file (ignores permissions)
        # The test expects exit(-1) when file has no read permission, but root gets exit(0)
        exclude_test_cmd = """
for script in /home/test-run.sh /home/fix-run.sh /home/run.sh; do
    if [ -f "$script" ]; then
        # Add Maven exclude parameter for the permission test
        sed -i 's|mvn clean test|mvn clean test -Dtest=!MainTest#testExistingTargetFileButWithoutReadAccess|g' "$script"
    fi
done
"""
        await runtime.call_executor('run_command', {'command': exclude_test_cmd, 'cwd': '/'})

    # Create test.patch if it doesn't exist (needed by test_run scripts in SWEImageDefault images)
    # Note: only create test.patch (test patch), not fix.patch (gold patch)
    # Use shell command with base64 encoding to avoid special character issues
    test_patch = instance.get('test_patch', '')
    if test_patch:
        import base64

        encoded_patch = base64.b64encode(test_patch.encode('utf-8')).decode('ascii')
        write_patch_cmd = f'[ ! -f /home/test.patch ] && echo "{encoded_patch}" | base64 -d > /home/test.patch || true'
        await runtime.call_executor('run_command', {'command': write_patch_cmd, 'cwd': '/'})

    # Execute prepare.sh if it exists (installs dependencies like npm ci, pip install, etc.)
    # Skip for specific repos where image already has dependencies installed
    # and re-running prepare.sh causes gyp compilation failures
    repo = instance.get('repo', '')

    # Repos to skip prepare.sh (dependencies pre-installed in image)
    skip_prepare_repos = {'vscode'}

    if repo in skip_prepare_repos:
        # Skip prepare.sh for these repos - dependencies already installed during image build
        pass
    else:
        prepare_script = '/home/prepare.sh'
        run_prepare_cmd = f"""
if [ -f {prepare_script} ]; then
    # Create modified prepare script with correct paths
    sed 's|/home/{repo}|{working_dir}|g' {prepare_script} > /tmp/prepare_modified.sh
    chmod +x /tmp/prepare_modified.sh
    cd {working_dir} && bash /tmp/prepare_modified.sh || true
fi
"""
        await runtime.call_executor('run_command', {'command': run_prepare_cmd, 'cwd': working_dir})


def eval_write_test_output(instance: SWEbenchInstance, path: Path, stdout: str | None) -> None:
    path.write_text(stdout or '', encoding='utf-8')
    return


def get_problem_statement(instance) -> str:
    resolved_issues = instance.get('resolved_issues', [])
    if resolved_issues and len(resolved_issues) > 0:
        return resolved_issues[0].get('body', '')
    return ''


async def infer_runtime_init(
    instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
) -> None:
    """Initialize runtime for multi-swe-bench inference.

    1. Reset repository to base_commit
    2. Run prepare.sh to install dependencies (so agent can run tests)
    3. Optionally truncate git history to prevent peeking
    """

    base_commit = instance.get('base_commit', '')
    if not base_commit:
        base_commit = instance.get('base', {}).get('sha', '')

    working_dir = instance.get('workdir', '/testbed')

    # Reset to base commit
    # Use 'git clean -fd' instead of 'git clean -fdx' to preserve .gitignore'd files
    # like vendor/ (PHP), node_modules/ (Node.js), which contain installed dependencies
    if base_commit:
        reset_cmd = f'cd {working_dir} && git reset --hard {base_commit} && git clean -fd'
        await runtime.call_executor('run_command', {'command': reset_cmd, 'cwd': working_dir})

    # Execute prepare.sh to install dependencies (agent may need to run tests)
    # Skip for specific repos where image already has dependencies installed
    repo = instance.get('repo', '')

    # Repos to skip prepare.sh (dependencies pre-installed in image)
    skip_prepare_repos = {'vscode'}

    if repo in skip_prepare_repos:
        # Skip prepare.sh for these repos - dependencies already installed during image build
        pass
    else:
        prepare_script = '/home/prepare.sh'
        run_prepare_cmd = f"""
if [ -f {prepare_script} ]; then
    sed 's|/home/{repo}|{working_dir}|g' {prepare_script} > /tmp/prepare_modified.sh
    chmod +x /tmp/prepare_modified.sh
    cd {working_dir} && bash /tmp/prepare_modified.sh || true
fi
"""
        await runtime.call_executor('run_command', {'command': run_prepare_cmd, 'cwd': working_dir})

    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    # Truncate git history to prevent peeking at future changes
    remove_remotes_cmd = "git remote remove origin && rm -rf .git/refs/remotes && sed -i '/refs\\/remotes\\//d' .git/packed-refs 2>/dev/null || true"
    await runtime.call_executor('run_command', {'command': remove_remotes_cmd, 'cwd': working_dir})

    if not skip_git_truncate:
        from nexus.extensions.tasks.swe.infer import run_git_truncate

        await run_git_truncate(runtime, base_commit, working_dir)
