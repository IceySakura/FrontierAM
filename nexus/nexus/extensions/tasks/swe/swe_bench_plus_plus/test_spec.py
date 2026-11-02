"""
Test spec for SWE-Bench-plus-plus dataset.
Aligned with SWE-Bench-plus-plus/swebench/harness/test_spec/ original implementation.
"""

import json
import os
import re

from dataclasses import dataclass, field
from typing import Any

from unidiff import PatchSet

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench_plus_plus.constants import (
    END_TEST_OUTPUT,
    KEY_INSTANCE_ID,
    LANGUAGES_STR_MAP,
    NON_TEST_EXTS,
    START_TEST_OUTPUT,
    get_ext_from_language,
)
from nexus.extensions.tasks.swe.swe_bench_plus_plus.log_parsers import LANGUAGE_PARSER_MAP


@dataclass
class TestSpec:
    """
    A dataclass that represents a test specification for a single instance of SWE-bench-plus-plus.
    """

    instance_id: str
    repo: str
    version: str
    eval_script_list: list[str]
    FAIL_TO_PASS: list[str]
    PASS_TO_PASS: list[str]
    language: str
    environment_config: dict = field(default_factory=dict)
    log_parser: Any = None

    def _get_env_setup_commands(self) -> list[str]:
        """
        Get environment setup commands based on language.
        This ensures the correct PATH and environment variables are set
        when running eval scripts via subprocess (which doesn't inherit Docker ENV).
        """
        env_cmds = []

        if self.language == 'go':
            # Go: ensure go binary is in PATH
            env_cmds.extend([
                'export PATH="/usr/local/go/bin:$PATH"',
                'export GOPATH="${GOPATH:-/root/go}"',
            ])
        elif self.language == 'py':
            # Python: activate conda environment if available, otherwise use system python
            env_cmds.extend([
                'source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true',
                'conda activate testbed 2>/dev/null || true',
                'export PATH="/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:$PATH"',
                # Fallback: create python symlink if only python3 exists (for non-conda images)
                'if ! command -v python &>/dev/null && command -v python3 &>/dev/null; then ln -sf "$(which python3)" /usr/local/bin/python 2>/dev/null || true; fi',
            ])
        elif self.language in ('js', 'ts'):
            # JavaScript/TypeScript: ensure node/npm are in PATH
            env_cmds.extend([
                'export PATH="/usr/local/bin:/usr/bin:$PATH"',
            ])
        elif self.language == 'java':
            # Java: ensure mvnd/maven are in PATH
            env_cmds.extend([
                'export MVND_HOME="${MVND_HOME:-/opt/mvnd}"',
                'export MAVEN_HOME="${MAVEN_HOME:-/opt/maven}"',
                'export PATH="/opt/mvnd/bin:/opt/maven/bin:$PATH"',
            ])
        elif self.language == 'rs':
            # Rust: ensure cargo/rustc are in PATH
            env_cmds.extend([
                'source "$HOME/.cargo/env" 2>/dev/null || true',
                'export PATH="$HOME/.cargo/bin:/usr/local/cargo/bin:$PATH"',
            ])
        elif self.language == 'rb':
            # Ruby: ensure ruby/gem are in PATH
            env_cmds.extend([
                'export PATH="/usr/local/bundle/bin:/usr/local/bin:$PATH"',
            ])
        elif self.language == 'php':
            # PHP: ensure php/composer are in PATH
            env_cmds.extend([
                'export PATH="/usr/local/bin:$PATH"',
            ])

        return env_cmds

    @property
    def eval_script(self):
        env_setup = self._get_env_setup_commands()
        script_parts = ['#!/bin/bash', 'set -uxo pipefail'] + env_setup + self.eval_script_list
        return '\n'.join(script_parts) + '\n'


def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    if not patch or not patch.strip():
        return []
    try:
        source_files = []
        for file in PatchSet(patch):
            if file.source_file != '/dev/null':
                source_files.append(file.source_file)
        return [x[2:] for x in source_files if x.startswith('a/')]
    except Exception:
        return []


def _from_json_or_obj(instance: SWEbenchInstance, key: str) -> Any:
    """If key points to string, load with json or eval Python literal"""
    value = instance.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            import ast

            try:
                return ast.literal_eval(value)
            except Exception:
                return []
    return value


def get_test_directives_go(instance: SWEbenchInstance) -> list[str]:
    """
    Given a SWE-bench task instance, return CLI fragments to append to a go test
    command that will run only the test packages modified in the PR.
    """
    diff_pat = r'diff --git a/.* b/(.*)'
    changed_paths = re.findall(diff_pat, instance.get('test_patch', ''))

    test_paths = [p for p in changed_paths if p.endswith('.go') and ('_test.go' in p or 'test' in p or '/test/' in p)]

    if not test_paths:
        return []

    pkg_paths = {f'./{os.path.dirname(p)}' if os.path.dirname(p) else '.' for p in test_paths}

    return sorted(pkg_paths)


def get_test_directives_py(instance: SWEbenchInstance) -> list[str]:
    """
    Get test directives from the test_patch of a task instance for Python
    """
    if instance.get('repo') == 'swe-bench/humaneval':
        return ['test.py']

    diff_pat = r'diff --git a/.* b/(.*)'
    test_patch = instance.get('test_patch', '')
    directives = re.findall(diff_pat, test_patch)
    directives = [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]

    if instance.get('repo') == 'django/django':
        directives_transformed = []
        for d in directives:
            d = d[: -len('.py')] if d.endswith('.py') else d
            d = d[len('tests/') :] if d.startswith('tests/') else d
            d = d.replace('/', '.')
            directives_transformed.append(d)
        directives = directives_transformed

    return directives


def get_test_directives_js(instance: SWEbenchInstance) -> list[str]:
    """
    Get test directives from the test_patch of a task instance for JavaScript/TypeScript
    """
    diff_pat = r'diff --git a/.* b/(.*)'
    test_patch = instance.get('test_patch', '')
    directives = re.findall(diff_pat, test_patch)
    return [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]


def get_test_directives_java(instance: SWEbenchInstance) -> list[str]:
    """
    Get test directives from the test_patch of a task instance for Java.
    Returns Maven test class FQCN format: ["-Dtest=com.example.TestClass1,TestClass2"]
    """
    diff_pat = r'diff --git a/.* b/(.*)'
    test_patch = instance.get('test_patch', '')
    test_paths = re.findall(diff_pat, test_patch)

    fqcns = []
    for path in test_paths:
        if not path.endswith('.java'):
            continue
        # Convert path to FQCN: src/test/java/com/example/TestClass.java -> com.example.TestClass
        fqcn = re.sub(r'^.*?src/(?:test|integration-test)/java/', '', path)
        fqcn = fqcn[: -len('.java')].replace('/', '.')
        fqcns.append(fqcn)

    if not fqcns:
        return []
    fqcns_arg = ','.join(sorted(set(fqcns)))
    return [f'-Dtest={fqcns_arg}']


def get_test_directives_php(instance: SWEbenchInstance) -> list[str]:
    """
    Get test directives from the test_patch of a task instance for PHP
    """
    diff_pat = r'diff --git a/.* b/(.*)'
    test_patch = instance.get('test_patch', '')
    directives = re.findall(diff_pat, test_patch)
    return [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]


def get_test_directives_ruby(instance: SWEbenchInstance) -> list[str]:
    """
    Get test directives from the test_patch of a task instance for Ruby
    """
    diff_pat = r'diff --git a/.* b/(.*)'
    test_patch = instance.get('test_patch', '')
    directives = re.findall(diff_pat, test_patch)
    return [d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)]


def make_eval_script_list_common(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests.
    Common implementation for most languages (non-Python).
    Aligned with SWE-Bench-plus-plus original: no conda activation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    test_cmd = specs.get('test_cmd', 'echo "No test command specified"')
    test_commands = test_cmd if isinstance(test_cmd, list) else [test_cmd]

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        *test_commands,
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_go(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for Go.
    Aligned with SWE-Bench-plus-plus original implementation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'go test -v')
    no_test_directives = specs.get('no_test_directives', False)
    directives = [] if (run_all_tests or no_test_directives) else get_test_directives_go(instance)
    test_commands = ' '.join([base_cmd, *directives])

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_commands})',
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_js(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for JavaScript/TypeScript.
    Aligned with SWE-Bench-plus-plus original implementation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'npm test')
    no_test_directives = specs.get('no_test_directives', False)
    directives = [] if run_all_tests or no_test_directives else get_test_directives_js(instance)
    test_commands = ' '.join([base_cmd, *directives])

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_commands})',
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_py(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for Python.
    Aligned with SWE-Bench-plus-plus original implementation.
    Uses conda environment from /opt/miniconda3.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)
    reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}' if test_files else 'echo "No test files"'
    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'pytest -rA')
    no_test_directives = specs.get('no_test_directives', False)
    directives = [] if run_all_tests or no_test_directives else get_test_directives_py(instance)
    test_command = ' '.join([base_cmd, *directives])

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    eval_commands = [
        'source /opt/miniconda3/bin/activate',
        f'conda activate {env_name}',
        f'cd {repo_directory}',
    ]
    if 'eval_commands' in specs:
        eval_commands += specs['eval_commands']
    eval_commands += [
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        'git status',
        'git show',
        f'git -c core.fileMode=false diff {base_commit}',
        'source /opt/miniconda3/bin/activate',
        f'conda activate {env_name}',
    ]
    if 'install' in specs:
        if isinstance(specs['install'], list):
            eval_commands.extend(specs['install'])
        else:
            eval_commands.append(specs['install'])
    eval_commands += [
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_command})',
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]
    return eval_commands


def make_eval_script_list_java(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for Java.
    Aligned with SWE-Bench-plus-plus original implementation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'mvn test')
    no_test_directives = specs.get('no_test_directives', False)
    directives = [] if run_all_tests or no_test_directives else get_test_directives_java(instance)
    test_commands = ' '.join([base_cmd, *directives])

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_commands})',
        'sleep 1',  # Allow stdout buffers to flush to prevent output interleaving
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_rust(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for Rust.
    Aligned with SWE-Bench-plus-plus original implementation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'cargo test')
    # Rust does not support test directives yet
    test_commands = base_cmd

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_commands})',
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_c(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for C/C++.
    Aligned with SWE-Bench-plus-plus original implementation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'make test')
    # C does not support test directives yet
    test_commands = base_cmd

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_commands})',
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_php(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for PHP.
    Aligned with SWE-Bench-plus-plus original implementation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'phpunit')
    no_test_directives = specs.get('no_test_directives', False)
    directives = [] if run_all_tests or no_test_directives else get_test_directives_php(instance)
    test_commands = ' '.join([base_cmd, *directives])

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_commands})',
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list_ruby(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests for Ruby.
    Aligned with SWE-Bench-plus-plus original implementation.
    """
    heredoc_delimiter = 'EOF_114329324912'
    test_files = get_modified_files(test_patch)

    if test_files:
        reset_tests_command = f'git checkout {base_commit} {" ".join(test_files)}'
    else:
        reset_tests_command = 'echo "No test files to reset"'

    build_commands = []
    if 'build' in specs:
        build_commands.extend(specs['build'])

    apply_test_patch_command = (
        f"git apply --verbose --reject - <<'{heredoc_delimiter}'\n{test_patch}\n{heredoc_delimiter}"
    )

    base_cmd = specs.get('test_cmd', 'bundle exec rake test')
    no_test_directives = specs.get('no_test_directives', False)
    directives = [] if run_all_tests or no_test_directives else get_test_directives_ruby(instance)
    test_commands = ' '.join([base_cmd, *directives])

    return [
        f'cd {repo_directory}',
        f'git config --global --add safe.directory {repo_directory}',
        f'cd {repo_directory}',
        reset_tests_command,
        apply_test_patch_command,
        *build_commands,
        f": '{START_TEST_OUTPUT}'",
        f'({test_commands})',
        f"set +x; echo '{END_TEST_OUTPUT}'",
        reset_tests_command,
    ]


def make_eval_script_list(
    instance: SWEbenchInstance,
    specs: dict,
    env_name: str,
    repo_directory: str,
    base_commit: str,
    test_patch: str,
    run_all_tests: bool = False,
) -> list[str]:
    """
    Applies the test patch and runs the tests.
    Dispatches to language-specific implementations.
    Aligned with SWE-Bench-plus-plus original create_scripts.py.
    """
    language = instance.get('language', 'Python')
    ext = get_ext_from_language(language)

    func = {
        'py': make_eval_script_list_py,
        'go': make_eval_script_list_go,
        'js': make_eval_script_list_js,
        'ts': make_eval_script_list_js,
        'java': make_eval_script_list_java,
        'rs': make_eval_script_list_rust,
        'c': make_eval_script_list_c,
        'php': make_eval_script_list_php,
        'rb': make_eval_script_list_ruby,
    }.get(ext, make_eval_script_list_common)

    return func(instance, specs, env_name, repo_directory, base_commit, test_patch, run_all_tests)


def make_test_spec(instance: SWEbenchInstance) -> TestSpec:
    """
    Create a TestSpec from a SWEbenchInstance.
    """
    instance_id = instance[KEY_INSTANCE_ID]
    repo = instance['repo']
    version = instance.get('version', instance_id.split('-')[-1])
    base_commit = instance['base_commit']
    test_patch = instance.get('test_patch', '')
    language = instance.get('language', 'Python')

    pass_to_pass = _from_json_or_obj(instance, 'PASS_TO_PASS')
    fail_to_pass = _from_json_or_obj(instance, 'FAIL_TO_PASS')
    environment_config = _from_json_or_obj(instance, 'environment_config')

    env_name = 'testbed'
    repo_directory = f'/{env_name}'

    specs = environment_config if environment_config else {}

    eval_script_list = make_eval_script_list(instance, specs, env_name, repo_directory, base_commit, test_patch)

    ext = get_ext_from_language(language)

    # Get log parser
    log_parser = None
    log_parser_name = specs.get('log_parser_name', '')
    if log_parser_name:
        language_constant = None
        for lang_const, lang_ext in LANGUAGES_STR_MAP.items():
            if lang_ext == ext:
                language_constant = lang_const
                break
        if language_constant and language_constant in LANGUAGE_PARSER_MAP:
            if log_parser_name == 'custom':
                log_parser = specs.get('log_parser_code')
            else:
                log_parser = LANGUAGE_PARSER_MAP[language_constant](log_parser_name)

    return TestSpec(
        instance_id=instance_id,
        repo=repo,
        version=version,
        eval_script_list=eval_script_list,
        FAIL_TO_PASS=fail_to_pass,
        PASS_TO_PASS=pass_to_pass,
        language=ext,
        environment_config=specs,
        log_parser=log_parser,
    )
