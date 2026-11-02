"""
通用 Instance 实现，从 instance_dict 最外层读取配置信息

配置字段直接在 instance_dict 最外层，包括：
- dataset_type: 数据集类型
- docker: Docker 配置
- scripts: 脚本配置
- test_parser: 测试解析器配置
- language: 编程语言
"""

import re

from typing import Any

from nexus.extensions.tasks.swe.multi_swe_bench.generic_image import GenericImage
from nexus.extensions.tasks.swe.multi_swe_bench.image import Config, Image
from nexus.extensions.tasks.swe.multi_swe_bench.instance import Instance
from nexus.extensions.tasks.swe.multi_swe_bench.pull_request import PullRequest
from nexus.extensions.tasks.swe.multi_swe_bench.test_result import TestResult


class GenericInstance(Instance):
    """从配置驱动的通用 Instance 实现（配置字段在 instance_dict 最外层）"""

    def __init__(self, pr: PullRequest, config: Config, instance_dict: dict[str, Any]):
        self._pr = pr
        self._config = config
        self._instance_dict = instance_dict
        self._image = None

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Image:
        if self._image is None:
            self._image = GenericImage(self._pr, self._config, self._instance_dict)
        return self._image

    def run(self) -> str:
        """返回 run.sh 脚本内容"""
        return self._instance_dict.get('scripts', {}).get('run', '')

    def test_patch_run(self) -> str:
        """返回 test-run.sh 脚本内容"""
        return self._instance_dict.get('scripts', {}).get('test_run', '')

    def fix_patch_run(self, fix_patch_run_cmd: str = '') -> str:
        """返回 fix-run.sh 脚本内容"""
        if fix_patch_run_cmd:
            return fix_patch_run_cmd
        return self._instance_dict.get('scripts', {}).get('fix_run', '')

    def _build_test_result(self, passed_tests: set[str], failed_tests: set[str], skipped_tests: set[str]) -> TestResult:
        """构建 TestResult 对象的辅助方法"""
        return TestResult(
            passed_count=len(passed_tests),
            failed_count=len(failed_tests),
            skipped_count=len(skipped_tests),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
        )

    def _parse_scalameta_log(self, test_log: str, rule: dict[str, Any] | None = None) -> TestResult:
        """Special parser for scalameta repo.

        The scalameta test_parser config in dataset has overlapping pass/fail/skip patterns,
        causing all tests to be incorrectly marked as failed. This method uses the correct
        parsing logic from multi_swe_bench/harness/repos/scala/scalameta/scalameta.py

        Args:
            test_log: The test log to parse
            rule: Optional scalameta_pr_conditional rule from test_parser.special_rules
        """
        passed_tests: set[str] = set()
        failed_tests: set[str] = set()
        skipped_tests: set[str] = set()

        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')

        # Get filter pattern from rule or use default
        filter_pattern_str = (
            rule.get('filter_pattern', r'^meta[cp] /tmp/[^ ]+_[^ ]+') if rule else r'^meta[cp] /tmp/[^ ]+_[^ ]+'
        )
        metac_sourceroot_re = re.compile(filter_pattern_str)

        # PR number determines which regex patterns to use
        pr_number = self._pr.number if self._pr else 0

        # Get threshold and patterns from rule or use defaults
        pr_threshold = rule.get('pr_threshold', 1936) if rule else 1936

        if pr_number <= pr_threshold:
            if rule and 'old_patterns' in rule:
                passed_res = [re.compile(p) for p in rule['old_patterns'].get('pass', [])]
                failed_res = [re.compile(p) for p in rule['old_patterns'].get('fail', [])]
            else:
                passed_res = [re.compile(r'^\[info\] [+-] (?!.*\*\*\* FAILED \*\*\*)(.*)$')]
                failed_res = [re.compile(r'^\[info\] - (.*) \*\*\* FAILED \*\*\*$')]
        else:
            if rule and 'new_patterns' in rule:
                passed_res = [re.compile(p) for p in rule['new_patterns'].get('pass', [])]
                failed_res = [re.compile(p) for p in rule['new_patterns'].get('fail', [])]
            else:
                passed_res = [re.compile(r'^\s*\+\s*(.*?)(?:\s+\d+\.\d+s)?$')]
                failed_res = [re.compile(r'^-([^\n]*)')]

        for line in test_log.splitlines():
            line = ansi_escape.sub('', line)

            for passed_re in passed_res:
                m = passed_re.match(line)
                if m and m.group(1) not in failed_tests and not metac_sourceroot_re.match(m.group(1)):
                    passed_tests.add(m.group(1))

            for failed_re in failed_res:
                m = failed_re.match(line)
                if m:
                    failed_tests.add(m.group(1))
                    if m.group(1) in passed_tests:
                        passed_tests.remove(m.group(1))

        return self._build_test_result(passed_tests, failed_tests, skipped_tests)

    def _apply_rule_result(
        self, rule_config: dict[str, Any], passed_tests: set[str], failed_tests: set[str], skipped_tests: set[str]
    ) -> TestResult:
        """应用规则配置中的测试结果"""
        passed_tests.update(rule_config.get('passed_tests', []))
        failed_tests.update(rule_config.get('failed_tests', []))
        skipped_tests.update(rule_config.get('skipped_tests', []))
        return self._build_test_result(passed_tests, failed_tests, skipped_tests)

    def _check_pr_range(self, rule: dict[str, Any]) -> bool:
        """检查当前 PR 是否在规则指定的范围内"""
        pr_range = rule.get('pr_range', [])
        if len(pr_range) == 2:
            min_pr, max_pr = pr_range
            return min_pr <= self._pr.number <= max_pr
        return False

    def _handle_pr_range_rule(
        self,
        rule: dict[str, Any],
        condition_matched: bool,
        passed_tests: set[str],
        failed_tests: set[str],
        skipped_tests: set[str],
    ) -> TestResult | None:
        """处理 PR 范围相关的规则，返回 TestResult 或 None"""
        if not self._check_pr_range(rule):
            return None

        result_key = 'on_match' if condition_matched else 'on_mismatch'
        return self._apply_rule_result(rule.get(result_key, {}), passed_tests, failed_tests, skipped_tests)

    def _parse_json_log(self, test_log: str, rule: dict[str, Any]) -> TestResult:
        """Parse test log using JSON extraction (for mui/material-ui style output).

        The test output contains JSON objects with stats, passes, failures, pending arrays.
        """
        import json

        from json import JSONDecoder

        passed_tests: set[str] = set()
        failed_tests: set[str] = set()
        skipped_tests: set[str] = set()

        def extract_json_objects(text: str, decoder=JSONDecoder()):
            """Extract all JSON objects from text."""
            pos = 0
            while True:
                match = text.find('{', pos)
                if match == -1:
                    break
                try:
                    result, index = decoder.raw_decode(text[match:])
                    yield result
                    pos = match + index
                except (ValueError, json.JSONDecodeError):
                    pos = match + 1

        # Preprocess: remove error messages and newlines
        if 'Building new' in test_log:
            test_log = test_log[test_log.find('Building new') :]

        test_log = test_log.replace('\r\n', '')
        test_log = test_log.replace('\n', '')

        # Get test format from rule, default to "{file}:{fullTitle}"
        rule.get('test_format', '{test.file}:{test.fullTitle}')

        for obj in extract_json_objects(test_log):
            # Try to extract passes, failures, pending from JSON object
            passes = obj.get('passes', [])
            failures = obj.get('failures', [])
            pending = obj.get('pending', [])

            for test in passes:
                if isinstance(test, dict):
                    test_name = f'{test.get("file", "")}:{test.get("fullTitle", "")}'
                    passed_tests.add(test_name)

            for test in failures:
                if isinstance(test, dict):
                    test_name = f'{test.get("file", "")}:{test.get("fullTitle", "")}'
                    failed_tests.add(test_name)

            for test in pending:
                if isinstance(test, dict):
                    test_name = f'{test.get("file", "")}:{test.get("fullTitle", "")}'
                    skipped_tests.add(test_name)

        # Remove conflicts: failed > skipped > passed
        passed_tests -= failed_tests
        passed_tests -= skipped_tests
        skipped_tests -= failed_tests

        return self._build_test_result(passed_tests, failed_tests, skipped_tests)

    def _parse_maven_state_machine(self, test_log: str, rule: dict[str, Any]) -> TestResult:
        """Parse Maven test log using state machine.

        Maven test output format:
        [INFO] Running com.example.TestClass
        [INFO] Tests run: X, Failures: 0, Errors: 0, Skipped: 0, Time elapsed: ... -- in com.example.TestClass

        Or for failed tests:
        [INFO] Running com.example.TestClass
        [ERROR] Tests run: X, Failures: Y, Errors: Z, Skipped: 0, Time elapsed: ... <<< FAILURE! - in com.example.TestClass

        Logic: Track "Running X" lines, then check if next test result line contains [ERROR] or [INFO].
        """
        passed_tests: set[str] = set()
        failed_tests: set[str] = set()
        skipped_tests: set[str] = set()

        # Get patterns from rule
        running_pattern = rule.get('running_pattern', r'\[INFO\] Running (.+)')
        error_pattern = rule.get('error_pattern', r'\[ERROR\] Tests run')

        running_regex = re.compile(running_pattern)

        lines = test_log.split('\n')

        for i, line in enumerate(lines):
            # Check if this line starts a new test
            running_match = running_regex.search(line)
            if running_match:
                test_name = running_match.group(1).strip()

                # Look ahead in the next few lines to find the test result
                found_result = False
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j]
                    if 'Tests run:' in next_line:
                        # Check if it's an error line (contains [ERROR])
                        if error_pattern in next_line:
                            failed_tests.add(test_name)
                        elif '[INFO]' in next_line:
                            # This is a success line
                            passed_tests.add(test_name)
                        found_result = True
                        break

                # If no result line found, assume passed (defensive)
                if not found_result:
                    passed_tests.add(test_name)

        # Remove conflicts: failed > passed
        passed_tests -= failed_tests

        return self._build_test_result(passed_tests, failed_tests, skipped_tests)

    def parse_log(self, test_log: str) -> TestResult:
        """解析测试日志，返回测试结果"""
        test_parser = self._instance_dict.get('test_parser', {})
        special_rules = test_parser.get('special_rules', [])

        # Check for json_parsing rule (for mui/material-ui style output)
        for rule in special_rules:
            if rule.get('type') == 'json_parsing':
                return self._parse_json_log(test_log, rule)

        # Check for scalameta_pr_conditional rule (generated by convert.py)
        for rule in special_rules:
            if rule.get('type') == 'scalameta_pr_conditional':
                return self._parse_scalameta_log(test_log, rule)

        # Check for maven_state_machine rule (for Maven test output)
        for rule in special_rules:
            if rule.get('type') == 'maven_state_machine':
                return self._parse_maven_state_machine(test_log, rule)

        # Legacy: if repo is scalameta but no scalameta_pr_conditional rule, use default parser
        repo = self._instance_dict.get('repo', '')
        if repo == 'scalameta':
            return self._parse_scalameta_log(test_log)

        patterns = test_parser.get('patterns', {})
        passed_tests = set()
        failed_tests = set()
        skipped_tests = set()

        # 预处理：默认移除 ANSI 转义序列（颜色代码等）
        # 这是必要的，因为很多测试输出包含 ANSI 转义序列，会干扰正则匹配
        ansi_pattern = r'\x1B\[[0-?9;]*[mK]'
        for rule in special_rules:
            if rule.get('type') == 'ansi_escape_removal':
                ansi_pattern = rule.get('pattern', ansi_pattern)
                break
        test_log = re.sub(ansi_pattern, '', test_log)

        # 收集 line_preprocessing 规则
        line_preprocessing_ops = []
        for rule in special_rules:
            if rule.get('type') == 'line_preprocessing':
                operation = rule.get('operation')
                if operation:
                    line_preprocessing_ops.append(operation)

        # 优先处理 special_rules
        for rule in special_rules:
            rule_type = rule.get('type')

            # 处理 PR 范围 + 关键字计数检查
            if rule_type == 'pr_range_keyword_count':
                keyword = rule.get('keyword', '')
                expected_count = rule.get('expected_count', 1)
                condition_matched = test_log.count(keyword) == expected_count

                result = self._handle_pr_range_rule(rule, condition_matched, passed_tests, failed_tests, skipped_tests)
                if result is not None:
                    return result

            # 处理 PR 范围 + 关键字存在性检查
            elif rule_type == 'pr_range_keyword':
                keyword = rule.get('keyword', '')
                condition_matched = keyword in test_log

                result = self._handle_pr_range_rule(rule, condition_matched, passed_tests, failed_tests, skipped_tests)
                if result is not None:
                    return result

            elif rule_type == 'ctest_summary' or rule_type == 'maven_test_result':
                pass

        pass_regexes = [re.compile(p) for p in patterns.get('pass', [])]
        fail_regexes = [re.compile(p) for p in patterns.get('fail', [])]
        skip_regexes = [re.compile(p) for p in patterns.get('skip', [])]

        # 检测是否是 Maven 格式的测试输出
        # Maven 格式: Tests run: X, Failures: Y, Errors: Z, Skipped: W, Time elapsed: ... in TestClassName
        maven_pattern = None
        for p in patterns.get('pass', []):
            if 'Tests run:' in p and 'Failures:' in p and 'Errors:' in p and 'Skipped:' in p:
                maven_pattern = re.compile(p, re.MULTILINE)
                break

        # 如果是 Maven 格式，使用跨行匹配（因为正则表达式包含 \n）
        if maven_pattern:
            for match in maven_pattern.finditer(test_log):
                if len(match.groups()) >= 5:
                    # 根据正则表达式的捕获组顺序确定字段位置
                    # 常见格式1: (test_name).*Tests run: (\d+), ... -> test_name在group(1)
                    # 常见格式2: Tests run: (\d+), ..., in (test_name) -> test_name在group(5)
                    # 通过检查group(1)是否为纯数字来判断顺序
                    if match.group(1).isdigit():
                        # 格式2: Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+), ... in (test_name)
                        tests_run = int(match.group(1))
                        failures = int(match.group(2))
                        errors = int(match.group(3))
                        skipped = int(match.group(4))
                        test_name = match.group(5)
                    else:
                        # 格式1: (test_name).*Tests run: (\d+), Failures: (\d+), Errors: (\d+), Skipped: (\d+)
                        test_name = match.group(1)
                        tests_run = int(match.group(2))
                        failures = int(match.group(3))
                        errors = int(match.group(4))
                        skipped = int(match.group(5))

                    if tests_run > 0 and failures == 0 and errors == 0 and skipped != tests_run:
                        passed_tests.add(test_name)
                    elif failures > 0 or errors > 0:
                        failed_tests.add(test_name)
                    elif skipped == tests_run:
                        skipped_tests.add(test_name)

        # 额外支持 Maven [INFO] 前缀格式（单行格式）
        # 格式: [INFO] Tests run: X, Failures: Y, Errors: Z, Skipped: W, Time elapsed: ... -- in TestClassName
        maven_info_pattern = re.compile(
            r'\[INFO\]\s+Tests run:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+),\s*Skipped:\s*(\d+),\s*Time elapsed:.*?--\s*in\s+(.+?)(?:\s|$)'
        )
        for match in maven_info_pattern.finditer(test_log):
            tests_run = int(match.group(1))
            failures = int(match.group(2))
            errors = int(match.group(3))
            skipped = int(match.group(4))
            test_name = match.group(5).strip()

            if tests_run > 0 and failures == 0 and errors == 0 and skipped != tests_run:
                passed_tests.add(test_name)
            elif failures > 0 or errors > 0:
                failed_tests.add(test_name)
            elif skipped == tests_run:
                skipped_tests.add(test_name)
        else:
            # 收集 context_prefix 规则（用于 Jest/Vitest 等框架的文件级上下文拼接）
            context_prefix_regexes: list[re.Pattern] = []
            context_separator = ':'
            for rule in special_rules:
                if rule.get('type') == 'context_prefix':
                    context_prefix_regexes = [re.compile(p) for p in rule.get('context_patterns', [])]
                    context_separator = rule.get('separator', ':')
                    break

            current_context: str | None = None

            # 非 Maven 格式，逐行处理
            for line in test_log.split('\n'):
                processed_line = line
                for op in line_preprocessing_ops:
                    if op == 'strip':
                        processed_line = processed_line.strip()
                    elif op == 'lower':
                        processed_line = processed_line.lower()

                # 检查当前行是否匹配 context_prefix 的上下文模式（如 PASS/FAIL file.test.js）
                is_context_line = False
                if context_prefix_regexes:
                    for ctx_regex in context_prefix_regexes:
                        ctx_match = ctx_regex.search(processed_line)
                        if ctx_match:
                            current_context = ctx_match.group(1) if ctx_match.groups() else ctx_match.group(0)
                            is_context_line = True
                            break

                # 检查 pass 模式
                for regex in pass_regexes:
                    match = regex.search(processed_line)
                    if match:
                        test_name = match.group(1) if match.groups() else match.group(0)
                        passed_tests.add(test_name)
                        if current_context and not is_context_line:
                            combined = current_context + context_separator + test_name
                            passed_tests.add(combined)

                # 检查 fail 模式
                for regex in fail_regexes:
                    match = regex.search(processed_line)
                    if match:
                        test_name = match.group(1) if match.groups() else match.group(0)
                        failed_tests.add(test_name)
                        if current_context and not is_context_line:
                            combined = current_context + context_separator + test_name
                            failed_tests.add(combined)

                # 检查 skip 模式
                for regex in skip_regexes:
                    match = regex.search(processed_line)
                    if match:
                        test_name = match.group(1) if match.groups() else match.group(0)
                        skipped_tests.add(test_name)
                        if current_context and not is_context_line:
                            combined = current_context + context_separator + test_name
                            skipped_tests.add(combined)

        # 清理冲突：如果一个测试同时出现在多个集合中，按优先级处理
        # 优先级：failed > skipped > passed
        # 原因：失败的测试更重要，应该优先保留
        passed_tests -= failed_tests
        passed_tests -= skipped_tests
        skipped_tests -= failed_tests

        return self._build_test_result(passed_tests, failed_tests, skipped_tests)
