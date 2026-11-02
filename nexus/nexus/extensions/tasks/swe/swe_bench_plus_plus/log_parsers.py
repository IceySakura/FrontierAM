"""
Log parsers for SWE-Bench-plus-plus dataset.
Aligned with SWE-Bench-plus-plus/swebench/harness/log_parsers/ original implementation.
"""

import json
import re
import xml.etree.ElementTree as ET

from collections.abc import Iterable

from nexus.extensions.tasks.swe.swe_bench_plus_plus.constants import (
    CSHARP_SBP_STR,
    C_PLUS_SBP_STR,
    C_SBP_STR,
    GO_SBP_STR,
    JAVASCRIPT_SBP_STR,
    JAVA_SBP_STR,
    PHP_SBP_STR,
    PYTHON_SBP_STR,
    RUBY_SBP_STR,
    RUST_SBP_STR,
    TYPESCRIPT_SBP_STR,
    TestStatus,
)


# ============== Utility Functions ==============
def ansi_escape(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_pattern.sub('', text)


def strip_ansi(s: str) -> str:
    """Alias for ansi_escape."""
    return ansi_escape(s)


# ============== Go Parser ==============
def parse_log_gotest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with 'go test'

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    # Pattern to match test result lines
    pattern = r'^--- (PASS|FAIL|SKIP): (.+) \((.+)\)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name, _duration = match.groups()
            if status == 'PASS':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'FAIL':
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status == 'SKIP':
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


# ============== Python Parsers ==============
def parse_log_pytest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split('\n'):
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = ' - '.join(line.split(' - ')[:-1])
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r'(.*?)\[(.*)\]')
    test_status_map = {}
    for line in log.split('\n'):
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if option.startswith('/') and not option.startswith('//') and '*' not in option:
                    option = '/' + option.split('/')[-1]
                test_name = f'{main}[{option}]'
            else:
                test_name = test_case[1]
            test_status_map[test_name] = test_case[0]
    return test_status_map


def parse_log_django(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split('\n')

    prev_test = None
    for line in lines:
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if '--version is equivalent to version' in line:
            test_status_map['--version is equivalent to version'] = TestStatus.PASSED.value

        # Log it in case of error
        if ' ... ' in line:
            prev_test = line.split(' ... ')[0]

        pass_suffixes = (' ... ok', ' ... OK', ' ...  OK')
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                if line.strip().startswith('Applying sites.0002_alter_domain_unique...test_no_migrations'):
                    line = line.split('...', 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED.value
                break
        if ' ... skipped' in line:
            test = line.split(' ... skipped')[0]
            test_status_map[test] = TestStatus.SKIPPED.value
        if line.endswith(' ... FAIL'):
            test = line.split(' ... FAIL')[0]
            test_status_map[test] = TestStatus.FAILED.value
        if line.startswith('FAIL:'):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED.value
        if line.endswith(' ... ERROR'):
            test = line.split(' ... ERROR')[0]
            test_status_map[test] = TestStatus.ERROR.value
        if line.startswith('ERROR:'):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR.value

        if line.lstrip().startswith('ok') and prev_test is not None:
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    patterns = [
        r'^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$',
        r'^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$',
        r'^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$',
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED.value
    return test_status_map


def parse_log_pytest_v2(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = ''.join([chr(char) for char in range(1, 32)])
    for line in log.split('\n'):
        line = re.sub(r'\[(\d+)m', '', line)
        translator = str.maketrans('', '', escapes)
        line = line.translate(translator)
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value) and ' - ' in line:
                line = ' - '.join(line.split(' - ')[:-1])
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[' '.join(test_case[1:])] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any(line.endswith(x.value) for x in TestStatus):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = test_case[1]

    progress_pattern = re.compile(r'^\[\s*\d+%]$')  # e.g. "[ 2%]" or "[98%]"
    return {k: v for k, v in test_status_map.items() if not progress_pattern.match(k)}


def parse_log_seaborn(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework
    """
    test_status_map = {}
    for line in log.split('\n'):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED.value
        elif f' {TestStatus.PASSED.value} ' in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED.value
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED.value
    return test_status_map


def parse_log_sympy(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework
    """
    test_status_map = {}
    pattern = r'(_*) (.*)\.py:(.*) (_*)'
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f'{match[1]}.py:{match[2]}'
        test_status_map[test_case] = TestStatus.FAILED.value
    for line in log.split('\n'):
        line = line.strip()
        if line.startswith('test_'):
            if line.endswith(' E'):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR.value
            if line.endswith(' F'):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED.value
            if line.endswith(' ok'):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED.value
    return test_status_map


def parse_log_matplotlib(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework for matplotlib
    """
    test_status_map = {}
    for line in log.split('\n'):
        line = line.replace('MouseButton.LEFT', '1')
        line = line.replace('MouseButton.RIGHT', '3')
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


# ============== JavaScript/TypeScript Parsers ==============
def parse_log_calypso(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by Calypso test suite
    """
    test_status_map = {}
    suite = []

    def get_test_name(suite, match_pattern, line):
        return ' - '.join([
            ' - '.join([x[0] for x in suite]),
            re.match(match_pattern, line).group(1),
        ]).strip()

    for log_part in log.split(' ./node_modules/.bin/jest ')[1:]:
        for line in log_part.split('\n'):
            if any(line.startswith(x) for x in ['Test Suites', '  ● ']):
                break
            if line.strip().startswith('✓'):
                match_pattern = (
                    r'^\s+✓\s(.*)\(\d+ms\)$' if re.search(r'\(\d+ms\)', line) is not None else r'^\s+✓\s(.*)'
                )
                test_status_map[get_test_name(suite, match_pattern, line)] = TestStatus.PASSED.value
            elif line.strip().startswith('✕'):
                match_pattern = (
                    r'^\s+✕\s(.*)\(\d+ms\)$' if re.search(r'\(\d+ms\)', line) is not None else r'^\s+✕\s(.*)'
                )
                test_status_map[get_test_name(suite, match_pattern, line)] = TestStatus.FAILED.value
            elif len(line) - len(line.lstrip()) > 0:
                indent = len(line) - len(line.lstrip())
                if len(suite) == 0:
                    suite = [(line.strip(), indent)]
                else:
                    while len(suite) > 0 and suite[-1][-1] >= indent:
                        suite.pop()
                    suite.append([line.strip(), indent])

    return test_status_map


def parse_log_mocha(log: str) -> dict[str, str]:
    """
    Parse Mocha CLI output and extract test statuses with full suite paths.
    """
    test_status_map = {}
    suite_stack = []

    for line in log.splitlines():
        stripped = line.strip()

        # Match for PASSED tests
        if stripped.startswith('✓') or stripped.startswith('✔'):
            pass_pattern = (
                r'^\s*(?:✓|✔)\s+(.*)\s+\(\d+ms\)$' if re.search(r'\(\d+ms\)$', stripped) else r'^\s*(?:✓|✔)\s+(.*)'
            )
            match = re.match(pass_pattern, stripped)
            if match:
                test_name_part = match.group(1).strip()
                suite_path = [item[0] for item in suite_stack]
                full_test_name = '.'.join(suite_path + [test_name_part])
                test_status_map[full_test_name] = TestStatus.PASSED.value
            continue

        # Match for FAILED tests
        if stripped.startswith('✕') or stripped.startswith('x'):
            fail_pattern = r'^\s*(?:✕|x)\s+(.*)'
            match = re.match(fail_pattern, stripped)
            if match:
                test_name_part = match.group(1).strip()
                suite_path = [item[0] for item in suite_stack]
                full_test_name = '.'.join(suite_path + [test_name_part])
                test_status_map[full_test_name] = TestStatus.FAILED.value
            continue

        # Match for SKIPPED tests
        if stripped.startswith('-'):
            skip_pattern = r'^\s*-\s+(.*)'
            match = re.match(skip_pattern, stripped)
            if match:
                test_name_part = match.group(1).strip()
                suite_path = [item[0] for item in suite_stack]
                full_test_name = '.'.join(suite_path + [test_name_part])
                test_status_map[full_test_name] = TestStatus.SKIPPED.value
            continue

        # Logic for handling suites
        if line.startswith(' ') and not stripped.startswith(('✓', '✔', '✕', 'x', '-')):
            indent = len(line) - len(line.lstrip(' '))
            while suite_stack and suite_stack[-1][1] >= indent:
                suite_stack.pop()
            suite_stack.append((stripped, indent))
        elif stripped:
            if not any(stripped.startswith(char) for char in ['✓', '✔', '✕', 'x', '-']):
                suite_stack = [(stripped, 0)]

    return test_status_map


def parse_log_mocha_v2(log: str) -> dict[str, str]:
    """
    Parse Mocha 'spec' output, returning
        { "Suite - SubSuite - Test": "PASSED"/"FAILED"/"PENDING" }
    """
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    pass_re = re.compile(r'^\s*[✓√✔]\s+(.*?)(?:\s+\(\d+ms\))?\s*$')
    fail_re = re.compile(r'^\s{4,}\d+\)\s+(.*)')
    cross_re = re.compile(r'^\s*[×✕]\s+(.*)')
    pend_re = re.compile(r'^\s*[-•]\s+(.*)')
    summary_re = re.compile(r'^\s*\d+\s+(passing|failing|pending)')
    dur_tail_re = re.compile(r'\s+\([\d\.]+ ?[a-zA-Z]+\)$')

    def strip_ansi_local(txt: str) -> str:
        return ansi_re.sub('', txt)

    test_status_map = {}
    suite_stack = []
    count_empty_lines = 0

    for raw in log.splitlines():
        line = strip_ansi_local(raw.rstrip())

        if not line:
            count_empty_lines += 1
            if count_empty_lines >= 2:
                count_empty_lines = 0
                suite_stack = []
            continue

        m = summary_re.match(line)
        if m:
            suite_stack = []
            continue

        m = pass_re.match(line)
        if m:
            test_name = m.group(1).strip()
            test_name = dur_tail_re.sub('', test_name)
            test_status_map[' - '.join(suite_stack + [test_name])] = TestStatus.PASSED.value
            continue

        m = fail_re.match(line) or cross_re.match(line)
        if m:
            test_name = m.group(1).strip()
            test_name = dur_tail_re.sub('', test_name)
            test_status_map[' - '.join(suite_stack + [test_name])] = TestStatus.FAILED.value
            continue

        m = pend_re.match(line)
        if m:
            test_name = m.group(1).strip()
            test_name = dur_tail_re.sub('', test_name)
            test_status_map[' - '.join(suite_stack + [test_name])] = TestStatus.SKIPPED.value
            continue

        indent = len(line) - len(line.lstrip())
        level = (indent // 2) - 1
        if indent >= 2:
            suite_stack[level:] = [line.strip()]

    return test_status_map


def parse_log_chart_js(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by ChartJS test suite
    """
    test_status_map = {}
    failure_case_patterns = [
        (r'Chrome\s[\d\.]+\s\(.*?\)\s(.*)FAILED$', re.MULTILINE),
    ]
    for failure_case_pattern, flags in failure_case_patterns:
        failures = re.findall(failure_case_pattern, log, flags)
        if len(failures) == 0:
            continue
        for failure in failures:
            test_status_map[failure] = TestStatus.FAILED.value
    return test_status_map


def parse_log_marked(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by Marked test suite
    """
    test_status_map = {}
    for line in log.split('\n'):
        match = re.search(r'^\d+\)\s(.*)', line)
        if match:
            test = match.group(1)
            test_status_map[test.strip()] = TestStatus.FAILED.value
    return test_status_map


def parse_log_p5js(log_content: str) -> dict[str, str]:
    def remove_json_blocks(log):
        filtered_lines = []
        in_json_block = False
        in_json_list_block = False
        for line in log.split('\n'):
            stripped_line = line.rstrip()
            if stripped_line.endswith('{'):
                in_json_block = True
                continue
            if stripped_line.endswith('['):
                in_json_list_block = True
                continue
            if stripped_line == '}' and in_json_block:
                in_json_block = False
                continue
            if stripped_line == ']' and in_json_list_block:
                in_json_list_block = False
                continue
            if in_json_block or in_json_list_block:
                continue
            if stripped_line.startswith('{') and stripped_line.endswith('}'):
                continue
            if stripped_line.startswith('[') and stripped_line.endswith(']'):
                continue
            filtered_lines.append(line)
        return '\n'.join(filtered_lines)

    def remove_xml_blocks(log):
        xml_pat = re.compile(r'<(\w+)>[\s\S]*?<\/\1>', re.MULTILINE)
        match = xml_pat.search(log)
        while match:
            opening_tags = match.group().count(rf'<{match.group(1)}>') - 1
            opening_tags = max(opening_tags, 0)
            start = match.start()
            end = match.end()
            log = log[:start] + f'<{match.group(1)}>' * opening_tags + log[end:]
            match = xml_pat.search(log)
        return log

    def is_valid_fail(match):
        last_line_indent = 0
        for line in match.group(2).split('\n'):
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= last_line_indent:
                return False
            last_line_indent = line_indent
        return True

    log_content = ansi_escape(log_content)
    log_content = remove_json_blocks(log_content)
    log_content = remove_xml_blocks(log_content)
    test_results = {}

    fail_pattern = re.compile(r'^\s*(\d+)\)(.{0,1000}?):', re.MULTILINE | re.DOTALL)
    for match in fail_pattern.finditer(log_content):
        if is_valid_fail(match):
            test_names = list(map(str.strip, match.group(2).split('\n')))
            full_name = ':'.join(test_names)
            test_results[full_name] = TestStatus.FAILED.value

    return test_results


def parse_log_react_pdf(log: str) -> dict[str, str]:
    """
    Parser for test logs generated by react-pdf test suite
    """
    test_status_map = {}
    for line in log.split('\n'):
        for pattern in [
            (r'^PASS\s(.*)\s\([\d\.]+ms\)', TestStatus.PASSED.value),
            (r'^PASS\s(.*)\s\([\d\.]+\ss\)', TestStatus.PASSED.value),
            (r'^PASS\s(.*)\s\([\d\.]+s\)', TestStatus.PASSED.value),
            (r'^PASS\s(.*)', TestStatus.PASSED.value),
            (r'^FAIL\s(.*)\s\([\d\.]+ms\)', TestStatus.FAILED.value),
            (r'^FAIL\s(.*)\s\([\d\.]+\ss\)', TestStatus.FAILED.value),
            (r'^FAIL\s(.*)\s\([\d\.]+s\)', TestStatus.FAILED.value),
            (r'^FAIL\s(.*)', TestStatus.FAILED.value),
        ]:
            match = re.match(pattern[0], line)
            if match:
                test_name = match.group(1)
                test_status_map[test_name] = pattern[1]
                break
    return test_status_map


def parse_log_jest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Jest. Assumes --verbose flag.
    """
    test_status_map = {}

    pattern = r'^\s*(✓|✕|○)\s(.+?)(?:\s+\((\d+\s*m?s)\))?$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status_symbol, test_name, _duration = match.groups()
            if status_symbol == '✓':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status_symbol == '✕':
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status_symbol == '○':
                test_status_map[test_name] = TestStatus.SKIPPED.value
    return test_status_map


def parse_log_jest_json(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Jest --json flag.
    """
    test_status_map = {}

    pattern = r'^\[(PASSED|FAILED)\]\s(.+)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == 'PASSED':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'FAILED':
                test_status_map[test_name] = TestStatus.FAILED.value
    return test_status_map


def parse_log_vitest(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with vitest. Assumes --reporter=verbose flag.
    """
    test_status_map = {}

    pattern = r'^\s*(✓|×|↓)\s(.+?)(?:\s+(\d+\s*m?s?|\[skipped\]))?$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status_symbol, test_name, _duration_or_skipped = match.groups()
            if status_symbol == '✓':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status_symbol == '×':
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status_symbol == '↓':
                test_status_map[test_name] = TestStatus.SKIPPED.value
    return test_status_map


def parse_log_karma(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with Karma.
    """
    test_status_map = {}
    current_indent = -1
    current_suite = []
    started = False

    pattern = r'^(\s*)?([✔✖])?\s(.*)$'

    for line in log.split('\n'):
        if line.startswith('SUMMARY:'):
            return test_status_map

        if 'Starting browser' in line:
            started = True
            continue

        if not started:
            continue

        match = re.match(pattern, line)
        if match:
            indent, status, name = match.groups()

            if indent and not status:
                new_indent = len(indent)
                if new_indent > current_indent:
                    current_indent = new_indent
                    current_suite.append(name)
                elif new_indent < current_indent:
                    current_indent = new_indent
                    current_suite.pop()
                    continue

            if status in ('✔', '✖'):
                full_test_name = ' > '.join(current_suite + [name])
                test_status_map[full_test_name] = TestStatus.PASSED.value if status == '✔' else TestStatus.FAILED.value

    return test_status_map


def parse_log_tap(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with TAP
    """
    test_status_map = {}

    pattern = r'^(ok|not ok) (\d+) (.+)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, _test_number, test_name = match.groups()
            if status == 'ok':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'not ok':
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


# ============== Java Parsers ==============
ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def merge_status(existing: str | None, new: str) -> str:
    """Prefer worst outcome when duplicates appear: FAILED > SKIPPED > PASSED"""
    if existing is None:
        return new
    priority = {'FAILED': 3, 'SKIPPED': 2, 'PASSED': 1}
    return existing if priority.get(existing, 0) >= priority.get(new, 0) else new


def mk_id(classname: str, name: str) -> str:
    """Normalize a test identifier as 'com.foo.BarTest.testMethod'"""
    return f'{classname}.{name}'.strip()


def parse_log_maven_v2(log: str) -> dict[str, str]:
    """
    Parse Maven Surefire console output.
    """
    test_status: dict[str, str] = {}

    line_re = re.compile(
        r'^(?:\[[^\]]+\]\s+)?\[(?:INFO|ERROR|WARN(?:ING)?)\]\s+(.+?)\s+(?:--\s+)?Time elapsed:\s*[\d.]+\s*s(?:\s*<<<\s+(ERROR|FAILURE)!)?'
    )

    for raw in log.splitlines():
        m = line_re.match(raw.strip())
        if not m:
            continue

        fq_test = m.group(1).strip()

        if fq_test.startswith('Tests run:'):
            continue

        if '.' not in fq_test:
            continue
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*\.', fq_test):
            continue

        failed_marker = m.group(2)
        status = TestStatus.FAILED.value if failed_marker else TestStatus.PASSED.value
        test_status[fq_test] = status

    return test_status


def parse_log_maven(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with 'mvn test'.
    """
    test_status_map = {}
    current_test_name = '---NO TEST NAME FOUND YET---'

    test_name_pattern = r'^.*-Dtest=(\S+).*$'
    result_pattern = r'^.*BUILD (SUCCESS|FAILURE)$'

    for line in log.split('\n'):
        test_name_match = re.match(test_name_pattern, line.strip())
        if test_name_match:
            current_test_name = test_name_match.groups()[0]

        result_match = re.match(result_pattern, line.strip())
        if result_match:
            status = result_match.groups()[0]
            if status == 'SUCCESS':
                test_status_map[current_test_name] = TestStatus.PASSED.value
            elif status == 'FAILURE':
                test_status_map[current_test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_ant(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r'^\s*\[junit\]\s+\[(PASS|FAIL|ERR)\]\s+(.*)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == 'PASS':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status in ['FAIL', 'ERR']:
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_gradle_custom(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with 'gradle test'.
    """
    test_status_map = {}

    pattern = r'^([^>].+)\s+(PASSED|FAILED)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            test_name, status = match.groups()
            if status == 'PASSED':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'FAILED':
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


# Gradle JSON lines parsing
JSON_PREFIX = '__TEST_JSON__ '


def parse_gradle_json_lines(log: str) -> dict[str, str]:
    """Parse structured Gradle JSON lines."""
    results: dict[str, str] = {}
    for line in log.splitlines():
        line = line.strip()
        if JSON_PREFIX not in line:
            continue
        prefix_idx = line.index(JSON_PREFIX)
        payload = line[prefix_idx + len(JSON_PREFIX) :].strip()
        try:
            obj = json.loads(payload)
            cls = obj.get('class') or obj.get('classname') or ''
            name = obj.get('name') or ''
            status_raw = (obj.get('status') or '').upper()
            if not cls or not name:
                continue
            if status_raw in ('SUCCESS', 'PASSED'):
                status = 'PASSED'
            elif status_raw in ('FAILURE', 'FAILED', 'ERROR'):
                status = 'FAILED'
            elif status_raw in ('SKIPPED', 'IGNORE', 'IGNORED'):
                status = 'SKIPPED'
            else:
                status = 'FAILED'
            test_id = mk_id(cls, name)
            results[test_id] = merge_status(results.get(test_id), status)
        except json.JSONDecodeError:
            continue
    return results


# JUnit XML parsing
XML_FILE_BEGIN = '__JUNIT_XML_FILE_BEGIN__'
XML_FILE_END = '__JUNIT_XML_FILE_END__'
FILE_BEGIN_RE = re.compile(r'__JUNIT_XML_FILE_BEGIN__\s+([^\r\n]+)')
FILE_END_RE = re.compile(r'__JUNIT_XML_FILE_END__\s+([^\r\n]+)')


def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def _sanitize_xml_text(xml: str) -> str:
    """Remove bash xtrace lines and trim to XML bounds."""
    cleaned_lines = []
    for line in xml.splitlines():
        ls = line.lstrip()
        if ls.startswith('+ ') and (' echo ' in ls or ' printf ' in ls):
            continue
        cleaned_lines.append(line)
    xml = '\n'.join(cleaned_lines)

    start = xml.find('<')
    end = xml.rfind('>')
    if start != -1 and end != -1 and end >= start:
        xml = xml[start : end + 1]
    return xml.strip()


def iter_junit_xml_strings_from_log(log: str) -> Iterable[tuple[str, str]]:
    """Extract (path, xml_string) pairs between FILE_BEGIN/FILE_END markers."""
    s = strip_ansi(log)
    pos = 0
    max_iterations = len(s) // 100 + 1000

    for _ in range(max_iterations):
        m_begin = FILE_BEGIN_RE.search(s, pos)
        if not m_begin:
            break

        path_raw = m_begin.group(1)
        path = _strip_quotes(path_raw)

        marker_start = m_begin.start()
        line_start = s.rfind('\n', max(0, marker_start - 500), marker_start)
        if line_start == -1:
            line_start = 0
        line_before_marker = s[line_start:marker_start]

        is_command_line = (
            '; cat' in line_before_marker
            or '; echo' in line_before_marker
            or 'done <' in line_before_marker
            or ('$' in line_before_marker and ('file' in line_before_marker or 'abs' in line_before_marker))
        )

        has_command_fragments = (
            any(char in path for char in [';', '"', "'", '`', '&&', '||'])
            or (' -D' in path)
            or (' done' in path)
            or (' echo' in path)
            or (' cat ' in path)
            or (' exit ' in path)
        )

        if is_command_line or has_command_fragments:
            new_pos = m_begin.end()
            pos = max(new_pos, pos + 1)
            continue

        nl_after_begin = s.find('\n', m_begin.end())
        if nl_after_begin == -1:
            break
        content_start = nl_after_begin + 1

        next_nl = s.find('\n', content_start)
        first_line = s[content_start : (next_nl if next_nl != -1 else content_start + 200)]
        if XML_FILE_BEGIN in first_line:
            content_start = (next_nl + 1) if next_nl != -1 else content_start

        m_end = FILE_END_RE.search(s, content_start)
        if not m_end:
            next_begin = FILE_BEGIN_RE.search(s, content_start)
            search_upto = next_begin.start() if next_begin else len(s)
            end_tag_idx = s.rfind('</testsuite>', content_start, search_upto)
            if end_tag_idx == -1:
                end_tag_idx = s.rfind('</testsuites>', content_start, search_upto)
            if end_tag_idx != -1:
                xml_text = s[content_start : end_tag_idx + len('</testsuite>')]
                yield (path, _sanitize_xml_text(xml_text))
            else:
                xml_text = s[content_start:search_upto]
                sanitized = _sanitize_xml_text(xml_text)
                if '<testcase' in sanitized:
                    yield (path, sanitized)
            pos = max(content_start, m_begin.end() + 1)
            continue

        content_end = m_end.start()
        xml_text = s[content_start:content_end]
        xml_text = _sanitize_xml_text(xml_text)
        yield (path, xml_text)
        pos = m_end.end()


def _parse_testcase_with_regex_fallback(xml_text: str) -> dict[str, str]:
    """Fallback parser using regex to extract testcases when XML parsing fails."""
    results: dict[str, str] = {}

    pattern1 = re.compile(
        r'<testcase[^>]*classname\s*=\s*["\']([^"\']+)["\'][^>]*name\s*=\s*["\']([^"\']+)["\']',
        re.MULTILINE | re.DOTALL,
    )
    pattern2 = re.compile(
        r'<testcase[^>]*name\s*=\s*["\']([^"\']+)["\'][^>]*classname\s*=\s*["\']([^"\']+)["\']',
        re.MULTILINE | re.DOTALL,
    )

    def decode_xml_entities(s: str) -> str:
        return (
            s
            .replace('&apos;', "'")
            .replace('&quot;', '"')
            .replace('&amp;', '&')
            .replace('&lt;', '<')
            .replace('&gt;', '>')
        )

    def extract_testcase_status(xml_text: str, start_pos: int, end_pos: int) -> str:
        elem_start = max(0, start_pos - 100)
        elem_end = min(len(xml_text), end_pos + 500)
        context = xml_text[elem_start:elem_end]

        if re.search(r'<skipped\s*/?>', context, re.IGNORECASE):
            return 'SKIPPED'
        if re.search(r'<(failure|error)\s*/?>', context, re.IGNORECASE):
            return 'FAILED'
        return 'PASSED'

    for match in pattern1.finditer(xml_text):
        cls = match.group(1)
        name = match.group(2)
        if cls and name and '.' in cls:
            cls = decode_xml_entities(cls).strip()
            name = decode_xml_entities(name).strip()
            start_pos = match.start()
            end_pos = xml_text.find('>', start_pos)
            if end_pos == -1:
                end_pos = start_pos + 500
            status = extract_testcase_status(xml_text, start_pos, end_pos)
            test_id = mk_id(cls, name)
            results[test_id] = merge_status(results.get(test_id), status)

    for match in pattern2.finditer(xml_text):
        name = match.group(1)
        cls = match.group(2)
        if cls and name and '.' in cls:
            cls = decode_xml_entities(cls).strip()
            name = decode_xml_entities(name).strip()
            start_pos = match.start()
            end_pos = xml_text.find('>', start_pos)
            if end_pos == -1:
                end_pos = start_pos + 500
            status = extract_testcase_status(xml_text, start_pos, end_pos)
            test_id = mk_id(cls, name)
            results[test_id] = merge_status(results.get(test_id), status)

    return results


def parse_junit_xml_string(xml_text: str) -> dict[str, str]:
    """Parse a JUnit XML document and return { "class.method": STATUS }."""
    results: dict[str, str] = {}
    xml_text = xml_text.strip()
    if not xml_text:
        return results

    root = None
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        xml_text2 = _sanitize_xml_text(xml_text)
        try:
            root = ET.fromstring(xml_text2)
            xml_text = xml_text2
        except ET.ParseError:
            return _parse_testcase_with_regex_fallback(xml_text)

    if root is None:
        return _parse_testcase_with_regex_fallback(xml_text)

    testcases = root.findall('.//testcase')

    if not testcases and '<testcase' in xml_text:
        regex_results = _parse_testcase_with_regex_fallback(xml_text)
        if regex_results:
            return regex_results

    for tc in testcases:
        cls = (tc.attrib.get('classname') or '').strip()
        name = (tc.attrib.get('name') or '').strip()
        if not cls or not name:
            continue

        status = 'PASSED'
        if tc.find('skipped') is not None:
            status = 'SKIPPED'
        if tc.find('failure') is not None or tc.find('error') is not None:
            status = 'FAILED'

        results[mk_id(cls, name)] = merge_status(results.get(mk_id(cls, name)), status)

    regex_results = _parse_testcase_with_regex_fallback(xml_text)
    for test_id, status in regex_results.items():
        if test_id not in results:
            results[test_id] = status

    return results


def parse_junit_xml_blocks_from_log(log: str) -> dict[str, str]:
    collected: dict[str, str] = {}
    for _path, xml_text in iter_junit_xml_strings_from_log(log):
        per_file = parse_junit_xml_string(xml_text)
        for k, v in per_file.items():
            collected[k] = merge_status(collected.get(k), v)
    return collected


def parse_ci_log(log: str) -> dict[str, str]:
    """
    Orchestrator: try in order → JSON lines, XML, console
    """
    json_results = parse_gradle_json_lines(log)
    if json_results:
        return json_results

    xml_results = parse_junit_xml_blocks_from_log(log)
    if xml_results:
        return xml_results

    maven_results = parse_log_maven_v2(log)
    gradle_results = parse_log_gradle_custom(log)

    merged: dict[str, str] = {}
    for k, v in {**maven_results, **gradle_results}.items():
        merged[k] = merge_status(merged.get(k), v)

    return merged


# ============== Ruby Parsers ==============
def parse_log_minitest(log: str) -> dict[str, str]:
    """
    Parses Minitest logs from classic or verbose format.
    """
    test_status_map = {}

    patterns = [
        re.compile(r'^(.+)\. .*=.*(\.|F|E).*$'),
        re.compile(r'^(.*?#.*?) = [\d.]+ s = (\.|F|E)$'),
    ]

    for line in log.split('\n'):
        line = line.strip()
        for pattern in patterns:
            match = pattern.match(line)
            if match:
                test_name, outcome = match.groups()
                if outcome == '.':
                    test_status_map[test_name] = TestStatus.PASSED.value
                elif outcome in {'F', 'E'}:
                    test_status_map[test_name] = TestStatus.FAILED.value
                break

    return test_status_map


def parse_log_cucumber(log: str) -> dict[str, str]:
    """
    Parse Cucumber test execution report.
    """

    def parse_json_report(data):
        def _has_failed(items) -> bool:
            for item in items:
                status = item.get('result', {}).get('status', '').lower()
                if status and status == 'failed':
                    return True
            return False

        test_status_map = {}

        for feature in data:
            if not isinstance(feature, dict):
                continue
            feature_name = feature.get('name', '<feature>')
            last_background_failed = False
            for element in feature.get('elements', []):
                if not isinstance(element, dict):
                    continue
                element_type = element.get('type', '').lower()
                element_keyword = element.get('keyword', '').lower()

                if element_type == 'background' or element_keyword == 'background':
                    last_background_failed = _has_failed(element)
                    continue

                if element_type in {'scenario', 'scenario_outline'} or element_keyword == 'scenario':
                    scenario_name = element.get('name', '<scenario>')
                    test_name = f'{feature_name} > {scenario_name}'
                    if (
                        last_background_failed
                        or _has_failed(element.get('before', []))
                        or _has_failed(element.get('steps', []))
                        or _has_failed(element.get('after', []))
                    ):
                        test_status_map[test_name] = TestStatus.FAILED.value
                    else:
                        test_status_map[test_name] = TestStatus.PASSED.value

        return test_status_map

    json_match = re.search(r'\[\s*\{.*\}\s*\]', log, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
        except ValueError:
            pass
        else:
            try:
                return parse_json_report(data)
            except KeyError:
                pass

    return {}


def parse_log_ruby_unit(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r'^\s*(?:test: )?(.+):\s+(\.|E\b|F\b|O\b)'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == '.':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome in ['E', 'F']:
                test_status_map[test_name] = TestStatus.FAILED.value
            elif outcome == 'O':
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_rspec(log: str) -> dict[str, str]:
    """
    Parse RSpec test execution report.
    """

    def parse_json_report(data):
        test_status_map = {}

        for item in data.get('examples', []):
            test_name = item['full_description']
            status = item['status']
            if status == 'passed':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'failed':
                test_status_map[test_name] = TestStatus.FAILED.value

        return test_status_map

    rspec_json_line = [line for line in log.split('\n') if '"examples":[{' in line]
    if rspec_json_line:
        json_pattern = r'\{.*"examples".*\}'
        json_match = re.search(json_pattern, rspec_json_line[0], re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
            except ValueError:
                pass
            else:
                try:
                    return parse_json_report(data)
                except KeyError:
                    pass

    return {}


def parse_log_tap_ruby(log: str) -> dict[str, str]:
    """
    Parser for TAP test execution report (Ruby version).
    """
    test_status_map = {}

    pattern = re.compile(
        r'^\s*(?:\d+:\s*)*'
        r'(not[ \t]+ok|ok)\b'
        r'(?:\s+(\d+))?'
        r'\s+-\s+([^#\n]*?)'
        r'(?:\s*#\s*(SKIP|TODO)\b'
        r'(?:\s+(.*))?)?'
        r'\s*$',
        re.IGNORECASE,
    )

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, _, test_name, directive, _ = match.groups()
            test_name = test_name.strip()
            status = status.strip().lower()
            if directive:
                continue
            if status == 'ok':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'not ok':
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


# ============== Rust Parsers ==============
def parse_log_cargo(log: str) -> dict[str, str]:
    """
    Parser for test logs generated with cargo test.
    """
    test_status_map = {}

    pattern_complete = r'^test\s+(\S+)\s+\.\.\.\s+(ok|FAILED)$'
    pattern_embedded = r'test\s+(\S+)\s+\.\.\.\s+(ok|FAILED)'
    pattern_incomplete = r'^test\s+(\S+)\s+\.\.\.'
    pattern_result = r'^(ok|FAILED)$'
    pattern_result_concatenated = r'^(ok|FAILED)(?=\S)'
    pattern_result_with_whitespace = r'^(ok|FAILED)\s+\S'
    pattern_result_at_end = r'(ok|FAILED)$'

    pending_test = None

    for line in log.split('\n'):
        line_stripped = strip_ansi(line).strip()

        match = re.match(pattern_complete, line_stripped)
        if match:
            test_name, outcome = match.groups()
            test_status_map[test_name] = TestStatus.PASSED.value if outcome == 'ok' else TestStatus.FAILED.value
            pending_test = None
            continue

        match = re.search(pattern_embedded, line_stripped)
        if match:
            test_name, outcome = match.groups()
            test_status_map[test_name] = TestStatus.PASSED.value if outcome == 'ok' else TestStatus.FAILED.value
            pending_test = None
            continue

        if not pending_test:
            match = re.match(pattern_incomplete, line_stripped)
            if match:
                test_name = match.group(1)
                result_check = re.match(r'^test\s+\S+\s+\.\.\.\s+(ok|FAILED)', line_stripped)
                if not result_check:
                    pending_test = test_name
                continue

            match = re.search(r'test\s+(\S+)\s+\.\.\.', line_stripped)
            if match:
                test_name = match.group(1)
                result_check = re.search(r'test\s+\S+\s+\.\.\.\s+(ok|FAILED)', line_stripped)
                if not result_check:
                    pending_test = test_name
                continue

        if pending_test:
            for pat in [pattern_result, pattern_result_concatenated, pattern_result_with_whitespace]:
                match = re.match(pat, line_stripped)
                if match:
                    outcome = match.group(1)
                    test_status_map[pending_test] = (
                        TestStatus.PASSED.value if outcome == 'ok' else TestStatus.FAILED.value
                    )
                    pending_test = None
                    break
            else:
                match = re.search(pattern_result_at_end, line_stripped)
                if match:
                    outcome = match.group(1)
                    test_status_map[pending_test] = (
                        TestStatus.PASSED.value if outcome == 'ok' else TestStatus.FAILED.value
                    )
                    pending_test = None

    return test_status_map


# ============== PHP Parsers ==============
def parse_log_phpunit(log: str) -> dict[str, str]:
    """
    Parser for phpunit logs with the --testdox option.
    """
    test_status_map = {}
    suite = None

    suite_pattern = r'^(\w.+) \(.+\)$'
    test_pattern = r'^\s*([✔✘↩])\s*(.*)$'

    for line in log.split('\n'):
        suite_match = re.match(suite_pattern, line)
        if suite_match:
            suite = suite_match.groups()[0]
            continue

        test_match = re.match(test_pattern, line)
        if test_match:
            status, test_name = test_match.groups()
            full_test_name = f'{suite} > {test_name}'

            if status == '✔':
                test_status_map[full_test_name] = TestStatus.PASSED.value
            elif status == '✘':
                test_status_map[full_test_name] = TestStatus.FAILED.value
            elif status == '↩':
                test_status_map[full_test_name] = TestStatus.SKIPPED.value

    return test_status_map


# ============== C/C++ Parsers ==============
def parse_log_redis(log: str) -> dict[str, str]:
    """
    Parser for Redis test logs.
    """
    test_status_map = {}

    pattern = r'^\[(ok|err|skip|ignore)\]:\s(.+?)(?:\s\((\d+\s*m?s)\))?$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name, _duration = match.groups()
            if status == 'ok':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'err':
                test_name = re.sub(r'\s+in\s+\S+$', '', test_name)
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status in ('skip', 'ignore'):
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_jq(log: str) -> dict[str, str]:
    """
    Parser for jq test logs.
    """
    test_status_map = {}

    pattern = r'^\s*(PASS|FAIL):\s(.+)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == 'PASS':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'FAIL':
                test_status_map[test_name] = TestStatus.FAILED.value
    return test_status_map


def parse_log_micropython_test(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r'^(pass|FAIL|skip)\s+(.+)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == 'pass':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'FAIL':
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status == 'skip':
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_googletest(log: str) -> dict[str, str]:
    test_status_map = {}

    pattern = r'^.*\[\s*(OK|FAILED)\s*\]\s(.*)\s\(.*\)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == 'OK':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'FAILED':
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_ctest(log: str) -> dict[str, str]:
    """
    Parser for ctest test runner report.
    """
    test_status_map = {}

    pattern = re.compile(r'^.*Test\s+#\d+:\s+(.*\.+)\s*(.*)\s+\d+\.\d+\s+sec$')

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            test_name, status = match.groups()
            test_name = test_name.strip('.').strip()
            status = status.strip().lower()
            if status == 'passed':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'failed' or status.startswith('***'):
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_doctest(log: str) -> dict[str, str]:
    """
    Parser for Doctest test execution report.
    """

    def parse_log_xml(log: str) -> dict[str, str]:
        def parse_testcase_section(section):
            section_name = section.get('name', '<section>')
            section_expressions = section.findall('Expression')
            if section_expressions is not None:
                section_result = all(expr.get('success') == 'true' for expr in section_expressions)
            else:
                section_result = True

            nested_section = section.find('SubCase')
            if nested_section is not None:
                nested_section_name, nested_section_result = parse_testcase_section(nested_section)
                section_name = f'{section_name} > {nested_section_name}'
                section_result = section_result and nested_section_result

            return section_name, section_result

        test_status_map = {}

        root = ET.fromstring(log)

        for testcase in root.findall('.//TestCase'):
            if testcase.get('skipped', '') == 'true':
                continue
            testcase_name = testcase.get('name', '<testcase>')
            testcase_overall_result = testcase.find('OverallResultsAsserts')
            if testcase_overall_result is not None:
                testcase_passed = testcase_overall_result.get('test_case_success') == 'true'
                if testcase_passed:
                    test_status_map[testcase_name] = TestStatus.PASSED.value
                else:
                    test_status_map[testcase_name] = TestStatus.FAILED.value
                for section in testcase.findall('SubCase'):
                    section_name, section_passed = parse_testcase_section(section)
                    testcase_section_name = f'{testcase_name} > {section_name}'
                    if section_passed:
                        test_status_map[testcase_section_name] = TestStatus.PASSED.value
                    else:
                        test_status_map[testcase_section_name] = TestStatus.FAILED.value

        return test_status_map

    start_tag = '<doctest'
    end_tag = '</doctest>'
    start_index = log.find(start_tag)
    end_index = log.find(end_tag, start_index) + len(end_tag) if start_index != -1 else -1
    if start_index != -1 and end_index != -1:
        return parse_log_xml(log[start_index:end_index])

    return parse_log_ctest(log)


def parse_log_catch2(log: str) -> dict[str, str]:
    """
    Parser for Catch2 test execution report.
    """

    def parse_log_xml(log: str) -> dict[str, str]:
        def parse_testcase_section(section):
            section_name = section.get('name', '<section>')
            section_expressions = section.findall('Expression')
            if section_expressions is not None:
                section_result = all(expr.get('success') == 'true' for expr in section_expressions)
            else:
                section_result = True

            nested_section = section.find('Section')
            if nested_section is not None:
                nested_section_name, nested_section_result = parse_testcase_section(nested_section)
                section_name = f'{section_name} > {nested_section_name}'
                section_result = section_result and nested_section_result

            return section_name, section_result

        test_status_map = {}

        root = ET.fromstring(log)

        for testcase in root.findall('.//TestCase'):
            testcase_name = testcase.get('name', '<testcase>')
            testcase_overall_result = testcase.find('OverallResult')
            if testcase_overall_result is not None:
                if testcase_overall_result.get('status', '') == 'skipped':
                    continue
                testcase_passed = testcase_overall_result.get('success') == 'true'
                if testcase_passed:
                    test_status_map[testcase_name] = TestStatus.PASSED.value
                else:
                    test_status_map[testcase_name] = TestStatus.FAILED.value
                for section in testcase.findall('Section'):
                    section_overall_result = section.find('OverallResults')
                    if section_overall_result is not None and section_overall_result.get('skipped', '') == 'true':
                        continue
                    section_name, section_passed = parse_testcase_section(section)
                    testcase_section_name = f'{testcase_name} > {section_name}'
                    if section_passed:
                        test_status_map[testcase_section_name] = TestStatus.PASSED.value
                    else:
                        test_status_map[testcase_section_name] = TestStatus.FAILED.value

        return test_status_map

    start_tag = '<Catch2TestRun'
    end_tag = '</Catch2TestRun>'
    start_index = log.find(start_tag)
    end_index = log.find(end_tag, start_index) + len(end_tag) if start_index != -1 else -1
    if start_index != -1 and end_index != -1:
        return parse_log_xml(log[start_index:end_index])

    return parse_log_ctest(log)


def parse_log_tap_c(log: str) -> dict[str, str]:
    """
    Parser for TAP test execution report (C version).
    """
    test_status_map = {}

    pattern = re.compile(
        r'^\s*(?:\d+:\s*)*'
        r'(not[ \t]+ok|ok)\b'
        r'(?:\s+(\d+))?'
        r'\s+-\s+([^#\n]*?)'
        r'(?:\s*#\s*(SKIP|TODO)\b'
        r'(?:\s+(.*))?)?'
        r'\s*$',
        re.IGNORECASE,
    )

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            status, _, test_name, directive, _ = match.groups()
            test_name = test_name.strip()
            status = status.strip().lower()
            if directive:
                continue
            if status == 'ok':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'not ok':
                test_status_map[test_name] = TestStatus.FAILED.value

    if test_status_map:
        return test_status_map

    return parse_log_ctest(log)


def parse_log_common_c(log: str) -> dict[str, str]:
    """
    Try all C parsers and return results.
    """
    parsers = [
        parse_log_googletest,
        parse_log_doctest,
        parse_log_catch2,
        parse_log_tap_c,
        parse_log_ctest,
    ]
    results = {}
    for parser in parsers:
        result = parser(log)
        if result:
            results[parser.__name__] = result

    if len(results) == 1:
        return list(results.values())[0]

    first_result = None
    for result in results.values():
        if first_result is None:
            first_result = result
        elif result != first_result:
            return {}
    return first_result if first_result is not None else {}


# ============== C# Parsers ==============
def parse_log_dotnet(log: str) -> dict[str, str]:
    """
    Parser for .NET test logs (supports xUnit, NUnit, MSTest).
    """
    test_status_map = {}

    nunit_pattern = r'Passed:\s*(\d+),\s*Failed:\s*(\d+),\s*Skipped:\s*(\d+)'
    nunit_match = re.search(nunit_pattern, log)
    if nunit_match:
        passed, failed, skipped = map(int, nunit_match.groups())
        for i in range(passed):
            test_status_map[f'nunit_passed_{i}'] = TestStatus.PASSED.value
        for i in range(failed):
            test_status_map[f'nunit_failed_{i}'] = TestStatus.FAILED.value
        for i in range(skipped):
            test_status_map[f'nunit_skipped_{i}'] = TestStatus.SKIPPED.value
        return test_status_map

    xunit_pattern = r'Total tests:\s*\d+.*?Passed:\s*(\d+).*?Failed:\s*(\d+).*?Skipped:\s*(\d+)'
    xunit_match = re.search(xunit_pattern, log, re.DOTALL)
    if xunit_match:
        passed, failed, skipped = map(int, xunit_match.groups())
        for i in range(passed):
            test_status_map[f'xunit_passed_{i}'] = TestStatus.PASSED.value
        for i in range(failed):
            test_status_map[f'xunit_failed_{i}'] = TestStatus.FAILED.value
        for i in range(skipped):
            test_status_map[f'xunit_skipped_{i}'] = TestStatus.SKIPPED.value
        return test_status_map

    mstest_pattern = r'Total tests:\s*(\d+).*?Passed:\s*(\d+).*?Failed:\s*(\d+).*?Skipped:\s*(\d+)'
    mstest_match = re.search(mstest_pattern, log, re.DOTALL)
    if mstest_match:
        _total, passed, failed, skipped = map(int, mstest_match.groups())
        for i in range(passed):
            test_status_map[f'mstest_passed_{i}'] = TestStatus.PASSED.value
        for i in range(failed):
            test_status_map[f'mstest_failed_{i}'] = TestStatus.FAILED.value
        for i in range(skipped):
            test_status_map[f'mstest_skipped_{i}'] = TestStatus.SKIPPED.value
        return test_status_map

    for line in log.splitlines():
        if 'Passed' in line:
            test_status_map[line.strip()] = TestStatus.PASSED.value
        elif 'Failed' in line:
            test_status_map[line.strip()] = TestStatus.FAILED.value
        elif 'Skipped' in line:
            test_status_map[line.strip()] = TestStatus.SKIPPED.value

    return test_status_map


# ============== Parser Registry ==============
def get_py_parser_by_name(name: str):
    if name == 'pytest':
        return parse_log_pytest_v2
    if name == 'django':
        return parse_log_django
    if name == 'seaborn':
        return parse_log_seaborn
    if name == 'sympy':
        return parse_log_sympy
    if name == 'matplotlib':
        return parse_log_matplotlib
    return parse_log_pytest_v2


def get_go_parser_by_name(name: str):
    if name == 'gotest':
        return parse_log_gotest
    return parse_log_gotest


def get_js_parser_by_name(name: str):
    if name == 'calypso':
        return parse_log_calypso
    if name == 'chartjs':
        return parse_log_chart_js
    if name == 'marked':
        return parse_log_marked
    if name == 'p5js':
        return parse_log_p5js
    if name == 'reactpdf':
        return parse_log_react_pdf
    if name == 'vitest':
        return parse_log_vitest
    if name == 'jest':
        return parse_log_jest
    if name == 'mocha':
        return parse_log_mocha_v2
    if name == 'karma':
        return parse_log_karma
    if name == 'tap':
        return parse_log_tap
    return parse_log_vitest


def get_java_parser_by_name(name: str):
    if name == 'maven':
        return parse_ci_log
    if name == 'ant':
        return parse_log_ant
    if name == 'gradle':
        return parse_ci_log
    return parse_ci_log


def get_ruby_parser_by_name(name: str):
    if name == 'rubyunit':
        return parse_log_ruby_unit
    if name == 'minitest':
        return parse_log_minitest
    if name == 'rspec':
        return parse_log_rspec
    if name == 'cucumber':
        return parse_log_cucumber
    if name == 'tap':
        return parse_log_tap_ruby
    return parse_log_ruby_unit


def get_rust_parser_by_name(name: str):
    if name == 'cargo':
        return parse_log_cargo
    return parse_log_cargo


def get_php_parser_by_name(name: str):
    if name == 'phpunit':
        return parse_log_phpunit
    return parse_log_phpunit


def get_c_parser_by_name(name: str):
    if name == 'non_agentic':
        return parse_log_common_c
    if name == 'googletest':
        return parse_log_googletest
    if name == 'doctest':
        return parse_log_doctest
    if name == 'catch2':
        return parse_log_catch2
    if name == 'tap':
        return parse_log_tap_c
    return parse_log_ctest


def get_cs_parser_by_name(name: str):
    if name == 'dotnet':
        return parse_log_dotnet
    return parse_log_dotnet


LANGUAGE_PARSER_MAP = {
    C_SBP_STR: get_c_parser_by_name,
    CSHARP_SBP_STR: get_cs_parser_by_name,
    C_PLUS_SBP_STR: get_c_parser_by_name,
    JAVA_SBP_STR: get_java_parser_by_name,
    GO_SBP_STR: get_go_parser_by_name,
    JAVASCRIPT_SBP_STR: get_js_parser_by_name,
    TYPESCRIPT_SBP_STR: get_js_parser_by_name,
    PHP_SBP_STR: get_php_parser_by_name,
    PYTHON_SBP_STR: get_py_parser_by_name,
    RUST_SBP_STR: get_rust_parser_by_name,
    RUBY_SBP_STR: get_ruby_parser_by_name,
}


# Static repo-to-parser map for backward compatibility
MAP_REPO_TO_PARSER = {
    # Python repos
    'astropy/astropy': parse_log_pytest_v2,
    'django/django': parse_log_django,
    'marshmallow-code/marshmallow': parse_log_pytest,
    'matplotlib/matplotlib': parse_log_matplotlib,
    'mwaskom/seaborn': parse_log_seaborn,
    'pallets/flask': parse_log_pytest,
    'psf/requests': parse_log_pytest_options,
    'pvlib/pvlib-python': parse_log_pytest,
    'pydata/xarray': parse_log_pytest,
    'pydicom/pydicom': parse_log_pytest_options,
    'pylint-dev/astroid': parse_log_pytest,
    'pylint-dev/pylint': parse_log_pytest_options,
    'pytest-dev/pytest': parse_log_pytest,
    'pyvista/pyvista': parse_log_pytest,
    'scikit-learn/scikit-learn': parse_log_pytest_v2,
    'sqlfluff/sqlfluff': parse_log_pytest,
    'sphinx-doc/sphinx': parse_log_pytest_v2,
    'sympy/sympy': parse_log_sympy,
    'redis/redis-py': parse_log_pytest,
    'ipython/ipython': parse_log_pytest,
    'Textualize/rich': parse_log_pytest,
    'python-pillow/Pillow': parse_log_pytest,
    'dask/dask': parse_log_pytest_v2,
    'pandas-dev/pandas': parse_log_pytest,
    'celery/celery': parse_log_pytest,
    'reflex-dev/reflex': parse_log_pytest,
    'getmoto/moto': parse_log_pytest_v2,
    'saltstack/salt': parse_log_pytest,
    'Pyomo/pyomo': parse_log_pytest_v2,
    # Go repos
    'caddyserver/caddy': parse_log_gotest,
    'hashicorp/terraform': parse_log_gotest,
    'prometheus/prometheus': parse_log_gotest,
    'gohugoio/hugo': parse_log_gotest,
    'gin-gonic/gin': parse_log_gotest,
    'kubernetes/kubernetes': parse_log_gotest,
    'gravitational/teleport': parse_log_gotest,
    # JavaScript repos
    'Automattic/wp-calypso': parse_log_calypso,
    'chartjs/Chart.js': parse_log_chart_js,
    'markedjs/marked': parse_log_marked,
    'processing/p5.js': parse_log_p5js,
    'diegomura/react-pdf': parse_log_react_pdf,
    'axios/axios': parse_log_marked,
    'tjw-lint/vue3-snapshot-serializer': parse_log_vitest,
    # Java repos
    'google/gson': parse_log_maven_v2,
    'apache/druid': parse_log_maven,
    'javaparser/javaparser': parse_log_maven,
    'projectlombok/lombok': parse_log_ant,
    'apache/lucene': parse_log_gradle_custom,
    'reactivex/rxjava': parse_log_gradle_custom,
    'apache/commons-lang': parse_log_maven_v2,
    'spring-projects/spring-framework': parse_log_gradle_custom,
    # Rust repos
    'BurntSushi/ripgrep': parse_log_cargo,
    'sharkdp/bat': parse_log_cargo,
    'astral-sh/ruff': parse_log_cargo,
    'tokio-rs/tokio': parse_log_cargo,
    'uutils/coreutils': parse_log_cargo,
    'nushell/nushell': parse_log_cargo,
    'tokio-rs/axum': parse_log_cargo,
    # Ruby repos
    'fluent/fluentd': parse_log_ruby_unit,
    'fastlane/fastlane': parse_log_rspec,
    'jordansissel/fpm': parse_log_rspec,
    'faker-ruby/faker': parse_log_ruby_unit,
    'rubocop/rubocop': parse_log_rspec,
    # PHP repos
    'phpoffice/phpspreadsheet': parse_log_phpunit,
    'laravel/framework': parse_log_phpunit,
    'php-cs-fixer/php-cs-fixer': parse_log_phpunit,
    'briannesbitt/carbon': parse_log_phpunit,
    # C/C++ repos
    'redis/redis': parse_log_redis,
    'jqlang/jq': parse_log_jq,
    'nlohmann/json': parse_log_doctest,
    'micropython/micropython': parse_log_micropython_test,
    'valkey-io/valkey': parse_log_redis,
    'fmtlib/fmt': parse_log_googletest,
    # C# repos
    'dotnet/runtime': parse_log_dotnet,
    'dotnet/aspnetcore': parse_log_dotnet,
    'dotnet/mstest': parse_log_dotnet,
    # Default
    'default': parse_log_pytest_v2,
}
