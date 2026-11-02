"""
Log parsers for different programming languages and test frameworks
Imported from SWE-bench harness
"""

import re
import xml.etree.ElementTree as ET

from collections.abc import Callable

from nexus.extensions.tasks.swe.swe_bench_multilingual.constants import TestStatus
from nexus.extensions.tasks.swe.swe_bench_multilingual.test_spec import TestSpec


# ============================================================================
# PYTHON LOG PARSERS
# ============================================================================


def parse_log_pytest(log: str, test_spec: TestSpec) -> dict[str, str]:
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
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


def parse_log_pytest_options(log: str, test_spec: TestSpec) -> dict[str, str]:
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


def parse_log_django(log: str, test_spec: TestSpec) -> dict[str, str]:
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
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
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
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            test_status_map[test] = TestStatus.PASSED.value

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
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


def parse_log_pytest_v2(log: str, test_spec: TestSpec) -> dict[str, str]:
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
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[1]] = test_case[0]
        # Support older pytest versions by checking if the line ends with the test status
        elif any(line.endswith(x.value) for x in TestStatus):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = test_case[1]
    return test_status_map


def parse_log_seaborn(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
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


def parse_log_sympy(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
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


def parse_log_matplotlib(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split('\n'):
        line = line.replace('MouseButton.LEFT', '1')
        line = line.replace('MouseButton.RIGHT', '3')
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(' - ', ' ')
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = test_case[0]
    return test_status_map


parse_log_astroid = parse_log_pytest
parse_log_flask = parse_log_pytest
parse_log_marshmallow = parse_log_pytest
parse_log_pvlib = parse_log_pytest
parse_log_pyvista = parse_log_pytest
parse_log_sqlfluff = parse_log_pytest
parse_log_xarray = parse_log_pytest

parse_log_pydicom = parse_log_pytest_options
parse_log_requests = parse_log_pytest_options
parse_log_pylint = parse_log_pytest_options

parse_log_astropy = parse_log_pytest_v2
parse_log_scikit = parse_log_pytest_v2
parse_log_sphinx = parse_log_pytest_v2


MAP_REPO_TO_PARSER_PY = {
    'astropy/astropy': parse_log_astropy,
    'django/django': parse_log_django,
    'marshmallow-code/marshmallow': parse_log_marshmallow,
    'matplotlib/matplotlib': parse_log_matplotlib,
    'mwaskom/seaborn': parse_log_seaborn,
    'pallets/flask': parse_log_flask,
    'psf/requests': parse_log_requests,
    'pvlib/pvlib-python': parse_log_pvlib,
    'pydata/xarray': parse_log_xarray,
    'pydicom/pydicom': parse_log_pydicom,
    'pylint-dev/astroid': parse_log_astroid,
    'pylint-dev/pylint': parse_log_pylint,
    'pytest-dev/pytest': parse_log_pytest,
    'pyvista/pyvista': parse_log_pyvista,
    'scikit-learn/scikit-learn': parse_log_scikit,
    'sqlfluff/sqlfluff': parse_log_sqlfluff,
    'sphinx-doc/sphinx': parse_log_sphinx,
    'sympy/sympy': parse_log_sympy,
}


# ============================================================================
# JAVA LOG PARSERS
# ============================================================================


def parse_log_maven(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with 'mvn test'.
    Annoyingly maven will not print the tests that have succeeded. For this log
    parser to work, each test must be run individually, and then we look for
    BUILD (SUCCESS|FAILURE) in the logs.

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    current_test_name = '---NO TEST NAME FOUND YET---'

    # Get the test name from the command used to execute the test.
    # Assumes we run evaluation with set -x
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


def parse_log_ant(log: str, test_spec: TestSpec) -> dict[str, str]:
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


def parse_log_gradle_custom(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with 'gradle test'. Assumes that the
    pre-install script to update the gradle config has run.

    Handles race conditions where test name and status appear on different lines
    due to interleaved log output from concurrent processes.
    """
    test_status_map = {}

    # Pattern for normal case: test name and status on the same line
    # e.g., "com.example.Test > testMethod PASSED"
    # [^>] ensures we don't match lines starting with > (shell prompts, etc.)
    full_pattern = r'^([^>].+)\s+(PASSED|FAILED)$'

    # Pattern for test name without status (race condition case)
    # e.g., "com.example.Test > testMethod" followed by warnings, then "PASSED"
    # Must also start with [^>] for consistency
    test_name_pattern = r'^([^>]\S*\s+>\s+\S+)$'

    # Pattern for standalone status line
    status_only_pattern = r'^(PASSED|FAILED)$'

    pending_test_name = None

    for line in log.split('\n'):
        stripped = line.strip()

        # Check for full match (test name + status on same line)
        match = re.match(full_pattern, stripped)
        if match:
            test_name, status = match.groups()
            if status == 'PASSED':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif status == 'FAILED':
                test_status_map[test_name] = TestStatus.FAILED.value
            pending_test_name = None
            continue

        # Check for test name without status
        test_name_match = re.match(test_name_pattern, stripped)
        if test_name_match:
            pending_test_name = test_name_match.group(1)
            continue

        # Check for standalone status (applies to pending test name)
        if pending_test_name:
            status_match = re.match(status_only_pattern, stripped)
            if status_match:
                status = status_match.group(1)
                if status == 'PASSED':
                    test_status_map[pending_test_name] = TestStatus.PASSED.value
                elif status == 'FAILED':
                    test_status_map[pending_test_name] = TestStatus.FAILED.value
                pending_test_name = None

    # Warn if there's a pending test without a result
    if pending_test_name:
        print(f'[WARNING] Gradle log parser: test had no status result: {pending_test_name}')

    return test_status_map


MAP_REPO_TO_PARSER_JAVA = {
    'google/gson': parse_log_maven,
    'apache/druid': parse_log_maven,
    'javaparser/javaparser': parse_log_maven,
    'projectlombok/lombok': parse_log_ant,
    'apache/lucene': parse_log_gradle_custom,
    'reactivex/rxjava': parse_log_gradle_custom,
}


# ============================================================================
# JAVASCRIPT LOG PARSERS
# ============================================================================


def parse_log_calypso(log: str, test_spec: TestSpec) -> dict[str, str]:
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

    for log_chunk in log.split(' ./node_modules/.bin/jest ')[1:]:
        for line in log_chunk.split('\n'):
            if any(line.startswith(x) for x in ['Test Suites', '  ● ']):
                break
            if line.strip().startswith('✓'):
                # Test passed
                match_pattern = (
                    r'^\s+✓\s(.*)\(\d+ms\)$' if re.search(r'\(\d+ms\)', line) is not None else r'^\s+✓\s(.*)'
                )
                test_status_map[get_test_name(suite, match_pattern, line)] = TestStatus.PASSED.value
            elif line.strip().startswith('✕'):
                # Test failed
                match_pattern = (
                    r'^\s+✕\s(.*)\(\d+ms\)$' if re.search(r'\(\d+ms\)', line) is not None else r'^\s+✕\s(.*)'
                )
                test_status_map[get_test_name(suite, match_pattern, line)] = TestStatus.FAILED.value
            elif len(line) - len(line.lstrip()) > 0:
                # Adjust suite name
                indent = len(line) - len(line.lstrip())
                if len(suite) == 0:
                    # If suite is empty, initialize it
                    suite = [(line.strip(), indent)]
                else:
                    while len(suite) > 0 and suite[-1][-1] >= indent:
                        # Pop until the last element with indent less than current indent
                        suite.pop()
                    suite.append([line.strip(), indent])

    return test_status_map


def parse_log_chart_js(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated by ChartJS test suite
    """
    log = ansi_escape(log)
    test_status_map = {}
    failure_case_patterns = [
        # use [^\S\r\n] to avoid overlapping Chrome groups on separate lines
        (r'Chrome\s[\d\.]+[^\S\r\n]\(.+?\)[^\S\r\n](.*)FAILED$', re.MULTILINE),
    ]
    for failure_case_pattern, flags in failure_case_patterns:
        failures = re.findall(failure_case_pattern, log, flags)
        if len(failures) == 0:
            continue
        for failure in failures:
            test_status_map[failure] = TestStatus.FAILED.value
    return test_status_map


def parse_log_marked(log: str, test_spec: TestSpec) -> dict[str, str]:
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


def parse_log_p5js(log: str, test_spec: TestSpec) -> dict[str, str]:
    def remove_json_blocks(log_content):
        filtered_lines = []
        in_json_block = False
        in_json_list_block = False
        for line in log_content.split('\n'):
            stripped_line = line.rstrip()  # Remove trailing whitespace
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

    def remove_xml_blocks(log_content):
        xml_pat = re.compile(r'<(\w+)>[\s\S]*?<\/\1>', re.MULTILINE)
        match = xml_pat.search(log_content)
        while match:
            # count the number of opening tags in the match
            opening_tags = match.group().count(rf'<{match.group(1)}>') - 1
            opening_tags = max(opening_tags, 0)
            start = match.start()
            end = match.end()
            log_content = log_content[:start] + f'<{match.group(1)}>' * opening_tags + log_content[end:]
            match = xml_pat.search(log_content)
        return log_content

    def is_valid_fail(match):
        last_line_indent = 0
        for line in match.group(2).split('\n'):
            line_indent = len(line) - len(line.lstrip())
            if line_indent <= last_line_indent:
                return False
            last_line_indent = line_indent
        return True

    log = ansi_escape(log)
    log = remove_json_blocks(log)
    log = remove_xml_blocks(log)
    test_results = {}

    # Parse failing tests
    fail_pattern = re.compile(r'^\s*(\d+)\)(.{0,1000}?):', re.MULTILINE | re.DOTALL)
    for match in fail_pattern.finditer(log):
        if is_valid_fail(match):
            test_names = list(map(str.strip, match.group(2).split('\n')))
            full_name = ':'.join(test_names)
            test_results[full_name] = TestStatus.FAILED.value

    return test_results


def parse_log_react_pdf(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated by Carbon test suite
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


def parse_log_jest(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Jest. Assumes --verbose flag.

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r'^\s*(✓|✕|○)\s(.+?)(?:\s\((\d+\s*m?s)\))?$'

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


def parse_log_jest_json(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Jest. Assumes the --json flag has been
    piped into JEST_JSON_JQ_TRANSFORM. Unlike --verbose, tests with the same name
    in different describe blocks print with different names.
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


def parse_log_vitest(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with vitest. Assumes --reporter=verbose flag.
    """
    test_status_map = {}

    pattern = r'^\s*(✓|×|↓)\s(.+?)(?:\s(\d+\s*m?s?|\[skipped\]))?$'

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


def parse_log_karma(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Karma. Handles duplicate test names in
    different describe blocks. Logic is brittle.
    """
    test_status_map = {}
    current_indent = -1
    current_suite = []
    started = False

    pattern = r'^(\s*)?([✔✖])?\s(.*)$'

    for line in log.split('\n'):
        if line.startswith('SUMMARY:'):
            # Individual test logs end here
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


def parse_log_tap(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with TAP

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    # Pattern to match TAP result lines
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


def parse_log_immutable_js(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Different immutable.js instances use different test runners and log formats.
    This function selects the appropriate log parser based on the instance id.
    """
    pr_number = test_spec.instance_id.split('-')[-1]

    if pr_number in ['2006']:
        return parse_log_jest(log, test_spec)
    if pr_number in ['2005']:
        return parse_log_jest_json(log, test_spec)
    raise ValueError(f'Unknown instance id: {test_spec.instance_id}')


MAP_REPO_TO_PARSER_JS = {
    'Automattic/wp-calypso': parse_log_calypso,
    'chartjs/Chart.js': parse_log_chart_js,
    'markedjs/marked': parse_log_marked,
    'processing/p5.js': parse_log_p5js,
    'diegomura/react-pdf': parse_log_react_pdf,
    'babel/babel': parse_log_jest,
    'vuejs/core': parse_log_vitest,
    'facebook/docusaurus': parse_log_jest,
    'immutable-js/immutable-js': parse_log_immutable_js,
    'mrdoob/three.js': parse_log_tap,
    'preactjs/preact': parse_log_karma,
    'axios/axios': parse_log_tap,
}


# ============================================================================
# C LOG PARSERS
# ============================================================================


def parse_log_redis(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
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
                # Strip out file path information from failed test names
                test_name = re.sub(r'\s+in\s+\S+$', '', test_name)
                test_status_map[test_name] = TestStatus.FAILED.value
            elif status == 'skip' or status == 'ignore':
                test_status_map[test_name] = TestStatus.SKIPPED.value

    return test_status_map


def parse_log_jq(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
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


def parse_log_doctest(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Assumes test binary runs with -s -r=xml.
    """
    test_status_map = {}

    # Extract XML content
    start_tag = '<doctest'
    end_tag = '</doctest>'
    start_index = log.find(start_tag)
    end_index = log.find(end_tag, start_index) + len(end_tag) if start_index != -1 else -1

    if start_index != -1 and end_index != -1:
        xml_string = log[start_index:end_index]
        root = ET.fromstring(xml_string)

        for testcase in root.findall('.//TestCase'):
            testcase_name = testcase.get('name')
            for subcase in testcase.findall('.//SubCase'):
                subcase_name = subcase.get('name')
                name = f'{testcase_name} > {subcase_name}'

                expressions = subcase.findall('.//Expression')
                subcase_passed = all(expr.get('success') == 'true' for expr in expressions)

                if subcase_passed:
                    test_status_map[name] = TestStatus.PASSED.value
                else:
                    test_status_map[name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_micropython_test(log: str, test_spec: TestSpec) -> dict[str, str]:
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


def parse_log_googletest(log: str, test_spec: TestSpec) -> dict[str, str]:
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


MAP_REPO_TO_PARSER_C = {
    'redis/redis': parse_log_redis,
    'jqlang/jq': parse_log_jq,
    'nlohmann/json': parse_log_doctest,
    'micropython/micropython': parse_log_micropython_test,
    'valkey-io/valkey': parse_log_redis,
    'fmtlib/fmt': parse_log_googletest,
}


# ============================================================================
# GO LOG PARSERS
# ============================================================================


def parse_log_gotest(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for test logs generated with 'go test'

    Args:
        log (str): log content
        test_spec (TestSpec): test spec (unused)
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


MAP_REPO_TO_PARSER_GO = {
    'caddyserver/caddy': parse_log_gotest,
    'hashicorp/terraform': parse_log_gotest,
    'prometheus/prometheus': parse_log_gotest,
    'gohugoio/hugo': parse_log_gotest,
    'gin-gonic/gin': parse_log_gotest,
}


# ============================================================================
# PHP LOG PARSERS
# ============================================================================


def parse_log_phpunit(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Parser for phpunit logs with the --testdox option.
    Args:
        log (str): log content
        test_spec (TestSpec): test spec (unused)
    Returns:
        dict: test case to test status mapping
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


MAP_REPO_TO_PARSER_PHP = {
    'phpoffice/phpspreadsheet': parse_log_phpunit,
    'laravel/framework': parse_log_phpunit,
    'php-cs-fixer/php-cs-fixer': parse_log_phpunit,
    'briannesbitt/carbon': parse_log_phpunit,
}


# ============================================================================
# RUBY LOG PARSERS
# ============================================================================


def parse_log_minitest(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r'^(.+)\. .*=.*(\.|F|E).*$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == '.':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome in ['F', 'E']:
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_cucumber(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Assumes --format progress is used.
    """
    test_status_map = {}

    pattern = r'^(.*) \.+(\.|F)'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == '.':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome == 'F':
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


def parse_log_ruby_unit(log: str, test_spec: TestSpec) -> dict[str, str]:
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


def parse_log_rspec_transformed_json(log: str, test_spec: TestSpec) -> dict[str, str]:
    test_status_map = {}

    pattern = r'(.+) - (passed|failed)'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == 'passed':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome == 'failed':
                test_status_map[test_name] = TestStatus.FAILED.value
            elif outcome == 'pending':
                test_status_map[test_name] = TestStatus.SKIPPED.value
            else:
                raise ValueError(f'Unknown outcome: {outcome}')

    return test_status_map


def parse_log_jekyll(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Different jekyll instances use different test runners and log formats.
    This function selects the appropriate log parser based on the instance id.
    """
    pr_number = test_spec.instance_id.split('-')[1]

    if pr_number in ['9141', '8047', '8167']:
        return parse_log_minitest(log, test_spec)
    if pr_number in ['8761', '8771']:
        return parse_log_cucumber(log, test_spec)
    raise ValueError(f'Unknown instance id: {test_spec.instance_id}')


MAP_REPO_TO_PARSER_RUBY = {
    'jekyll/jekyll': parse_log_jekyll,
    'fluent/fluentd': parse_log_ruby_unit,
    'fastlane/fastlane': parse_log_rspec_transformed_json,
    'jordansissel/fpm': parse_log_rspec_transformed_json,
    'faker-ruby/faker': parse_log_ruby_unit,
    'rubocop/rubocop': parse_log_rspec_transformed_json,
}


# ============================================================================
# RUST LOG PARSERS
# ============================================================================


def parse_log_cargo(log: str, test_spec: TestSpec) -> dict[str, str]:
    """
    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r'^test\s+(\S+)\s+\.\.\.\s+(\w+)$'

    for line in log.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            test_name, outcome = match.groups()
            if outcome == 'ok':
                test_status_map[test_name] = TestStatus.PASSED.value
            elif outcome == 'FAILED':
                test_status_map[test_name] = TestStatus.FAILED.value

    return test_status_map


MAP_REPO_TO_PARSER_RUST = {
    'burntsushi/ripgrep': parse_log_cargo,
    'sharkdp/bat': parse_log_cargo,
    'astral-sh/ruff': parse_log_cargo,
    'tokio-rs/tokio': parse_log_cargo,
    'uutils/coreutils': parse_log_cargo,
    'nushell/nushell': parse_log_cargo,
    'tokio-rs/axum': parse_log_cargo,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def ansi_escape(text: str) -> str:
    """Remove ANSI escape sequences from text"""
    ansi_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_pattern.sub('', text)


# ============================================================================
# COMBINED REPOSITORY TO PARSER MAPPING
# ============================================================================

MAP_REPO_TO_PARSER: dict[str, Callable[[str, TestSpec], dict[str, str]]] = {
    **MAP_REPO_TO_PARSER_PY,
    **MAP_REPO_TO_PARSER_JAVA,
    **MAP_REPO_TO_PARSER_JS,
    **MAP_REPO_TO_PARSER_C,
    **MAP_REPO_TO_PARSER_GO,
    **MAP_REPO_TO_PARSER_PHP,
    **MAP_REPO_TO_PARSER_RUBY,
    **MAP_REPO_TO_PARSER_RUST,
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_parser_for_repo(repo: str) -> Callable[[str, TestSpec], dict[str, str]]:
    """
    Get the appropriate log parser function for a given repository.

    Args:
        repo: Repository name (e.g., 'apache/druid')

    Returns:
        Parser function that takes (log_content, test_spec) and returns status_map

    Raises:
        ValueError: If no parser is configured for the repository
    """
    if repo not in MAP_REPO_TO_PARSER:
        raise ValueError(
            f"No log parser configured for repository '{repo}'. Available repositories: {len(MAP_REPO_TO_PARSER)} total"
        )

    return MAP_REPO_TO_PARSER[repo]
