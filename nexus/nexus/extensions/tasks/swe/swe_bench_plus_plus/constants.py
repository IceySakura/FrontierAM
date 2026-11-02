"""
Constants for SWE-Bench-plus-plus dataset.
Copied from SWE-Bench-plus-plus/swebench/harness/constants/__init__.py
"""

from enum import Enum
from pathlib import Path


# Constants - Evaluation Log Directories
BASE_IMAGE_BUILD_DIR = Path('logs/build_images/base')
ENV_IMAGE_BUILD_DIR = Path('logs/build_images/env')
INSTANCE_IMAGE_BUILD_DIR = Path('logs/build_images/instances')
RUN_EVALUATION_LOG_DIR = Path('logs/run_evaluation')
RUN_VALIDATION_LOG_DIR = Path('logs/run_validation')


# Constants - Test Types, Statuses, Commands
FAIL_TO_PASS = 'FAIL_TO_PASS'
FAIL_TO_FAIL = 'FAIL_TO_FAIL'
PASS_TO_PASS = 'PASS_TO_PASS'
PASS_TO_FAIL = 'PASS_TO_FAIL'


class ResolvedStatus(Enum):
    NO = 'RESOLVED_NO'
    PARTIAL = 'RESOLVED_PARTIAL'
    FULL = 'RESOLVED_FULL'


class TestStatus(Enum):
    FAILED = 'FAILED'
    PASSED = 'PASSED'
    SKIPPED = 'SKIPPED'
    ERROR = 'ERROR'
    XFAIL = 'XFAIL'


class EvalType(Enum):
    PASS_AND_FAIL = 'pass_and_fail'
    FAIL_ONLY = 'fail_only'


# Constants - Evaluation Keys
KEY_INSTANCE_ID = 'instance_id'
KEY_MODEL = 'model_name_or_path'
KEY_PREDICTION = 'model_patch'

# Constants - Harness
DOCKER_PATCH = '/tmp/patch.diff'
DOCKER_USER = 'root'
DOCKER_WORKDIR = '/testbed'
LOG_REPORT = 'report.json'
LOG_INSTANCE = 'run_instance.log'
LOG_TEST_OUTPUT = 'test_output_after.txt'
LOG_TEST_BEFORE_OUTPUT = 'test_output_before.txt'
LOG_TEST_BASE_OUTPUT = 'test_output_base.txt'
UTF8 = 'utf-8'

# Constants - Logging
APPLY_PATCH_FAIL = '>>>>> Patch Apply Failed'
APPLY_PATCH_PASS = '>>>>> Applied Patch'
INSTALL_FAIL = '>>>>> Init Failed'
INSTALL_PASS = '>>>>> Init Succeeded'
INSTALL_TIMEOUT = '>>>>> Init Timed Out'
RESET_FAILED = '>>>>> Reset Failed'
TESTS_ERROR = '>>>>> Tests Errored'
TESTS_FAILED = '>>>>> Some Tests Failed'
TESTS_PASSED = '>>>>> All Tests Passed'
TESTS_TIMEOUT = '>>>>> Tests Timed Out'
START_TEST_OUTPUT = '>>>>> Start Test Output'
END_TEST_OUTPUT = '>>>>> End Test Output'

# Language constants
C_SBP_STR = 'C'
C_PLUS_SBP_STR = 'C++'
GO_SBP_STR = 'Go'
JAVA_SBP_STR = 'Java'
JAVASCRIPT_SBP_STR = 'JavaScript'
TYPESCRIPT_SBP_STR = 'TypeScript'
PHP_SBP_STR = 'PHP'
PYTHON_SBP_STR = 'Python'
RUBY_SBP_STR = 'Ruby'
RUST_SBP_STR = 'Rust'
CSHARP_SBP_STR = 'C#'

LANGUAGES_STR_MAP = {
    C_SBP_STR: 'c',
    C_PLUS_SBP_STR: 'c',
    GO_SBP_STR: 'go',
    JAVA_SBP_STR: 'java',
    JAVASCRIPT_SBP_STR: 'js',
    TYPESCRIPT_SBP_STR: 'ts',
    PHP_SBP_STR: 'php',
    PYTHON_SBP_STR: 'py',
    RUBY_SBP_STR: 'rb',
    RUST_SBP_STR: 'rs',
    CSHARP_SBP_STR: 'cs',
}


# Constants - Patch Types
class PatchType(Enum):
    PATCH_GOLD = 'gold'
    PATCH_PRED = 'pred'
    PATCH_PRED_TRY = 'pred_try'
    PATCH_PRED_MINIMAL = 'pred_minimal'
    PATCH_PRED_MINIMAL_TRY = 'pred_minimal_try'
    PATCH_TEST = 'test'

    def __str__(self):
        return self.value


# Constants - Miscellaneous
NON_TEST_EXTS = [
    '.json',
    '.png',
    'csv',
    '.txt',
    '.md',
    '.jpg',
    '.jpeg',
    '.pkl',
    '.yml',
    '.yaml',
    '.toml',
    '.out',
]
SWE_BENCH_URL_RAW = 'https://raw.githubusercontent.com/'
DEFAULT_DOCKER_SPECS = {
    'conda_version': 'py311_23.11.0-2',
    'node_version': '21.6.2',
    'pnpm_version': '9.5.0',
    'python_version': '3.9',
    'ubuntu_version': '22.04',
}
FAIL_ONLY_REPOS = {
    'chartjs/Chart.js',
    'processing/p5.js',
    'markedjs/marked',
}


def get_ext_from_language(language: str) -> str:
    return LANGUAGES_STR_MAP.get(language, 'unknown')


LATEST = 'latest'
USE_X86 = set()  # Empty set since we no longer have hard-coded USE_X86
