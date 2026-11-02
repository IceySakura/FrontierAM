"""Constants for SWE Builder task."""

from __future__ import annotations


# ============== Task Phase States ==============


class TaskPhase:
    """Task phase states (numeric for comparison and ordering)."""

    INIT = 0  # Initialization
    SETUP = 1  # Environment setup in progress
    EVAL_SUBMITTED = 2  # Eval script submitted and image uploaded
    IMAGE_EXPORTED = 3  # Image exported (alias for EVAL_SUBMITTED)
    COMPLETED = 6  # Task completed
    FAILED = -1  # Task failed

    _PHASE_NAMES = {
        -1: 'failed',
        0: 'init',
        1: 'setup',
        2: 'eval_submitted',
        3: 'image_exported',
        6: 'completed',
    }

    @classmethod
    def to_string(cls, phase: int) -> str:
        """Convert numeric phase to string representation."""
        return cls._PHASE_NAMES.get(phase, 'unknown')


# ============== Directory and File Paths ==============

# 工作目录常量
# Unified to use testbed: workspace_dir defaults to testbed_dir
DEFAULT_WORKSPACE_DIR = '/testbed'
DEFAULT_TESTBED_DIR = '/testbed'

# 文件路径常量
# All agent-generated output files live in /tmp to avoid interference with git operations in /testbed
# (git reset --hard / git clean -fdx would delete files placed under /testbed)
DOCKERFILE_PATH = '/tmp/Dockerfile'
EVAL_DOCKERFILE_SCRIPT_PATH = '/tmp/eval_dockerfile.sh'
EVAL_SCRIPT_PATH = '/tmp/eval.sh'
INSTALL_SCRIPT_PATH = '/tmp/install.sh'
REVERT_SCRIPT_PATH = '/tmp/revert.sh'

# 补丁文件路径
TEST_PATCH_PATH = '/tmp/test_patch.diff'
GOLDEN_PATCH_PATH = '/tmp/golden_patch.diff'


# ============== Agent Configuration ==============

# Agent name (v2 uses the same agent as v1)
DEFAULT_AGENT_NAME = 'swe_setup_agent'
DEFAULT_MAX_ITERATIONS = 50
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_MAX_BASE_IMAGE_SWITCHES = 3


# ============== Docker Configuration ==============

# Docker 配置
DOCKER_HEREDOC_DELIMITER = 'EOF_1399519320'  # 用于 heredoc 的唯一分隔符
DOCKER_DEFAULT_PLATFORM = 'linux/x86_64'
DOCKER_DEFAULT_MEM_LIMIT = '50g'


# ============== Available Base Images ==============

# Note: All images below are git-enabled variants built by 02_build_git_images.sh
# Ubuntu-git images include comprehensive common dependencies (see below)

# Common dependencies in ubuntu-git:20.04 and ubuntu-git:22.04:
# - Core tools: git, ca-certificates, curl, wget
# - Build toolchain: build-essential, gcc, g++, make, cmake, autoconf, automake, libtool, pkg-config
# - SSL/TLS: libssl-dev
# - Compression: zlib1g-dev, libbz2-dev, liblzma-dev
# - FFI: libffi-dev
# - Parsing: libyaml-dev, libreadline-dev, libxml2-dev, libxslt1-dev
# - Database: libsqlite3-dev
# - Repository: software-properties-common, apt-transport-https
# - Utilities: patch, unzip, zip

AVAILABLE_BASE_IMAGES = {
    'python': {
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps (Python 3.8 default), best for legacy Python',
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps (Python 3.10 default), recommended for most projects',
        'python-git:2.7-slim': 'Python 2.7 pre-installed (EOL, Debian Buster base)',
        'python-git:3.5-slim': 'Python 3.5 pre-installed (EOL, Debian Stretch base)',
        'python-git:3.6-slim': 'Python 3.6 pre-installed (slim, minimal deps)',
        'python-git:3.7-slim': 'Python 3.7 pre-installed (slim, minimal deps)',
        'python-git:3.8-slim': 'Python 3.8 pre-installed (slim, minimal deps)',
        'python-git:3.9-slim': 'Python 3.9 pre-installed (slim, minimal deps)',
        'python-git:3.10-slim': 'Python 3.10 pre-installed (slim, minimal deps)',
        'python-git:3.11-slim': 'Python 3.11 pre-installed (slim, minimal deps, recommended)',
        'python-git:3.12-slim': 'Python 3.12 pre-installed (slim, minimal deps, latest stable)',
        'python-git:3.13-slim': 'Python 3.13 pre-installed (slim, minimal deps, bleeding edge)',
        'python-git:3.6-bullseye': 'Python 3.6 + Debian Bullseye + C extension build tools',
        'python-git:3.7-bullseye': 'Python 3.7 + Debian Bullseye + C extension build tools',
        'python-git:3.8-bullseye': 'Python 3.8 + Debian Bullseye + C extension build tools',
        'python-git:3.9-bullseye': 'Python 3.9 + Debian Bullseye + C extension build tools',
        'python-git:3.10-bullseye': 'Python 3.10 + Debian Bullseye + C extension build tools',
        'python-git:3.11-bullseye': 'Python 3.11 + Debian Bullseye + C extension build tools (recommended for C extensions)',
        'python-git:3.12-bullseye': 'Python 3.12 + Debian Bullseye + C extension build tools',
        'python-git:3.13-bullseye': 'Python 3.13 + Debian Bullseye + C extension build tools',
    },
    'javascript': {
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps (manual Node.js installation required)',
        'node-git:12-bullseye': 'Node.js 12 LTS pre-installed',
        'node-git:14-bullseye': 'Node.js 14 LTS pre-installed',
        'node-git:16-bullseye': 'Node.js 16 LTS pre-installed',
        'node-git:18-bullseye': 'Node.js 18 LTS pre-installed (recommended)',
        'node-git:20-bullseye': 'Node.js 20 LTS pre-installed',
        'node-git:20-slim': 'Node.js 20 LTS pre-installed (slim, minimal deps)',
        'node-git:22-bullseye': 'Node.js 22 pre-installed',
        'node-git:22-bookworm': 'Node.js 22 + Debian Bookworm (latest)',
    },
    'typescript': {
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps (manual Node.js + TypeScript installation required)',
        'node-git:16-bullseye': 'Node.js 16 LTS (install TypeScript with: npm install -g typescript)',
        'node-git:18-bullseye': 'Node.js 18 LTS (install TypeScript with: npm install -g typescript, recommended)',
        'node-git:20-bullseye': 'Node.js 20 LTS (install TypeScript with: npm install -g typescript)',
        'node-git:22-bookworm': 'Node.js 22 + Debian Bookworm (install TypeScript with: npm install -g typescript)',
    },
    'ruby': {
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps (Ruby 2.7 default, or manual installation)',
        'ruby-git:2.3-slim': 'Ruby 2.3 pre-installed (EOL, Debian Stretch base)',
        'ruby-git:2.4-slim': 'Ruby 2.4 pre-installed (EOL, Debian Stretch base)',
        'ruby-git:2.5-slim': 'Ruby 2.5 pre-installed (EOL, Debian Buster base)',
        'ruby-git:2.6-slim': 'Ruby 2.6 pre-installed (slim, minimal deps)',
        'ruby-git:2.7-slim': 'Ruby 2.7 pre-installed (slim, minimal deps, requires OpenSSL 1.1)',
        'ruby-git:3.0-slim': 'Ruby 3.0 pre-installed (slim, minimal deps)',
        'ruby-git:3.1-slim': 'Ruby 3.1 pre-installed (slim, minimal deps, recommended)',
        'ruby-git:3.2-slim': 'Ruby 3.2 pre-installed (slim, minimal deps)',
        'ruby-git:3.3-slim': 'Ruby 3.3 pre-installed (slim, minimal deps)',
        'ruby-git:3.4-slim': 'Ruby 3.4 pre-installed (slim, minimal deps, latest)',
        'ruby-git:2.7-bullseye': 'Ruby 2.7 + Debian Bullseye + C extension build tools',
        'ruby-git:3.1-bullseye': 'Ruby 3.1 + Debian Bullseye + C extension build tools (recommended for C extensions)',
        'ruby-git:3.2-bullseye': 'Ruby 3.2 + Debian Bullseye + C extension build tools',
        'ruby-git:3.3-bullseye': 'Ruby 3.3 + Debian Bullseye + C extension build tools',
    },
    'go': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps (manual Go installation with: wget https://go.dev/dl/..., recommended)',
        'golang-local:1.11': 'Go 1.11 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.12': 'Go 1.12 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.13': 'Go 1.13 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.14': 'Go 1.14 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.15': 'Go 1.15 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.16': 'Go 1.16 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.17': 'Go 1.17 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.18': 'Go 1.18 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.19': 'Go 1.19 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.20': 'Go 1.20 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.21': 'Go 1.21 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.22': 'Go 1.22 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.23': 'Go 1.23 pre-installed (based on ubuntu-git:22.04)',
    },
    'golang': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps (manual Go installation with: wget https://go.dev/dl/..., recommended)',
        'golang-local:1.11': 'Go 1.11 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.12': 'Go 1.12 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.13': 'Go 1.13 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.14': 'Go 1.14 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.15': 'Go 1.15 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.16': 'Go 1.16 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.17': 'Go 1.17 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.18': 'Go 1.18 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.19': 'Go 1.19 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.20': 'Go 1.20 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.21': 'Go 1.21 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.22': 'Go 1.22 pre-installed (based on ubuntu-git:22.04)',
        'golang-local:1.23': 'Go 1.23 pre-installed (based on ubuntu-git:22.04)',
    },
    'rust': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps (manual Rust installation with: curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh)',
        'rust-git:1.70-slim': 'Rust 1.70 pre-installed (slim, minimal deps)',
        'rust-git:1.75-slim': 'Rust 1.75 pre-installed (slim, minimal deps)',
        'rust-git:1.80-slim': 'Rust 1.80 pre-installed (slim, minimal deps)',
        'rust-git:1.85-slim': 'Rust 1.85 pre-installed (slim, minimal deps)',
        'rust-git:latest': 'Rust latest stable version pre-installed (slim, minimal deps, recommended)',
    },
    'c#': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps (manual .NET SDK installation required)',
        'dotnet-git:2.1': '.NET Core 2.1 SDK pre-installed (EOL, Debian Stretch base)',
        'dotnet-git:core-2.2': '.NET Core 2.2 SDK pre-installed (EOL, Debian Stretch base)',
        'dotnet-git:3.1': '.NET Core 3.1 SDK pre-installed (EOL, Debian Buster base)',
        'dotnet-git:5.0': '.NET 5.0 SDK pre-installed (EOL, Debian Buster base)',
        'dotnet-git:6.0': '.NET 6.0 SDK pre-installed (LTS)',
        'dotnet-git:7.0': '.NET 7.0 SDK pre-installed',
        'dotnet-git:8.0': '.NET 8.0 SDK pre-installed (LTS, recommended)',
        'dotnet-git:9.0': '.NET 9.0 SDK pre-installed (latest)',
    },
    'csharp': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps (manual .NET SDK installation required)',
        'dotnet-git:2.1': '.NET Core 2.1 SDK pre-installed (EOL, Debian Stretch base)',
        'dotnet-git:core-2.2': '.NET Core 2.2 SDK pre-installed (EOL, Debian Stretch base)',
        'dotnet-git:3.1': '.NET Core 3.1 SDK pre-installed (EOL, Debian Buster base)',
        'dotnet-git:5.0': '.NET 5.0 SDK pre-installed (EOL, Debian Buster base)',
        'dotnet-git:6.0': '.NET 6.0 SDK pre-installed (LTS)',
        'dotnet-git:7.0': '.NET 7.0 SDK pre-installed',
        'dotnet-git:8.0': '.NET 8.0 SDK pre-installed (LTS, recommended)',
        'dotnet-git:9.0': '.NET 9.0 SDK pre-installed (latest)',
    },
    'java': {
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps (install JDK with: apt-get install -y openjdk-8-jdk or openjdk-11-jdk)',
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps (install JDK with: apt-get install -y openjdk-11-jdk or openjdk-17-jdk, recommended)',
    },
    'c++': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps including g++, cmake, make (recommended)',
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps including g++, cmake, make',
        'ubuntu-cpp:22.04': 'Ubuntu 22.04 + g++/cmake + ninja-build/libgtest-dev/libgmock-dev/libboost/libcurl/libpq/python3 pre-installed',
        'ubuntu-cpp:20.04': 'Ubuntu 20.04 + same C++ packages as ubuntu-cpp:22.04',
        'debian-git:bullseye-slim': 'Debian Bullseye + git + curl + wget (need to install: apt-get install -y build-essential)',
        'debian-git:bookworm-slim': 'Debian Bookworm + git + curl + wget + gcc 12 (need to install: apt-get install -y build-essential)',
    },
    'cpp': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps including g++, cmake, make (recommended)',
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps including g++, cmake, make',
        'ubuntu-cpp:22.04': 'Ubuntu 22.04 + g++/cmake + ninja-build/libgtest-dev/libgmock-dev/libboost/libcurl/libpq/python3 pre-installed',
        'ubuntu-cpp:20.04': 'Ubuntu 20.04 + same C++ packages as ubuntu-cpp:22.04',
        'debian-git:bullseye-slim': 'Debian Bullseye + git + curl + wget (need to install: apt-get install -y build-essential)',
        'debian-git:bookworm-slim': 'Debian Bookworm + git + curl + wget + gcc 12 (need to install: apt-get install -y build-essential)',
    },
    'c': {
        'ubuntu-git:22.04': 'Ubuntu 22.04 + all common deps including gcc, cmake, make (recommended)',
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps including gcc, cmake, make',
        'debian-git:bullseye-slim': 'Debian Bullseye + git + curl + wget (need to install: apt-get install -y build-essential)',
        'debian-git:bookworm-slim': 'Debian Bookworm + git + curl + wget + gcc 12 (need to install: apt-get install -y build-essential)',
    },
    'php': {
        'ubuntu-git:20.04': 'Ubuntu 20.04 + all common deps (manual PHP installation required)',
        'php-git:5.6-cli': 'PHP 5.6 CLI pre-installed (EOL, Debian Jessie base)',
        'php-git:7.0-cli': 'PHP 7.0 CLI pre-installed (EOL, Debian Jessie base)',
        'php-git:7.1-cli': 'PHP 7.1 CLI pre-installed (EOL, Debian Stretch base)',
        'php-git:7.2-cli': 'PHP 7.2 CLI pre-installed (EOL, Debian Buster base)',
        'php-git:7.3-cli': 'PHP 7.3 CLI pre-installed',
        'php-git:7.4-cli': 'PHP 7.4 CLI pre-installed',
        'php-git:8.0-cli': 'PHP 8.0 CLI pre-installed',
        'php-git:8.1-cli': 'PHP 8.1 CLI pre-installed (recommended)',
        'php-git:8.2-cli': 'PHP 8.2 CLI pre-installed',
        'php-git:8.3-cli': 'PHP 8.3 CLI pre-installed (latest)',
    },
}


def get_available_images_for_language(language: str) -> dict[str, str]:
    """Get available base images for a specific language.

    Args:
        language: Programming language (case-insensitive)

    Returns:
        Dictionary mapping image names to descriptions
    """
    lang_key = language.lower().strip()
    return AVAILABLE_BASE_IMAGES.get(lang_key, {})


def format_available_images_list(language: str) -> str:
    """Format available images list for prompt display.

    Args:
        language: Programming language

    Returns:
        Formatted string with image list
    """
    images = get_available_images_for_language(language)
    if not images:
        return f'No language-specific images available for {language}.\nRecommend using ubuntu-git:22.04 and manual installation.'

    lines = []
    for img, desc in images.items():
        lines.append(f'  - `{img}`: {desc}')
    return '\n'.join(lines)
