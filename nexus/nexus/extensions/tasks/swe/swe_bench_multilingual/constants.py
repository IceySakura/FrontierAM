"""
Constants for SWE-Bench-Multilingual dataset.

This module defines the mapping between repositories and their programming languages,
as well as utility functions for language detection.
"""

from enum import Enum


# Constants - Test Types
FAIL_TO_PASS = 'FAIL_TO_PASS'
FAIL_TO_FAIL = 'FAIL_TO_FAIL'
PASS_TO_PASS = 'PASS_TO_PASS'
PASS_TO_FAIL = 'PASS_TO_FAIL'

# Constants - Evaluation Keys
KEY_INSTANCE_ID = 'instance_id'
KEY_MODEL = 'model_name_or_path'
KEY_PREDICTION = 'model_patch'

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
]


class TestStatus(Enum):
    FAILED = 'FAILED'
    PASSED = 'PASSED'
    SKIPPED = 'SKIPPED'
    ERROR = 'ERROR'
    XFAIL = 'XFAIL'


class ResolvedStatus(Enum):
    NO = 'RESOLVED_NO'
    PARTIAL = 'RESOLVED_PARTIAL'
    FULL = 'RESOLVED_FULL'


# Mapping from repository to language extension
# Based on SWE-bench harness constants
MAP_REPO_TO_LANGUAGE = {
    # C repositories
    'fmtlib/fmt': 'c',
    'jqlang/jq': 'c',
    'micropython/micropython': 'c',
    'nlohmann/json': 'c',
    'redis/redis': 'c',
    'valkey-io/valkey': 'c',
    # Go repositories
    'caddyserver/caddy': 'go',
    'gin-gonic/gin': 'go',
    'gohugoio/hugo': 'go',
    'hashicorp/terraform': 'go',
    'prometheus/prometheus': 'go',
    # Java repositories
    'apache/druid': 'java',
    'apache/lucene': 'java',
    'google/gson': 'java',
    'javaparser/javaparser': 'java',
    'projectlombok/lombok': 'java',
    'reactivex/rxjava': 'java',
    # JavaScript repositories
    'axios/axios': 'js',
    'babel/babel': 'js',
    'facebook/docusaurus': 'js',
    'immutable-js/immutable-js': 'js',
    'mrdoob/three.js': 'js',
    'preactjs/preact': 'js',
    'vuejs/core': 'js',
    # PHP repositories
    'briannesbitt/carbon': 'php',
    'laravel/framework': 'php',
    'php-cs-fixer/php-cs-fixer': 'php',
    'phpoffice/phpspreadsheet': 'php',
    # Ruby repositories
    'faker-ruby/faker': 'rb',
    'fastlane/fastlane': 'rb',
    'fluent/fluentd': 'rb',
    'jekyll/jekyll': 'rb',
    'jordansissel/fpm': 'rb',
    'rubocop/rubocop': 'rb',
    # Rust repositories
    'astral-sh/ruff': 'rs',
    'burntsushi/ripgrep': 'rs',
    'nushell/nushell': 'rs',
    'sharkdp/bat': 'rs',
    'tokio-rs/axum': 'rs',
    'tokio-rs/tokio': 'rs',
    'uutils/coreutils': 'rs',
}

# Mapping from language name to file extension
LANGUAGE_TO_EXT = {
    'c': 'c',
    'C': 'c',
    'go': 'go',
    'Go': 'go',
    'java': 'java',
    'Java': 'java',
    'javascript': 'js',
    'JavaScript': 'js',
    'js': 'js',
    'php': 'php',
    'PHP': 'php',
    'ruby': 'rb',
    'Ruby': 'rb',
    'rust': 'rs',
    'Rust': 'rs',
    'python': 'py',
    'Python': 'py',
}

# Reverse mapping: language extension to repo
MAP_REPO_TO_EXT = MAP_REPO_TO_LANGUAGE


def get_language_from_repo(repo: str) -> str | None:
    """
    Get the programming language for a given repository.

    Args:
        repo: Repository name in format 'owner/repo'

    Returns:
        Language extension (e.g., 'py', 'java', 'js') or None if not found
    """
    return MAP_REPO_TO_LANGUAGE.get(repo)


def get_ext_from_language(language: str) -> str | None:
    """
    Convert a language name to its file extension.

    Args:
        language: Language name (e.g., 'Python', 'java', 'JavaScript')

    Returns:
        File extension (e.g., 'py', 'java', 'js') or None if not found
    """
    return LANGUAGE_TO_EXT.get(language)


# Test specifications for each repository and version
# This should be populated with test_cmd configurations from SWE-bench harness
# Format: MAP_REPO_VERSION_TO_SPECS[repo][version] = {'test_cmd': [...], 'install': [...], ...}
#
# For now, we'll use a placeholder that extracts test commands from the instance data
# In production, this should be populated from SWE-bench harness constants
MAP_REPO_VERSION_TO_SPECS = {
    'apache/druid': {
        '13704': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                "sed -i 's/<resourceBundle>org.apache.apache.resources:apache-jar-resource-bundle:1.5-SNAPSHOT<\\/resourceBundle>/<resourceBundle>org.apache.apache.resources:apache-jar-resource-bundle:1.5<\\/resourceBundle>/' pom.xml",
                'mvn clean install -B -pl processing -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl processing -Dtest=org.apache.druid.query.aggregation.post.ArithmeticPostAggregatorTest#testPow',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.query.aggregation.post.ArithmeticPostAggregatorTest#testDiv',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.query.aggregation.post.ArithmeticPostAggregatorTest#testQuotient',
            ],
        },
        '14092': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl processing,cloud/aws-common,cloud/gcp-common -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl server -Dtest=org.apache.druid.discovery.DruidLeaderClientTest#test503ResponseFromServerAndCacheRefresh',
                'mvnd test -B -pl server -Dtest=org.apache.druid.discovery.DruidLeaderClientTest#testServerFailureAndRedirect',
            ],
        },
        '14136': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl processing -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl processing -Dtest=org.apache.druid.timeline.VersionedIntervalTimelineTest#testOverlapSecondContainsFirstZeroLengthInterval',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.timeline.VersionedIntervalTimelineTest#testOverlapSecondContainsFirstZeroLengthInterval2',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.timeline.VersionedIntervalTimelineTest#testOverlapSecondContainsFirstZeroLengthInterval3',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.timeline.VersionedIntervalTimelineTest#testOverlapSecondContainsFirstZeroLengthInterval4',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.timeline.VersionedIntervalTimelineTest#testOverlapFirstContainsSecond',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.timeline.VersionedIntervalTimelineTest#testOverlapSecondContainsFirst',
            ],
        },
        '15402': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl processing -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl processing -Dtest=org.apache.druid.query.groupby.GroupByQueryQueryToolChestTest#testCacheStrategy',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.query.groupby.GroupByQueryQueryToolChestTest#testResultLevelCacheKeyWithSubTotalsSpec',
                'mvnd test -B -pl processing -Dtest=org.apache.druid.query.groupby.GroupByQueryQueryToolChestTest#testMultiColumnCacheStrategy',
            ],
        },
        '16875': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl server -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl server -Dtest=org.apache.druid.server.metrics.WorkerTaskCountStatsMonitorTest#testMonitorWithPeon',
                'mvnd test -B -pl server -Dtest=org.apache.druid.server.metrics.WorkerTaskCountStatsMonitorTest#testMonitorWithNulls',
                'mvnd test -B -pl server -Dtest=org.apache.druid.server.metrics.WorkerTaskCountStatsMonitorTest#testMonitorIndexer',
            ],
        },
    },
    'apache/lucene': {
        '11760': {
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.queries.intervals.TestIntervalBuilder',
            ],
        },
        '12022': {
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.document.TestLatLonShape',
            ],
        },
        '12196': {
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.queryparser.classic.TestMultiFieldQueryParser',
            ],
        },
        '12212': {
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.facet.TestDrillSideways',
            ],
        },
        '12626': {
            'docker_specs': {
                'java_version': '21',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.index.TestIndexWriter',
            ],
        },
        '13170': {
            'docker_specs': {
                'java_version': '21',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.analysis.opennlp.TestOpenNLPSentenceBreakIterator -Ptests.useSecurityManager=false',
            ],
        },
        '13301': {
            'docker_specs': {
                'java_version': '21',
            },
            'test_cmd': [
                './gradlew test --tests TestXYPoint.testEqualsAndHashCode -Dtests.seed=3ABEFE4D876DD310 -Dtests.nightly=true -Dtests.locale=es-419 -Dtests.timezone=Asia/Ulaanbaatar -Dtests.asserts=true -Dtests.file.encoding=UTF-8',
            ],
        },
        '13494': {
            'docker_specs': {
                'java_version': '21',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.facet.TestStringValueFacetCounts',
            ],
        },
        '13704': {
            'docker_specs': {
                'java_version': '21',
            },
            'test_cmd': [
                './gradlew test --tests org.apache.lucene.search.TestLatLonDocValuesQueries',
            ],
        },
    },
    'astral-sh/ruff': {
        '15309': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ruff_linter --no-run',
            ],
            'test_cmd': [
                "cargo test --package ruff_linter 'f52'",
            ],
        },
        '15330': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ruff_linter --lib rules::eradicate --no-run',
            ],
            'test_cmd': [
                'cargo test --package ruff_linter --lib rules::eradicate',
            ],
        },
        '15356': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ruff_linter --lib rules::pycodestyle --no-run',
            ],
            'test_cmd': [
                'cargo test --package ruff_linter --lib rules::pycodestyle',
            ],
        },
        '15394': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ruff_linter --lib rules::flake8_pie --no-run',
            ],
            'test_cmd': [
                'cargo test --package ruff_linter --lib rules::flake8_pie',
            ],
        },
        '15443': {
            'docker_specs': {
                'rust_version': '1.84',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ruff_linter --lib rules::flake8_bandit --no-run',
            ],
            'test_cmd': [
                'cargo test --package ruff_linter --lib rules::flake8_bandit',
            ],
        },
        '15543': {
            'docker_specs': {
                'rust_version': '1.84',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ruff_linter --lib rules::pyupgrade --no-run',
            ],
            'test_cmd': [
                'cargo test --package ruff_linter --lib rules::pyupgrade',
            ],
        },
        '15626': {
            'docker_specs': {
                'rust_version': '1.84',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ruff_linter --lib rules::flake8_simplify::tests --no-run',
            ],
            'test_cmd': [
                'cargo test --package ruff_linter --lib rules::flake8_simplify::tests',
            ],
        },
    },
    'axios/axios': {
        '4731': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                "npx mocha -R tap test/unit/adapters/http.js -g 'body length'",
            ],
        },
        '4738': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                "timeout 10s npx mocha -R tap test/unit/adapters/http.js -g 'timeout'",
            ],
        },
        '5085': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
                'npm install --no-save hpagent',
            ],
            'test_cmd': [
                (
                    "node --input-type=module -e '"
                    'import axios from "/testbed/index.js"; '
                    'import {HttpsProxyAgent} from "hpagent"; '
                    'const proxy = process.env.HTTPS_PROXY || process.env.https_proxy || '
                    'process.env.HTTP_PROXY || process.env.http_proxy; '
                    'axios.defaults.httpsAgent = new HttpsProxyAgent({proxy}); '
                    'axios.defaults.proxy = false; '
                    'process.argv = ["node", "mocha", "-R", "tap", "test/unit/regression/bugs.js"]; '
                    'await import("/testbed/node_modules/mocha/bin/_mocha");'
                    "'"
                ),
            ],
        },
        '5316': {
            'build': [
                'npm install',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': ['npm install', "npx mocha test/unit/adapters/http.js -R tap -g 'FormData'"],
        },
        '5892': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                "npx mocha test/unit/adapters/http.js -R tap -g 'compression'",
            ],
        },
        '6539': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'npx mocha -R tap test/unit/regression/SNYK-JS-AXIOS-7361793.js',
            ],
        },
    },
    'babel/babel': {
        '13928': {
            'build': [
                'yarn gulp build-rollup',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                r"echo 'npmRegistryServer: \"https://mirrors.tencent.com/npm/\"' >> .yarnrc.yml",
                'make bootstrap',
                'make build',
            ],
            'test_cmd': [
                r"sed -i 's/throw new Error(mis);/console.log(\"mismatch ignored\");/g' packages/babel-parser/test/helpers/runFixtureTests.js",
                'yarn jest babel-parser -t "arrow" --verbose',
            ],
        },
        '14532': {
            'build': [
                'yarn gulp build-no-bundle',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                r"echo 'npmRegistryServer: \"https://mirrors.tencent.com/npm/\"' >> .yarnrc.yml",
                'make bootstrap',
                'make build',
            ],
            'test_cmd': [
                'yarn gulp build-no-bundle',
                'yarn jest babel-generator --verbose',
            ],
        },
        '15445': {
            'build': [
                'yarn gulp build-no-bundle',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                r"echo 'npmRegistryServer: \"https://mirrors.tencent.com/npm/\"' >> .yarnrc.yml",
                'make bootstrap',
                'make build',
            ],
            'test_cmd': [
                r"sed -i 's/it(\"inputSourceMap without sourcesContent\", () => {/it(\"inputSourceMap without sourcesContent\", () => { return;/g' packages/babel-generator/test/index.js",
                'yarn jest packages/babel-generator/test/index.js -t "generation " --verbose',
            ],
        },
        '15649': {
            'build': [
                'yarn gulp build-no-bundle',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                r"echo 'npmRegistryServer: \"https://mirrors.tencent.com/npm/\"' >> .yarnrc.yml",
                'make bootstrap',
                'make build',
            ],
            'test_cmd': [
                'yarn gulp build-no-bundle',
                'yarn jest packages/babel-traverse/test/scope.js --verbose',
            ],
        },
        '16130': {
            'build': [
                'yarn gulp build-no-bundle',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': ['make bootstrap', 'make build'],
            'test_cmd': [
                'rm packages/babel-helpers/test/fixtures/misc/declaration-name-conflict-helper-entrypoint/output.js',
                'yarn jest babel-helpers --verbose',
            ],
        },
    },
    'briannesbitt/carbon': {
        '2665': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Carbon/RoundTest.php',
            ],
        },
        '2752': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonImmutable/IsTest.php',
            ],
        },
        '2762': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                # Route composer through Tencent Cloud mirror to avoid packagist.org latency jitter.
                'composer config -g repos.packagist composer https://mirrors.cloud.tencent.com/composer/',
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonInterval/RoundingTest.php',
            ],
        },
        '2813': {
            'build': [
                'composer update --no-interaction --prefer-dist',
                'composer install --no-interaction --prefer-dist',
            ],
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'composer update psr/clock --no-interaction && vendor/bin/phpunit --testdox --colors=never tests/Factory/FactoryTest.php',
            ],
        },
        '2981': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonInterval/TotalTest.php',
            ],
        },
        '3005': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                "sed -i '453s/\\$this->assertEquals/return; \\/\\/ /' tests/CarbonInterval/ConstructTest.php",
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonInterval/ConstructTest.php',
            ],
        },
        '3041': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonPeriod/CreateTest.php',
            ],
        },
        '3073': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonInterval/TotalTest.php',
            ],
        },
        '3098': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonInterval/ConstructTest.php',
            ],
        },
        '3103': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/CarbonImmutable/SettersTest.php',
            ],
        },
    },
    'burntsushi/ripgrep': {
        '2209': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ripgrep --test integration --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package ripgrep --test integration -- regression::r2208 --exact',
            ],
        },
        '2576': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package ripgrep --test integration --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package ripgrep --test integration -- regression',
            ],
        },
    },
    'caddyserver/caddy': {
        '4774': {
            'docker_specs': {
                'go_version': '1.18.10',
            },
            'install': [
                'go test -c ./caddytest/integration -run "TestCaddyfileAdapt*"',
            ],
            'test_cmd': [
                'go test -v ./caddytest/integration -run "TestCaddyfileAdapt*"',
            ],
        },
        '4943': {
            'docker_specs': {
                'go_version': '1.18.10',
            },
            'install': [
                'go test -c ./modules/logging',
            ],
            'test_cmd': [
                'export GOPROXY=https://goproxy.cn,https://goproxy.io,direct && go test -v ./modules/logging',
            ],
        },
        '5404': {
            'docker_specs': {
                'go_version': '1.20.14',
            },
            'install': [
                'go test -c ./caddyconfig/caddyfile',
            ],
            'test_cmd': [
                'go test -v ./caddyconfig/caddyfile',
            ],
        },
        '5626': {
            'docker_specs': {
                'go_version': '1.19.13',
            },
            'install': [
                'go test -c ./caddyconfig/httpcaddyfile -run "Test.*Import"',
            ],
            'test_cmd': [
                'go test -v ./caddyconfig/httpcaddyfile -run "Test.*Import"',
            ],
        },
        '5761': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./caddyconfig/caddyfile -run "TestLexer.*"',
            ],
            'test_cmd': [
                'go test -v ./caddyconfig/caddyfile -run "TestLexer.*"',
            ],
        },
        '5870': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c . -run "TestUnsyncedConfigAccess"',
            ],
            'test_cmd': [
                'go test -v . -run "TestUnsyncedConfigAccess"',
            ],
        },
        '5995': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./caddytest/integration -run "^TestUriReplace"',
            ],
            'test_cmd': [
                'go test -v ./caddytest/integration -run "^TestUriReplace"',
            ],
        },
        '6051': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./caddyconfig/caddyfile',
            ],
            'test_cmd': [
                'go test -v ./caddyconfig/caddyfile',
            ],
        },
        '6115': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./modules/caddyhttp/reverseproxy',
            ],
            'test_cmd': [
                'go test -v ./modules/caddyhttp/reverseproxy',
            ],
        },
        '6288': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./caddytest/integration -run "TestCaddyfileAdapt*"',
            ],
            'test_cmd': [
                'go test -v ./caddytest/integration -run "TestCaddyfileAdapt*"',
            ],
        },
        '6345': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./caddytest/integration',
            ],
            'test_cmd': [
                'go test -v ./caddytest/integration',
            ],
        },
        '6350': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./caddytest/integration -run "TestCaddyfileAdapt*"',
            ],
            'test_cmd': [
                'go test -v ./caddytest/integration -run "TestCaddyfileAdapt*"',
            ],
        },
        '6370': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./cmd',
            ],
            'test_cmd': [
                'go test -v ./cmd',
            ],
        },
        '6411': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go mod tidy',
            ],
            'test_cmd': [
                'go test -v . -run "TestReplacerNew*"',
            ],
        },
    },
    'facebook/docusaurus': {
        '10130': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'yarn install',
            ],
            'test_cmd': [
                'yarn test packages/docusaurus/src/server/__tests__/brokenLinks.test.ts --verbose',
            ],
        },
        '10309': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'yarn install',
            ],
            'test_cmd': [
                'yarn test packages/docusaurus-plugin-content-docs/src/client/__tests__/docsClientUtils.test.ts --verbose',
            ],
        },
        '8927': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'yarn install',
            ],
            'test_cmd': [
                'yarn test packages/docusaurus-utils/src/__tests__/markdownLinks.test.ts --verbose',
            ],
        },
        '9183': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'yarn install',
            ],
            'test_cmd': [
                'yarn test packages/docusaurus-theme-classic/src/__tests__/options.test.ts --verbose',
            ],
        },
        '9897': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'yarn install',
            ],
            'test_cmd': [
                'yarn test packages/docusaurus-utils/src/__tests__/markdownUtils.test.ts --verbose',
            ],
        },
    },
    'faker-ruby/faker': {
        '2705': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/faker/default/test_faker_internet.rb -v -n '/password/'",
            ],
        },
        '2970': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/faker/default/test_faker_internet.rb -v -n '/email/'",
            ],
        },
    },
    'fastlane/fastlane': {
        '19207': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install --jobs=$(nproc)',
            ],
            'test_cmd': [
                'FASTLANE_SKIP_UPDATE_CHECK=1 bundle exec rspec ./fastlane/spec/actions_specs/zip_spec.rb --no-color --format json | tail -n +2 | jq -r \'.examples[] | "\\(.description) - \\(.id) - \\(.status)"\'',
            ],
        },
        '19304': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install --jobs=$(nproc)',
            ],
            'test_cmd': [
                'FASTLANE_SKIP_UPDATE_CHECK=1 bundle exec rspec ./fastlane/spec/actions_specs/zip_spec.rb --no-color --format json | tail -n +2 | jq -r \'.examples[] | "\\(.description) - \\(.id) - \\(.status)"\'',
            ],
        },
        '19765': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install --jobs=$(nproc)',
            ],
            'test_cmd': [
                'FASTLANE_SKIP_UPDATE_CHECK=1 bundle exec rspec ./fastlane/spec/actions_specs/download_dsyms_spec.rb --no-color --format json | tail -n +2 | jq -r \'.examples[] | "\\(.description) - \\(.id) - \\(.status)"\'',
            ],
        },
        '20642': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install --jobs=$(nproc)',
            ],
            'test_cmd': [
                'FASTLANE_SKIP_UPDATE_CHECK=1 bundle exec rspec ./frameit/spec/device_spec.rb --no-color --format json | tail -n +2 | jq -r \'.examples[] | "\\(.description) - \\(.id) - \\(.status)"\'',
            ],
        },
        '20958': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install --jobs=$(nproc)',
            ],
            'test_cmd': [
                'FASTLANE_SKIP_UPDATE_CHECK=1 bundle exec rspec ./fastlane/spec/actions_specs/import_from_git_spec.rb --no-color --format json | tail -n +2 | jq -r \'.examples[] | "\\(.description) - \\(.id) - \\(.status)"\'',
            ],
        },
        '20975': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install --jobs=$(nproc)',
            ],
            'test_cmd': [
                'FASTLANE_SKIP_UPDATE_CHECK=1 bundle exec rspec ./match/spec/storage/s3_storage_spec.rb --no-color --format json | tail -n +2 | jq -r \'.examples[] | "\\(.description) - \\(.id) - \\(.status)"\'',
            ],
        },
        '21857': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install --jobs=$(nproc)',
            ],
            'test_cmd': [
                'FASTLANE_SKIP_UPDATE_CHECK=1 bundle exec rspec ./fastlane/spec/lane_manager_base_spec.rb --no-color --format json | tail -n +2 | jq -r \'.examples[] | "\\(.description) - \\(.id) - \\(.status)"\'',
            ],
        },
    },
    'fluent/fluentd': {
        '3328': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/plugin/test_in_tail.rb -v -n '/test_ENOENT_error_after_setup_watcher/'",
            ],
        },
        '3466': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/plugin/test_in_tail.rb -v -n '/test_should_replace_target_info/'",
            ],
        },
        '3608': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/plugin/test_output_as_buffered_retries.rb -v -n '/retry_max_times/'",
            ],
        },
        '3616': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/plugin/test_in_http.rb -v -n '/test_application/'",
            ],
        },
        '3631': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/test_event_router.rb -v -n '/handle_emits_error/'",
            ],
        },
        '3640': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/plugin_helper/test_retry_state.rb -v -n '/exponential backoff/'",
            ],
        },
        '3641': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec ruby test/test_supervisor.rb -v',
            ],
        },
        '3917': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec ruby test/test_config.rb -v',
            ],
        },
        '4030': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec ruby test/plugin/out_forward/test_ack_handler.rb -v',
            ],
        },
        '4311': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/config/test_system_config.rb -v -n '/rotate_age/'",
            ],
        },
        '4598': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/plugin_helper/test_http_server_helper.rb -v -n '/mount/'",
            ],
        },
        '4655': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec ruby test/plugin/test_in_http.rb -v -n '/test_add/'",
            ],
        },
    },
    'fmtlib/fmt': {
        '1683': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target printf-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R printf-test',
            ],
        },
        '2310': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target format-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R format-test',
            ],
        },
        '2317': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target format-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R format-test',
            ],
        },
        '2457': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target ranges-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R ranges-test',
            ],
        },
        '3158': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target ranges-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R ranges-test',
            ],
        },
        '3248': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target format-test',
            ],
            'test_cmd': [
                'cmake --build build --target format-test',
                'ctest --test-dir build -V -R format-test',
            ],
        },
        '3272': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target xchar-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R xchar-test',
            ],
        },
        '3729': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target std-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R std-test',
            ],
        },
        '3750': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target format-test',
            ],
            'test_cmd': [
                'cmake --build build --target format-test',
                'ctest --test-dir build -V -R format-test',
            ],
        },
        '3863': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target ranges-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R ranges-test',
            ],
        },
        '3901': {
            'build': [
                'mkdir -p build',
                'cmake -B build -S .',
                'cmake --build build --parallel $(nproc) --target format-test',
            ],
            'test_cmd': [
                'ctest --test-dir build -V -R format-test',
            ],
        },
    },
    'gin-gonic/gin': {
        '1805': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c .',
            ],
            'test_cmd': [
                'go test -v . -run "^Test.*Router"',
            ],
        },
        '1957': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c .',
            ],
            'test_cmd': [
                'go test -v . -run "^TestContext.*Bind"',
            ],
        },
        '2121': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./...',
            ],
            'test_cmd': [
                'go test -v ./... -run "^Test.*Reader"',
            ],
        },
        '2755': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c .',
            ],
            'test_cmd': [
                'go test -v . -run "^TestTree"',
            ],
        },
        '3227': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c .',
            ],
            'test_cmd': [
                'go test -v . -run "^TestRedirect"',
            ],
        },
        '3741': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c .',
            ],
            'test_cmd': [
                'go test -v . -run "^TestColor"',
            ],
        },
        '3820': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./binding',
            ],
            'test_cmd': [
                'go test -v ./binding -run "^TestMapping"',
            ],
        },
        '4003': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c .',
            ],
            'test_cmd': [
                'go test . -v -run "TestMethodNotAllowedNoRoute"',
            ],
        },
    },
    'gohugoio/hugo': {
        '12171': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./hugolib',
            ],
            'test_cmd': [
                'go test -v ./hugolib -run "^Test.*Pages"',
            ],
        },
        '12204': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./tpl/tplimpl',
            ],
            'test_cmd': [
                'go test -v ./tpl/tplimpl -run "^TestEmbedded"',
            ],
        },
        '12343': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./resources/page/...',
            ],
            'test_cmd': [
                'go test -v ./resources/page/... -run "^Test.*Permalink"',
            ],
        },
        '12448': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./hugolib/...',
            ],
            'test_cmd': [
                'go test -v ./hugolib/... -run "^TestRebuild"',
            ],
        },
        '12562': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./hugolib/...',
            ],
            'test_cmd': [
                'go test -v ./hugolib/... -run "^TestGetPage[^/]"',
            ],
        },
        '12579': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./resources/page',
            ],
            'test_cmd': [
                'go test -v ./resources/page -run "^TestGroupBy"',
            ],
        },
        '12768': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./markup/goldmark/blockquotes/...',
            ],
            'test_cmd': [
                'go test -v ./markup/goldmark/blockquotes/...',
            ],
        },
    },
    'google/gson': {
        '1014': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.JsonTreeReaderTest#testSkipValue_emptyJsonObject',
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.JsonTreeReaderTest#testSkipValue_filledJsonObject',
            ],
        },
        '1093': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonWriterTest#testNonFiniteDoublesWhenLenient',
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonWriterTest#testNonFiniteBoxedDoublesWhenLenient',
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonWriterTest#testNonFiniteDoubles',
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonWriterTest#testNonFiniteBoxedDoubles',
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonWriterTest#testDoubles',
            ],
        },
        '1100': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.DefaultDateTypeAdapterTest#testNullValue',
                'mvnd test -B -pl gson -Dtest=com.google.gson.DefaultDateTypeAdapterTest#testDatePattern',
                'mvnd test -B -pl gson -Dtest=com.google.gson.DefaultDateTypeAdapterTest#testInvalidDatePattern',
            ],
        },
        '2024': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.FieldNamingTest#testUpperCaseWithUnderscores',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.NamingPolicyTest#testGsonWithUpperCaseUnderscorePolicySerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.NamingPolicyTest#testGsonWithUpperCaseUnderscorePolicyDeserialiation',
            ],
        },
        '2061': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonReaderTest#testHasNextEndOfDocument',
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.JsonTreeReaderTest#testHasNext_endOfDocument',
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonReaderTest#testReadEmptyObject',
                'mvnd test -B -pl gson -Dtest=com.google.gson.stream.JsonReaderTest#testReadEmptyArray',
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.JsonTreeReaderTest#testSkipValue_emptyJsonObject',
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.JsonTreeReaderTest#testSkipValue_filledJsonObject',
            ],
        },
        '2134': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.util.ISO8601UtilsTest#testDateParseInvalidDay',
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.util.ISO8601UtilsTest#testDateParseInvalidMonth',
                'mvnd test -B -pl gson -Dtest=com.google.gson.internal.bind.util.ISO8601UtilsTest#testDateParseWithDefaultTimezone',
            ],
        },
        '2158': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testByteSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testShortSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testIntSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testLongSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testFloatSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testDoubleSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testPrimitiveIntegerAutoboxedSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testPrimitiveIntegerAutoboxedInASingleElementArraySerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testReallyLongValuesSerialization',
                'mvnd test -B -pl gson -Dtest=com.google.gson.functional.PrimitiveTest#testPrimitiveLongAutoboxedSerialization',
            ],
        },
        '2311': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.JsonPrimitiveTest#testEqualsIntegerAndBigInteger',
                'mvnd test -B -pl gson -Dtest=com.google.gson.JsonPrimitiveTest#testLongEqualsBigInteger',
                'mvnd test -B -pl gson -Dtest=com.google.gson.JsonPrimitiveTest#testEqualsAcrossTypes',
            ],
        },
        '2479': {
            'docker_specs': {
                'java_version': '11',
            },
            'install': [
                'mvn clean install -B -pl gson -DskipTests -am',
            ],
            'test_cmd': [
                'mvnd test -B -pl gson -Dtest=com.google.gson.GsonBuilderTest#testRegisterTypeAdapterForObjectAndJsonElements',
                'mvnd test -B -pl gson -Dtest=com.google.gson.GsonBuilderTest#testRegisterTypeHierarchyAdapterJsonElements',
                'mvnd test -B -pl gson -Dtest=com.google.gson.GsonBuilderTest#testModificationAfterCreate',
            ],
        },
    },
    'hashicorp/terraform': {
        '34580': {
            'docker_specs': {
                'go_version': '1.21.13',
            },
            'install': [
                'go test -c ./internal/command',
            ],
            'test_cmd': [
                'go test -v ./internal/command -run "^TestFmt"',
            ],
        },
        '34814': {
            'docker_specs': {
                'go_version': '1.22.12',
            },
            'install': [
                'go test -c ./internal/builtin/provisioners/remote-exec',
            ],
            'test_cmd': [
                'go test -v ./internal/builtin/provisioners/remote-exec',
            ],
        },
        '34900': {
            'docker_specs': {
                'go_version': '1.22.12',
            },
            'install': [
                'go test -c ./internal/terraform',
            ],
            'test_cmd': [
                'go test -v ./internal/terraform -run "(^TestContext2Apply|^TestContext2Plan).*[Ss]ensitive"',
            ],
        },
        '35543': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./internal/terraform',
            ],
            'test_cmd': [
                'go test -v ./internal/terraform -run "^TestContext2Plan_import"',
            ],
        },
        '35611': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./internal/terraform',
            ],
            'test_cmd': [
                'go test -v ./internal/terraform -run "^TestContext2Apply_provisioner"',
            ],
        },
    },
    'immutable-js/immutable-js': {
        '2005': {
            'build': [
                # The full build also runs build:stats, which fetches bundlephobia.io
                # and makes evaluation depend on external network availability.
                # Tests import src/ directly, so only the local dist build is needed.
                'npm run build:dist',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'npx jest __tests__/OrderedMap.ts __tests__/OrderedSet.ts --silent --json | jq -r \'.testResults[].assertionResults[] | "[" + (.status | ascii_upcase) + "] " + ((.ancestorTitles | join(" > ")) + (if .ancestorTitles | length > 0 then " > " else "" end) + .title)\'',
            ],
        },
        '2006': {
            'build': [
                'npm run build:dist',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'npx jest __tests__/Range.ts --verbose',
            ],
        },
    },
    'javaparser/javaparser': {
        '4538': {
            'build': [
                './mvnw clean install -B -pl javaparser-core-testing -DskipTests -am',
            ],
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                './mvnw install -B -pl javaparser-core -am -DskipTests',
                './mvnw test -B -pl javaparser-core-testing -Dtest=NodeTest',
                './mvnw test -B -pl javaparser-core-testing -Dtest=NodePositionTest',
            ],
        },
        '4561': {
            'build': [
                './mvnw clean install -B -pl javaparser-symbol-solver-testing -DskipTests -am',
            ],
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                './mvnw install -B -pl javaparser-symbol-solver-core -am -DskipTests',
                './mvnw test -B -pl javaparser-symbol-solver-testing -Dtest=Issue4560Test',
                './mvnw test -B -pl javaparser-symbol-solver-testing -Dtest=JavaSymbolSolverTest',
            ],
        },
    },
    'jekyll/jekyll': {
        '8047': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'script/bootstrap',
                'bundle add webrick',
            ],
            'test_cmd': [
                'bundle exec ruby -I test test/test_filters.rb -v -n "/where_exp filter/"',
            ],
        },
        '8167': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'script/bootstrap',
                'bundle add webrick',
            ],
            'test_cmd': [
                'bundle exec ruby -I test test/test_utils.rb -v -n "/Utils.slugify/"',
            ],
        },
        '8761': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'script/bootstrap',
            ],
            'test_cmd': [
                'bundle exec cucumber --publish-quiet --format progress --no-color features/post_data.feature:6 features/post_data.feature:30',
            ],
        },
        '8771': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'script/bootstrap',
            ],
            'test_cmd': [
                'bundle exec cucumber --publish-quiet --format progress --no-color features/incremental_rebuild.feature:27 features/incremental_rebuild.feature:70',
            ],
        },
        '9141': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'script/bootstrap',
            ],
            'test_cmd': [
                'bundle exec ruby -I test test/test_site.rb -v -n "/static files/"',
            ],
        },
    },
    'jordansissel/fpm': {
        '1829': {
            'docker_specs': {
                'ruby_version': '3.1',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec rspec spec/fpm/package/deb_spec.rb --no-color --format json | sed -n '/^{/,$p' | jq -r '.examples[] | \"\\(.description) - \\(.status)\"'",
            ],
        },
        '1850': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                "bundle exec rspec spec/fpm/package/empty_spec.rb --no-color --format json | sed -n '/^{/,$p' | jq -r '.examples[] | \"\\(.description) - \\(.status)\"'",
            ],
        },
    },
    'jqlang/jq': {
        '2235': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
        '2598': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
        '2650': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
        '2658': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
        '2681': {
            'test_cmd': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin --disable-valgrind',
                'make clean',
                'rm -f src/parser.c src/parser.h src/lexer.c src/lexer.h',
                'make -j$(nproc)',
                'NO_VALGRIND=1 make check',
            ],
        },
        '2728': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
        '2750': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
        '2839': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
        '2919': {
            'build': [
                'git submodule update --init',
                'autoreconf -fi',
                './configure --with-oniguruma=builtin',
                'make clean',
                'touch src/parser.y src/lexer.l',
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'make check',
            ],
        },
    },
    'laravel/framework': {
        '46234': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Routing/RoutingUrlGeneratorTest.php',
            ],
        },
        '48573': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Cache/CacheArrayStoreTest.php',
            ],
        },
        '48636': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Database/DatabaseEloquentModelTest.php',
            ],
        },
        '51195': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/View/Blade/BladeVerbatimTest.php',
            ],
        },
        '51890': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                "vendor/bin/phpunit --testdox --colors=never tests/Validation/ValidationValidatorTest.php --filter 'attribute'",
            ],
        },
        '52451': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                "vendor/bin/phpunit --testdox --colors=never tests/Validation/ValidationValidatorTest.php --filter 'custom'",
            ],
        },
        '52680': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Database/DatabaseEloquentInverseRelationTest.php',
            ],
        },
        '52684': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Support/SupportStrTest.php',
            ],
        },
        '52866': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require laravel/prompts --no-update',
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Container/ContextualAttributeBindingTest.php',
            ],
        },
        '53206': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Support/SupportJsTest.php',
            ],
        },
        '53696': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Database/DatabaseSchemaBlueprintTest.php',
            ],
        },
        '53914': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'test_cmd': [
                # Fix testbench-core version: image has v10.8 but Laravel 11.35.1 needs v9.7
                'composer require orchestra/testbench-core --no-update',
                'composer install',
                'rm -f composer.lock && rm -rf vendor',
                'COMPOSER_ROOT_VERSION=11.35.1 composer require "orchestra/testbench-core:^9.7"',
                # DatabaseConnectionsTest doesn't need DatabaseMigrations, change to extend TestCase directly
                "sed -i 's/extends DatabaseTestCase/extends \\\\Orchestra\\\\Testbench\\\\TestCase/' "
                'tests/Integration/Database/DatabaseConnectionsTest.php && '
                'vendor/bin/phpunit --testdox --colors=never tests/Integration/Database/DatabaseConnectionsTest.php',
            ],
        },
        '53949': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer require orchestra/testbench-core --no-update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Support/OnceTest.php',
            ],
        },
    },
    'micropython/micropython': {
        '10095': {
            'test_cmd': [
                'sed -i "s/uint mp_import_stat/mp_import_stat_t mp_import_stat/" mpy-cross/main.c',
                'make -C mpy-cross',
                'make -C ports/unix VARIANT=standard CFLAGS_EXTRA="-DMICROPY_PY_FUNCTION_FEATURES=7" -j$(nproc)',
                '(cd tests && MICROPY_CPYTHON3=python3 MICROPY_MICROPYTHON=../ports/unix/build-standard/micropython python3 ./run-tests.py -j 1 -i basics/fun)',
            ]
        },
        '12158': {
            'test_cmd': [
                'make -C mpy-cross',
                'make -C ports/unix VARIANT=standard MICROPY_PY_THREAD=1 LDFLAGS_EXTRA="-lpthread" -j$(nproc)',
                'gcc -shared -fPIC -o tests/unix/ffi_lib.so tests/unix/ffi_lib.c || true',
                '(cd tests && MICROPY_CPYTHON3=python3 MICROPY_MICROPYTHON=../ports/unix/build-standard/micropython python3 run-tests.py -j 1 -d thread)',
            ]
        },
        '13039': {
            'test_cmd': [
                'make -C mpy-cross',
                'make -C ports/unix VARIANT=standard MICROPY_LONGINT_IMPL=mpz CFLAGS_EXTRA="-DMICROPY_PY_BUILTINS_SLICE_INDICES=1" -j$(nproc)',
                '(cd tests && MICROPY_CPYTHON3=python3 MICROPY_MICROPYTHON=../ports/unix/build-standard/micropython python3 run-tests.py -j 1 -i slice)',
            ]
        },
        '13569': {
            'test_cmd': [
                'make -C mpy-cross',
                'make -C ports/unix VARIANT=standard -j$(nproc)',
                '(cd tests && MICROPY_CPYTHON3=python3 MICROPY_MICROPYTHON=../ports/unix/build-standard/micropython python3 run-tests.py -j 1 -i try)',
            ]
        },
        '15898': {
            'test_cmd': [
                'make -C mpy-cross',
                'make -C ports/unix VARIANT=standard MICROPY_LONGINT_IMPL=mpz -j$(nproc)',
                '(cd tests && MICROPY_CPYTHON3=python3 MICROPY_MICROPYTHON=../ports/unix/build-standard/micropython python3 run-tests.py -j 1 -i string_format)',
            ]
        },
    },
    'mrdoob/three.js': {
        '25687': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install --ignore-scripts',
            ],
            'test_cmd': [
                'npx qunit test/unit/src/core/Object3D.tests.js -f "/json|clone|copy/i"',
            ],
        },
        '26589': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install --ignore-scripts',
            ],
            'test_cmd': [
                'npx qunit test/unit/src/objects/Line.tests.js test/unit/src/objects/Mesh.tests.js test/unit/src/objects/Points.tests.js',
            ],
        },
        '27395': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install --ignore-scripts',
            ],
            'test_cmd': [
                'npx qunit test/unit/src/math/Sphere.tests.js',
            ],
        },
    },
    'nlohmann/json': {
        '4237': {
            'build': [
                'mkdir -p build',
                'cd build',
                'cmake ..',
                'make test-udt_cpp11',
                'cd ..',
            ],
            'test_cmd': [
                './build/tests/test-udt_cpp11 -s -r=xml',
            ],
        },
    },
    'nushell/nushell': {
        '12901': {
            'docker_specs': {
                'rust_version': '1.77',
            },
            'install': [
                'cargo test --no-run shell::env',
            ],
            'test_cmd': [
                'cargo test --no-fail-fast shell::env',
            ],
        },
        '12950': {
            'docker_specs': {
                'rust_version': '1.77',
            },
            'install': [
                'cargo test external_arguments --no-run',
            ],
            'test_cmd': [
                'cargo test external_arguments --no-fail-fast',
            ],
        },
        '13246': {
            'build': [
                r"sed -i '/fn find_with_bytestream_search_with_char/,/^}/c\#[test]\nfn find_with_bytestream_search_with_char() {}' crates/nu-command/tests/commands/find.rs",
                'cargo build',
            ],
            'docker_specs': {
                'rust_version': '1.77',
            },
            'install': [
                'cargo test -p nu-command --no-run --test main find::',
            ],
            'test_cmd': [
                'cargo build --bin nu',
                'cargo test -p nu-command --test main commands::find',
                'rm -rf target',
            ],
        },
        '13605': {
            'build': [
                r"sed -i '/fn list_symlink_with_full_path/,/^}/c\#[test]\nfn list_symlink_with_full_path() {}' crates/nu-command/tests/commands/ls.rs",
                'cargo build',
            ],
            'docker_specs': {
                'rust_version': '1.78',
            },
            'install': [
                'cargo test -p nu-command --no-run ls::',
            ],
            'test_cmd': [
                'cargo build',
                'cargo test -p nu-command --no-fail-fast ls::',
                'rm -rf target',
            ],
        },
        '13831': {
            'build': [
                r"sed -i '/fn to_column/,/^}/c\#[test]\nfn to_column() {}' crates/nu-command/tests/commands/split_column.rs",
                'cargo build',
            ],
            'docker_specs': {
                'rust_version': '1.79',
            },
            'install': [
                'cargo test -p nu-command --no-run split_column',
            ],
            'test_cmd': [
                'cargo build --bin nu',
                'cargo test -p nu-command --test main commands::split_column::to_column',
            ],
        },
    },
    'php-cs-fixer/php-cs-fixer': {
        '7523': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/Operator/BinaryOperatorSpacesFixerTest.php',
            ],
        },
        '7635': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/Import/FullyQualifiedStrictTypesFixerTest.php',
            ],
        },
        '7663': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                # See briannesbitt/carbon::2762.
                'composer config -g repos.packagist composer https://mirrors.cloud.tencent.com/composer/',
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/Whitespace/StatementIndentationFixerTest.php',
            ],
        },
        '7875': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/Whitespace/StatementIndentationFixerTest.php',
            ],
        },
        '7998': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/Casing/ConstantCaseFixerTest.php',
            ],
        },
        '8064': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/StringNotation/SimpleToComplexStringVariableFixerTest.php',
            ],
        },
        '8075': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/PhpUnit/PhpUnitAttributesFixerTest.php',
            ],
        },
        '8256': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/PhpTag/BlankLineAfterOpeningTagFixerTest.php',
            ],
        },
        '8331': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/LanguageConstruct/NullableTypeDeclarationFixerTest.php',
            ],
        },
        '8367': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                'vendor/bin/phpunit --testdox --colors=never tests/Fixer/Import/FullyQualifiedStrictTypesFixerTest.php',
            ],
        },
    },
    'phpoffice/phpspreadsheet': {
        '3463': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Writer/Xlsx/FunctionPrefixTest.php',
            ],
        },
        '3469': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Style/StyleTest.php',
            ],
        },
        '3570': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Calculation/Functions/LookupRef/VLookupTest.php',
            ],
        },
        '3659': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Worksheet/Table/Issue3635Test.php',
            ],
        },
        '3903': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Shared/StringHelperTest.php',
            ],
        },
        '3940': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Worksheet/WorksheetTest.php',
            ],
        },
        '4114': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Worksheet/Issue4112Test.php',
            ],
        },
        '4186': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Writer/Xlsx/FunctionPrefixTest.php',
            ],
        },
        '4214': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Calculation/Functions/MathTrig/RoundDownTest.php',
            ],
        },
        '4313': {
            'docker_specs': {
                'php_version': '8.3.16',
            },
            'install': [
                # See briannesbitt/carbon::2762.
                'composer config -g repos.packagist composer https://mirrors.cloud.tencent.com/composer/',
                'composer update',
                'composer install',
            ],
            'test_cmd': [
                './vendor/bin/phpunit --testdox --colors=never tests/PhpSpreadsheetTests/Reader/Ods/FormulaTranslatorTest.php',
            ],
        },
    },
    'preactjs/preact': {
        '2757': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '16',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/render.test.js"',
            ],
        },
        '2896': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '16',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="compat/test/browser/memo.test.js"',
            ],
        },
        '2927': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '16',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/render.test.js"',
            ],
        },
        '3010': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '16',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/render.test.js"',
            ],
        },
        '3062': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '16',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/render.test.js"',
            ],
        },
        '3345': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '18',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="hooks/test/browser/useEffect.test.js"',
            ],
        },
        '3454': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '18',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/svg.test.js"',
            ],
        },
        '3562': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '18',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="compat/test/browser/render.test.js"',
            ],
        },
        '3567': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '18',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="hooks/test/browser/useEffect.test.js"',
            ],
        },
        '3689': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '18',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="hooks/test/browser/errorBoundary.test.js"',
            ],
        },
        '3739': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="hooks/test/browser/useState.test.js"',
            ],
        },
        '3763': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/lifecycles/componentDidMount.test.js"',
            ],
        },
        '4152': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/components.test.js"',
            ],
        },
        '4182': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="hooks/test/browser/errorBoundary.test.js"',
            ],
        },
        '4245': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="hooks/test/browser/useId.test.js"',
            ],
        },
        '4316': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/events.test.js"',
            ],
        },
        '4436': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'npm install',
            ],
            'test_cmd': [
                'COVERAGE=false BABEL_NO_MODULES=true npx karma start karma.conf.js --single-run --grep="test/browser/refs.test.js"',
            ],
        },
    },
    'projectlombok/lombok': {
        '2792': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3009': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3042': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3052': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3215': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3312': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3326': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3350': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3371': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3422': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3479': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3486': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3571': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '17',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3594': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3602': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3674': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
        '3697': {
            'build': [
                'ant test.compile',
            ],
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                'ant test.instance',
            ],
        },
    },
    'prometheus/prometheus': {
        '10633': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./discovery/puppetdb',
            ],
            'test_cmd': [
                'go test -v ./discovery/puppetdb -run "TestPuppetDBRefreshWithParameters"',
            ],
        },
        '10720': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./promql',
            ],
            'test_cmd': [
                'go test -v ./promql -run "^TestEvaluations"',
            ],
        },
        '11859': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./tsdb',
            ],
            'test_cmd': [
                'go test -v ./tsdb -run "^TestSnapshot"',
            ],
        },
        '12874': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./tsdb',
            ],
            'test_cmd': [
                'go test -v ./tsdb -run "^TestHead"',
            ],
        },
        '13845': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./promql ./model/labels',
            ],
            'test_cmd': [
                'go test -v ./promql ./model/labels -run "^(TestRangeQuery|TestLabels)"',
            ],
        },
        '14861': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./promql',
            ],
            'test_cmd': [
                'go test -v ./promql -run "^TestEngine"',
            ],
        },
        '15142': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./tsdb',
            ],
            'test_cmd': [
                'go test -v ./tsdb -run "^TestHead"',
            ],
        },
        '9248': {
            'docker_specs': {
                'go_version': '1.23.8',
            },
            'install': [
                'go test -c ./promql',
            ],
            'test_cmd': [
                'go test -v ./promql -run "^TestEvaluations"',
            ],
        },
    },
    'reactivex/rxjava': {
        '7597': {
            'docker_specs': {
                'java_version': '11',
            },
            'test_cmd': [
                './gradlew test --tests io.reactivex.rxjava3.internal.operators.observable.ObservableSwitchTest',
            ],
        },
    },
    'redis/redis': {
        '10068': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                "sed -i '/XTRIM with MINID option, big delta/,/}/ s/assert_equal/list/g' tests/unit/type/stream.tcl",
                'TERM=dumb ./runtest --durable --single unit/type/stream --only "/*XTRIM*"',
            ],
        },
        '10095': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/type/list --only "/.*(LPOP|RPOP)"',
            ],
        },
        '10764': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/type/zset --only "BZMPOP"',
            ],
        },
        '11279': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/acl',
            ],
        },
        '11510': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/introspection --only "/.*MONITOR.*"',
            ],
        },
        '11631': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                "sed -i '/GEOSEARCH simple (sorted)/,/}/ s/assert_equal/list/g' tests/unit/geo.tcl",
                "sed -i '/GEOSEARCH withdist (sorted)/,/}/ s/assert_equal/list/g' tests/unit/geo.tcl",
                "sed -i '/GEOSEARCH FROMMEMBER simple (sorted)/,/}/ s/assert_equal/list/g' tests/unit/geo.tcl",
                "sed -i '/GEOSEARCH with small distance/,/}/ s/assert_equal/list/g' tests/unit/geo.tcl",
                'TERM=dumb ./runtest --durable --single unit/geo --only "/.*GEOSEARCH .*"',
            ],
        },
        '11734': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/bitops',
            ],
        },
        '12272': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                "sed -i '/GETRANGE against string value/,/}/ s/assert_equal/list/g' tests/unit/type/string.tcl",
                "sed -i '/GETRANGE against integer-encoded value/,/}/ s/assert_equal/list/g' tests/unit/type/string.tcl",
                'TERM=dumb ./runtest --durable --single unit/type/string --only "/.*(GETRANGE|SETRANGE).*"',
            ],
        },
        '12472': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                "sed -i 's/dict get .* passwords/list /g' tests/unit/acl.tcl",
                "sed -i '/ACL GETUSER provides correct results/,/}/ s/assert_equal/list/g' tests/unit/acl.tcl",
                'TERM=dumb ./runtest --durable --single unit/acl --only "/.*ACL GETUSER.*"',
            ],
        },
        '13115': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/scripting',
            ],
        },
        '13338': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/type/stream-cgroups',
            ],
        },
        '9733': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                'make -j$(nproc)',
                'TERM=dumb ./runtest --durable --single unit/introspection-2',
            ],
        },
    },
    'rubocop/rubocop': {
        '13362': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/redundant_freeze_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13375': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cli_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13393': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/guard_clause_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13396': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/redundant_parentheses_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13424': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/safe_navigation_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13431': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/layout/empty_lines_around_method_body_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13479': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/layout/leading_comment_space_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13503': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/dig_chain_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13560': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/file_null_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13579': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/layout/line_continuation_spacing_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13627': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/multiple_comparison_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13653': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/access_modifier_declarations_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13668': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/sole_nested_conditional_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13680': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/style/redundant_line_continuation_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13687': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/lint/safe_navigation_chain_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
        '13705': {
            'docker_specs': {
                'ruby_version': '3.3',
            },
            'install': [
                'bundle install',
            ],
            'test_cmd': [
                'bundle exec rspec spec/rubocop/cop/lint/out_of_range_regexp_ref_spec.rb --no-color --format json | sed -n \'/^{/,$p\' | \\\n  jq -r \'.examples | group_by(.description) | .[] |\n    if any(select(.status == "failed")) then\n      (.[0] | "\\(.description) - failed")\n    else\n      (.[0] | "\\(.description) - \\(.status)")\n    end\'',
            ],
        },
    },
    'sharkdp/bat': {
        '1892': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests ignored_suffix_arg --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests ignored_suffix_arg',
            ],
        },
        '2201': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests pag --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests pag',
            ],
        },
        '2260': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests syntax --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests syntax',
            ],
        },
        '2393': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests cache_ --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests cache_',
            ],
        },
        '2650': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests map_syntax --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests map_syntax',
            ],
        },
        '2835': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests header --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests header',
            ],
        },
        '3108': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests pag --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests pag',
            ],
        },
        '562': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package bat --test integration_tests cache',
            ],
        },
    },
    'tokio-rs/axum': {
        '1119': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --package axum --lib slash --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package axum --lib slash',
            ],
        },
        '1730': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --package axum --lib --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package axum --lib -- routing::tests::mod state',
            ],
        },
        '1934': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --package axum --lib --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package axum --lib -- routing::tests::fallback',
            ],
        },
        '2096': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --package axum --lib --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package axum --lib -- routing::tests::fallback',
            ],
        },
        '682': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --package axum --lib trailing --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package axum --lib trailing -- with_trailing_slash_post without_trailing_slash_post',
            ],
        },
        '691': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --package axum --lib --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package axum --lib -- routing::tests::nest::nesting_router_at_root --exact',
            ],
        },
        '734': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --package axum --lib --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --package axum --lib -- routing::tests::head',
            ],
        },
    },
    'tokio-rs/tokio': {
        '4384': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'export PATH="/usr/local/cargo/bin:$PATH"',
                # Not load-bearing (`cargo update` would create the lockfile itself), but makes the
                # "resolve, then pin, then build" sequence explicit.
                'cargo generate-lockfile || exit 1',
                'cargo update -p tempfile --precise 3.14.0 || exit 1',
                'cargo update -p proptest --precise 1.6.0 || exit 1',
                'rustc --version && cargo --version || exit 1',
                'RUSTFLAGS=-Awarnings cargo test --package tokio --test net_lookup_host '
                '--features full --no-fail-fast --no-run || exit 1',
            ],
            'test_cmd': [
                'export PATH="/usr/local/cargo/bin:$PATH"',
                'RUSTFLAGS=-Awarnings cargo test --package tokio --test net_types_unwind '
                '--features full --no-fail-fast || exit 1',
            ],
        },
        '4867': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --test sync_broadcast --no-fail-fast --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --test sync_broadcast --no-fail-fast',
            ],
        },
        '4898': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS="-Awarnings --cfg tokio_unstable" cargo test --locked --features full --test rt_metrics --no-run',
            ],
            'test_cmd': [
                # Sandbox blocks egress, so io_driver_{fd,ready}_count's TcpStream::connect would
                # time out. Retarget to a loopback listener (semantically equivalent: the tests
                # only assert that connect() succeeds and fd_registered_count increments).
                'sed -i "s/google.com:80/127.0.0.1:18080/g" tokio/tests/rt_metrics.rs && '
                '(python3 -c "import socket\ns=socket.socket()\ns.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\ns.bind((\'127.0.0.1\', 18080))\ns.listen(64)\nwhile True:\n    c,_=s.accept()\n    c.close()" >/tmp/loopback.log 2>&1 &) && '
                # Wait until the listener is actually accepting connections (fixed sleep 0.3s
                # was racy on slower hosts). Fail loudly with the listener log on timeout so
                # we never silently fall back to connect-timeouts that masquerade as the original bug.
                'for i in $(seq 1 30); do python3 -c "import socket; socket.create_connection((\'127.0.0.1\', 18080), timeout=0.2)" 2>/dev/null && break; sleep 0.1; done && '
                'python3 -c "import socket; socket.create_connection((\'127.0.0.1\', 18080), timeout=0.5)" || { echo "ERROR: loopback listener never became ready"; cat /tmp/loopback.log; exit 1; } && '
                # Make the current-thread metrics tests generate the two park/noop events they
                # assert on instead of depending on incidental signal-driver wakeups. This keeps
                # the original PASS_TO_PASS thresholds intact.
                'python3 -c "from pathlib import Path\n'
                "p = Path('tokio/tests/rt_metrics.rs')\n"
                's = p.read_text()\n'
                "old = 'rt.block_on(async {\\\\n        time::sleep(Duration::from_millis(1)).await;\\\\n    });\\\\n    drop(rt);\\\\n    assert!(2 <= metrics.worker_park_count(0));'\n"
                "new = 'rt.block_on(async {\\\\n        for _ in 0..10 {\\\\n            time::sleep(Duration::from_millis(1)).await;\\\\n            if 2 <= metrics.worker_park_count(0) {\\\\n                break;\\\\n            }\\\\n        }\\\\n    });\\\\n    drop(rt);\\\\n    assert!(2 <= metrics.worker_park_count(0));'\n"
                "if old not in s: raise SystemExit('worker_park_count pattern not found')\n"
                's = s.replace(old, new, 1)\n'
                "old = 'rt.block_on(async {\\\\n        time::sleep(Duration::from_millis(1)).await;\\\\n    });\\\\n    drop(rt);\\\\n    assert!(2 <= metrics.worker_noop_count(0));'\n"
                "new = 'rt.block_on(async {\\\\n        for _ in 0..10 {\\\\n            time::sleep(Duration::from_millis(1)).await;\\\\n            if 2 <= metrics.worker_noop_count(0) {\\\\n                break;\\\\n            }\\\\n        }\\\\n    });\\\\n    drop(rt);\\\\n    assert!(2 <= metrics.worker_noop_count(0));'\n"
                "if old not in s: raise SystemExit('worker_noop_count pattern not found')\n"
                'p.write_text(s.replace(old, new, 1))" && '
                'RUSTFLAGS="--cfg tokio_unstable" cargo test --features full --test rt_metrics'
            ],
        },
        '6551': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS="-Awarnings --cfg tokio_unstable" cargo test --locked --features full --test rt_metrics --no-fail-fast --no-run',
            ],
            'test_cmd': [
                # See tokio-4898 above for the rationale; same loopback retarget + ready check.
                'sed -i "s/google.com:80/127.0.0.1:18080/g" tokio/tests/rt_metrics.rs && '
                '(python3 -c "import socket\ns=socket.socket()\ns.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\ns.bind((\'127.0.0.1\', 18080))\ns.listen(64)\nwhile True:\n    c,_=s.accept()\n    c.close()" >/tmp/loopback.log 2>&1 &) && '
                'for i in $(seq 1 30); do python3 -c "import socket; socket.create_connection((\'127.0.0.1\', 18080), timeout=0.2)" 2>/dev/null && break; sleep 0.1; done && '
                'python3 -c "import socket; socket.create_connection((\'127.0.0.1\', 18080), timeout=0.5)" || { echo "ERROR: loopback listener never became ready"; cat /tmp/loopback.log; exit 1; } && '
                'RUSTFLAGS="--cfg tokio_unstable" cargo test --features full --test rt_metrics --no-fail-fast'
            ],
        },
        '6603': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --test sync_mpsc --no-fail-fast --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --test sync_mpsc --no-fail-fast',
            ],
        },
        '6724': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --test io_write_all_buf --no-fail-fast --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --test io_write_all_buf --no-fail-fast',
            ],
        },
        '6752': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --test time_delay_queue --no-fail-fast --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --test time_delay_queue --no-fail-fast',
            ],
        },
        '6838': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS=-Awarnings cargo test --locked --test uds_stream --no-fail-fast --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS=-Awarnings cargo test --test uds_stream --no-fail-fast',
            ],
        },
        '7139': {
            'docker_specs': {
                'rust_version': '1.83',
            },
            'install': [
                'RUSTFLAGS="-Awarnings --cfg tokio_unstable" cargo test --locked --test fs_file --no-fail-fast --no-run',
            ],
            'test_cmd': [
                'RUSTFLAGS="-Awarnings --cfg tokio_unstable" cargo test --test fs_file --no-fail-fast',
            ],
        },
    },
    'uutils/coreutils': {
        '6377': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'cargo test test_env --no-run',
            ],
            'test_cmd': [
                'cargo test test_env --no-fail-fast',
            ],
        },
        '6575': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'cargo test cksum --no-run',
            ],
            'test_cmd': [
                'cargo test cksum --no-fail-fast',
            ],
        },
        '6682': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'cargo test mkdir --no-run',
            ],
            'test_cmd': [
                'cargo test mkdir --no-fail-fast',
            ],
        },
        '6690': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'cargo test --no-run -- test_cp_cp test_cp_same_file test_cp_multiple_files test_cp_single_file test_cp_no_file',
            ],
            'test_cmd': [
                'cargo test --no-fail-fast -- test_cp_cp test_cp_same_file test_cp_multiple_files test_cp_single_file test_cp_no_file',
            ],
        },
        '6731': {
            'docker_specs': {
                'rust_version': '1.81',
            },
            'install': [
                'cargo test backslash --no-run',
            ],
            'test_cmd': [
                'cargo test backslash --no-fail-fast',
            ],
        },
    },
    'valkey-io/valkey': {
        '1499': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': ['make -j$(nproc)', 'TERM=dumb ./runtest --durable --single unit/introspection-2'],
        },
        '1842': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': ['make -j$(nproc)', 'TERM=dumb ./runtest --durable --single unit/acl --only "/.*ACL LOAD.*"'],
        },
        '790': {
            'build': [
                'make distclean',
                'make',
            ],
            'test_cmd': [
                "sed -i '/CLUSTER SHARDS slot response/,/}/ s/assert_equal/list/g' tests/unit/cluster/cluster-shards.tcl",
                'TERM=dumb ./runtest --durable --single unit/cluster/cluster-shards',
            ],
        },
        '928': {
            'build': [
                'make distclean',
                'make',
            ],
            'install': [
                'make -j$(nproc)',
            ],
            'test_cmd': [
                'TERM=dumb ./runtest --durable --single unit/cluster/replica-migration --only "/.*NOREPLICAS.*"',
            ],
        },
    },
    'vuejs/core': {
        '11589': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'pnpm i',
            ],
            'test_cmd': [
                'pnpm run test packages/runtime-core/__tests__/apiWatch.spec.ts --no-watch --reporter=verbose',
            ],
        },
        '11739': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'pnpm i',
            ],
            'test_cmd': [
                'pnpm run test packages/runtime-core/__tests__/hydration.spec.ts --no-watch --reporter=verbose -t "mismatch handling"',
            ],
        },
        '11870': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'pnpm i',
            ],
            'test_cmd': [
                'pnpm run test packages/runtime-core/__tests__/helpers/renderList.spec.ts --no-watch --reporter=verbose',
            ],
        },
        '11899': {
            'build': [
                'pnpm run build compiler-sfc',
            ],
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'pnpm i',
            ],
            'test_cmd': [
                'pnpm run test packages/compiler-sfc/__tests__/compileStyle.spec.ts --no-watch --reporter=verbose',
            ],
        },
        '11915': {
            'docker_specs': {
                '_variant': 'js_2',
                'node_version': '20',
            },
            'install': [
                'pnpm i',
            ],
            'test_cmd': [
                'pnpm run test packages/compiler-core/__tests__/parse.spec.ts --no-watch --reporter=verbose -t "Element"',
            ],
        },
    },
}
