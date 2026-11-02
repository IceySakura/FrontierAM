## PHP

### Available Images

| PHP version | Image | Notes |
|-------------|-------|-------|
| 5.6 | `php-git:5.6-cli` | EOL; `zip` NOT available (Debian jessie `zlib1g` conflict) |
| 7.0 | `php-git:7.0-cli` | EOL; `zip` NOT available |
| 7.1 | `php-git:7.1-cli` | |
| 7.2 | `php-git:7.2-cli` | |
| 7.3 | `php-git:7.3-cli` | |
| 7.4 | `php-git:7.4-cli` | |
| 8.0 | `php-git:8.0-cli` | |
| 8.1 | `php-git:8.1-cli` | **default** (most common) |
| 8.2 | `php-git:8.2-cli` | |
| 8.3 | `php-git:8.3-cli` | |
| 8.4 | `php-git:8.4-cli` | |

`php-git` images are based on the official `php:X.Y-cli` Docker image.

**Pre-built extensions** (do NOT reinstall): `mbstring`, `xml`, `curl`, `pdo`, `json`, `tokenizer`, `openssl`, `intl`, `bcmath` (all images), `zip` (7.1+ only)

**Install additional extensions with `docker-php-ext-install`** (NOT `apt-get install php-*`):
```bash
docker-php-ext-install zip                                              # zip (7.1+ only)
apt-get install -y libicu-dev && docker-php-ext-install intl            # intl (requires libicu-dev)
apt-get install -y libpq-dev && docker-php-ext-install pdo_mysql pdo_pgsql  # database drivers
docker-php-ext-install bcmath                                           # bcmath
```
`apt-get install php-zip` will NOT work — PHP extensions inside `php:X.Y-cli` must use `docker-php-ext-install`. Check what's already loaded: `php -m`

### Phase 1 — Image Selection

**Default**: `php-git:8.1-cli`. Detect PHP version from `composer.json` `require.php` constraint:

```bash
cat /testbed/composer.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('require',{}).get('php','unspecified'))"
php --version
```

If the project requires a specific PHP version (e.g. `>=7.4`, `^8.0`), call `switch_runtime_image` to the matching `php-git:X.Y-cli` tag. If the project requires `zip` and targets PHP 5.6 or 7.0, switch to `php-git:7.1-cli` or higher.

### Phase 2 — Installation

**Composer is NOT pre-installed.** Install it first, then run `composer install`:

```bash
#!/bin/bash
set -xo pipefail
echo "==[1/3] Install composer=="
# Primary: getcomposer.org; Fallback: GitHub releases (getcomposer.org may time out)
curl -sS --connect-timeout 10 https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer 2>/dev/null \
  || curl -fsSL --connect-timeout 10 https://github.com/composer/composer/releases/latest/download/composer.phar -o /usr/local/bin/composer
chmod +x /usr/local/bin/composer
composer --version || { echo "FAILED composer install"; exit 1; }

echo "==[2/3] PHP extensions (only if not already built-in)=="
# Example: install zip if needed
# docker-php-ext-install zip
# For intl: apt-get install -y libicu-dev && docker-php-ext-install intl

echo "==[3/3] Composer install=="
cd /testbed
# ⚠️ Do NOT use --no-dev — phpunit/pest/behat are in require-dev and needed for testing.
# --no-dev skips test framework installation → vendor/bin/phpunit not found → eval.sh fails.
composer install --no-interaction --prefer-dist || { echo "FAILED composer"; exit 1; }
echo "==INSTALL COMPLETE=="
```

### Phase 3 — eval.sh

```bash
#!/bin/bash
set -xo pipefail
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# Regenerate autoloader after git apply (new test classes need to be autoloaded)
composer dump-autoload -q 2>/dev/null || true

vendor/bin/phpunit --testdox tests/path/to/TestClass.php
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Behat tests** (`.feature` files):
```bash
vendor/bin/behat --suite=<suite> --tags=@tag features/path/to/Feature.feature
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**PHPUnit invocation patterns:**
```bash
vendor/bin/phpunit --testdox tests/path/to/TestClass.php      # --testdox shows test names
vendor/bin/phpunit --filter TestMethodName tests/             # run specific test
vendor/bin/pest tests/path/to/test.php                       # Pest test framework
```

**Never use `php /tmp/test.php` as the test runner** — the eval validator rejects eval.sh that doesn't use a real test framework (phpunit/behat/pest/codeception).

**⚠️ Do NOT use `--no-dev` in install.sh** — `phpunit`, `pest`, and `behat` are `require-dev` dependencies. If you install with `--no-dev`, `vendor/bin/phpunit` will not exist and eval.sh will fail.

### Hints

**Read CI workflow for authoritative test commands:**
```bash
cat /testbed/.github/workflows/build.yml | head -150
cat /testbed/.github/workflows/build.yml | grep -A 30 "composer test"
```

**Find test scripts via composer.json:**
```bash
cat /testbed/composer.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('scripts', {}), indent=2))"
# Common: "test:unit", "test:behavioral", "test:behat-cli", "phpunit", "pest"
```

**Behat / behavioral tests with Flow-based frameworks:**
```bash
cat /testbed/.github/workflows/build.yml | grep -B5 -A 50 "Setup Flow configuration" | head -100
```

**Composer install with specific commit from GitHub (when composer.lock references a GitHub commit):**
```bash
ls ~/.composer/cache/vcs/ | head -20
cd ~/.composer/cache/vcs/https---github.com-org-repo.git && git fetch origin 2>&1 | tail -5
composer update <package>/<name> --no-interaction 2>&1 | tail -20
```

**Legacy Zend Framework to Laminas migration:**
Many PHP repos (pre-2020) use `Zend\*` namespaces migrated to `Laminas\*`. If `composer install` fails with `zendframework/*` not found:
```bash
cd /testbed
sed -i 's/"zendframework\/zend-mvc"/"laminas\/laminas-mvc"/g' composer.json
sed -i 's/"zendframework\/zend-cache"/"laminas\/laminas-cache"/g' composer.json
sed -i 's/"zendframework\/zend-router"/"laminas\/laminas-router"/g' composer.json
sed -i 's/"zendframework\/zend-servicemanager"/"laminas\/laminas-servicemanager"/g' composer.json
sed -i 's/"zendframework\/zend-http"/"laminas\/laminas-http"/g' composer.json
sed -i 's/"zfcampus\//`"mezzio\//g' composer.json
rm -f composer.lock
composer config --no-plugins allow-plugins.laminas/laminas-dependency-plugin true 2>/dev/null || true
composer install --no-interaction --prefer-dist
```
Write tests using new Laminas namespaces (not Zend):
```php
use Laminas\Router\RouteMatch;          // not Zend\Mvc\Router\Http\RouteMatch
use Laminas\EventManager\EventManager;  // not Zend\EventManager\EventManager
```

**Common Issues:**
- `composer: command not found` → install composer first (see Phase 2 template above). If `getcomposer.org` times out, use the GitHub fallback: `curl -fsSL https://github.com/composer/composer/releases/latest/download/composer.phar -o /usr/local/bin/composer && chmod +x /usr/local/bin/composer`
- `Extension not found: zip` → `docker-php-ext-install zip` (NOT `apt-get install php-zip`)
- `Class not found` → `composer dump-autoload` before running tests
- PHPUnit version: use `vendor/bin/phpunit` (project's version), not system phpunit
- `--testdox` makes output verbose enough for test identification
- `vendor/bin/phpunit` exits 0 even when 0 tests run → always add `--filter TestName` to ensure the specific test runs

#### Anti-Patterns (will be REJECTED)
- `composer install --no-dev` — skips test framework installation, `vendor/bin/phpunit` not found
- `php /tmp/test.php` or custom PHP scripts as test runner — always use `vendor/bin/phpunit` or `vendor/bin/pest`
- Missing `composer dump-autoload` after `git apply` — new test classes from test_patch may not be autoloaded
- Using `grep`/`diff`/file-existence checks instead of a real test framework
