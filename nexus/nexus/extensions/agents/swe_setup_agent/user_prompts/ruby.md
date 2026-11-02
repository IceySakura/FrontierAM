## Ruby

### Available Images

| .ruby-version | Image | Notes |
|---------------|-------|-------|
| 2.3.x | `ruby-git:2.3-slim` | EOL |
| 2.4.x | `ruby-git:2.4-slim` | EOL |
| 2.5.x | `ruby-git:2.5-slim` | EOL |
| 2.6.x | `ruby-git:2.6-slim` | EOL |
| 2.7.x | `ruby-git:2.7-slim` | |
| 3.0.x | `ruby-git:3.0-slim` | |
| 3.1.x | `ruby-git:3.1-slim` | |
| 3.2.x | `ruby-git:3.2-slim` | |
| 3.3.x | `ruby-git:3.3-slim` | |
| 3.4.x | `ruby-git:3.4-slim` | |
| 3.5.x | `ruby-git:3.4-slim` | no 3.5 image yet; use 3.4 |
| unspecified | `ubuntu-git:20.04` | Ruby via apt |

Bullseye variants (`ruby-git:2.7-bullseye`, `3.1-bullseye`, `3.2-bullseye`, `3.3-bullseye`) are available for native gem builds that need the full Debian toolchain.

**Note**: `ruby-git` images are built from the official `ruby:X.Y-slim` base. Ruby is compiled from source at `/usr/local/bin/ruby` — **NO rbenv**. Do NOT use rbenv commands — rbenv does not exist in these images.

- `ruby`, `gem`, `bundle` are directly in PATH — no sourcing or init needed
- `apt-get install ruby-dev` does NOT work in ruby-git (not available in apt) — omit it
- For native gem dependencies: `apt-get install -y build-essential libssl-dev libreadline-dev libpq-dev libsqlite3-dev libxml2-dev libxslt1-dev` etc.
- `bundler` is pre-installed — but you may need to install a specific version to match `Gemfile.lock`

### Phase 1 — Image Selection

Detect `.ruby-version` and `Gemfile.lock` to determine the correct image:

```bash
cat /testbed/.ruby-version 2>/dev/null || echo "no .ruby-version"
cat /testbed/Gemfile.lock | head -5
cat /testbed/Gemfile.lock | grep -A5 "RUBY VERSION"
```

Match the detected Ruby version to the table above and call `switch_runtime_image` if the default does not match.

### Phase 2 — Installation

```bash
#!/bin/bash
set -xo pipefail
echo "==[1/2] System packages (native gem deps)=="
apt-get install -y build-essential libssl-dev libreadline-dev 2>/dev/null || true
# Add more if needed: libpq-dev libsqlite3-dev libxml2-dev libxslt1-dev
echo "==[2/2] Bundle install=="
cd /testbed
export BUNDLE_PATH=/root/bundle   # keep gems outside /testbed (git clean -fd safe)
gem install bundler -v "$(grep -A1 'BUNDLED WITH' Gemfile.lock 2>/dev/null | tail -1 | tr -d ' ')" 2>/dev/null \
  || gem install bundler
bundle install || { echo "FAILED bundle install"; exit 1; }
echo "==INSTALL COMPLETE=="
```

**Match bundler version to Gemfile.lock:**
```bash
gem install bundler -v 2.6.8 2>/dev/null || true    # exact version from Gemfile.lock "BUNDLED WITH"
bundle install > /tmp/bundle.log 2>&1 && echo "SUCCESS" || { cat /tmp/bundle.log | grep -i error | tail -10; }
```

**Native extension failures — install system libraries first:**
```bash
apt-get install -y libsqlite3-dev pkg-config libpq-dev libxml2-dev libxslt1-dev \
  libcurl4-openssl-dev libyaml-dev zlib1g-dev 2>&1 | tail -5
```

### Phase 3 — eval.sh

**Standard (RSpec):**
```bash
#!/bin/bash
set -xo pipefail
export BUNDLE_PATH=/root/bundle
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

bundle exec rspec spec/path/to_spec.rb --format documentation
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Standard (Minitest):**
```bash
#!/bin/bash
set -xo pipefail
export BUNDLE_PATH=/root/bundle
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

bundle exec ruby -Ilib:test test/path/test_file.rb --verbose
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Rails projects — full eval.sh template:**
```bash
#!/bin/bash
set -xo pipefail
export BUNDLE_PATH=/root/bundle
export RAILS_ENV=test

# Start database
service postgresql start 2>/dev/null || true
sleep 2

cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# Load schema / run pending migrations
bundle exec rails db:schema:load RAILS_ENV=test 2>/dev/null \
  || bundle exec rails db:migrate RAILS_ENV=test 2>/dev/null || true

# Run targeted spec
bundle exec rspec spec/path/to_spec.rb --format documentation
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

### Hints

**Detect test framework:**
```bash
cat /testbed/Gemfile | grep "rspec\|minitest\|cucumber" | head -5
ls /testbed/spec/ 2>/dev/null | head -20 || ls /testbed/test/ | head -20
```

**Rails — detect database engine:**
```bash
grep -E "adapter|gem 'pg'|gem 'mysql2'|gem 'sqlite3'" /testbed/config/database.yml /testbed/Gemfile 2>/dev/null | head -10
```

**Rails — PostgreSQL setup in eval.sh:**
```bash
service postgresql start || pg_ctlcluster $(pg_lsclusters -h | awk '{print $1}') main start
sleep 2
su postgres -c "psql -c \"CREATE USER rails_test WITH SUPERUSER PASSWORD 'rails_test';\" 2>/dev/null || true"
su postgres -c "psql -c \"CREATE DATABASE $(grep -A2 'test:' /testbed/config/database.yml | grep 'database:' | awk '{print $2}') OWNER rails_test;\" 2>/dev/null || true"
export DATABASE_URL=postgresql://rails_test:rails_test@localhost/$(grep -A2 'test:' /testbed/config/database.yml | grep 'database:' | awk '{print $2}')
```

**Rails — SQLite fallback (simpler, no service needed):**
```bash
export DATABASE_URL="sqlite3::memory:"
export RAILS_ENV=test
bundle exec rails db:schema:load RAILS_ENV=test 2>/dev/null || true
```

**Common Issues:**
- `rbenv: command not found` → do NOT use rbenv; ruby-git images have ruby at `/usr/local/bin/ruby` directly, no rbenv
- `Bundler::GemNotFound` in eval.sh → **`export BUNDLE_PATH=/root/bundle` MUST appear in eval.sh** (not just install.sh); without it, bundle exec can't find the installed gems
- `Bundler::GemNotFound` from native gem → add system library via `apt-get install -y libpq-dev libxml2-dev libxslt1-dev`, etc.
- `Could not find gem 'X'` → BUNDLE_PATH inconsistency between install.sh and eval.sh; ensure both use `/root/bundle`
- DB-dependent tests → start service + run `db:migrate` in eval.sh (see Rails template above)
- `bundle exec rspec` exits 0 with 0 examples → wrong spec path; verify path from test_patch and add `--require spec_helper`
- `Errno::EACCES` / permission errors on DB socket → PostgreSQL running as wrong user; use `su postgres -c "..."` for setup commands
- `PendingMigrationError` → missing `db:migrate` in eval.sh — add it immediately
- `ActiveRecord::NoDatabaseError` → DB not started or not created — add service start + db:create
- `could not connect to server` → PostgreSQL not running — add `service postgresql start; sleep 2`
- `Gem::LoadError` / `cannot load such file` → BUNDLE_PATH missing in eval.sh
