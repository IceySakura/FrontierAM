## JavaScript

### Available Images

| Node version | Image | Notes |
|---|---|---|
| 6 | `node-git:6-stretch` | very old projects |
| 8 | `node-git:8-stretch` | |
| 10 | `node-git:10-buster` | |
| 12 | `node-git:12-bullseye` | |
| 14 | `node-git:14-bullseye` | |
| 16 | `node-git:16-bullseye` | |
| 18 | `node-git:18-bullseye` | **default** |
| 20 | `node-git:20-bullseye` | |
| 20 (slim) | `node-git:20-slim` | lighter, for projects that don't need build tools |
| 22 | `node-git:22-bullseye` | |
| 22 (bookworm) | `node-git:22-bookworm` | newer Debian base |

**Note**: `node-git` images have Node and npm pre-installed — **NO nvm**. Do NOT `source ~/.nvm/nvm.sh` — it will silently fail.

### Phase 1 — Image Selection

**Default**: `node-git:18-bullseye`. Switch version based on `.nvmrc`, `.node-version`, or `engines.node` in `package.json`.

```bash
cat /testbed/.nvmrc 2>/dev/null || echo "no .nvmrc"
cat /testbed/.node-version 2>/dev/null || echo "no .node-version"
node --version
cat /testbed/package.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('engines',{}))"
```

If the detected version does not match the default image, call `switch_runtime_image` to the matching `node-git:X-bullseye` tag.

### Phase 2 — Installation

```bash
#!/bin/bash
set -xo pipefail
node --version || { echo "FAILED node not found — check runtime image"; exit 1; }
echo "==[1/2] npm install=="
cd /testbed && npm install --registry https://mirrors.tencent.com/npm/ || { echo "FAILED npm"; exit 1; }
echo "==[2/2] build=="
npm run build 2>/dev/null || true
echo "==INSTALL COMPLETE=="
```

**pnpm workspaces (monorepo with pnpm):**
```bash
npm install -g pnpm@8 2>/dev/null && pnpm --version
cd /testbed && pnpm install 2>/dev/null || npm install
```

**yarn.lock with GitHub-hosted packages (codeload.github.com blocked):**
```bash
# If yarn install fails on codeload.github.com — substitute with npm registry
grep -B5 "codeload.github.com" /testbed/yarn.lock | head -20
# Manual approach: remove conflicting packages from package.json temporarily and use --ignore-scripts
yarn install --ignore-scripts --network-timeout 100000 > /tmp/yarn.log 2>&1
```

### Phase 3 — eval.sh

```bash
#!/bin/bash
set -xo pipefail
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# npm install ensures freshness — test_patch may change package.json; also ensures completeness
npm install --registry https://mirrors.tencent.com/npm/ 2>/dev/null || true

npx jest path/to/test.spec.js --no-coverage
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Vitest (modern alternative to Jest):**
```bash
# Check for vitest config
cat /testbed/vitest.config.js 2>/dev/null || cat /testbed/vitest.config.ts 2>/dev/null
# Run vitest
./node_modules/.bin/vitest run path/to/test.spec.js
# Or via npm script: npx vitest run --reporter=verbose
```

**Chrome/Puppeteer in Docker (root) — add `--no-sandbox` flag:**
```bash
CHROMIUM_FLAGS="--no-sandbox --disable-setuid-sandbox" npx jest ...
# OR set env var:
export PUPPETEER_ARGS='--no-sandbox'
```

### Hints

**Monorepo — find test scripts per package:**
```bash
cat /testbed/packages/<plugin>/package.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('scripts',{}), indent=2))"
ls /testbed/packages/<plugin>/node_modules/.bin/ | grep -E "jest|vitest|mocha" | head -5
```

**Find what test framework a package uses:**
```bash
cat /testbed/package.json | python3 -c "import sys,json; d=json.load(sys.stdin); devd=d.get('devDependencies',{}); [print(k) for k in devd if any(t in k for t in ['jest','vitest','mocha','jasmine','karma'])]"
```

**Common Issues:**
- `node: command not found` → switch to correct `node-git:X-bullseye` image
- `Cannot find module` → `npm install` not run or package not in node_modules
- Karma/Mocha browser tests → add `--no-sandbox` to Chrome args
