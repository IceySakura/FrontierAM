## TypeScript

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
npx tsc --noEmit 2>/dev/null || npm run build 2>/dev/null || true
echo "==INSTALL COMPLETE=="
```

**yarn.lock with GitHub-hosted packages (codeload.github.com blocked):**
```bash
grep -B5 "codeload.github.com" /testbed/yarn.lock | head -20
# Update specific package from npm:
npm view buffer@5.6.0 dist.tarball 2>/dev/null    # get npm tarball URL
# Remove the github-only package from package.json and use --ignore-scripts
python3 -c "
import json
with open('/testbed/package.json') as f: d = json.load(f)
del d['dependencies']['problematic-github-pkg']
with open('/testbed/package.json', 'w') as f: json.dump(d, f, indent=2)
"
yarn install --ignore-scripts --network-timeout 100000
```

### Phase 3 — eval.sh

```bash
#!/bin/bash
set -xo pipefail
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# npm install ensures freshness — test_patch may change package.json or add @types/* deps
npm install --registry https://mirrors.tencent.com/npm/ 2>&1 | tail -5 || echo "[WARN] npm install failed"

npx jest path/to/test.spec.ts --no-coverage
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**How TypeScript tests run** — there are two patterns, check `package.json` `jest` config:
- **ts-jest** (most common): jest runs `.ts` files directly via `ts-jest` transform — no manual `tsc` needed in eval.sh
- **Pre-compiled**: project compiles to `dist/` or `lib/` first, then jest runs on `.js` output — add a `npm run build` step before jest in eval.sh

**Chrome/Puppeteer (same as JavaScript):**
Add `--no-sandbox` to Chrome/Chromium invocations when running as root in Docker.

### Hints

**Identify test runner and TS patterns:**
```bash
cat /testbed/packages/<module>/package.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('scripts',{}), indent=2))"
```

**Stencil projects (web component libraries):**
```bash
cat /testbed/packages/<module>/stencil.config.ts | head -30   # stencil uses its own test runner
cat /testbed/packages/<module>/node_modules/@stencil/core/package.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('version',''))"
# Run stencil test with path pattern:
cd /testbed/packages/<module> && ./node_modules/.bin/stencil test --spec --testPathPattern="ic-data-table/test/basic"
```

**TypeScript path aliases and jest module mapper:**
```bash
cat /testbed/tsconfig.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('compilerOptions',{}).get('paths',{}), indent=2))"
cat /testbed/jest.config.ts 2>/dev/null | grep -A 20 "moduleNameMapper"
# If aliases not configured in jest — add to jest.config:
# moduleNameMapper: { '^@/(.*)$': '<rootDir>/src/$1' }
```

**Large monorepos (Nx, Lerna) — install only the needed package:**
```bash
grep -r '"name"' /testbed/packages/*/package.json | grep "relevant-package" | head -5
cd /testbed/packages/<module> && npm install && npx jest path/to/test.spec.ts --no-coverage
```

**Common Issues:**
- `node: command not found` → switch to correct `node-git:X-bullseye` image
- `Cannot find module '@types/...'` → already in devDependencies, just run `npm install`
- `error TS...` type errors during test → project uses ts-jest; these errors come from the test file in test_patch — they are the **expected FAIL state** on base_commit. Do NOT fix type errors; submit as-is.
- `Jest: Cannot find module` → path alias not configured; check `tsconfig.json` `paths` and add `moduleNameMapper` to jest config, or use the project's existing jest config via `npx jest --config jest.config.ts`
- Pre-compiled project: `Cannot find module './dist/...'` → add `npm run build` before `npx jest` in eval.sh
- `ts-jest` not installed → check `package.json` devDependencies; if missing, use `npx ts-jest` or compile first with `npx tsc`
