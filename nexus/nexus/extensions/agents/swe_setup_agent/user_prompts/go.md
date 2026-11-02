## Go

### Available Images

| Image | Go version |
|-------|------------|
| `golang-local:1.11` | 1.11 |
| `golang-local:1.12` | 1.12 |
| `golang-local:1.13` | 1.13 |
| `golang-local:1.14` | 1.14 |
| `golang-local:1.15` | 1.15 |
| `golang-local:1.16` | 1.16 |
| `golang-local:1.17` | 1.17 |
| `golang-local:1.18` | 1.18 |
| `golang-local:1.19` | 1.19 |
| `golang-local:1.20` | 1.20 |
| `golang-local:1.21` | 1.21 |
| `golang-local:1.22` | 1.22 |
| `golang-local:1.23` | 1.23 |
| `golang-local:1.24` | 1.24 |
| `golang-local:1.25` | 1.25 |
| `golang-local:1.26` | 1.26 |

All versions 1.11–1.26 have individual images. **Always use exact version match.**

### Phase 1 — Image Selection (CRITICAL)

**`ubuntu-git:22.04` has NO Go binary** — you **MUST** switch to a `golang-local` image. Any Go project will fail on the default image.

1. `explore_repo` detects `go.mod` version. Match it to the corresponding `golang-local:X.X` image.
2. If no `go.mod` exists (pre-modules repo), pick a version appropriate for the codebase age (e.g. 1.11–1.13 for pre-2019 repos).
3. Switch to the matching image before doing anything else.

### Phase 2 — Installation

**Standard repos (with go.mod):**
```bash
#!/bin/bash
set -xo pipefail
echo "==[1/2] Go environment=="
export GOPATH=$(go env GOPATH 2>/dev/null || echo /go)
export PATH=$GOPATH/bin:/usr/local/go/bin:$PATH
export GOPROXY=https://goproxy.cn,direct
go version || { echo "FAILED: go not found"; exit 1; }
echo "==[2/2] Download modules=="
cd /testbed && go mod download || { echo "FAILED: go mod download"; exit 1; }
echo "==INSTALL COMPLETE=="
```

**Pre-modules repos (no go.mod) — GOPATH setup:**

Old Go repos (pre-2018, no `go.mod`) use GOPATH-style imports. `go test ./...` fails because the package path doesn't match the filesystem path.

Detect: `ls /testbed/go.mod 2>/dev/null || echo "NO_GOMOD"` — if no go.mod, use GOPATH mode.

Fix: create a GOPATH symlink so import paths resolve:
```bash
#!/bin/bash
set -xo pipefail
export GOPATH=/go
export PATH=$GOPATH/bin:/usr/local/go/bin:$PATH

# Find the import path from any .go file's package/import declarations
# Common pattern: github.com/org/repo or k8s.io/component
# Check: head -5 /testbed/*.go or cat /testbed/Makefile | grep IMPORT
IMPORT_PATH="github.com/<org>/<repo>"   # fill in from repo name

mkdir -p $GOPATH/src/$(dirname $IMPORT_PATH)
ln -sfn /testbed $GOPATH/src/$IMPORT_PATH
echo "==INSTALL COMPLETE=="
```

### Phase 3 — eval.sh

**Standard repos (with go.mod):**
```bash
#!/bin/bash
set -xo pipefail
export GOPATH=$(go env GOPATH 2>/dev/null || echo /go)
export PATH=$GOPATH/bin:/usr/local/go/bin:$PATH
export GOPROXY=https://goproxy.cn,direct
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# Run only the specific package/test from test_patch
go test -v ./path/to/package/... -run TestSpecificName
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Pre-modules repos (GOPATH-style):**
```bash
#!/bin/bash
set -xo pipefail
export GOPATH=/go
export PATH=$GOPATH/bin:/usr/local/go/bin:$PATH

# Use GOPROXY=off if vendor/ exists (offline, no downloads needed)
# Use GOPROXY=https://goproxy.cn,direct if fetching is needed
export GOPROXY=off   # or goproxy.cn,direct

cd $GOPATH/src/<import-path>   # NOT /testbed directly — must match import path

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

go test -v -mod=vendor ./<pkg>/... -run TestSpecificName   # -mod=vendor if vendor/ exists
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

### Hints

**Vendor directory detection:**
```bash
ls /testbed/vendor/ 2>/dev/null | head -5 && echo "vendor exists"   # if vendor/, use -mod=vendor
cat /testbed/go.mod | head -5                                         # go version and module path
```
- If `vendor/` exists, add `-mod=vendor` to `go test` commands.
- Apply golden_patch excluding vendor directory (vendor files often cause "already patched" conflicts):
```bash
git apply --ignore-whitespace \
  --exclude='vendor/k8s.io/client-go/pkg/version/base.go' \
  --exclude='vendor/k8s.io/component-base/version/base.go' \
  /tmp/golden_patch.patch 2>&1
# OR: apply with 3-way merge to skip already-applied vendor chunks
git apply --ignore-whitespace --3way /tmp/golden_patch.patch 2>&1
```

**Build tags:**
- If tests require build tags, add `-tags tag1,tag2` to both `go build` and `go test` commands.

**Pre-modules GOPATH pattern:**
- `import path does not match package path` → use the GOPATH symlink pattern from Phase 2.
- `cd $GOPATH/src/<import-path>` in eval.sh, NOT `/testbed` directly.

**CI entrypoint check — find exact test invocation:**
```bash
cat /testbed/.github/actions/go-test/entrypoint.sh 2>/dev/null | head -50
cat /testbed/.github/workflows/*.yml | grep -A5 "go test" | head -30
```

**Vendor dependency versions (useful for k8s ecosystem projects):**
```bash
cat /testbed/vendor/k8s.io/client-go/pkg/version/base.go | head -20
grep "gitVersion\|gitCommit" /testbed/vendor/k8s.io/*/version/base.go | head -10
```

**Common issues:**
- `go: command not found` → wrong image, switch to `golang-local:X.X`
- `cannot find module` → run `go mod tidy` first; if network unavailable, check `ls vendor/` — if vendor/ exists use `-mod=vendor`
- Module download fails (no network) → `go mod vendor` to create vendor dir, then use `go test -mod=vendor`; if go.sum is missing entries use `GONOSUMCHECK=* go mod download`
- Build tags required → add `-tags tag1,tag2`
- `GOFLAGS=-mod=mod` already set in session; use `-mod=vendor` only if `vendor/` exists
- Pre-modules repo: `import path does not match package path` → use GOPATH symlink pattern above
