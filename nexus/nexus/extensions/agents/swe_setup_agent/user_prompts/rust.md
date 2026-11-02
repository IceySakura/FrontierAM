## Rust

### Available Images

**`rust-git` images** — pick the closest version that is ≥ the required toolchain:

| rust-toolchain channel | Image | rustc version |
|------------------------|-------|---------------|
| < 1.60 | `rust-git:1.60-slim` | 1.60 (use closest; exact not available) |
| 1.60.x | `rust-git:1.60-slim` | 1.60 |
| 1.61–1.64 | `rust-git:1.65-slim` | 1.65 |
| 1.65.x | `rust-git:1.65-slim` | 1.65 |
| 1.66–1.69 | `rust-git:1.70-slim` | 1.70 |
| 1.70.x | `rust-git:1.70-slim` | 1.70 |
| 1.71–1.72 | `rust-git:1.73-slim` | 1.73 |
| 1.73.x | `rust-git:1.73-slim` | 1.73 |
| 1.74 | `rust-git:1.75-slim` | 1.75 |
| 1.75.x | `rust-git:1.75-slim` | 1.75 |
| 1.76 | `rust-git:1.77-slim` | 1.77 |
| 1.77.x | `rust-git:1.77-slim` | 1.77 |
| 1.78–1.79 | `rust-git:1.80-slim` | 1.80 |
| 1.80.x | `rust-git:1.80-slim` | 1.80 |
| 1.81–1.82 | `rust-git:1.83-slim` | 1.83 |
| 1.83.x | `rust-git:1.83-slim` | 1.83 |
| 1.84 | `rust-git:1.85-slim` | 1.85 |
| 1.85.x | `rust-git:1.85-slim` | 1.85 |
| stable / unspecified / 1.86+ | `rust-git:latest` | latest stable |
| no rust-toolchain file | `ubuntu-git:22.04` | — |

**⚠️ No exact match < 1.60.** Use `rust-git:1.60-slim` and check if the project compiles — most do unless MSRV is strictly enforced in CI.

### Phase 1 — Image Selection

`explore_repo` detects `rust-toolchain.toml`. Switch to the matching `rust-git` image.

**Step 1 — always read the toolchain file first:**
```bash
cat /testbed/rust-toolchain.toml 2>/dev/null || cat /testbed/rust-toolchain 2>/dev/null || echo "no toolchain file"
```

**Step 2 — infer Rust version from commit date when toolchain file is missing:**
```bash
cd /testbed && git log --format="%H %ad" --date=short -3
# Commits from ~2022-05 → stable was 1.60; 2023-05 → ~1.69-1.70; 2024-01 → ~1.75
```

**Rust in rust-git images:**
- `rust-git` images set `CARGO_HOME=/usr/local/cargo`. The cargo binary is at `/usr/local/cargo/bin/cargo`, already in PATH. **There is no `~/.cargo/env` file** — do not try to source it.
- `pkg-config` and `libssl-dev` are pre-installed in rust-git images — no need to install them.

**Always export both Cargo bin paths:**
```bash
export PATH="/root/.cargo/bin:/usr/local/cargo/bin:$PATH"
# rust-git images use /usr/local/cargo; ubuntu-git images use /root/.cargo
```

### CRITICAL — Incremental Build: NEVER Use `cargo clean` in eval.sh

`submit_eval` runs eval.sh **multiple times** (pre-check on base state, full-state validation, base-state validation). Each run switches git state via `git reset --hard` + `git clean -fd`, which:
- Reverts tracked source files to the target commit
- Removes untracked non-ignored files
- **Preserves** `.gitignore`d directories: `target/`, `~/.cargo/registry`

Cargo's incremental compilation detects source-file changes and recompiles **only** affected crates. This means:
- **First eval.sh run**: full compilation (may take 3–15 min for large workspaces)
- **Subsequent runs**: incremental compilation (typically **10–60 seconds**)

**If you add `cargo clean` to eval.sh, every run rebuilds from scratch — turning a 30s run into a 15-min run, easily causing timeout.**

```bash
# WRONG — forces full rebuild every run (3–15 min each × 3 runs = timeout)
cargo clean && cargo test -p crate_name -- test_name
rm -rf /testbed/target

# CORRECT — incremental compilation (first run: full build; subsequent runs: seconds)
cargo test -p crate_name -- test_name --nocapture
```

**What about stale compiled artifacts?** After `git reset --hard`, source files change but compiled artifacts from the previous build remain in `target/`. Cargo tracks source changes via fingerprints and automatically recompiles affected crates. You do **NOT** need to run `cargo clean` — Cargo handles this correctly.

### Phase 2 — Installation

#### ⚠️ CRITICAL — Workspace Repos: Always Scope to the Target Crate

**First, check if the repo is a workspace** (root `Cargo.toml` contains `[workspace]`):
- **Single-crate repo** (no `[workspace]`): `cargo build` and `cargo test` work fine — they only build the one crate.
- **Workspace repo** (has `[workspace]` with multiple members): `cargo build` builds **every** crate → OOM / timeout on large workspaces (polars, databend, omicron, prisma-engines, etc.). **Always use `-p <crate_name>`.**

```bash
# Workspace repos — ALWAYS scope with -p:
cargo build -p <crate_name>                                    # ✅
cargo test -p <crate_name> -- <test_name> --nocapture         # ✅
cargo build          # ❌ on workspace — builds all crates → OOM
cargo build --workspace  # ❌ NEVER

# Single-crate repos — both forms are fine:
cargo build          # ✅
cargo build -p <crate_name>  # ✅ equivalent
```

Read `test_patch` to find which crate and test function to target.

#### ⚠️ NEVER use `cargo install` to install build tools

`cargo install` **compiles from source** — takes 30–60 minutes per tool and will exhaust all iterations.

Instead, use apt equivalents:

| Build tool needed | Wrong | Right |
|-------------------|-------|-------|
| bindgen (C FFI) | `cargo install bindgen-cli` | `apt-get install -y libclang-dev llvm-dev` |
| protobuf compiler | `cargo install protoc` | `apt-get install -y protobuf-compiler libprotobuf-dev` |
| cmake | `cargo install cmake` | `apt-get install -y cmake` |
| cargo-nextest | `cargo install cargo-nextest` | use `cargo test` directly |

**How it works**: projects that use `bindgen` list it as a `build-dependency` in Cargo.toml — cargo resolves it automatically during `cargo build`. The only requirement is that `libclang-dev` is installed so bindgen can parse C headers. **`libclang-dev` is already pre-installed in all `rust-git` images** — you do NOT need to install anything, and you do NOT need `bindgen-cli`.

#### Install Script Template
```bash
#!/bin/bash
set -xo pipefail
echo "==[1/3] System packages (if needed beyond pre-installed)=="
# pkg-config and libssl-dev already pre-installed in rust-git
apt-get install -y build-essential 2>/dev/null || true
echo "==[2/3] Rust toolchain=="
export PATH=/usr/local/cargo/bin:$PATH
rustup show
echo "==[3/3] Compile target crate only (no run)=="
# Workspace repo: use -p <crate_name> (find crate name from test_patch path)
# Single-crate repo: use plain `cargo build`
cd /testbed && cargo build -p <crate_name> 2>&1 | tail -10 || { echo "FAILED build"; exit 1; }
echo "==INSTALL COMPLETE=="
```

### Phase 3 — eval.sh

```bash
#!/bin/bash
set -xo pipefail
export PATH=/usr/local/cargo/bin:$PATH
# Note: CARGO_HTTP_MULTIPLEXING=false is auto-injected by submit_eval — no need to set manually
cd /testbed
git config --global --add safe.directory /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# ⚠️ NO `cargo clean` here — incremental build is critical for performance.
# First run compiles everything; subsequent runs (submit_eval validation) only recompile changes.

# Feature flags: -F must come BEFORE --
# Do NOT set CARGO_NET_OFFLINE=true — it blocks downloading crates needed by test_patch
# ALWAYS use -p <crate_name> for workspace repos (without -p, cargo builds ALL crates → OOM/timeout)
cargo test -p <crate_name> -- <specific_test_name> --nocapture 2>&1
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Key rules for eval.sh performance:**
- **NEVER** use `cargo clean` or `rm -rf target` — destroys incremental build cache, causes 10x slowdown
- **ALWAYS** use `-p <crate_name>` for workspace repos — without it, all crates are compiled (OOM on polars, databend, reth, etc.)
- **ALWAYS** target a specific test: `-- test_name --nocapture`
- Do NOT compile the whole workspace — `cargo test` without `-p` compiles ALL crates

#### Anti-Patterns (will cause REJECTION)

**1. Hard-coding test counts** — NEVER do this:
```bash
# ❌ WRONG — count changes when test_patch adds tests
grep -q 'test result: ok. 65 passed' /tmp/out.txt
# ✅ CORRECT — use exit code directly
```

**2. Running the entire test suite** — target specific tests only:
```bash
# ❌ WRONG — builds ALL crates in workspace → OOM / timeout
cargo test
cargo test --workspace
# ✅ CORRECT — target specific crate and test
cargo test -p specific_crate -- test_function_name --nocapture
```

**3. Feature flags in wrong position**:
```bash
# ❌ WRONG: cargo test -- -F some_feature
# ✅ CORRECT: cargo test -F some_feature -- test_name
```

**4. Writing snapshot files (.snap) with bash** — NEVER do this:
```bash
# ❌ WRONG — writing .snap directly is treated as hacking
cat > crates/foo/snapshots/foo.snap << 'EOF'
...
EOF
# ✅ CORRECT — .snap files are part of test_patch; embed the FULL test_patch via git apply heredoc
# The test_patch already contains the .snap diffs — copy it in its entirety
```

**5. Destroying build cache** — NEVER do this:
```bash
# ❌ WRONG — forces full rebuild every eval.sh run (3–15 min × 3 runs = timeout)
cargo clean
rm -rf /testbed/target
rm -rf target/debug
```

### Hints

#### Cargo Registry Issues — Fix Immediately

If install.sh or full state fails with `failed to get ... as a dependency`, `Unable to update registry`, or `failed to select a version for`:

```bash
# Override to USTC git index (works for all Rust versions)
cat > ${CARGO_HOME:-$HOME/.cargo}/config.toml << 'CARGOEOF'
[source.crates-io]
replace-with = "ustc"
[source.ustc]
registry = "https://mirrors.ustc.edu.cn/crates.io-index"
[net]
git-fetch-with-cli = true
CARGOEOF
```

If USTC is also broken, try direct crates.io sparse (Rust >= 1.68):
```bash
cat > ${CARGO_HOME:-$HOME/.cargo}/config.toml << 'CARGOEOF'
[net]
git-fetch-with-cli = true
CARGOEOF
# No replace-with = cargo uses default sparse+https://index.crates.io/
```

For git dependencies (`Unable to update https://github.com/...`):
```bash
# Try --offline if dep is already in cache:
cargo test -p <crate> --offline -- <test> --nocapture
# Or vendor the dependency tree:
cd /testbed && cargo vendor /tmp/vendor
mkdir -p .cargo
cat > .cargo/config.toml << 'EOF'
[source.crates-io]
replace-with = "vendored-sources"
[source.vendored-sources]
directory = "/tmp/vendor"
EOF
```

For old projects (pre-2020) that use `ahash ^0.X` or other version conflicts:
```bash
# Try updating specific dependency to satisfy both constraints
cargo update -p ahash --precise 0.8.3 2>/dev/null || true
```

#### Insta Snapshot Tests

For insta/cargo-insta snapshot tests: the fail2pass mechanism is that on base_commit the snapshot doesn't match → test fails; after golden_patch the output matches → test passes. Set `INSTA_UPDATE=no` (never `always`).

#### Toolchain Pin Inside Image

If project needs exact toolchain version:
```bash
# Pin via env var (works if toolchain version is installed in the image)
export RUSTUP_TOOLCHAIN=1.70.0
# For nightly:
rustup override set nightly-2024-01-15 --path /testbed 2>/dev/null || true
# Check available toolchains:
rustup toolchain list
```

If `rustup override set` fails (toolchain not installed): switch to a closer rust-git image version instead.

**Override toolchain to a specific version (if image has it installed):**
```bash
rustup toolchain list            # see what's available
rustup toolchain install 1.70.0 --profile minimal 2>&1 | tail -5 || true
cd /testbed && cat > rust-toolchain.toml << 'EOF'
[toolchain]
channel = "1.70.0"
targets = ["wasm32-unknown-unknown"]   # add only if project needs wasm
components = ["rustc", "cargo", "rustfmt", "clippy", "rust-src"]
EOF
```

#### Common Issues

- `error: toolchain not found` → wrong image, check rust-toolchain.toml
- `linker cc not found` → `apt-get install -y build-essential`
- `pkg-config not found` → already pre-set in session, also try `apt-get install -y pkg-config`
- `libclang shared library not found` / bindgen errors → `apt-get install -y libclang-dev llvm-dev`
- `protoc: command not found` → `apt-get install -y protobuf-compiler`
- Compile fails on base_commit using new APIs from golden_patch → this IS the correct FAIL state
- `usage of sparse registries requires -Z sparse-registry` → old Rust + sparse registry; fix cargo config to use git protocol (see above)
- `failed to query replaced source registry` → rsproxy issue; override to direct crates.io (see above)

#### Additional World Knowledge

**Silence common warnings that block build:**
```bash
# elided-named-lifetimes warning became hard error in newer Rust — suppress when older code
RUSTFLAGS="-A dead_code -A unused -A elided-named-lifetimes" cargo build ...
```

**Fetch dependencies before going offline:**
```bash
cargo fetch 2>&1 | tail -5          # fetches to local cache
cargo build --offline ...           # then build offline (faster, avoids network errors)
```

**Wasm targets — add if needed:**
```bash
rustup target add wasm32-unknown-unknown --toolchain 1.70.0
```

**Binary/wasm inspection (when test involves parsing compiled artifacts):**
```bash
strings /testbed/path/to/file.wasm | grep -i "meta\|version\|contract" | head -10
xxd /testbed/path/to/file.wasm | head -20
```

**Blockchain/crypto projects (Stellar/Stacks/etc.) — check env.json and meta.rs:**
```bash
cat /testbed/soroban-env-common/env.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(list(d.keys())[:10])"
grep "pre_release_version\|INTERFACE_VERSION\|interface_version" /testbed/*/src/meta.rs 2>/dev/null | head -10
```

#### Full State Fails Repeatedly — Escape Path

If full state fails 5+ times with the same error pattern:
- **`error[E...]` compile error**: this MAY be the expected FAIL state if golden_patch adds the missing API/type. Submit as-is — compile failure on base is a valid fail2pass signal.
- **Registry/network error**: fix cargo config (see above), do not retry same config
- **Test panics/asserts on base**: review test command — target correct `-p crate` and test name

**⚠️ Do NOT fall back to `grep` on source files** — eval is always rejected. If you cannot get full state to pass after 10 attempts, diagnose the root error before trying again.
