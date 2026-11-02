## C# / .NET

### Available Images

| SDK in global.json | Image | Notes |
|--------------------|-------|-------|
| 2.1.x | `dotnet-git:2.1` | |
| 2.2.x | `dotnet-git:core-2.2` | |
| 3.1.x | `dotnet-git:3.1` | |
| 5.0.x | `dotnet-git:5.0` | |
| 6.0.x | `dotnet-git:6.0` | |
| 7.0.x | `dotnet-git:7.0` | |
| 8.0.x | `dotnet-git:8.0` | |
| 9.0.x | `dotnet-git:9.0` | |
| 10.0.x | `dotnet-git:10.0` | |
| .NET Framework / Mono | `ubuntu-git:22.04` + `apt-get install mono-complete` | |

**Always prefer `dotnet-git` over `ubuntu-git`** — `ubuntu-git:22.04` succeeds only 53% of the time for C# projects.

**NuGet mirror**: The init command writes `~/.nuget/NuGet/NuGet.Config` pointing to `api.nuget.org` (accessible from k8s). Do **NOT** overwrite `NuGet.Config` — it will break package downloads.

### Phase 1 — Image Selection (CRITICAL)

**`ubuntu-git` images do NOT have `dotnet` installed** — you **MUST** switch to a `dotnet-git` image. Any C# project will fail on the default image.

**Default**: `dotnet-git:8.0`. Detect SDK version from `global.json`:

```bash
cat /testbed/global.json 2>/dev/null || echo "No global.json"
# Also check TargetFramework in .csproj files:
grep -r 'TargetFramework' /testbed/*.csproj /testbed/**/*.csproj 2>/dev/null | head -10
dotnet --version 2>/dev/null || echo "dotnet not found"
```

Match the detected SDK version to the table above and call `switch_runtime_image` to the matching `dotnet-git:X.0` tag. If `global.json` specifies an exact SDK version like `6.0.xxx`, use `dotnet-git:6.0`. For .NET Framework / Mono projects, use `ubuntu-git:22.04` with `apt-get install mono-complete`.

### CRITICAL — Incremental Build: NEVER Use `dotnet clean` in eval.sh

`submit_eval` runs eval.sh **multiple times** (pre-check on base state, full-state validation, base-state validation). Each run switches git state via `git reset --hard` + `git clean -fd`, which:
- Reverts tracked source files to the target commit
- Removes untracked non-ignored files
- **Preserves** `.gitignore`d directories: `bin/`, `obj/`, `~/.nuget/packages`

`dotnet build` is **incremental** — it detects source-file changes and recompiles only affected assemblies. This means:
- **First eval.sh run**: full compilation (may take 1–5 min for large solutions)
- **Subsequent runs**: incremental compilation (typically **5–30 seconds**)

**If you add `dotnet clean` to eval.sh, every run rebuilds from scratch — turning a 15s run into a 5-min run, easily causing timeout.**

```bash
# WRONG — forces full rebuild every run
dotnet clean && dotnet test path/to/Test.csproj
rm -rf /testbed/**/bin /testbed/**/obj

# CORRECT — incremental compilation (first run: full build; subsequent runs: seconds)
dotnet build path/to/Test.csproj -c Release
dotnet test path/to/Test.csproj --no-build -c Release --filter "FullyQualifiedName~TestClass"
```

### Phase 2 — Installation

**Partial build**: check `test_patch` to find which `.csproj` contains the test. Build only that project — do NOT build the entire solution.

```bash
#!/bin/bash
set -xo pipefail
echo "==[1/2] NuGet restore=="
cd /testbed && dotnet restore || { echo "FAILED restore"; exit 1; }
echo "==[2/2] Build (target test project only)=="
# Find the test project from test_patch, e.g. Src/Foo.Tests/Foo.Tests.csproj
# dotnet build path/to/Foo.Tests.csproj --no-restore -c Release
# If project structure is unclear, build the whole solution:
dotnet build --no-restore -c Release || { echo "FAILED build"; exit 1; }
echo "==INSTALL COMPLETE=="
```

Once `==INSTALL COMPLETE==` is shown, **immediately write eval.sh** — do NOT explore test structure further.

### Phase 3 — eval.sh

```bash
#!/bin/bash
set -xo pipefail
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# ⚠️ MUST build after git apply — test_patch adds new test code that needs compilation.
# Do NOT use --no-build here — it skips compilation and the new test code won't be compiled.
# dotnet build is incremental: first run compiles everything; subsequent runs only recompile changes.
dotnet build path/to/Test.csproj -c Release
dotnet test path/to/Test.csproj --no-build -c Release \
  --verbosity normal --filter "FullyQualifiedName~TestClass"
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Key rules for eval.sh performance:**
- **ALWAYS** `dotnet build` before `dotnet test --no-build` — test_patch adds new code that must be compiled
- **NEVER** use `dotnet clean` — destroys incremental build cache, causes 5x slowdown
- **NEVER** delete `bin/` or `obj/` directories — same effect as `dotnet clean`
- **ALWAYS** target a specific `.csproj` — `dotnet test` without a project path builds/tests the entire solution

**dotnet test filter patterns:**
```bash
# Filter by full class name:
dotnet test /testbed/tests/Project.Tests/ --filter "FullyQualifiedName~TestClassName"
# Filter by method:
dotnet test /testbed/tests/Project.Tests/ --filter "FullyQualifiedName~TestClass.MethodName"
```

**⚠️ `--no-build` is ONLY safe after `dotnet build`** — never use `--no-build` alone in eval.sh:
```bash
# ❌ WRONG — skips compilation, test_patch's new code is NOT compiled
dotnet test path/to/Test.csproj --no-build -c Release --filter "..."

# ✅ CORRECT — build first (incremental), then test without re-building
dotnet build path/to/Test.csproj -c Release
dotnet test path/to/Test.csproj --no-build -c Release --filter "..."

# ✅ ALSO CORRECT — dotnet test without --no-build (builds implicitly, but slower)
dotnet test path/to/Test.csproj -c Release --filter "..."
```

**Never use `python3` in C# eval.sh** — dotnet-git images do NOT include python.

### Hints

**Find specific test class/method (CSharpier/Roslyn-style):**
```bash
ls /testbed/Src/CSharpier.Tests/FormattingTests/TestFiles/
grep -r "TestMethodName\|ClassName" /testbed/Src/*/tests/ --include="*.cs" -l | head -10
cat /testbed/Src/<Project>.Tests/obj/Release/net*/generated/**/*.cs 2>/dev/null | head -50
```

**NuGet restore issues — check what's cached:**
```bash
ls ~/.nuget/packages/ 2>/dev/null | head -20
dotnet nuget locals all --list 2>/dev/null
cat /testbed/NuGet.Config
```

**Check for test data files referenced in tests:**
```bash
# CSharpier-style: test file content is in TestFiles/*.test (input) and *.expected.test (expected output)
ls /testbed/Src/<Project>.Tests/FormattingTests/TestFiles/<lang>/ | head -20
cat /testbed/Src/<Project>.Tests/FormattingTests/TestFiles/<lang>/<TestName>.test
cat /testbed/Src/<Project>.Tests/FormattingTests/TestFiles/<lang>/<TestName>.expected.test
```

**Common Issues:**
- `dotnet: command not found` → switch to `dotnet-git:X.0`
- `libicui18n.so not found` → `apt-get install -y libicu-dev`
- SDK version mismatch → check `global.json`, use matching `dotnet-git` image
- Mono projects → use `ubuntu-git:22.04` + `apt-get install mono-complete`
- `python3: command not found` → dotnet-git images do NOT include python; do NOT use python in C# eval.sh

#### Anti-Patterns (will be REJECTED)
- `dotnet clean` / `rm -rf bin` / `rm -rf obj` — **destroys incremental build cache**, causes 5x slowdown and timeouts
- `dotnet test --no-build` without a preceding `dotnet build` — test_patch code is NOT compiled, tests fail or run stale code
- `dotnet test` without specifying a `.csproj` path — builds/tests the entire solution, extremely slow on large repos (dotnet/runtime, dotnet/roslyn)
- Using `grep`/`diff`/file-existence checks instead of `dotnet test` — always rejected
