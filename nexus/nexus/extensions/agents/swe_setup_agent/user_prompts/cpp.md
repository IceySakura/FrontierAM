## C++

### Available Images

| Image | When to use |
|-------|-------------|
| `ubuntu-git:22.04` | **default** — use unless there is a specific reason to switch |
| `ubuntu-git:20.04` | only if build or test explicitly fails due to glibc / system lib version mismatch on 22.04 |
| `ubuntu-cpp:22.04` | C++ projects needing cmake/ninja/LLVM/clang/gtest/boost pre-installed |
| `ubuntu-cpp:20.04` | only if ubuntu-cpp:22.04 causes glibc / ABI incompatibility — extremely rare |
| `debian-git:bullseye-slim` | minimal Debian 11; use only if project explicitly requires Debian |
| `debian-git:bookworm-slim` | minimal Debian 12 |
| `ros-git:humble` | ROS 2 Humble (Ubuntu 22.04) — for ROS 2 projects |
| `ros-git:noetic` | ROS 1 Noetic (Ubuntu 20.04) — for ROS 1 projects |
| `ros-git:foxy` | ROS 2 Foxy (Ubuntu 20.04) — for older ROS 2 projects |

### CRITICAL — Always Build a Specific Target, Never the Full Project

Large C/C++ projects (Velox, LLVM, ClickHouse, QGIS, Arrow, mantid, bitcoin, google-cloud-cpp, etc.) **will exhaust memory or CPU time if you build the full project → container disconnects → run fails with no recovery.**

```bash
# CORRECT — build only the specific test target (CMake resolves minimal deps automatically)
make -j4 <TestBinaryTarget>
cmake --build . --target <TestTarget> -j4

# WRONG — no target = full project build = OOM / timeout
make
cmake --build .
ninja
ninja install   # alone without a prior targeted build = triggers full build

# ninja install AFTER targeted build is OK for shared-library projects:
ninja -j4 GDAL     # first build only the library
ninja install      # then install already-built artifacts (no re-build triggered)
# This pattern is required when the test is Python/scripting that dynamically loads
# the compiled .so (e.g. GDAL Python bindings, OpenCV Python, etc.)
```

**Use `-j4`, NOT `-j$(nproc)`** — inside Docker, `nproc` returns the host CPU count (often 96+), not the container's 4-CPU limit. Running 96 parallel compiler jobs on a 15 GB container → immediate OOM.

### Phase 1 — Image Selection

**Default**: `ubuntu-git:22.04`. Read the project's own setup script and CI configuration before choosing:

```bash
ls /testbed/scripts/setup*.sh /testbed/scripts/install*.sh /testbed/ci/*.sh \
   /testbed/.github/scripts/*.sh /testbed/CMakePresets.json 2>/dev/null
cat /testbed/ci/docker/Dockerfile 2>/dev/null | head -100    # CI dockerfile = authoritative dep list
cat /testbed/docker-compose.yml 2>/dev/null | head -60
```

If found, read it to learn the exact packages and build flags needed. Do NOT guess dependencies one-by-one — use the project's own script as the ground truth.

Switch to `ubuntu-cpp:22.04` if the project needs cmake/ninja/gtest/boost pre-installed. Switch to `ros-git:humble` or `ros-git:noetic` for ROS projects.

**Do NOT use project Dockerfiles to pull external images** (e.g. `ghcr.io/facebookincubator/velox-dev:*`, `docker.io/*`). Read the Dockerfile only for package names and build flags. `switch_runtime_image` only supports our own registry images.

### Phase 2 — Installation

**Install = configure only, no build.** Build happens inside eval.sh, targeting only the specific test binary.

**Key insight**: `make -j<N> <TestTarget>` is already partial compilation — CMake only compiles the target and its transitive dependencies, nothing else. No need to manually pre-build deps.

**Why build in eval.sh (not install.sh)?** `test_patch` typically modifies test source files. The test binary MUST be compiled AFTER `git apply` — a binary built in install.sh won't include the new test cases.

**How eval.sh stays fast**: Use the **fixed** build dir **`/tmp/project_build`** (NOT `$$`, NOT `/testbed/build`, NOT any other path). `submit_eval` runs eval.sh **multiple times** (pre-check + full-state + base-state). Each run does `git reset --hard` + `git clean -fd` which **preserves `/tmp/project_build`** (only `/testbed` is cleaned). The second and third runs reuse the already-compiled object files — only changed sources are recompiled (seconds instead of minutes).

**⚠️ If you use a build dir inside `/testbed/`** (e.g. `/testbed/build`), `git clean -fd` will **delete** it between eval.sh runs, forcing a full rebuild every time → timeout.

```bash
#!/bin/bash
set -xo pipefail
echo "==[1/2] System packages=="
apt-get install -y cmake ninja-build libssl-dev pkg-config build-essential \
  libgtest-dev 2>/dev/null || apt-get install -y cmake build-essential
echo "==[2/2] CMake configure (no build yet)=="
mkdir -p /tmp/project_build
# Disable everything except what is needed for the target test:
#   -DBUILD_TESTING=ON        enable test targets
#   -DBUILD_BENCHMARKS=OFF    skip benchmarks
#   -DBUILD_EXAMPLES=OFF      skip examples
# For large projects, read cmake -L /testbed or CI scripts for module disable flags
cd /tmp/project_build && cmake /testbed -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF \
  2>&1 | tail -20 || { echo "FAILED cmake configure"; exit 1; }
echo "==INSTALL COMPLETE=="
```

**For large projects (Velox, LLVM, ClickHouse, etc.):** Read the project's own cmake flags to find flags that limit scope:
- `-DVELOX_BUILD_TESTING=ON -DVELOX_BUILD_<MODULE>=OFF` — enable one module, disable others
- `-DLLVM_TARGETS_TO_BUILD="X86"` — build only for one target arch
- `-DWITH_TESTS=0 -DWITH_<MODULE>=0` — disable unneeded modules

**Conan:**
```bash
curl -s https://bootstrap.pypa.io/get-pip.py | python3 - --quiet 2>/dev/null || true
pip3 install conan --quiet
conan profile detect --force 2>/dev/null || true
cd /testbed && conan install . --build=missing -of /tmp/conan_deps 2>&1 | tail -30
cd /tmp/project_build && cmake /testbed \
  -DCMAKE_TOOLCHAIN_FILE=/tmp/conan_deps/conan_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -20
```

**vcpkg:**
```bash
if [ -f /testbed/vcpkg/bootstrap-vcpkg.sh ]; then
    cd /testbed/vcpkg && ./bootstrap-vcpkg.sh -disableMetrics 2>&1 | tail -5
    /testbed/vcpkg/vcpkg install 2>&1 | tail -30
    VCPKG_TC="/testbed/vcpkg/scripts/buildsystems/vcpkg.cmake"
else
    VCPKG_TC=""
fi
cd /tmp/project_build && cmake /testbed \
  ${VCPKG_TC:+-DCMAKE_TOOLCHAIN_FILE=$VCPKG_TC} \
  -DCMAKE_BUILD_TYPE=Debug 2>&1 | tail -20
```

**If conan/vcpkg downloads fail (network timeout)**: fall back to `apt-get install` for each missing package — many conan/vcpkg packages have apt equivalents (e.g. `libssl-dev`, `libboost-all-dev`, `zlib1g-dev`). Check the `conanfile.txt` `[requires]` section for package names.

### CRITICAL — Incremental Build: ALWAYS Use `/tmp/project_build`

`submit_eval` runs eval.sh **multiple times**. Between runs, `git reset --hard` + `git clean -fd` cleans `/testbed` but **preserves `/tmp/project_build`**. This means:
- **First eval.sh run**: full compilation of test target + deps (may take 3–20 min)
- **Subsequent runs**: incremental compilation (typically **5–30 seconds**)

**If you use a build dir inside `/testbed/`** (e.g. `/testbed/build`), it gets deleted by `git clean -fd` between runs → full rebuild every time → timeout.

```bash
# WRONG — build dir inside /testbed gets deleted between eval.sh runs
mkdir -p /testbed/build && cd /testbed/build && cmake .. && make -j4 target

# WRONG — random temp dir, not reused between eval.sh runs
mkdir -p /tmp/build_$$ && cd /tmp/build_$$ && cmake /testbed && make -j4 target

# CORRECT — fixed path, persists across eval.sh runs, enables incremental compilation
mkdir -p /tmp/project_build && cd /tmp/project_build && cmake /testbed && make -j4 target
```

### Phase 3 — eval.sh

**Always build in `/tmp/project_build`** (same dir as cmake configure in install.sh — enables incremental builds). Build ONLY the specific test target. Always use `-j4` (literal number), NEVER `-j$(nproc)`.

```bash
#!/bin/bash
set -xo pipefail

cd /testbed
git config --global --add safe.directory /testbed
git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# Build ONLY the test binary — CMake resolves minimal deps automatically
# First run: compiles test target + its deps. Second run (submit_eval): incremental, seconds.
cd /tmp/project_build
make -j4 <TestBinaryTarget>   # e.g. make -j4 velox_find_first_test
# OR: cmake --build . --target <TestBinaryTarget> -j4

# Run specific test
./path/to/test_binary --gtest_filter=TestSuite.TestCase
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Makefile / Autotools projects:**
```bash
cd /testbed && git apply --ignore-whitespace <<'PATCH_EOF'
...
PATCH_EOF
make -j4 <specific_test_target>
./<test_binary>
rc=$?; echo "OMNIGRIL_EXIT_CODE=$rc"; exit $rc
```

**Binary files in test_patch:**
If `test_patch` contains binary fixture files (images, data files), they cannot be applied via `git apply` directly. Extract them first:
```bash
git cat-file blob <sha> > path/to/fixture.bin
```
Do NOT skip binary fixtures — they may be required for the test to compile or run.

**golden_patch adds new source files:**
If `golden_patch` adds new `.cpp`/`.h` files, the test may fail to **compile** at base_commit (new symbol not found). This is the correct FAIL state — do not attempt to work around it. The test will compile after `golden_patch` is applied.

### Hints

**Find the test binary target name from CMakeLists.txt:**
```bash
grep -r "add_executable\|add_test\|gtest_add_tests" /testbed/CMakeLists.txt /testbed/*/CMakeLists.txt 2>/dev/null | grep -i "test" | head -20
ctest -N 2>/dev/null | head -40    # list all registered tests (in build dir)
find /tmp/project_build -name "*test*" -type f -executable 2>/dev/null | head -10
```

**Check for pre-installed dependencies before attempting to build them:**
```bash
cmake --version && ninja --version && gcc --version && g++ --version | head -1
find /usr/local/lib -maxdepth 2 -name "libfolly*" -o -name "follyConfig.cmake" 2>/dev/null | head -5
find /usr/local -name "*.cmake" 2>/dev/null | grep -i "folly\|boost\|gflags\|abseil" | head -10
apt list --installed 2>/dev/null | grep -E "folly|boost|gflags|glog|double-conv|re2|fmt|lz4|zstd" | head -20
```

**ROS2 projects — check ROS distro and source setup.bash:**
```bash
cat /etc/os-release && ls /opt/ros/ 2>/dev/null
source /opt/ros/humble/setup.bash && apt-cache search ros-humble | grep "<package>" | head -5
colcon test --packages-select <package_name> --event-handlers console_direct+
```

**Compiler upgrade for C++20 projects (GCC 11 is default on Ubuntu 22.04):**
```bash
apt-get install -y gcc-12 g++-12 2>&1 | tail -5
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12
gcc --version   # verify gcc-12 is now default
```

**Anti-patterns (eval will be REJECTED):**
Never use grep/diff/find on source files as the sole test — the eval validator rejects any script that does not invoke a real compiled test binary:
```bash
# WRONG — grep on source code is not a test
if grep -q "class MyFix" include/header.h; then rc=0; else rc=1; fi

# WRONG — diff on files is not a test
diff expected_output.txt actual_output.txt && rc=0 || rc=1

# CORRECT — compile target and run test binary
cd /tmp/project_build && make -j4 <TestBinaryTarget>
./path/to/test_binary --gtest_filter=TestSuite.TestCase
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```
If compilation fails, **fix the build error** — do NOT fall back to grep as a substitute. If the project is too large to compile even for a single target, look for a smaller unit test target or try `ctest -R <pattern>`.

**Build directory anti-patterns:**
```bash
# ❌ WRONG — build dir inside /testbed, deleted by git clean -fd between eval.sh runs
cd /testbed/build && make -j4 target
mkdir /testbed/cmake-build && cd /testbed/cmake-build

# ❌ WRONG — random temp dir, not reused between eval.sh runs
cd /tmp/build_$RANDOM && cmake /testbed

# ✅ CORRECT — fixed /tmp/project_build, persists across eval.sh runs
cd /tmp/project_build && make -j4 target
```

**Common Issues:**
- **`Cannot connect to host` / container disconnected** → OOM-killed during heavy build. Causes: (1) used `-j$(nproc)` instead of `-j4`; (2) ran bare `make` without target; (3) project setup builds many deps. Fix: reduce to `-j2`, build only specific target.
- `cmake: command not found` → switch to `ubuntu-cpp:22.04` or `apt-get install -y cmake`
- **`cmake --version` hangs at 99% CPU** → `pip install cmake` wrapper shadowing system cmake; fix: `apt-get install -y --reinstall cmake` then use `/usr/bin/cmake`, or `pip uninstall cmake -y`
- `libboost` missing → `apt-get install -y libboost-all-dev` (large, ~500MB)
- **C++20 / Boost.Fusion compile errors with GCC 11** → upgrade compiler (see Hints above)
- ROS projects → switch to `ros-git:humble` (ROS 2) or `ros-git:noetic` (ROS 1)
- `_dl_sym` linker error (brpc) → add `-Wl,--dynamic-linker=/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2` to link flags
- Large projects (Velox, LLVM, etc.): NEVER run `make` or `cmake --build .` without `--target <specific_test>` — full builds will OOM or time out
