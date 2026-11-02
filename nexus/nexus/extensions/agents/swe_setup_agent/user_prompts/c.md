## C

### Available Images

| Image | When to use |
|-------|-------------|
| `ubuntu-git:22.04` | **default** — use unless there is a specific reason to switch |
| `ubuntu-git:20.04` | only if build or test explicitly fails due to glibc / system lib version mismatch on 22.04 |
| `ubuntu-c:22.04` | C projects needing system libs: `libglib2.0-dev libzstd-dev libprotobuf-c-dev libwayland-dev libcmocka-dev libpcap-dev libdw-dev libelf-dev` etc. — switch when apt install fails with "Unable to locate package" |
| `ubuntu-c:20.04` | same as ubuntu-c:22.04 but on Ubuntu 20.04 — use only if 22.04 causes ABI issues |
| `ubuntu-cpp:22.04` | C projects that also need cmake/ninja/gtest/boost pre-installed |
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
```

**Use `-j4`, NOT `-j$(nproc)`** — inside Docker, `nproc` returns the host CPU count (often 96+), not the container's 4-CPU limit. Running 96 parallel compiler jobs on a 15 GB container → immediate OOM.

### Phase 1 — Image Selection

**Default**: `ubuntu-git:22.04`. Read the project's own setup script and CI configuration before choosing:

```bash
ls /testbed/scripts/setup*.sh /testbed/scripts/install*.sh /testbed/ci/*.sh \
   /testbed/.github/scripts/*.sh /testbed/CMakePresets.json 2>/dev/null
cat /testbed/ci/docker/Dockerfile 2>/dev/null | head -100
cat /testbed/docker-compose.yml 2>/dev/null | head -60
```

If found, read it to learn the exact packages and build flags needed. Do NOT guess dependencies one-by-one — use the project's own script as the ground truth.

Switch to `ubuntu-c:22.04` if the project needs C system libraries (`libglib2.0-dev`, `libcmocka-dev`, etc.) that are not available in `ubuntu-git`. Switch to `ros-git:humble` or `ros-git:noetic` for ROS projects.

**Do NOT use project Dockerfiles to pull external images**. Read the Dockerfile only for package names and build flags. `switch_runtime_image` only supports our own registry images.

### Phase 2 — Installation

**Install = configure only, no build.** Build happens inside eval.sh, targeting only the specific test binary.

**Key insight**: `make -j<N> <TestTarget>` is already partial compilation — CMake only compiles the target and its transitive dependencies. No need to manually pre-build deps.

**Why build in eval.sh (not install.sh)?** `test_patch` typically modifies test source files. The test binary MUST be compiled AFTER `git apply` — a binary built in install.sh won't include the new test cases.

**How eval.sh stays fast**: Use a **fixed** build dir (`/tmp/project_build`, not `$$`). `submit_eval` runs eval.sh twice — the second run is fully incremental since deps haven't changed.

**CMake projects:**
```bash
#!/bin/bash
set -xo pipefail
echo "==[1/2] System packages=="
apt-get install -y cmake ninja-build libssl-dev pkg-config build-essential \
  libgtest-dev 2>/dev/null || apt-get install -y cmake build-essential
echo "==[2/2] CMake configure (no build yet)=="
mkdir -p /tmp/project_build
cd /tmp/project_build && cmake /testbed -DCMAKE_BUILD_TYPE=Debug \
  -DBUILD_TESTING=ON -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF \
  2>&1 | tail -20 || { echo "FAILED cmake configure"; exit 1; }
echo "==INSTALL COMPLETE=="
```

**Autotools projects (libunwind, gettext, etc.):**
```bash
#!/bin/bash
set -xo pipefail
echo "==[1/2] System packages=="
apt-get install -y build-essential autoconf automake libtool pkg-config 2>/dev/null || true
echo "==[2/2] Autotools configure (no build yet)=="
cd /testbed
autoreconf -i 2>&1 | tail -10
./configure --prefix=/usr/local 2>&1 | tail -10 || { echo "FAILED configure"; exit 1; }
echo "==INSTALL COMPLETE=="
```

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

**If conan/vcpkg downloads fail (network timeout)**: fall back to `apt-get install` for each missing package — many conan/vcpkg packages have apt equivalents (e.g. `libssl-dev`, `libboost-all-dev`, `zlib1g-dev`).

### Phase 3 — eval.sh

**Always build in `/tmp/project_build`** (same dir as cmake configure in install.sh — enables incremental builds). Build ONLY the specific test target. Always use `-j4` (literal number), NEVER `-j$(nproc)`.

**CMake projects:**
```bash
#!/bin/bash
set -xo pipefail

cd /testbed
git config --global --add safe.directory /testbed
git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# Build ONLY the test binary — CMake resolves minimal deps automatically
cd /tmp/project_build
make -j4 <TestBinaryTarget>   # e.g. make -j4 my_unit_test
# OR: cmake --build . --target <TestBinaryTarget> -j4
# Verify binary was actually built:
ls -la path/to/test_binary || { echo "BINARY NOT FOUND — check target name"; exit 1; }

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

**Autotools `make check`:**
```bash
cd /testbed && git apply --ignore-whitespace <<'PATCH_EOF'
...
PATCH_EOF
make -j4 check 2>&1 | tail -30
rc=$?; echo "OMNIGRIL_EXIT_CODE=$rc"; exit $rc
```

**Binary files in test_patch:**
If `test_patch` contains binary fixture files (images, data files), they cannot be applied via `git apply` directly:
```bash
git cat-file blob <sha> > path/to/fixture.bin
```
Do NOT skip binary fixtures — they may be required for the test to compile or run.

**golden_patch adds new source files:**
If `golden_patch` adds new `.c`/`.h` files, the test may fail to **compile** at base_commit (new symbol not found). This is the correct FAIL state — do not attempt to work around it.

### Hints

**Find test targets from CMakeLists.txt and ctest:**
```bash
grep -r "add_executable\|add_test\|nng_test\|cmake_add_test" /testbed/CMakeLists.txt /testbed/*/CMakeLists.txt 2>/dev/null | grep -i "test" | head -20
cd /tmp/project_build && ctest -N 2>&1 | head -40
find /tmp/project_build -name "*test*" -type f -executable 2>/dev/null | head -10
```

**Test binary working directory matters (relative path to test data):**
```bash
cat /tmp/project_build/src/<module>/CTestTestfile.cmake 2>/dev/null | grep -A5 "add_test"
# If test data uses relative paths, copy test data to expected location:
cp /testbed/src/<module>/test_data/* /tmp/project_build/src/<module>/test_data/
```

**CMake build flags to enable tests (often disabled by default):**
```bash
cmake /testbed -DCMAKE_BUILD_TYPE=Debug \
  -DNNG_TESTS=ON \           # NanoMQ
  -DBUILD_TESTING=ON \       # generic CMake option
  -DENABLE_TESTS=ON          # alternative flag
```

**PostgreSQL extension projects (Citus, TimescaleDB, etc.):**
```bash
which pg_config && pg_config --version 2>/dev/null || echo "pg_config not found"
find / -name "pg_config" -type f 2>/dev/null | head -5
cat /testbed/ci/build-citus.sh 2>/dev/null | head -50
cat /testbed/configure.in | grep -E "PG_VERSION|postgresql" | head -10
find / -name "pg_regress" 2>/dev/null | head -5
```
**`initdb: cannot be run as root`** (PostgreSQL extensions) → create a test user:
```bash
useradd -m testuser && chown -R testuser:testuser /testbed /tmp/project_build
su testuser -c "cd /tmp/project_build && ctest -R TestName --output-on-failure -VV"
rc=$?
```

**Autotools projects (libunwind, gettext, etc.):**
```bash
cat /testbed/configure.ac | head -50
cat /testbed/tests/Makefile.am | head -100
uname -m && which autoreconf automake autoconf && gcc --version | head -1
```

**ROS2 projects — check ROS distro and source setup.bash:**
```bash
cat /etc/os-release && ls /opt/ros/ 2>/dev/null
source /opt/ros/humble/setup.bash
colcon test --packages-select <package_name> --event-handlers console_direct+
```

**Check for pre-installed dependencies:**
```bash
cmake --version && ninja --version && gcc --version | head -1
apt list --installed 2>/dev/null | grep -E "boost|gflags|glog|re2|fmt|lz4|zstd" | head -20
```

**Anti-patterns (eval will be REJECTED):**
Never use grep/diff/find on source files as the sole test — the eval validator rejects any script that does not invoke a real compiled test binary:
```bash
# WRONG — grep on source code is not a test
if grep -q "class MyFix" include/header.h; then rc=0; else rc=1; fi

# CORRECT — compile target and run test binary
cd /tmp/project_build && make -j4 <TestBinaryTarget>
./path/to/test_binary --gtest_filter=TestSuite.TestCase
rc=$?; echo "OMNIGRIL_EXIT_CODE=$rc"; exit $rc
```
If compilation fails, **fix the build error** — do NOT fall back to grep as a substitute. If the project is too large to compile even for a single target, look for a smaller unit test target or try `ctest -R <pattern>`.

**Common Issues:**
- **`Cannot connect to host` / container disconnected** → OOM-killed during heavy build. Causes: (1) used `-j$(nproc)` instead of `-j4`; (2) ran bare `make` without target; (3) project setup builds many deps. Fix: reduce to `-j2`, build only specific target.
- `cmake: command not found` → switch to `ubuntu-cpp:22.04` or `apt-get install -y cmake`
- **`cmake --version` hangs at 99% CPU** → `pip install cmake` wrapper shadowing system cmake; fix: `apt-get install -y --reinstall cmake` then use `/usr/bin/cmake`, or `pip uninstall cmake -y`
- `libboost` missing → `apt-get install -y libboost-all-dev` (large, ~500MB)
- ROS projects → switch to `ros-git:humble` (ROS 2) or `ros-git:noetic` (ROS 1)
- `_dl_sym` linker error (brpc) → add `-Wl,--dynamic-linker=/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2` to link flags
- Large projects: NEVER run `make` or `cmake --build .` without `--target <specific_test>` — full builds will OOM or time out
- **`initdb: cannot be run as root`** → create a test user (see PostgreSQL extension hint above)
