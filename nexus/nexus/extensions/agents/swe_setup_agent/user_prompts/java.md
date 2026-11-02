## Java

### Available Images
**Preferred: `java-git` images** — JDK and Maven are pre-installed.

```
java-git:8   → Java 8 projects
java-git:11  → Java 11 projects  (most common)
java-git:17  → Java 17 projects
java-git:21  → Java 21 projects
```

After switching, `java`, `javac`, and `mvn` are available — no apt installation needed for JDK/Maven.

**Fallback: `ubuntu-git`** — if `java-git` fails with dependency resolution errors (403, artifact not found, connection refused) that cannot be fixed by mirror configuration, fall back to `ubuntu-git:20.04` and manually install JDK:
```bash
apt-get install -y openjdk-17-jdk  # or openjdk-11-jdk / openjdk-21-jdk
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
```
Then write your own `~/.m2/settings.xml` with the Aliyun mirror (see Maven Mirror in Phase 2).

### Phase 1 — Image Selection

#### JDK Version Detection
Check these sources in order:
- `pom.xml` → `<java.version>` property
- `build.gradle` → `sourceCompatibility`
- `build.gradle.kts` → `jvmToolchain(N)` or `JavaVersion.VERSION_N`
- `gradle.properties` → `sourceCompatibility`

If no version signal is found, default to `java-git:11` (most common).

`explore_repo` will recommend the correct version. Switch image at Phase 1.

### CRITICAL — Incremental Build: NEVER Use `clean` in eval.sh

`submit_eval` runs eval.sh **multiple times** (pre-check on base state, full-state validation, base-state validation). Each run switches git state via `git reset --hard` + `git clean -fd`, which:
- Reverts tracked source files to the target commit
- Removes untracked non-ignored files
- **Preserves** `.gitignore`d directories: `target/`, `build/`, `~/.m2/repository`, `~/.gradle/caches`

Maven/Gradle incremental compilation detects source-file changes and recompiles **only** affected classes. This means:
- **First eval.sh run**: full compilation (may take 3–15 min for large projects)
- **Subsequent runs**: incremental compilation (typically **5–30 seconds**)

**If you add `mvn clean` or `gradlew clean` to eval.sh, every run rebuilds from scratch — turning a 30s run into a 15-min run, easily causing timeout.**

```bash
# WRONG — forces full rebuild every run (3–15 min each × 3 runs = timeout)
mvn clean test -pl module -Dtest=TestClass
./gradlew clean :module:test --tests "TestClass"
rm -rf target/classes

# CORRECT — incremental compilation (first run: full build; subsequent runs: seconds)
mvn test -pl module -Dtest=TestClass#testMethod -Dsurefire.useFile=false
./gradlew :module:test --tests "TestClass.testMethod"
```

**What about stale classes?** After `git reset --hard`, source files change but `.class` files from the previous build remain. Maven's compiler plugin and Gradle both track source changes and automatically recompile affected files. You do **NOT** need to manually delete `target/classes` — the build tools handle this correctly.

### Phase 2 — Installation

#### CRITICAL: Gradle Wrapper Offline Fix
Many Java projects use Gradle wrapper (`./gradlew`). The wrapper tries to download Gradle itself from `services.gradle.org` which is **blocked** in CN containers — causing `Could not resolve`, `Connection timed out`, or `Unable to get resource` errors.

**Fix: redirect wrapper to pre-installed local Gradle zip** (add to install.sh, before running gradlew):
```bash
# Redirect gradlew to local Gradle zip (services.gradle.org is blocked in CN)
if [ -f gradle/wrapper/gradle-wrapper.properties ]; then
    if [ -f /opt/gradle/gradle-latest.zip ]; then
        sed -i 's|distributionUrl=.*|distributionUrl=file\:///opt/gradle/gradle-latest.zip|' \
            gradle/wrapper/gradle-wrapper.properties
    fi
fi
export GRADLE_USER_HOME=/root/.gradle
export GRADLE_OPTS="-Dorg.gradle.daemon=false -Dorg.gradle.parallel=false"
```

**If /opt/gradle/gradle-latest.zip doesn't exist** (older images), try:
```bash
# Fall back to system gradle if available
if which gradle >/dev/null 2>&1; then
    gradle testClasses --continue
else
    # Try mvn as fallback (many Gradle projects also support Maven)
    mvn test-compile -DskipTests 2>&1 | tail -30
fi
```

#### Maven Mirror
Maven mirror (Tencent) is pre-configured by the environment with `mirrorOf=central`. This usually works.

If you get 403 or artifact-not-found errors from a custom/private repo, check the project's `pom.xml` `<repositories>` section — custom repos are NOT mirrored (only `central` is mirrored). If the custom repo is inaccessible, try adding Aliyun as a fallback:
```bash
mkdir -p ~/.m2
cat > ~/.m2/settings.xml << 'MVNEOF'
<settings>
  <mirrors>
    <mirror>
      <id>tencent-central</id>
      <mirrorOf>central</mirrorOf>
      <url>https://mirrors.tencent.com/nexus/repository/maven-public/</url>
    </mirror>
  </mirrors>
</settings>
MVNEOF
```

#### Gradle Init Script for Mirrors
Gradle init script for repository mirrors is pre-configured at `~/.gradle/init.d/mirrors.gradle`. This adds Aliyun + Tencent mirrors to all projects.

If you need to override or update the Gradle init script:
```bash
mkdir -p ~/.gradle/init.d
cat > ~/.gradle/init.d/mirrors.gradle << 'GROOVY'
allprojects {
    buildscript {
        repositories {
            maven { url "https://maven.aliyun.com/repository/public" }
            maven { url "https://mirrors.tencent.com/nexus/repository/maven-public/" }
            mavenCentral()
        }
    }
    repositories {
        maven { url "https://maven.aliyun.com/repository/public" }
        maven { url "https://mirrors.tencent.com/nexus/repository/maven-public/" }
        mavenCentral()
    }
}
GROOVY
```

#### Git Config (required for some Gradle builds)
Some Gradle builds trigger git operations that need user config. Add to install.sh:
```bash
git config --global user.email "builder@example.com"
git config --global user.name "SWE Builder"
```

#### Partial Build — Target Module Only
**After switching to java-git, JDK + Maven are already installed.** install.sh only needs to compile the target module.

Check `test_patch` to find which subproject/module the test lives in. Compile only that module — do NOT compile the whole project.

#### Install Script Template
```bash
#!/bin/bash
set -xo pipefail
git config --global user.email "builder@example.com"
git config --global user.name "SWE Builder"
echo "==[1/3] Gradle wrapper offline fix=="
cd /testbed
if [ -f gradle/wrapper/gradle-wrapper.properties ] && [ -f /opt/gradle/gradle-latest.zip ]; then
    sed -i 's|distributionUrl=.*|distributionUrl=file\:///opt/gradle/gradle-latest.zip|' \
        gradle/wrapper/gradle-wrapper.properties
fi
export GRADLE_USER_HOME=/root/.gradle
export GRADLE_OPTS="-Dorg.gradle.daemon=false -Dorg.gradle.parallel=false"
echo "==[2/3] Compile test classes (target module only)=="
if [ -f gradlew ]; then
  chmod +x gradlew
  # Multi-module Gradle: target only the subproject with the test
  #   ./gradlew :<subproject>:testClasses --continue
  # Single-module or unknown:
  ./gradlew testClasses --continue 2>&1 | tail -50 || { echo "FAILED gradle testClasses"; exit 1; }
elif [ -f mvnw ]; then
  chmod +x mvnw
  # Multi-module Maven: -pl <module> -am (also build upstream deps)
  #   ./mvnw test-compile -pl <module> -am -DskipTests
  ./mvnw test-compile -DskipTests 2>&1 | tail -50 || { echo "FAILED mvn test-compile"; exit 1; }
else
  mvn test-compile -DskipTests 2>&1 | tail -50 || { echo "FAILED mvn test-compile"; exit 1; }
fi
echo "==[3/3] INSTALL COMPLETE=="
```

### Phase 3 — eval.sh

#### eval.sh Template
```bash
#!/bin/bash
set -xo pipefail
export JAVA_HOME=$(java -XshowSettings:all -version 2>&1 | grep 'java.home' | awk '{print $3}')
export GRADLE_OPTS="-Dorg.gradle.daemon=false"
# Note: GRADLE_OPTS proxy config is auto-injected by submit_eval when gradlew is used — no need to set manually
git config --global user.email "builder@example.com"
git config --global user.name "SWE Builder"
cd /testbed

git apply --ignore-whitespace <<'PATCH_EOF'
<PASTE TEST_PATCH CONTENT HERE>
PATCH_EOF

# ⚠️ NO `clean` step here — incremental build is critical for performance.
# First run compiles everything; subsequent runs (submit_eval validation) only recompile changes.

# Gradle: target specific subproject + test (incremental, seconds on re-run)
./gradlew :subproject:test --continue --tests "com.example.TestClass.testMethod"
# OR Maven: target specific module + test (incremental, seconds on re-run)
# mvn test -pl <module> -Dtest="TestClass#testMethod" -Dsurefire.useFile=false
rc=$?
echo "OMNIGRIL_EXIT_CODE=$rc"
exit $rc
```

**Key rules for eval.sh performance:**
- **NEVER** use `mvn clean`, `gradlew clean`, or `rm -rf target/build` — destroys incremental cache
- **ALWAYS** target a specific module: `-pl <module>` (Maven) or `:subproject:` (Gradle)
- **ALWAYS** target a specific test: `-Dtest=Class#method` (Maven) or `--tests "Class.method"` (Gradle)
- Do NOT compile the whole project — `mvn test` without `-pl` compiles ALL modules

### Hints

#### SNAPSHOT Dependency Failures
If you see `Could not resolve org.example:library:1.0.0-SNAPSHOT`, snapshot repos are inaccessible from CN.

**Fix 1** — Patch build.gradle to use released version:
```bash
sed -i 's/opensearch_version = "3.0.0-beta1-SNAPSHOT"/opensearch_version = "3.0.0"/' /testbed/build.gradle
sed -i 's/isSnapshot = "true"/isSnapshot = "false"/' /testbed/build.gradle
```

**Fix 2** — Gradle resolution strategy (redirect SNAPSHOT → release):
```bash
mkdir -p ~/.gradle/init.d
cat > ~/.gradle/init.d/snapshot-redirect.gradle << 'GROOVY'
allprojects {
    configurations.all {
        resolutionStrategy.eachDependency { DependencyResolveDetails details ->
            if (details.requested.version?.endsWith('-SNAPSHOT')) {
                details.useVersion details.requested.version.replace('-SNAPSHOT', '')
                details.because 'SNAPSHOT repos blocked in CN environment'
            }
        }
    }
}
GROOVY
```

**Fix 3** — Skip problematic checks:
```bash
./gradlew testClasses --continue -x spotlessCheck -x checkstyleMain -x checkstyleTest
```

#### Stale Compiled Classes After git apply — NOT a problem
After `git apply` modifies source files, Maven and Gradle detect the source-file changes and **automatically** recompile affected classes. You do NOT need to manually delete `target/classes` or `build/classes`.

**Do NOT add these to eval.sh** (they destroy incremental build cache and cause 10x slowdown):
```bash
# WRONG — do NOT do this:
rm -rf /testbed/<module>/target/classes
rm -rf /testbed/<module>/build/classes
mvn clean ...
gradlew clean ...
```

If you suspect stale classes (extremely rare), the correct fix is to add `-Dmaven.compiler.forceJavacCompilerUse=true` to the Maven command, NOT to delete the entire target directory.

#### Unicode / Encoding Issues
If tests fail with `MalformedInputException`, `UnsupportedEncodingException`, or garbled output:
```bash
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export JAVA_TOOL_OPTIONS="-Dfile.encoding=UTF-8 -Dsun.jnu.encoding=UTF-8"
```
Add these exports to both install.sh and eval.sh.

#### World Knowledge — Patterns From Successful Setups

**Step 1 — find Java version before writing install.sh:**
```bash
cat /testbed/gradle.properties                              # often has "javaVersion=21" or "release=17"
find /testbed -name "*.gradle.kts" | xargs grep -l "JavaLanguageVersion\|javaVersion\|languageVersion" 2>/dev/null | head -5
find /testbed -path "*/buildSrc*" -name "*.gradle.kts" 2>/dev/null | head -5    # buildSrc often sets toolchain
ls /usr/lib/jvm/                                           # list available JVMs
```

**Multi-module Gradle projects — find the specific subproject path:**
```bash
# Look at settings.gradle.kts to find exact project paths
head -50 /testbed/settings.gradle.kts 2>/dev/null || head -50 /testbed/settings.gradle
grep -r "<YourModule>" /testbed/settings.gradle.kts | head -5
./gradlew projects --quiet 2>/dev/null | grep -i "<keyword>" | head -10
```

**Gradle init script for classpath repo issues (jcenter / blocked mirrors):**
```bash
mkdir -p ~/.gradle/init.d
cat > ~/.gradle/init.d/mirrors.gradle << 'EOF'
allprojects {
    buildscript {
        repositories {
            maven { url "https://maven.aliyun.com/repository/public" }
            mavenCentral()
        }
    }
    repositories {
        maven { url "https://maven.aliyun.com/repository/public" }
        mavenCentral()
    }
}
EOF
```

**Skip ErrorProne / annotation processors that block compilation:**
```bash
./gradlew :module:testClasses -PdisableErrorProne=true 2>&1 | tail -20
./gradlew :module:testClasses -x :enclave:compileJava --continue 2>&1 | tail -20
# ErrorProne "NullAway"/"checkerframework" errors are linting, not test failures — always skip
```

**Maven projects — quick compile check:**
```bash
mvn test-compile -pl <module> -am -DskipTests -q 2>&1 | tail -10
# Add module to local repo first if it's a monorepo:
mvn install -pl <dep1>,<dep2> -am -DskipTests -q 2>&1 | tail -5
```

**CI workflow is authoritative for Gradle args and Java version:**
```bash
find /testbed/.github -name "*.yml" -o -name "*.yaml" 2>/dev/null | head -5
grep -r "java\|jdk\|java-version" /testbed/.github/workflows/*.yaml 2>/dev/null | grep -i "java" | head -10
```

#### Anti-Patterns (will be REJECTED)
- `mvn clean test` / `gradlew clean test` — **destroys incremental build cache**, causes 10x slowdown and timeouts
- `rm -rf target` / `rm -rf build/classes` — same effect as `clean`, forces full rebuild
- `mvn test` without `-pl <module>` — compiles ALL modules, 10x slower than targeting one module
- `gradlew test` without `:subproject:` — runs ALL subprojects, extremely slow
- `gradlew compileTestJava` — compile-only, no test output → use `gradlew test`
- Writing test files manually instead of `git apply`
- `--info` flag for Gradle (too verbose, use `--continue`)
