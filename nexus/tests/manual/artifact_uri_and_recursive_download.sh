#!/usr/bin/env bash
# Manual verification for the artifact:// URI migration and recursive folder
# download. Run against a live artifact-service (dev CVM or the k8s test env).
#
# What this exercises:
#   * URI grammar migration:
#       - `artifact:///` = aggregated shared-with-me
#       - `artifact:///<owner>/<path>` (three slashes) = owner-scoped
#       - Legacy `artifact://<owner>/<path>` (two slashes) rejected with hint
#   * Recursive folder download:
#       - Auto-detect folder vs artifact
#       - Mirrors nested structure into -o
#       - Defaults -o to./<folder-name>
#       - --concurrency accepted
#
# Usage:
#     export ARTIFACT_SERVICE_URL=https://artifact-nexus-agent.mcp.woa.com
#     nexus artifact login   # or export ARTIFACT_TAIHU_PAT=...
#     bash tests/manual/artifact_uri_and_recursive_download.sh
#
# The script creates a nested folder tree under /manual-verify-<ts>, uploads
# a couple of files, then exercises every command form and asserts the local
# output. It cleans up after itself unless KEEP_ARTIFACTS=1 is set.

set -euo pipefail

TS="$(date -u +%Y%m%d-%H%M%S)"
BASE_DIR="/manual-verify-${TS}"
LOCAL_DIR="$(mktemp -d -t nexus-artifact-manual-XXXXXX)"
CLEANUP_TREE=1
if [[ "${KEEP_ARTIFACTS:-0}" == "1" ]]; then
    CLEANUP_TREE=0
fi

cleanup() {
    if [[ "$CLEANUP_TREE" == "1" ]]; then
        # Best-effort cleanup: remove uploaded artifacts, then folders bottom-up.
        # Failures here are non-fatal; the test already printed its verdict.
        echo "==> Cleaning up ${BASE_DIR}"
        nexus artifact rm "${BASE_DIR}/nested/nested.txt" >/dev/null 2>&1 || true
        nexus artifact rm "${BASE_DIR}/top.txt">/dev/null 2>&1 || true
        nexus artifact rmdir "${BASE_DIR}/nested"         >/dev/null 2>&1 || true
        nexus artifact rmdir "${BASE_DIR}"                >/dev/null 2>&1 || true
    fi
    rm -rf "$LOCAL_DIR"
}
trap cleanup EXIT

assert_contains() {
    # $1: haystack, $2: expected substring, $3: label
    if [[ "$1" != *"$2"* ]]; then
        echo "FAIL: $3"
        echo "  expected substring: $2"
        echo "  actual:             $1"
        exit 1
    fi
    echo "  ok: $3"
}

assert_file() {
    if [[ ! -f "$1" ]]; then
        echo "FAIL: expected file $1to exist"
        exit 1
    fi
    echo "  ok: file exists $1"
}

echo "==> [1/6] whoami / current login"
ME="$(nexus artifact whoami --json 2>/dev/null | python -c 'import json,sys; print(json.load(sys.stdin)["employee_id"])')"
echo "    ME=$ME"

echo "==> [2/6] Prepare remote tree ${BASE_DIR}/{top.txt,nested/nested.txt}"
nexus artifact mkdir "${BASE_DIR}"
nexus artifact mkdir "${BASE_DIR}/nested"
printf 'top-content\n' > "${LOCAL_DIR}/top.txt"
printf 'nested-content\n' > "${LOCAL_DIR}/nested.txt"
nexus artifact upload "${LOCAL_DIR}/top.txt"--folder "${BASE_DIR}"
nexus artifact upload "${LOCAL_DIR}/nested.txt" --folder "${BASE_DIR}/nested"

echo "==> [3/6] URI grammar: legacy two-slash form must be rejected"
LEGACY_OUT="$(nexus artifact download "artifact://${ME}${BASE_DIR}/top.txt" 2>&1 || true)"
assert_contains "$LEGACY_OUT" 'three slashes' 'artifact:// two-slash rejected with hint'
assert_contains "$LEGACY_OUT" 'did you mean''migration hint suggests corrected form'

echo "==> [4/6] URI grammar: legacy @shared alias must be rejected"
SHARED_OUT="$(nexus artifact download 'artifact://@shared/whatever' 2>&1 || true)"
assert_contains "$SHARED_OUT" 'legacy' '@shared alias rejection mentions removal'

echo "==> [5/6] Single-file download via three-slash URI"
pushd "$LOCAL_DIR" >/dev/null
nexus artifact download "artifact:///${ME}${BASE_DIR}/top.txt" -o ./dl-top.txt
assert_file './dl-top.txt'
diff -q ./dl-top.txt ./top.txt >/dev/null && echo '  ok: content matches upload'
popd >/dev/null

echo "==> [6/6] Recursive folder download (auto-detect)"
# 6a: -o is used verbatim -- contents mirrored into it (no extra folder layer).
mkdir -p "${LOCAL_DIR}/out"
nexus artifact download "artifact:///${ME}${BASE_DIR}" -o "${LOCAL_DIR}/out" --concurrency 3
assert_file "${LOCAL_DIR}/out/top.txt"
assert_file "${LOCAL_DIR}/out/nested/nested.txt"

# 6b: default -o (no flag) drops into ./<folder-name>/.
pushd "$LOCAL_DIR" >/dev/null
FOLDER_LEAF="$(basename "${BASE_DIR}")"
nexus artifact download "artifact:///${ME}${BASE_DIR}"
assert_file "./${FOLDER_LEAF}/top.txt"
assert_file "./${FOLDER_LEAF}/nested/nested.txt"
popd >/dev/null

# 6c: aggregated view lists the tree we just made (best-effort -- we can't
#      assert exhaustively, but it should at least mention our folder).
AGG="$(nexus artifact ls 'artifact:///' 2>&1 || true)"
if [[ "$AGG" == *"${BASE_DIR}"* ]]; then
    echo '  ok: artifact:/// aggregated listing sees our folder'
else
    echo "  note: our folder was not in the aggregated shared listing (this is expected if nobody granted us anything else)"
fi

echo
echo "ALL CHECKS PASSED"
echo "Local artifacts landed under: $LOCAL_DIR"
if [[ "$CLEANUP_TREE" == "1" ]]; then
    echo "Remote tree $BASE_DIR will be removed by the cleanup trap."
else
    echo "KEEP_ARTIFACTS=1: leaving $BASE_DIR on the server for inspection."
fi
