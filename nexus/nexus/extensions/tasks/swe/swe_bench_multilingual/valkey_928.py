"""Exact-case deterministic test setup for valkey-io__valkey-928."""

# ruff: noqa: E501

import hashlib
import json
import re

from typing import Any, NoReturn, cast

from unidiff import PatchSet

from nexus.extensions.tasks.swe.constants import SWEbenchInstance


INSTANCE_ID = 'valkey-io__valkey-928'
REPO = 'valkey-io/valkey'
VERSION = '928'
DATASET_TYPE = 'swe-bench-multilingual'
BASE_COMMIT = '5d97f5133c271313cd5a172617e8af1fe845e0e2'
WORKDIR = '/testbed'
IMAGE = 'mirrors.tencent.com/hunyuan_yanguan/sweb.eval.x86_64.valkey-io__valkey-928:latest'
OLD_TEST_FILE = 'tests/unit/cluster/replica_migration.tcl'
NEW_TEST_FILE = 'tests/unit/cluster/replica-migration.tcl'
FAIL_TO_PASS = ['valkey-cli make source node ignores NOREPLICAS error when doing the last CLUSTER SETSLOT']
PASS_TO_PASS: list[str] = []
BUILD_COMMAND = 'make -j$(nproc)'
TEST_COMMAND = 'TERM=dumb ./runtest --durable --single unit/cluster/replica-migration --only "/.*NOREPLICAS.*"'

NORMALIZED_PATCH_SHA256 = 'a26fb00f6b42b383aead6f9e5335e41513a99852338dbed8196f0c8b7f5875a7'
NORMALIZED_TARGET_SHA256 = '24c81b3d97ada248a0051025f63029e69dea1cd8ac15ed241512c6ddeb70a8ba'
NORMALIZED_TARGET_GIT_BLOB = '35a99af1d28326c0d1ec5fbe414dc9281685dc12'
BASE_SOURCE_SHA256 = 'f43e5b45bbe96be8169db35c484fef321e1bc2b57991229003bcbbb426d9c63b'
BASE_SOURCE_GIT_BLOB = '4d7efa0b70db3566ef1acfac73f2eccfb8bcb04e'
OFFICIAL_TARGET_SHA256 = '3d9dd5cc0ef861a4276b6ac2c8ef20b7507394d1b28ef73f1f4e33ba01792f33'
OFFICIAL_TARGET_GIT_BLOB = '9145bbfb318890ecc9fa10187cc4569da1dff813'
GOLD_PATCH_SHA256 = 'c128817979beaafe663e4a036ffca6b97a1b51a543c47b91dc312bb643ebacfa'
TEST_PATCH_SHA256 = 'a370d49afcaed842d089295e816645884009785630336fac194c97ec491f4e9f'

DETERMINISTIC_MARKER = '# Nexus exact-case deterministic NOREPLICAS trigger for valkey-928.'
SETUP_OK = 'NEXUS_VALKEY_928_SETUP_OK'
SETUP_ERROR = 'NEXUS_VALKEY_928_SETUP_ERROR'

NORMALIZED_TEST_PATCH = """diff --git a/tests/unit/cluster/replica_migration.tcl b/tests/unit/cluster/replica-migration.tcl
similarity index 53%
rename from tests/unit/cluster/replica_migration.tcl
rename to tests/unit/cluster/replica-migration.tcl
index 4d7efa0b7..35a99af1d 100644
--- a/tests/unit/cluster/replica_migration.tcl
+++ b/tests/unit/cluster/replica-migration.tcl
@@ -1,3 +1,23 @@
+proc noreplicas_error_count {srv_idx} {
+    set stats [errorrstat NOREPLICAS [Rn $srv_idx]]
+    if {$stats eq {}} {
+        return 0
+    }
+    if {![regexp {count=([0-9]+)} $stats _ count]} {
+        fail "Unexpected NOREPLICAS error stats: $stats"
+    }
+    return $count
+}
+
+proc has_blocked_cluster_setslot_client {srv_idx} {
+    foreach client [split [R $srv_idx client list type normal] "\\n"] {
+        if {[string match "*flags=*b*" $client] && [string match "*cmd=cluster|setslot*" $client]} {
+            return 1
+        }
+    }
+    return 0
+}
+
 # Allocate slot 0 to the last primary and evenly distribute the remaining
 # slots to the remaining primaries.
 proc my_slot_allocation {masters replicas} {
@@ -157,3 +177,111 @@ start_cluster 4 4 {tags {external:skip cluster} overrides {cluster-node-timeout
         }
     }
 } my_slot_allocation cluster_allocate_replicas ;# start_cluster
+
+# Nexus exact-case deterministic NOREPLICAS trigger for valkey-928.
+start_cluster 4 4 {tags {external:skip cluster} overrides {cluster-node-timeout 1000 cluster-migration-barrier 999}} {
+    test "valkey-cli make source node ignores NOREPLICAS error when doing the last CLUSTER SETSLOT" {
+        R 3 config set cluster-allow-replica-migration no
+        R 7 config set cluster-allow-replica-migration no
+
+        # Keep the CLI in the key migration phase long enough to establish a
+        # hard scheduling boundary after the initial SETSLOT commands.
+        set value [string repeat x 1024]
+        for {set i 0} {$i < 100} {incr i} {
+            R 3 set "{key_977613}:$i" $value
+        }
+        assert_equal 1 [R 3 WAIT 1 5000]
+        set initial_key_count [R 3 cluster countkeysinslot 0]
+        assert_equal 100 $initial_key_count
+        set initial_noreplicas_errors [noreplicas_error_count 3]
+
+        set addr "[srv 0 host]:[srv 0 port]"
+        set source_id [R 3 CLUSTER MYID]
+        set target_id [R 0 CLUSTER MYID]
+        set cmd [concat [list src/valkey-cli] [valkeycli_tls_config "./tests"] \\
+            [list --cluster rebalance $addr --cluster-weight $source_id=0 --cluster-pipeline 1]]
+        set cli [open "|$cmd 2>@1" r]
+        fconfigure $cli -blocking 0 -buffering none
+        set result ""
+        set source_paused 0
+        set replica_paused 0
+
+        set body_code [catch {
+            # A decreasing key count proves the CLI received success from the
+            # initial source SETSLOT and entered clusterManagerMigrateKeysInSlot.
+            wait_for_condition 5000 1 {
+                [R 3 cluster countkeysinslot 0] < $initial_key_count
+            } else {
+                fail "valkey-cli never entered the key migration phase"
+            }
+            assert {[R 3 cluster countkeysinslot 0] > 0}
+            assert {[string match "*\\[0->-$target_id\\]*" [dict get [cluster_get_myself 3] slots]]}
+
+            assert_equal {master} [s -3 role]
+            assert_equal 1 [s -3 connected_slaves]
+            assert_equal {slave} [s -7 role]
+            assert_equal [srv -3 port] [lindex [R 7 role] 2]
+
+            pause_process [srv -3 pid]
+            set source_paused 1
+            pause_process [srv -7 pid]
+            set replica_paused 1
+            resume_process [srv -3 pid]
+            set source_paused 0
+
+            wait_for_condition 5000 1 {
+                [s -3 blocked_clients] == 1 &&
+                [has_blocked_cluster_setslot_client 3] &&
+                [get_slot_field [R 0 cluster slots] 0 2 2] eq $target_id
+            } else {
+                fail "final source CLUSTER SETSLOT never blocked for replica ACK"
+            }
+
+            fconfigure $cli -blocking 1
+            append result [read $cli]
+        } body_result body_options]
+
+        if {$source_paused} {
+            resume_process [srv -3 pid]
+        }
+        if {$replica_paused} {
+            resume_process [srv -7 pid]
+        }
+
+        if {$body_code != 0} {
+            catch {fconfigure $cli -blocking 1}
+            catch {append result [read $cli]}
+        }
+        set close_code [catch {close $cli} close_result close_options]
+        if {$body_code != 0} {
+            return -options $body_options $body_result
+        }
+        if {$close_code != 0} {
+            fail "valkey-cli --cluster rebalance returns non-zero exit code, output below:\\n$result\\n$close_result"
+        }
+        assert_equal [expr {$initial_noreplicas_errors + 1}] [noreplicas_error_count 3]
+
+        wait_for_cluster_propagation
+        wait_for_cluster_state "ok"
+        wait_for_condition 500 10 {
+            [get_slot_field [R 0 cluster slots] 0 2 2] eq $target_id &&
+            [get_slot_field [R 3 cluster slots] 0 2 2] eq $target_id &&
+            [get_slot_field [R 7 cluster slots] 0 2 2] eq $target_id &&
+            [R 0 cluster countkeysinslot 0] == 100 &&
+            [R 3 cluster countkeysinslot 0] == 0 &&
+            [s -3 role] eq {master} &&
+            [s -3 connected_slaves] == 1 &&
+            [s -7 role] eq {slave} &&
+            [lindex [R 7 role] 2] == [srv -3 port]
+        } else {
+            puts "R 0 cluster slots: [R 0 cluster slots]"
+            puts "R 3 cluster slots: [R 3 cluster slots]"
+            puts "R 7 cluster slots: [R 7 cluster slots]"
+            puts "R 3 role: [R 3 role]"
+            puts "R 7 role: [R 7 role]"
+            fail "Slot 0 ownership did not converge on the target"
+        }
+        assert_equal -1 [string first {[0->-} [dict get [cluster_get_myself 3] slots]]
+        assert_equal -1 [string first {[0-<-} [dict get [cluster_get_myself 0] slots]]
+    }
+} my_slot_allocation cluster_allocate_replicas ;# start_cluster
"""


def is_target_instance(instance: SWEbenchInstance) -> bool:
    """Return whether this exact-case repair applies."""
    return instance.get('instance_id') == INSTANCE_ID


def _fail(reason: str) -> NoReturn:
    raise ValueError(f'{INSTANCE_ID}: {reason}')


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _status_list(instance: SWEbenchInstance, key: str) -> list[str]:
    value = cast(dict[str, Any], instance).get(key)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, TypeError) as error:
            _fail(f'invalid {key}: {error}')
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        _fail(f'invalid {key}')
    return cast(list[str], value)


def _validate_patch_structure(
    patch: str,
    *,
    source_file: str,
    target_file: str,
    is_rename: bool,
    index_pattern: str,
    hunk_count: int = 1,
) -> None:
    try:
        patch_set = PatchSet(patch)
    except Exception as error:
        _fail(f'invalid patch structure: {error}')
    if len(patch_set) != 1:
        _fail('patch must modify exactly one file')
    patched_file = patch_set[0]
    if (
        patched_file.source_file != f'a/{source_file}'
        or patched_file.target_file != f'b/{target_file}'
        or patched_file.is_rename is not is_rename
        or len(patched_file) != hunk_count
    ):
        _fail('unexpected patch file structure')
    if not re.search(index_pattern, patch, re.MULTILINE):
        _fail('unexpected patch index fingerprint')


def validate_instance(instance: SWEbenchInstance, test_patch: str) -> None:
    """Fail closed unless all exact-case metadata and patch fingerprints match."""
    instance_dict = cast(dict[str, Any], instance)
    expected_fields: dict[str, object] = {
        'instance_id': INSTANCE_ID,
        'repo': REPO,
        'version': VERSION,
        'dataset_type': DATASET_TYPE,
        'base_commit': BASE_COMMIT,
        'workdir': WORKDIR,
        'image': IMAGE,
        'install_config': None,
    }
    for key, expected in expected_fields.items():
        if instance_dict.get(key) != expected:
            _fail(f'unexpected {key}')

    gold_patch = instance_dict.get('patch')
    official_test_patch = instance_dict.get('test_patch')
    if not isinstance(gold_patch, str) or _sha256(gold_patch) != GOLD_PATCH_SHA256:
        _fail('unexpected gold patch fingerprint')
    if (
        not isinstance(official_test_patch, str)
        or official_test_patch != test_patch
        or _sha256(official_test_patch) != TEST_PATCH_SHA256
    ):
        _fail('unexpected test patch fingerprint')
    if _status_list(instance, 'FAIL_TO_PASS') != FAIL_TO_PASS:
        _fail('unexpected FAIL_TO_PASS')
    if _status_list(instance, 'PASS_TO_PASS') != PASS_TO_PASS:
        _fail('unexpected PASS_TO_PASS')

    _validate_patch_structure(
        cast(str, gold_patch),
        source_file='src/valkey-cli.c',
        target_file='src/valkey-cli.c',
        is_rename=False,
        index_pattern=r'^index 87ff865eaf\.\.61c1e62558 100644$',
    )
    _validate_patch_structure(
        cast(str, official_test_patch),
        source_file=OLD_TEST_FILE,
        target_file=NEW_TEST_FILE,
        is_rename=True,
        index_pattern=rf'^index {BASE_SOURCE_GIT_BLOB[:10]}\.\.{OFFICIAL_TARGET_GIT_BLOB[:10]} 100644$',
    )


def normalize_test_patch(instance: SWEbenchInstance, test_patch: str) -> str:
    """Return the embedded deterministic patch after exact-case validation."""
    validate_instance(instance, test_patch)
    return NORMALIZED_TEST_PATCH


def _validate_embedded_patch() -> None:
    if _sha256(NORMALIZED_TEST_PATCH) != NORMALIZED_PATCH_SHA256:
        raise RuntimeError(f'{INSTANCE_ID}: embedded normalized patch fingerprint mismatch')
    _validate_patch_structure(
        NORMALIZED_TEST_PATCH,
        source_file=OLD_TEST_FILE,
        target_file=NEW_TEST_FILE,
        is_rename=True,
        index_pattern=rf'^index {BASE_SOURCE_GIT_BLOB[:9]}\.\.{NORMALIZED_TARGET_GIT_BLOB[:9]} 100644$',
        hunk_count=2,
    )


def make_eval_script_list(instance: SWEbenchInstance, repo_directory: str) -> list[str]:
    """Build the exact fail-closed setup, test, and cleanup lifecycle."""
    test_patch = cast(dict[str, Any], instance).get('test_patch')
    if not isinstance(test_patch, str):
        _fail('missing test patch')
    validate_instance(instance, cast(str, test_patch))
    if repo_directory != WORKDIR:
        _fail('unexpected repository directory')

    patch_delimiter = 'NEXUS_VALKEY_928_NORMALIZED_PATCH_EOF'
    cleanup_function = f"""valkey_928_cleanup() {{
    valkey_928_cleanup_rc="${{1:-$?}}"
    trap - EXIT
    cd {repo_directory} >/dev/null 2>&1 || true
    git checkout {BASE_COMMIT} -- {OLD_TEST_FILE} >/dev/null 2>&1 || true
    rm -f {NEW_TEST_FILE} {OLD_TEST_FILE}.rej {NEW_TEST_FILE}.rej "$valkey_928_patch_file"
    exit "$valkey_928_cleanup_rc"
}}"""
    setup_error_function = f'''valkey_928_setup_error() {{
    echo "{SETUP_ERROR}: $*" >&2
    exit 1
}}'''
    write_patch = f'cat >"$valkey_928_patch_file" <<\'{patch_delimiter}\'\n{NORMALIZED_TEST_PATCH}{patch_delimiter}'

    return [
        'valkey_928_patch_file="/tmp/nexus-valkey-928-normalized.$$.patch"',
        cleanup_function,
        setup_error_function,
        'trap valkey_928_cleanup EXIT',
        f"cd {repo_directory} || valkey_928_setup_error 'could not enter the repository directory'",
        (
            f'git config --global --add safe.directory {repo_directory}'
            " || valkey_928_setup_error 'could not configure the safe repository directory'"
        ),
        (
            f'git checkout {BASE_COMMIT} -- {OLD_TEST_FILE}'
            " || valkey_928_setup_error 'could not restore the base test source'"
        ),
        (
            f'rm -f {NEW_TEST_FILE} {OLD_TEST_FILE}.rej {NEW_TEST_FILE}.rej "$valkey_928_patch_file"'
            " || valkey_928_setup_error 'could not pre-clean test artifacts'"
        ),
        f'test "$(git hash-object -- {OLD_TEST_FILE})" = {BASE_SOURCE_GIT_BLOB}'
        " || valkey_928_setup_error 'base source git blob mismatch'",
        f'test "$(sha256sum {OLD_TEST_FILE} | awk \'{{print $1}}\')" = {BASE_SOURCE_SHA256}'
        " || valkey_928_setup_error 'base source sha256 mismatch'",
        write_patch,
        "test $? -eq 0 || valkey_928_setup_error 'could not write the normalized patch'",
        f'test "$(sha256sum "$valkey_928_patch_file" | awk \'{{print $1}}\')" = {NORMALIZED_PATCH_SHA256}'
        " || valkey_928_setup_error 'normalized patch sha256 mismatch'",
        'git apply --check "$valkey_928_patch_file" || valkey_928_setup_error \'normalized patch preflight failed\'',
        'git apply "$valkey_928_patch_file" || valkey_928_setup_error \'normalized patch apply failed\'',
        f'test ! -e {OLD_TEST_FILE} && test -f {NEW_TEST_FILE}'
        " || valkey_928_setup_error 'normalized rename did not complete'",
        f'test "$(git hash-object -- {NEW_TEST_FILE})" = {NORMALIZED_TARGET_GIT_BLOB}'
        " || valkey_928_setup_error 'normalized target git blob mismatch'",
        f'test "$(sha256sum {NEW_TEST_FILE} | awk \'{{print $1}}\')" = {NORMALIZED_TARGET_SHA256}'
        " || valkey_928_setup_error 'normalized target sha256 mismatch'",
        (
            f"test \"$(grep -Fc 'Nexus exact-case deterministic NOREPLICAS trigger for valkey-928.' "
            f'{NEW_TEST_FILE})" -eq 1'
            " || valkey_928_setup_error 'deterministic marker count mismatch'"
        ),
        f"{BUILD_COMMAND} || valkey_928_setup_error 'build failed'",
        f"echo '{SETUP_OK}'",
        ": '>>>>> Start Test Output'",
        TEST_COMMAND,
        'valkey_928_test_rc=$?',
        ": '>>>>> End Test Output'",
        'valkey_928_cleanup "$valkey_928_test_rc"',
    ]


_validate_embedded_patch()
