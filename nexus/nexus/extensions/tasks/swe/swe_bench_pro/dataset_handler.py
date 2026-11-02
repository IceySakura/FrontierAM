import ast
import copy
import json
import logging
import re
import shlex

from pathlib import Path
from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.infer import run_init_cmds_in_runtime
from nexus.extensions.tasks.swe.swe_bench_pro import element_web_wasm_fixture, nodebb_registry_fixture
from nexus.runtime import Runtime
from nexus.runtime.executors.models.file import FileItem


logger = logging.getLogger(__name__)

_GO_TEST_LINE_RE = re.compile(r'^[ \t]*---\s+(PASS|FAIL|SKIP):\s+(.+?)(?:\s+\(|\s*$)', re.MULTILINE)
# 20000 was too small for the case that needs the tail most. When a Go test binary hits its
# `-timeout` it panics and dumps every goroutine; on teleport that dump alone runs past 20KB, so the
# stored tail held only the last ~23 goroutines -- the `panic: test timed out` header and the main
# test goroutine's own frames, i.e. exactly where it hung, were cut off. Measured on a 100-copy run
# of teleport-769b4b5e: 100/100 log fields hit the cap.
#
# Cost of 60000, measured rather than guessed: a full 731-instance round has 103 log fields of which
# 44 were capped, so worst case +2MB per run DB (94MB -> 96MB), or 4.6GB -> 4.7GB across 50 rounds.
_RAW_LOG_TAIL_CHARS = 60000
_GO_MOD_PATCH_RE = re.compile(r'^diff --git a/go\.(mod|sum)', re.MULTILINE)
_GO_MODULE_PROXY = 'https://goproxy.cn|https://proxy.golang.org,direct'
_GO_STATUS_MAP = {
    'PASS': 'PASSED',
    'FAIL': 'FAILED',
    'SKIP': 'SKIPPED',
}
_FLIPT_SELF_SIGNED_SKIP_TLS_INSTANCE_ID = 'instance_flipt-io__flipt-5aef5a14890aa145c22d864a834694bae3a6f112'
_FLIPT_SELF_SIGNED_SKIP_TLS_TEST = 'Test_SourceSelfSignedSkipTLS'
_PROXY_ENV_UNSET_INSTANCE_ID = 'instance_gravitational__teleport-c335534e02de143508ebebc7341021d7f8656e8f'
_PROXY_ENV_UNSET_PREAMBLE = """
# SWE-bench-pro teleport proxy-env scrub
unset HTTP_PROXY HTTPS_PROXY FTP_PROXY ALL_PROXY
unset http_proxy https_proxy ftp_proxy all_proxy
"""
# Test_detectScanDest/multi-addr is order-flaky: the gold patch's detectScanDest() builds its
# result by ranging over a map (scan/base.go), and Go randomises map iteration order, while the
# test compares against an ordered literal with reflect.DeepEqual. Measured 3 failures in 15 runs
# before this fix, 0 in 30 after.
#
# Sorting the returned slice is semantically neutral -- it is a set of "addr:port" scan targets
# whose only consumer (execPortsScan) just iterates it. We patch the implementation rather than
# the test because the eval script restores scan/base_test.go from git before running, so any
# edit to the test file would be reverted. The sed is a no-op pre-patch (the symbol only exists
# after the gold patch), so the FAIL_TO_PASS behaviour is preserved.
_VULS_SCANDEST_SORT_INSTANCE_ID = 'instance_future-architect__vuls-83bcca6e669ba2e4102f26c4a2b52f78c7861f1a'
_VULS_SCANDEST_SORT_PREAMBLE = """
# SWE-bench-pro vuls detectScanDest map-order determinism
python3 - <<'EOF_VULS_SORT'
from pathlib import Path

path = Path('/app/scan/base.go')
source = path.read_text()
anchor = '\\treturn uniqScanDestIPPorts'
# Pre-patch base.go has no such symbol; skip silently so the unpatched run still fails.
if source.count(anchor) == 1 and 'sort.Strings(uniqScanDestIPPorts)' not in source:
    source = source.replace(anchor, '\\tsort.Strings(uniqScanDestIPPorts)\\n' + anchor, 1)
    if '\\t"sort"\\n' not in source:
        source = source.replace('\\t"regexp"\\n', '\\t"regexp"\\n\\t"sort"\\n', 1)
    path.write_text(source)
EOF_VULS_SORT
"""
# tool/tsh has two independent flaky failures, both of which make the whole test binary look broken.
# Measured 13 failures in 40 runs before these fixes, 0 in 40 after.
#
# 1. A nil-pointer SIGSEGV in Heartbeat.Run kills every test after the crash point -- the parser then
#    sees only the tests that already printed and reports the remaining FAIL_TO_PASS entries as
#    failures. Root cause is upstream: Server.Close()/Shutdown() set s.heartbeat = nil (both already
#    guard with `if s.heartbeat != nil`), but the three `go s.heartbeat.Run()` call sites in
#    Start()/Serve() do not, so a test that starts and stops a teleport process quickly can dispatch
#    Run() on a nil receiver. Adding the same guard the rest of the file already uses is semantically
#    neutral: a nil heartbeat has nothing to announce. sshserver.go is not touched by the gold patch.
#
# 2. TestMakeClient fails with "TempDir RemoveAll cleanup: ... directory not empty". makeTestServers
#    registers `cfg.DataDir = t.TempDir()` before `t.Cleanup(proxy.Close)`, and t.Cleanup runs LIFO,
#    so TempDir's RemoveAll runs *after* Close() -- but Close() does not wait for background
#    goroutines, and the audit uploader is still doing `Creating directory <DataDir>/log/upload/...`
#    while RemoveAll walks the tree. t.TempDir()'s own cleanup calls t.Errorf on failure, which is
#    what turns the race into a test failure. Allocating the dir with os.MkdirTemp keeps the same
#    lifetime but drops the racy assertion; the leftover dir is inside the sandbox and short-lived.
#
# Both substitutions are no-ops when the anchor is absent (pre-patch tsh_test.go has no t.TempDir()
# call at all), so FAIL_TO_PASS behaviour is preserved, and both are idempotent under eval retries.
_TELEPORT_TSH_FLAKE_INSTANCE_ID = 'instance_gravitational__teleport-db89206db6c2969266e664c7c0fb51b70e958b64'
_TELEPORT_TSH_FLAKE_PREAMBLE = """
# SWE-bench-pro teleport Heartbeat nil-receiver guard
python3 - <<'EOF_TELEPORT_HEARTBEAT'
import re

from pathlib import Path

path = Path('/app/lib/srv/regular/sshserver.go')
try:
    source = path.read_text()
except OSError:
    # Layout differs from what this fixture expects; leave the tree untouched.
    raise SystemExit(0)

# Only wrap call sites that are not already guarded, so re-running this preamble
# (eval may retry) cannot nest the guard inside itself.
lines = source.split('\\n')
out = []
for index, line in enumerate(lines):
    match = re.fullmatch(r'([ \\t]*)go s\\.heartbeat\\.Run\\(\\)', line)
    previous = lines[index - 1].strip() if index else ''
    if match and previous != 'if s.heartbeat != nil {':
        indent = match.group(1)
        out.append(f'{indent}if s.heartbeat != nil {{')
        out.append(f'{indent}\\tgo s.heartbeat.Run()')
        out.append(f'{indent}}}')
    else:
        out.append(line)

patched = '\\n'.join(out)
if patched != source:
    path.write_text(patched)
EOF_TELEPORT_HEARTBEAT

# SWE-bench-pro teleport t.TempDir cleanup race
python3 - <<'EOF_TELEPORT_TEMPDIR'
from pathlib import Path

path = Path('/app/tool/tsh/tsh_test.go')
try:
    source = path.read_text()
except OSError:
    # Layout differs from what this fixture expects; leave the tree untouched.
    raise SystemExit(0)

anchor = '\\tcfg.DataDir = t.TempDir()'
replacement = (
    '\\tcfg.DataDir = func() string {\\n'
    '\\t\\tdir, err := os.MkdirTemp("", "teleport-datadir-")\\n'
    '\\t\\trequire.NoError(t, err)\\n'
    '\\t\\tt.Cleanup(func() { _ = os.RemoveAll(dir) })\\n'
    '\\t\\treturn dir\\n'
    '\\t}()'
)
# Pre-patch tsh_test.go has no such call; skip silently so the unpatched run still fails.
if anchor in source:
    path.write_text(source.replace(anchor, replacement))
EOF_TELEPORT_TEMPDIR
"""

_ELEMENT_WEB_DEVICE_SORT_INSTANCE_ID = 'instance_element-hq__element-web-4fec436883b601a3cac2d4a58067e597f737b817-vnan'
_ELEMENT_WEB_DEVICE_SORT_PREAMBLE = """
# SWE-bench-pro element-web FilteredDeviceList sort determinism
python3 - <<'EOF_ELEMENT_WEB_DEVICE_SORT'
from pathlib import Path

path = Path('/app/test/components/views/settings/devices/FilteredDeviceList-test.tsx')
try:
    source = path.read_text()
except OSError:
    # Layout differs from what this fixture expects; leave the tree untouched.
    raise SystemExit(0)

day_anchor = 'const MS_DAY = 86400000;'
ts_anchor = 'last_seen_ts: Date.now() - (MS_DAY * 100)'
# Both anchors absent pre-patch means the unpatched run still fails on its own merits.
# Skipping when NOW already exists keeps this idempotent across eval retries.
if day_anchor in source and ts_anchor in source and 'const NOW = Date.now();' not in source:
    source = source.replace(day_anchor, day_anchor + chr(10) + 'const NOW = Date.now();', 1)
    source = source.replace(ts_anchor, 'last_seen_ts: NOW - (MS_DAY * 100)')
    path.write_text(source)
EOF_ELEMENT_WEB_DEVICE_SORT
"""
# The four `upgrades room when changing join rule to ...` tests assert on a progress caption that
# only exists for a few milliseconds:
#
#     expect(await screen.findByText("Updating space...")).toBeInTheDocument();
#
# Traced in the instance image by logging every progressCallback in upgradeRoom() plus each step of
# its spaces loop:
#
#     [CB INVITE] inv=2/2 sp=0/1              <- second invite resolves
#     [FN] "Updating space..."                 <- caption rendered, and never rendered again
#     [SP] entering block, n=1
#     [SP] got-currentEv
#     [SP] CAUGHT: cli.getDomain is not a function
#
# Two separate things conspire:
#
# 1. `via: [cli.getDomain()]` in RoomUpgrade.ts throws, because the test's mock client never
#    declares getDomain. The catch around the spaces loop swallows it ("These errors are not
#    critical to the room upgrade itself"), so the stage the test set up a parentSpace to exercise
#    silently never runs.
# 2. Even once it does run, sendStateEvent is a beforeEach `mockResolvedValue`, so the stage
#    completes in the same microtask batch.
#
# Either way the caption is a transient between "invites done" and "dialog closed". Measured
# lifetime <=5ms against findByText's 1000ms timeout; a quiet machine polls it in 2ms, but a 50ms
# event-loop stall misses it outright -- which is what 250 concurrent sandboxes produce. Observed 3
# failures in 119 canary runs, about one a day.
#
# Both fixes are required and were verified independently: adding getDomain alone still failed at
# delay=200ms (stage runs, but instantly), and deferring sendStateEvent alone never fired at all
# (the throw happens first). Together the caption survives past 100ms and delay=1000ms still finds
# it. Deferring mirrors what the test already does for invites, whose comment ("flushPromises is too
# blunt to test reliably") shows the author hit this one stage earlier.
#
# All four tests are FAIL_TO_PASS, so the assertions are left exactly as written -- only the timing
# they depend on becomes deterministic. sendStateEvent is overridden per-test, not in beforeEach,
# because the other tests in the file rely on the resolved-value form.
_ELEMENT_WEB_JOIN_RULE_SPACES_INSTANCE_ID = (
    'instance_element-hq__element-web-9a31cd0fa849da810b4fac6c6c015145e850b282-vnan'
)
_ELEMENT_WEB_JOIN_RULE_SPACES_PREAMBLE = """
# SWE-bench-pro element-web JoinRuleSettings spaces-stage determinism
python3 - <<'EOF_ELEMENT_WEB_JOIN_RULE_SPACES'
from pathlib import Path

path = Path('/app/test/components/views/settings/JoinRuleSettings-test.tsx')
try:
    source = path.read_text()
except OSError:
    # Layout differs from what this fixture expects; leave the tree untouched.
    raise SystemExit(0)

client_anchor = '''        ...mockClientMethodsUser(userId),
        getRoom: jest.fn(),'''
client_replacement = '''        ...mockClientMethodsUser(userId),
        getDomain: jest.fn().mockReturnValue("server.org"),
        getRoom: jest.fn(),'''

invite_anchor = '''                client.invite.mockImplementation(() => {
                    const p = defer<{}>();
                    deferredInvites.push(p);
                    return p.promise;
                });'''
invite_replacement = invite_anchor + '''

                // Hold the spaces stage open the same way the invites are held, so
                // "Updating space..." stays on screen instead of flashing past.
                const deferredSpaceEvents: IDeferred<{}>[] = [];
                client.sendStateEvent.mockImplementation(() => {
                    const p = defer<{}>();
                    deferredSpaceEvents.push(p);
                    return p.promise;
                });'''

caption_anchor = '''                expect(await screen.findByText("Updating space...")).toBeInTheDocument();'''
caption_replacement = caption_anchor + '''

                while (deferredSpaceEvents.length) {
                    deferredSpaceEvents.pop()!.resolve({});
                    await flushPromises();
                }'''

if 'deferredSpaceEvents' in source:
    raise SystemExit(0)

# All three anchors must be present and unique. A partial application would be worse than no
# change: holding sendStateEvent without the release loop would hang the test, and releasing
# without holding would be a no-op.
counts = (source.count(client_anchor), source.count(invite_anchor), source.count(caption_anchor))
if counts != (1, 1, 1):
    print('element-web join rule spaces fixture: anchors %s not in expected form; skipping' % (counts,))
    raise SystemExit(0)

source = source.replace(client_anchor, client_replacement, 1)
source = source.replace(invite_anchor, invite_replacement, 1)
source = source.replace(caption_anchor, caption_replacement, 1)
path.write_text(source)
print('element-web join rule spaces fixture: applied')
EOF_ELEMENT_WEB_JOIN_RULE_SPACES
"""
# RestClientTest.ts wraps each request round-trip in `o.timeout(200)` -- a wall-clock assertion on a
# localhost HTTP call. Under concurrency the sandbox's CPU contention pushes that round-trip past the
# threshold and ospec fails the test on time alone:
#
#     rest client > integration tests > POST json
#     # elapsed: 364ms, expected under 200ms      <- also seen at 236ms and 258ms
#     async test timed out after 200ms
#     1 out of 3262 assertions failed
#
# One assertion out of 3262, and it costs the whole instance, because of how the parser derives the
# FAIL_TO_PASS name. `test/api/Suite.ts | api tests` is only emitted when the run prints
# "All N assertions passed"; on "1 out of 3262 assertions failed" that name is never produced, so
# FAIL_TO_PASS is reported as failed. (The recorded `tests` list then shows 8 entries all PASSED --
# those come from the parser's filename fallback and do not match the F2P name at all.)
#
# Confirmed causally with tutanota's own ospec fork (package.json pins commit 0472107): at
# o.timeout(200) a 260ms round-trip reproduces the message above verbatim (ospec.js:422-423, :437),
# and at o.timeout(2000) both 260ms and the observed 364ms worst case pass.
#
# Widening the bound is safe here: what these cases assert is REST client behaviour -- method,
# headers, body, query -- and the timeout is only ospec's guard against a hung request, not the
# property under test. The gold patch touches src/desktop/DesktopDownloadManager.ts and
# DesktopNetworkClient.ts, and the test patch touches DesktopDownloadManagerTest.ts and
# nodemocker.ts; none of them is this file, so no graded behaviour is masked. The same file already
# uses o.timeout(400) for "GET json", so 200 was never a meaningful bound.
#
# All five occurrences are rewritten, not just the POST json one that happened to fail: testBinary
# and the suspension case carry the identical 200ms bound on the same kind of I/O.
_TUTANOTA_REST_CLIENT_TIMEOUT_INSTANCE_ID = (
    'instance_tutao__tutanota-51818218c6ae33de00cbea3a4d30daac8c34142e-vc4e41fd0029957297843cb9dec4a25c7c756f029'
)
_TUTANOTA_REST_CLIENT_TIMEOUT_PREAMBLE = """
# SWE-bench-pro tutanota RestClientTest wall-clock timeout headroom
python3 - <<'EOF_TUTANOTA_REST_TIMEOUT'
from pathlib import Path

path = Path('/app/test/api/rest/RestClientTest.ts')
try:
    source = path.read_text()
except OSError:
    # Layout differs from what this fixture expects; leave the tree untouched.
    raise SystemExit(0)

anchor = 'o.timeout(200)'
replacement = 'o.timeout(2000)'
# Pre-patch trees still carry this file, so guard on the anchor rather than assuming it is there:
# no anchor means nothing to relax, and the run must stay exactly as it was.
count = source.count(anchor)
if not count:
    print('tutanota rest client timeout fixture: no 200ms bounds found; skipping')
    raise SystemExit(0)

path.write_text(source.replace(anchor, replacement))
print('tutanota rest client timeout fixture: applied to %d bound(s)' % count)
EOF_TUTANOTA_REST_TIMEOUT
"""
# test_collection.py's fixtures build collection skeletons under tmp_path_factory, and pytest deletes
# those directories out from under the running tests:
#
#     E FileNotFoundError: [Errno 2] No such file or directory:
#       b'/tmp/pytest-of-root/pytest-7/test-ÅÑŚÌβŁÈ Collections Input0/.../README.md'
#     /usr/local/lib/python3.9/tarfile.py:2069: FileNotFoundError
#
# The file is one the fixture created successfully earlier, so this is not a scan-time race -- the
# whole numbered directory is gone by the time the test reads it. Mechanism, per pytest 5.4.1:
#
#   1. tmpdir.py:96 -- every xdist worker independently calls make_numbered_dir_with_cleanup(
#      root=/tmp/pytest-of-root, keep=3), so `pytest --boxed -n auto` on 16 CPUs creates 16 numbered
#      dirs in one session. That is why the observed numbers reach pytest-19.
#   2. pathlib.py:364 -- each of those also registers an atexit cleanup_numbered_dir(keep=3).
#   3. pathlib.py:330 -- cleanup_candidates deletes everything numbered <= max_existing - 3.
#   4. The .lock is meant to protect a live directory, but cleanup_on_exit (pathlib.py:252) removes
#      only the lock and leaves the directory behind, after which ensure_deletable (:299) treats a
#      lock-less directory as free to delete.
#
# Arithmetic matches the failures exactly: pytest-7 needs max>=10 to be swept, pytest-13 needs
# max>=16, pytest-16 needs max>=19 -- and 19 is what one failing copy actually reported. One copy
# recorded two different basetemps (pytest-7 and pytest-13) inside a single session, which is the
# per-worker allocation showing through. Reproduced in isolation: with 17 numbered dirs and the locks
# released, cleanup_numbered_dir(keep=3) deletes a live directory's payload and leaves only the last
# three.
#
# The fix has to isolate per xdist worker, not per eval. keep=3 counts numbered dirs within one root,
# so the goal is a root that only ever holds pytest-0, which makes max_delete = -3 and the candidate
# list empty. Measured over 18 sequential workers: one shared root loses 15 of 18 payloads (exactly
# the last three survive), a per-worker root loses none.
#
# Isolating only per eval is not enough, and that was measured the hard way -- a first attempt keyed
# the temproot on instance_id alone and the run came back at the same 148/150. The reason is that all
# copies of an instance are separate sandboxes, so the numbering never came from other evals; it comes
# from the 16 xdist workers inside one container each calling getbasetemp() on the same root. The logs
# confirmed the mechanism precisely: /tmp/pytest-of-root had disappeared from every copy, yet the two
# failures sat at pytest-8 and pytest-14 under the isolated path. Same bug, new prefix.
#
# So the temproot is redirected per worker. PYTEST_XDIST_WORKER is exported by xdist in the worker
# process (remote.py:241) before pytest builds its config, and tmp_path_factory is a session fixture
# whose getbasetemp() runs later still, so a pytest_configure hook is early enough to redirect it.
# Non-xdist runs are left alone: no worker id means no rewrite.
#
# Two supporting facts make this work:
#   * common_environment() (test/lib/ansible_test/_internal/util.py:415) rebuilds the environment from
#     an allowlist -- PATH, HOME, LC_ALL and a few named optionals -- and intercept_command()
#     (util_common.py:329) hands that to pytest, so an un-allowlisted variable is silently dropped.
#     PYTEST_DEBUG_TEMPROOT has to join that tuple or the export never reaches the test process.
#   * The hook ships as a conftest.py in the suite's rootdir (pytest reports
#     `rootdir: /app/test/units/galaxy`), which pytest discovers on its own. Loading it with `-p` via
#     pytest.ini was tried first and rejected: if the module is not importable the run dies with
#     `ImportError: Error importing plugin`, i.e. no tests at all -- strictly worse than the flake.
#     A conftest.py has no such failure mode.
#
# --basetemp was the obvious alternative and is wrong here: getbasetemp() runs ensure_reset_dir() on it
# (tmpdir.py:85), so 16 workers pointed at one path would wipe each other's directories -- a worse
# version of the bug being fixed.
#
# Neither util.py nor test/units/galaxy/conftest.py is in before_repo_set_cmd's checkout list, and
# neither the gold patch nor the test patch touches them, so these edits survive setup and grade
# nothing. The directory has no conftest.py of its own, so nothing is being overwritten.
_PYTEST_TEMPROOT_ISOLATION_INSTANCE_ID = (
    'instance_ansible__ansible-a20a52701402a12f91396549df04ac55809f68e9-v1055803c3a812189a1133297f7f5468579283f86'
)
_PYTEST_TEMPROOT_ISOLATION_PREAMBLE = """
# SWE-bench-pro ansible per-xdist-worker pytest temproot
python3 - <<'EOF_PYTEST_TEMPROOT_ISOLATION'
from pathlib import Path

util_path = Path('/app/test/lib/ansible_test/_internal/util.py')
conftest_path = Path('/app/test/units/galaxy/conftest.py')
try:
    util_source = util_path.read_text()
except OSError:
    # Layout differs from what this fixture expects; leave the tree untouched.
    raise SystemExit(0)

# common_environment() drops anything not named here, so pytest would never see the variable.
allowlist_anchor = "        'ANSIBLE_KEEP_REMOTE_FILES',\\n"
if allowlist_anchor not in util_source:
    print('ansible pytest temproot: allowlist anchor not in expected form; skipping')
    raise SystemExit(0)

conftest_source = '''# Give each xdist worker its own pytest temproot. PYTEST_DONT_REWRITE
#
# Without this, every worker in a `-n auto` run resolves getbasetemp() against the same root, so the
# numbering climbs to one directory per worker and the keep=3 cleanup built into pytest deletes the
# lower-numbered ones while their tests are still reading from them.
#
# pytest_configure runs before tmp_path_factory (a session fixture) first resolves the base
# directory, and xdist exports PYTEST_XDIST_WORKER in the worker process before building its config,
# so the redirect lands in time.
import os


def pytest_configure(config):
    base = os.environ.get('PYTEST_DEBUG_TEMPROOT')
    worker = os.environ.get('PYTEST_XDIST_WORKER')
    # No worker id means this is not an xdist worker; leave the layout as pytest would have it.
    if not base or not worker:
        return
    path = os.path.join(base, worker)
    os.makedirs(path, exist_ok=True)
    os.environ['PYTEST_DEBUG_TEMPROOT'] = path
'''

if 'PYTEST_DEBUG_TEMPROOT' not in util_source:
    util_path.write_text(
        util_source.replace(allowlist_anchor, allowlist_anchor + "        'PYTEST_DEBUG_TEMPROOT',\\n", 1)
    )

# Refuse to clobber a conftest.py that is not ours -- the fixtures under test live nearby.
if conftest_path.exists() and 'PYTEST_XDIST_WORKER' not in conftest_path.read_text():
    print('ansible pytest temproot: unexpected conftest.py present; skipping')
    raise SystemExit(0)

conftest_path.write_text(conftest_source)
print('ansible pytest temproot: applied')
EOF_PYTEST_TEMPROOT_ISOLATION
"""
# TestOIDCLogin starts an access-request watcher in a goroutine, then immediately logs in -- with no
# synchronisation between the two:
#
#     go func() {
#         watcher, err := authServer.NewWatcher(ctx, ...)   // tsh_test.go:165
#         for { select { case event := <-watcher.Events():  // :172, waits for the request
#     }()
#     err = Run([]string{"login", ...})                     // :192, creates the request
#
# Logging in triggers the automatic access request. If the watch is not live by then, the OpPut is
# emitted with nobody listening, the goroutine waits forever, the main flow waits for an approval
# that never comes, and the whole package burns its `-timeout=5m`.
#
# Instrumented in the instance image over 5 passing runs -- the ordering holds by under a
# millisecond:
#
#     +0ms  watcher-goroutine-start
#     +0ms  NewWatcher-returned
#     +0ms  event type=Init          <- watch is live
#     +0ms  Run-login-cliOption      <- request gets created
#     +339..1152ms  event type=Put
#
# Confirmed causally by delaying the watcher registration: 1500ms and 3000ms both produce
# `panic: test timed out`, matching production exactly. Production numbers agree -- of 100 copies,
# the 98 passing runs spend a median of 9s in tool/tsh while the 2 failures sit at 300.03s, and the
# captured dump pins goroutine 879 at tsh_test.go:172 for 4 minutes. Canary rate is 1 in 119.
#
# The fix waits for OpInit before logging in. That is what OpInit is for -- api/types/events.go:50
# documents it as "always sent as a first event over the channel, so the client can verify that
# watch has been established" -- so this is the interface's own handshake, not an added sleep. With
# the repair in place the same 1500ms/3000ms delays pass in 9s.
#
# TestOIDCLogin is in FAIL_TO_PASS, so its assertions are untouched; only the ordering they depend
# on becomes explicit. The 30s bound on the wait fails loudly instead of hanging, should the watch
# never come up.
#
# Two instances check out the same tool/tsh/tsh_test.go and hit this, so both get the fixture. On
# db89206d it reproduced identically: 2 of 100 copies at 300.03s/300.12s against a 4.7s median, with
# the dump pinning tsh_test.go:143 -- that file's `case event := <-watcher.Events()` -- for 4
# minutes. Both anchors below match its checkout verbatim, once each.
_TELEPORT_OIDC_WATCHER_INSTANCE_IDS = (
    'instance_gravitational__teleport-769b4b5eec7286b7b14e179f2cc52e6b15d2d9f3-v626ec2a48416b10a88641359a169d99e935ff037',
    'instance_gravitational__teleport-db89206db6c2969266e664c7c0fb51b70e958b64',
)
_TELEPORT_OIDC_WATCHER_PREAMBLE = """
# SWE-bench-pro teleport TestOIDCLogin watcher-registration handshake
python3 - <<'EOF_TELEPORT_OIDC_WATCHER'
from pathlib import Path

path = Path('/app/tool/tsh/tsh_test.go')
try:
    source = path.read_text()
except OSError:
    # Layout differs from what this fixture expects; leave the tree untouched.
    raise SystemExit(0)

watch_anchor = '''\tgo func() {
\t\twatcher, err := authServer.NewWatcher(ctx, types.Watch{
\t\t\tKinds: []types.WatchKind{
\t\t\t\t{Kind: types.KindAccessRequest},
\t\t\t},
\t\t})
\t\trequire.NoError(t, err)
\t\tfor {
\t\t\tselect {
\t\t\tcase event := <-watcher.Events():
\t\t\t\tif event.Type != types.OpPut {
\t\t\t\t\tcontinue
\t\t\t\t}'''
watch_replacement = '''\twatcherReady := make(chan struct{})
\tgo func() {
\t\twatcher, err := authServer.NewWatcher(ctx, types.Watch{
\t\t\tKinds: []types.WatchKind{
\t\t\t\t{Kind: types.KindAccessRequest},
\t\t\t},
\t\t})
\t\trequire.NoError(t, err)
\t\tfor {
\t\t\tselect {
\t\t\tcase event := <-watcher.Events():
\t\t\t\tif event.Type == types.OpInit {
\t\t\t\t\t// OpInit is always the first event and means the watch is established;
\t\t\t\t\t// only after it can a later OpPut be observed.
\t\t\t\t\tclose(watcherReady)
\t\t\t\t\tcontinue
\t\t\t\t}
\t\t\t\tif event.Type != types.OpPut {
\t\t\t\t\tcontinue
\t\t\t\t}'''

login_anchor = '''\terr = Run([]string{
\t\t"login",
\t\t"--insecure",
\t\t"--debug",
\t\t"--auth", connector.GetName(),
\t\t"--proxy", proxyAddr.String(),
\t\t"--user", "alice", // explicitly use wrong name
\t}, cliOption(func(cf *CLIConf) error {'''
login_replacement = '''\t// Logging in creates the access request; a watcher that registers afterwards never sees the
\t// OpPut and waits forever, taking the package to its -timeout. Measured gap between the watch
\t// going live and the login starting: under 1ms, so any scheduling hiccup can invert it.
\tselect {
\tcase <-watcherReady:
\tcase <-time.After(30 * time.Second):
\t\tt.Fatal("timed out waiting for access request watcher to be established")
\t}

''' + login_anchor

if 'watcherReady' in source:
    raise SystemExit(0)

# Both anchors must match exactly once. A half-applied rewrite would either declare a channel
# nobody closes (hanging every run) or wait on one that does not exist (a compile error).
counts = (source.count(watch_anchor), source.count(login_anchor))
if counts != (1, 1):
    print('teleport OIDC watcher fixture: anchors %s not in expected form; skipping' % (counts,))
    raise SystemExit(0)

source = source.replace(watch_anchor, watch_replacement, 1)
source = source.replace(login_anchor, login_replacement, 1)
path.write_text(source)
print('teleport OIDC watcher fixture: applied')
EOF_TELEPORT_OIDC_WATCHER
"""
_FLIPT_SELF_SIGNED_TLS_MOCK_SOURCE = r"""package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net"
	"net/http"
	"os"
	"time"
)

func main() {
	port := flag.Int("port", 443, "listen port")
	ready := flag.String("ready", "", "readiness marker path")
	flag.Parse()

	certificate, err := selfSignedCertificate()
	if err != nil {
		log.Fatal(err)
	}

	listener, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", *port))
	if err != nil {
		log.Fatal(err)
	}
	tlsListener := tls.NewListener(listener, &tls.Config{
		Certificates: []tls.Certificate{certificate},
		MinVersion:   tls.VersionTLS12,
	})

	if *ready != "" {
		if err := os.WriteFile(*ready, []byte("ready"), 0o600); err != nil {
			log.Fatal(err)
		}
	}

	server := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.NotFound(w, r)
		}),
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Fatal(server.Serve(tlsListener))
}

func selfSignedCertificate() (tls.Certificate, error) {
	privateKey, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, err
	}

	now := time.Now()
	template := x509.Certificate{
		SerialNumber: big.NewInt(now.UnixNano()),
		Subject: pkix.Name{
			CommonName: "self-signed.badssl.com",
		},
		DNSNames:              []string{"self-signed.badssl.com"},
		NotBefore:             now.Add(-time.Hour),
		NotAfter:              now.AddDate(10, 0, 0),
		KeyUsage:              x509.KeyUsageDigitalSignature,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
	}

	derBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		return tls.Certificate{}, err
	}
	keyBytes, err := x509.MarshalECPrivateKey(privateKey)
	if err != nil {
		return tls.Certificate{}, err
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: derBytes})
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes})
	return tls.X509KeyPair(certPEM, keyPEM)
}
"""


def _load_base_docker(iid):
    path = f'/tmp/swebench/{iid}'
    with open(f'{path}/baseDockerfile') as fp:
        return fp.read()


def _instance_docker(iid):
    path = f'/tmp/swebench/{iid}'
    with open(f'{path}/instanceDockerfile') as fp:
        return fp.read()


def _read_json_to_dict(file_path):
    try:
        with open(file_path, encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise ValueError(f'file not found: {file_path}') from None
    except json.JSONDecodeError as e:
        raise ValueError(f'json decode error: {e}') from e
    except Exception as e:
        raise ValueError(f'json read error: {e}') from e


def _parse_selected_tests(value: Any) -> list[str]:
    if value in (None, ''):
        return []
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f'invalid selected_test_files_to_run: {value!r}') from e
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    raise ValueError(f'invalid selected_test_files_to_run type: {type(value).__name__}')


def _needs_flipt_self_signed_tls_mock(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _FLIPT_SELF_SIGNED_SKIP_TLS_INSTANCE_ID
        and instance.get('repo') == 'flipt-io/flipt'
        and _parse_selected_tests(instance.get('selected_test_files_to_run')) == [_FLIPT_SELF_SIGNED_SKIP_TLS_TEST]
    )


def _needs_proxy_env_unset(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _PROXY_ENV_UNSET_INSTANCE_ID
        and instance.get('repo') == 'gravitational/teleport'
    )


def _needs_vuls_scandest_sort(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _VULS_SCANDEST_SORT_INSTANCE_ID
        and instance.get('repo') == 'future-architect/vuls'
    )


def _needs_teleport_tsh_flake_fixes(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _TELEPORT_TSH_FLAKE_INSTANCE_ID
        and instance.get('repo') == 'gravitational/teleport'
    )


def _needs_element_web_device_sort(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _ELEMENT_WEB_DEVICE_SORT_INSTANCE_ID
        and instance.get('repo') == 'element-hq/element-web'
    )


def _needs_element_web_join_rule_spaces(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _ELEMENT_WEB_JOIN_RULE_SPACES_INSTANCE_ID
        and instance.get('repo') == 'element-hq/element-web'
    )


def _needs_teleport_oidc_watcher(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() in _TELEPORT_OIDC_WATCHER_INSTANCE_IDS
        and instance.get('repo') == 'gravitational/teleport'
    )


def _needs_tutanota_rest_client_timeout(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _TUTANOTA_REST_CLIENT_TIMEOUT_INSTANCE_ID
        and instance.get('repo') == 'tutao/tutanota'
    )


def _needs_pytest_temproot_isolation(instance: SWEbenchInstance) -> bool:
    return (
        instance.get('instance_id', '').lower() == _PYTEST_TEMPROOT_ISOLATION_INSTANCE_ID
        and instance.get('repo') == 'ansible/ansible'
    )


def _needs_go_module_proxy(instance: SWEbenchInstance) -> bool:
    if not _is_go_instance(instance):
        return False
    return bool(_GO_MOD_PATCH_RE.search(instance.get('patch') or ''))


def _build_eval_workspace_files(instance: SWEbenchInstance, path: str) -> list[FileItem]:
    file_items = [
        FileItem(
            path=f'{path}/{file_path}',
            content=content,
        )
        for file_path, content in (instance.get('spec_dict') or {}).items()
    ]
    if _needs_flipt_self_signed_tls_mock(instance):
        file_items.append(
            FileItem(
                path=f'{path}/flipt_self_signed_tls_mock.go',
                content=_FLIPT_SELF_SIGNED_TLS_MOCK_SOURCE,
            )
        )
    file_items.extend(nodebb_registry_fixture.get_nodebb_registry_fixture_files(instance, path))
    return file_items


def _flipt_self_signed_tls_mock_preamble(path: str) -> str:
    return f"""
# SWE-bench-pro flipt local self-signed TLS fixture
set -e
flipt_tls_hosts_backup={path}/hosts.before-flipt-tls-mock
flipt_tls_ready={path}/flipt-self-signed-tls.ready
flipt_tls_log={path}/flipt-self-signed-tls.log
cp /etc/hosts "$flipt_tls_hosts_backup"

cleanup_flipt_tls_mock() {{
  if [ -n "${{flipt_tls_mock_pid:-}}" ]; then
    kill "$flipt_tls_mock_pid" 2>/dev/null || true
    wait "$flipt_tls_mock_pid" 2>/dev/null || true
  fi
  if [ -f "$flipt_tls_hosts_backup" ]; then
    cat "$flipt_tls_hosts_backup" > /etc/hosts 2>/dev/null || true
  fi
}}
trap cleanup_flipt_tls_mock EXIT

printf '\\n127.0.0.1 self-signed.badssl.com\\n' >> /etc/hosts
export NO_PROXY="${{NO_PROXY:+${{NO_PROXY}},}}self-signed.badssl.com"
export no_proxy="${{no_proxy:+${{no_proxy}},}}self-signed.badssl.com"
rm -f "$flipt_tls_ready" "$flipt_tls_log"
go build -o {path}/flipt_self_signed_tls_mock {path}/flipt_self_signed_tls_mock.go
{path}/flipt_self_signed_tls_mock \
  --port 443 \
  --ready "$flipt_tls_ready" > "$flipt_tls_log" 2>&1 &
flipt_tls_mock_pid=$!

for _ in {{1..100}}; do
  if [ -s "$flipt_tls_ready" ]; then
    break
  fi
  if ! kill -0 "$flipt_tls_mock_pid" 2>/dev/null; then
    cat "$flipt_tls_log" >&2
    exit 1
  fi
  sleep 0.05
done
if [ ! -s "$flipt_tls_ready" ]; then
  cat "$flipt_tls_log" >&2
  exit 1
fi
"""


def _is_go_instance(instance: SWEbenchInstance) -> bool:
    return (instance.get('repo_language') or instance.get('language') or '').lower() == 'go'


def _parse_go_test_statuses(*logs: str) -> list[tuple[str, str]]:
    statuses = []
    for log in logs:
        for match in _GO_TEST_LINE_RE.finditer(log or ''):
            raw_status, test_name = match.groups()
            statuses.append((test_name.strip(), _GO_STATUS_MAP[raw_status]))
    return statuses


def _tail_log(log: str) -> str:
    if len(log) <= _RAW_LOG_TAIL_CHARS:
        return log
    return log[-_RAW_LOG_TAIL_CHARS:]


def _attach_raw_logs(enriched: dict[str, Any], stdout_log: str, stderr_log: str) -> dict[str, Any]:
    raw_logs = {}
    if stdout_log:
        raw_logs['stdout_tail'] = _tail_log(stdout_log)
    if stderr_log:
        raw_logs['stderr_tail'] = _tail_log(stderr_log)
    if raw_logs:
        enriched['pro_raw_logs'] = raw_logs
    return enriched


def _enrich_test_output(
    instance: SWEbenchInstance, test_output: dict[str, Any], stdout_log: str, stderr_log: str
) -> dict[str, Any]:
    if not _is_go_instance(instance):
        return _attach_raw_logs(copy.deepcopy(test_output), stdout_log, stderr_log)

    go_statuses = _parse_go_test_statuses(stdout_log, stderr_log)
    if not go_statuses:
        return _attach_raw_logs(copy.deepcopy(test_output), stdout_log, stderr_log)

    original_tests = test_output.get('tests', [])
    parser_statuses_by_name: dict[str, set[str]] = {}
    for test in original_tests:
        name = test.get('name')
        status = test.get('status')
        if name and status:
            parser_statuses_by_name.setdefault(name, set()).add(status)

    raw_statuses_by_name: dict[str, set[str]] = {}
    for name, status in go_statuses:
        raw_statuses_by_name.setdefault(name, set()).add(status)

    additions = []
    status_priority = ('PASSED', 'FAILED', 'SKIPPED')
    for name, raw_statuses in raw_statuses_by_name.items():
        parser_statuses = parser_statuses_by_name.get(name, set())
        if 'PASSED' in parser_statuses:
            continue
        aggregate_status = next(status for status in status_priority if status in raw_statuses)
        if aggregate_status == 'PASSED' or not parser_statuses:
            additions.append({'name': name, 'status': aggregate_status})

    has_failed_status = any(status == 'FAILED' for _, status in go_statuses)
    if not (has_failed_status or additions):
        return _attach_raw_logs(copy.deepcopy(test_output), stdout_log, stderr_log)

    enriched = copy.deepcopy(test_output)
    enriched.setdefault('tests', []).extend(additions)
    enriched['pro_parser_fallback'] = {
        'go_test_statuses': len(go_statuses),
    }
    return _attach_raw_logs(enriched, stdout_log, stderr_log)


def get_eval_script(instance: SWEbenchInstance) -> str:
    before_repo_set_cmd_raw = instance.get('before_repo_set_cmd')
    before_repo_set_cmd = '' if before_repo_set_cmd_raw is None else before_repo_set_cmd_raw.strip().split('\n')[-1]

    selected_tests = _parse_selected_tests(instance.get('selected_test_files_to_run'))
    selected_test_files_to_run = ','.join(selected_tests)

    base_commit = instance['base_commit']
    base_dockerfile = _load_base_docker(instance['instance_id'].lower())
    instance_dockerfile = _instance_docker(instance['instance_id'].lower())

    # Extract ENV commands from dockerfiles
    env_cmds = []
    for dockerfile_content in [base_dockerfile, instance_dockerfile]:
        for line in dockerfile_content.split('\n'):
            line = line.strip()
            if line.startswith('ENV'):
                # Convert ENV commands to export statements
                env_cmd = line.replace('ENV', 'export', 1)
                env_cmds.append(env_cmd)

    # Cap Go linker parallelism on CGO-heavy repos (vuls/flipt/teleport): observed
    # Go 1.24 linker SIGSEGV in deadcode/OutBuf.WriteSym/memmove under 16-CPU
    # concurrent `go test ./...`. GOMAXPROCS=4 trades a bit of wall-time for
    # stable linking.
    if (instance.get('repo_language') or instance.get('language') or '').lower() == 'go':
        env_cmds.append('export GOMAXPROCS=4')

    if _needs_go_module_proxy(instance):
        env_cmds.append(f'export GOPROXY={shlex.quote(_GO_MODULE_PROXY)}')

    if _needs_pytest_temproot_isolation(instance):
        # Keyed on instance_id so concurrent copies of other instances cannot collide here either.
        # mkdir first: pytest only does mkdir(exist_ok=True) on <temproot>/pytest-of-<user>, without
        # parents, so a missing temproot fails the run before any test executes.
        temproot = f'/tmp/pytest-temproot-{instance["instance_id"].lower()}'
        env_cmds.append(f'mkdir -p {shlex.quote(temproot)}')
        env_cmds.append(f'export PYTEST_DEBUG_TEMPROOT={shlex.quote(temproot)}')

    env_cmds = '\n'.join(env_cmds)

    path = f'/tmp/swebench/{instance["instance_id"].lower()}'
    if _needs_flipt_self_signed_tls_mock(instance):
        eval_preamble = _flipt_self_signed_tls_mock_preamble(path)
    elif _needs_proxy_env_unset(instance):
        eval_preamble = _PROXY_ENV_UNSET_PREAMBLE
    elif _needs_vuls_scandest_sort(instance):
        eval_preamble = _VULS_SCANDEST_SORT_PREAMBLE
    elif _needs_teleport_tsh_flake_fixes(instance):
        eval_preamble = _TELEPORT_TSH_FLAKE_PREAMBLE
    elif _needs_element_web_device_sort(instance):
        eval_preamble = _ELEMENT_WEB_DEVICE_SORT_PREAMBLE
    elif _needs_element_web_join_rule_spaces(instance):
        eval_preamble = _ELEMENT_WEB_JOIN_RULE_SPACES_PREAMBLE
    else:
        eval_preamble = ''
    # Appended rather than another `elif`: db89206d already claims the tsh-flake branch above, so an
    # `elif` would leave it with the flakes it does not have and without the one it does. The
    # fixtures edit disjoint regions of tsh_test.go (makeTestServers' DataDir vs TestOIDCLogin's
    # watcher), so order does not matter; each is independently idempotent and anchor-guarded.
    if _needs_teleport_oidc_watcher(instance):
        eval_preamble += _TELEPORT_OIDC_WATCHER_PREAMBLE
    # Appended for the same reason: nothing above claims this instance today, but an `elif` would
    # silently stop applying the moment something did.
    if _needs_tutanota_rest_client_timeout(instance):
        eval_preamble += _TUTANOTA_REST_CLIENT_TIMEOUT_PREAMBLE
    if _needs_pytest_temproot_isolation(instance):
        eval_preamble += _PYTEST_TEMPROOT_ISOLATION_PREAMBLE
    element_web_wasm_commands = element_web_wasm_fixture.get_element_web_wasm_fixture_test_commands(
        instance,
        path,
        selected_tests,
    )
    if nodebb_registry_fixture.needs_nodebb_registry_fixture(instance):
        test_commands = nodebb_registry_fixture.get_nodebb_registry_fixture_test_commands(
            instance,
            path,
        )
    elif element_web_wasm_commands:
        test_commands = element_web_wasm_commands
    else:
        test_commands = f"""
# run test and save stdout and stderr to separate files
bash {path}/run_script.sh {selected_test_files_to_run} > {path}/stdout.log 2> {path}/stderr.log
# run parsing script
python {path}/parser.py {path}/stdout.log {path}/stderr.log {path}/test_output.txt
""".strip()

    return f"""
{env_cmds}
# apply patch
cd /app
#git reset --hard {base_commit}
#git checkout {base_commit}
#git apply -v {path}/patch.diff
{before_repo_set_cmd}
{eval_preamble}
{test_commands}
"""


def _fuzzy_match_in(name: str, passed_tests: set[str]) -> bool:
    """Check if a gold label name matches any passed test, with fuzzy fallback.

    Handles known encoding issues where gold labels stored as Python repr()
    inside JSON lose trailing characters (typically a '"') due to escaping.
    """
    if name in passed_tests:
        return True
    stripped = name.strip()
    if stripped != name and stripped in passed_tests:
        return True
    for actual in passed_tests:
        if len(actual) - len(stripped) in (1, 2) and actual.startswith(stripped):
            logger.debug(f'Fuzzy matched gold label: {name!r} → {actual!r}')
            return True
        if len(stripped) - len(actual) in (1, 2) and stripped.startswith(actual):
            logger.debug(f'Fuzzy matched gold label: {name!r} → {actual!r}')
            return True
    return False


def _parse_gold_tests(value: Any) -> list[str]:
    if value in (None, ''):
        return []
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, list | tuple):
        return [str(item) for item in value]
    if isinstance(value, set):
        return sorted(str(item) for item in value)
    return [str(value)]


def _build_test_group(test_names: list[str], passed_tests: set[str]) -> dict[str, list[str]]:
    success = []
    failure = []
    for name in test_names:
        if _fuzzy_match_in(name, passed_tests):
            success.append(name)
        else:
            failure.append(name)
    return {
        'success': success,
        'failure': failure,
    }


def get_eval_report_map(
    instance: SWEbenchInstance, prediction: dict[str, str], test_log_path: str, include_tests_status: bool
) -> dict[str, Any]:
    logger.debug(f'get_eval_report_map: {prediction}, {test_log_path}, {include_tests_status}')

    test_output = _read_json_to_dict(test_log_path)
    logger.debug(f'get_eval_report_map: test_output {test_output}')

    passed_tests = {x['name'] for x in test_output['tests'] if x['status'] == 'PASSED'}
    f2p = _parse_gold_tests(instance.get('FAIL_TO_PASS', []))
    p2p = _parse_gold_tests(instance.get('PASS_TO_PASS', []))
    result = all(_fuzzy_match_in(name, passed_tests) for name in f2p + p2p)
    logger.info(f'result: {result}')
    expected_tests = set(f2p + p2p)
    diff1 = {name for name in expected_tests if not _fuzzy_match_in(name, passed_tests)}
    logger.debug(f'diff1: {diff1}')
    diff2 = passed_tests - expected_tests
    logger.debug(f'diff2: {diff2}')
    report = {
        'resolved': result,
    }
    if include_tests_status:
        report['tests_status'] = {
            'FAIL_TO_PASS': _build_test_group(f2p, passed_tests),
            'PASS_TO_PASS': _build_test_group(p2p, passed_tests),
        }
    return {instance['instance_id']: report}


def get_repo_language(instance: SWEbenchInstance) -> str | None:
    repo_language_map = {
        'go': 'go',
        'js': 'js',
        'ts': 'ts',
        'python': 'py',
    }
    language = instance.get('language', '')  # need to translate to standard
    if language:
        return repo_language_map[language]
    language = instance.get('repo_language', '')
    if language:
        return repo_language_map[language]
    raise ValueError('get_repo_language error, not found language')


async def infer_runtime_init(
    instance: SWEbenchInstance, runtime: Runtime, *, skip_git_truncate: bool = False, **kwargs: Any
) -> None:
    """Initialize runtime for SWE-bench-pro inference.

    SWE-bench-pro images need to reset to base commit before inference.
    """
    from nexus.extensions.tasks.swe.infer import run_git_truncate

    base_commit = instance['base_commit']
    working_dir = instance.get('workdir', '/app')

    init_cmds = [
        f'git reset --hard {base_commit}',
        f'git checkout {base_commit}',
    ]

    logger.debug('Running infer_runtime_init for swe_bench_pro')

    log_output = await run_init_cmds_in_runtime(runtime, init_cmds, working_dir=working_dir)
    logger.debug('SWE-bench-pro init commands executed:\n%s', log_output)

    if kwargs.get('remove_git_dir', False):
        from nexus.extensions.tasks.swe.infer import remove_git_dir

        await remove_git_dir(runtime, working_dir)
        return

    remove_remotes_cmd = "git remote remove origin && rm -rf .git/refs/remotes && sed -i '/refs\\/remotes\\//d' .git/packed-refs 2>/dev/null || true"
    await runtime.call_executor('run_command', {'command': remove_remotes_cmd, 'cwd': working_dir})

    if not skip_git_truncate:
        await run_git_truncate(runtime, base_commit, working_dir)


def get_problem_statement(instance: SWEbenchInstance) -> str:
    def _load_json_value(value: str) -> str:
        if not value.startswith('"'):
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    problem_statement = instance.get('problem_statement') or instance.get('problem') or ''
    if isinstance(problem_statement, str):
        problem_statement = _load_json_value(problem_statement)
    requirements = instance.get('requirements', '')
    if isinstance(requirements, str):
        requirements = _load_json_value(requirements)
    interface = instance.get('interface', '')
    if isinstance(interface, str):
        interface = _load_json_value(interface)
    return f"""{problem_statement}

Requirements:
{requirements}

New interfaces introduced:
{interface}""".strip()


async def eval_runtime_init(instance: SWEbenchInstance, runtime: Runtime) -> None:
    path = f'/tmp/swebench/{instance["instance_id"].lower()}'

    mkdir_result = await runtime.run_command(command=f'mkdir -p {path}', timeout=60)
    logger.debug(f'Mkdir {path} result: {mkdir_result}')

    logger.debug(f'instance.spec_dict {instance.get("spec_dict") or {}}')
    file_items = _build_eval_workspace_files(instance, path)

    logger.debug(f'Uploading {len(file_items)} workspace files to runtime...')
    upload_result = await runtime.upload_file(files=file_items, overwrite=True)
    logger.debug(f'Upload workspace files result: {upload_result}')

    return


def _read_log(file_path):
    try:
        with open(file_path, encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise ValueError(f'file not found: {file_path}') from None
    except Exception as e:
        raise ValueError(f'read error: {e}') from e


def _read_first_existing(paths: list[Path]) -> str:
    errors = []
    for file_path in paths:
        try:
            return _read_log(file_path)
        except Exception as e:
            errors.append(f'{file_path}: {e}')
    raise ValueError('; '.join(errors))


def _read_first_existing_json_content(paths: list[Path]) -> tuple[str, dict[str, Any]]:
    errors = []
    for file_path in paths:
        try:
            content = _read_log(file_path)
            return content, json.loads(content)
        except Exception as e:
            errors.append(f'{file_path}: {e}')
    raise ValueError('; '.join(errors))


def eval_write_test_output(instance: SWEbenchInstance, path: Path, stdout: str | None) -> None:
    runtime_log_path = Path(f'/tmp/swebench/{instance["instance_id"].lower()}')
    adjacent_log_path = path.parent

    stdout_log = ''
    try:
        stdout_log = _read_first_existing([runtime_log_path / 'stdout.log', adjacent_log_path / 'stdout.log'])
        logger.debug(f'stdout.log: {stdout_log}')
    except Exception as e:
        logger.info(f'read stdout.log error: {e}')

    stderr_log = ''
    try:
        stderr_log = _read_first_existing([runtime_log_path / 'stderr.log', adjacent_log_path / 'stderr.log'])
        logger.debug(f'stderr.log: {stderr_log}')
    except Exception as e:
        logger.info(f'read stderr.log error: {e}')

    test_output_content, test_output = _read_first_existing_json_content([runtime_log_path / 'test_output.txt', path])
    enriched = _enrich_test_output(instance, test_output, stdout_log, stderr_log)
    enriched = nodebb_registry_fixture.enrich_nodebb_registry_test_output(instance, enriched, runtime_log_path)
    if enriched is test_output:
        path.write_text(test_output_content, encoding='utf-8')
    else:
        path.write_text(json.dumps(enriched), encoding='utf-8')

    return
