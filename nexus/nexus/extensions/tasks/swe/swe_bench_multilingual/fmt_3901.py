"""Exact-case repairs for fmtlib__fmt-3901.

Two independent defects in this instance's own test harness make it fail intermittently -- 3 of 17
oracle canary batches. Both take down the whole ``format-test`` binary, so every test after the
faulty one is reported as failed, including ``format_test.group_digits_view``, the only FAIL_TO_PASS
entry. The report therefore blames group_digits_view even though it never runs.

1. ``format_test.line_buffering`` races. Its reader thread calls ``cv.notify_one()`` without holding
   the mutex while the main thread waits with the predicate-less ``wait_for``, so a reader that
   finishes first -- easy, the pipe already holds "42\n" before the thread starts -- loses its
   notification and the wait burns its full 60s. ``ASSERT_EQ`` on the timeout then makes gtest
   ``return`` before ``reader.join()``, destroying a joinable ``std::thread``, which calls
   ``std::terminate()``. Measured in the instance image: the shipped pattern lost the wakeup 1999
   times out of 2000, and an unjoined thread exits with code 134.

2. ``output_redirect`` (test/gtest-extra.cc) deadlocks on a downgraded pipe. It backs every
   ``EXPECT_WRITE(file, ...)`` by dup2()ing a fresh pipe over the target fd and only draining it
   afterwards in ``restore_and_read()``; its own header admits "The output it can handle is limited
   by the pipe capacity". ``format_test.big_print`` writes 5000 bytes that way, which fits the usual
   65536-byte pipe. But the kernel silently downgrades new pipes to a single 4096-byte page once a
   user exceeds ``fs.pipe-user-pages-soft`` (16384 pages here), which 50 concurrent sandboxes on one
   host do reach -- reproduced in the image: of 1200 pipes opened in a loop the last 240 came back
   at 4096. At that size big_print blocks in write() *inside* the statement, before
   ``restore_and_read()`` can run, so stdout is never restored and the process wedges forever.
   Confirmed on a wedged sandbox: fd 1 and fd 5 pointed at one pipe inode held by no other process,
   ``queued=4096 capacity=4096``, and 11932 of 11992 output bytes written -- stuck on the final
   "[ PASSED ] N tests." line. ``F_SETPIPE_SZ`` cannot grow it back; past the soft limit the kernel
   returns EPERM.

Repair 1 publishes the flag under the mutex, waits on the predicate overload, and joins before
asserting. Repair 2 drains the pipe from a helper thread started in the constructor, so the writer
never blocks regardless of capacity; the captured bytes and their ordering are unchanged for every
caller. A/B check under a forced 4096-byte pipe: unpatched wedges until timeout, patched passes.

Both edits touch test-harness plumbing only, never the library code under grading, and both skip
silently when their anchor is missing. Each stamps the file's mtime forward: ``git checkout``
restores the original mtime, which can predate an existing object file and let cmake skip the
rebuild, silently dropping the repair.
"""

from nexus.extensions.tasks.swe.constants import SWEbenchInstance


INSTANCE_ID = 'fmtlib__fmt-3901'
REPO = 'fmtlib/fmt'
DATASET_TYPE = 'swe-bench-multilingual'

# Harness sources patched below. They are outside ``test_patch``, so the caller restores them.
_PATCHED_HARNESS_FILES = ('test/gtest-extra.h', 'test/gtest-extra.cc')


_OUTPUT_REDIRECT_REPAIR = r"""python3 - <<'EOF_FMT_3901_OUTPUT_REDIRECT'
import os
import sys
import time
from pathlib import Path

header = Path('test/gtest-extra.h')
source_file = Path('test/gtest-extra.cc')
try:
    header_text = header.read_text()
    source_text = source_file.read_text()
except OSError:
    print('fmt-3901 output_redirect repair: files not found; skipping', file=sys.stderr)
    raise SystemExit(0)

if 'NEXUS_FMT_3901_PIPE_DRAIN' in header_text:
    raise SystemExit(0)

header_needle = '''class output_redirect {
 private:
  FILE* file_;
  fmt::file original_;  // Original file passed to redirector.
  fmt::file read_end_;  // Read end of the pipe where the output is redirected.

  void flush();
  void restore();
'''

header_replacement = '''class output_redirect {
 private:
  FILE* file_;
  fmt::file original_;  // Original file passed to redirector.
  fmt::file read_end_;  // Read end of the pipe where the output is redirected.
  // NEXUS_FMT_3901_PIPE_DRAIN: drain the pipe on a helper thread so the redirected writer never
  // blocks when the kernel hands out a 4096-byte pipe (which it does once
  // fs.pipe-user-pages-soft is exceeded). Without this, writes larger than the capacity wedge
  // inside the statement under test and stdout is never restored.
  std::thread drain_thread_;
  std::string drained_;

  void flush();
  void restore();
  void start_drain();
  void finish_drain();
'''

if header_needle not in header_text:
    print('fmt-3901 output_redirect repair: header not in expected form; skipping', file=sys.stderr)
    raise SystemExit(0)

header_text = header_text.replace(header_needle, header_replacement, 1)

# The helper thread needs <thread> and <string>; <string> is already pulled in transitively but
# being explicit costs nothing and keeps the edit self-contained.
include_anchor = '#if FMT_USE_FCNTL'
if '#include <thread>' not in header_text:
    header_text = header_text.replace(
        include_anchor, '#include <string>\n#include <thread>\n\n' + include_anchor, 1
    )

source_needle = '''output_redirect::output_redirect(FILE* f, bool flush) : file_(f) {
  if (flush) this->flush();
  int fd = FMT_POSIX(fileno(f));
  // Create a file object referring to the original file.
  original_ = file::dup(fd);
  // Create a pipe.
  auto pipe = fmt::pipe();
  read_end_ = std::move(pipe.read_end);
  // Connect the passed FILE object to the write end of the pipe.
  pipe.write_end.dup2(fd);
}
'''

source_replacement = '''output_redirect::output_redirect(FILE* f, bool flush) : file_(f) {
  if (flush) this->flush();
  int fd = FMT_POSIX(fileno(f));
  // Create a file object referring to the original file.
  original_ = file::dup(fd);
  // Create a pipe.
  auto pipe = fmt::pipe();
  read_end_ = std::move(pipe.read_end);
  // Connect the passed FILE object to the write end of the pipe.
  pipe.write_end.dup2(fd);
  // NEXUS_FMT_3901_PIPE_DRAIN: begin draining immediately; see the header for why.
  start_drain();
}

void output_redirect::start_drain() {
  drain_thread_ = std::thread([this]() {
    enum { BUFFER_SIZE = 4096 };
    char buffer[BUFFER_SIZE];
    for (;;) {
      std::size_t count = 0;
      try {
        count = read_end_.read(buffer, BUFFER_SIZE);
      } catch (const std::exception&) {
        return;  // Pipe closed or errored; nothing more to collect.
      }
      if (count == 0) return;  // EOF: the write end is closed.
      drained_.append(buffer, count);
    }
  });
}

void output_redirect::finish_drain() {
  if (drain_thread_.joinable()) drain_thread_.join();
}
'''

if source_needle not in source_text:
    print('fmt-3901 output_redirect repair: ctor not in expected form; skipping', file=sys.stderr)
    raise SystemExit(0)

source_text = source_text.replace(source_needle, source_replacement, 1)

read_needle = '''std::string output_redirect::restore_and_read() {
  // Restore output.
  restore();

  // Read everything from the pipe.
  std::string content;
  if (read_end_.descriptor() == -1) return content;  // Already read.
  enum { BUFFER_SIZE = 4096 };
  char buffer[BUFFER_SIZE];
  size_t count = 0;
  do {
    count = read_end_.read(buffer, BUFFER_SIZE);
    content.append(buffer, count);
  } while (count != 0);
  read_end_.close();
  return content;
}
'''

read_replacement = '''std::string output_redirect::restore_and_read() {
  // NEXUS_FMT_3901_PIPE_DRAIN: restore the fd first, which drops this object's reference to the
  // pipe's write end. The drain thread then sees EOF and finishes; joining it yields everything
  // that was written, with no dependence on the pipe's capacity.
  restore();
  finish_drain();
  std::string content = std::move(drained_);
  drained_.clear();
  if (read_end_.descriptor() != -1) read_end_.close();
  return content;
}
'''

if read_needle not in source_text:
    print('fmt-3901 output_redirect repair: restore_and_read not in expected form; skipping',
          file=sys.stderr)
    raise SystemExit(0)

source_text = source_text.replace(read_needle, read_replacement, 1)

dtor_needle = '''output_redirect::~output_redirect() noexcept {
  try {
    restore();
  } catch (const std::exception& e) {
    std::fputs(e.what(), stderr);
  }
}
'''

dtor_replacement = '''output_redirect::~output_redirect() noexcept {
  try {
    restore();
  } catch (const std::exception& e) {
    std::fputs(e.what(), stderr);
  }
  // NEXUS_FMT_3901_PIPE_DRAIN: the reader must be joined even when restore_and_read() was never
  // called (e.g. an assertion failed mid-statement), otherwise a joinable std::thread is
  // destroyed and std::terminate() takes down the whole binary.
  try {
    if (read_end_.descriptor() != -1) read_end_.close();
  } catch (const std::exception&) {
  }
  finish_drain();
}
'''

if dtor_needle not in source_text:
    print('fmt-3901 output_redirect repair: dtor not in expected form; skipping', file=sys.stderr)
    raise SystemExit(0)

source_text = source_text.replace(dtor_needle, dtor_replacement, 1)

header.write_text(header_text)
source_file.write_text(source_text)
# Stamp both files into the future relative to any existing object file: git checkout restores the
# original mtime, so cmake would otherwise consider the stale .o up to date and skip the rebuild.
now = time.time() + 1
for path in (header, source_file):
    os.utime(path, (now, now))
print('fmt-3901 output_redirect repair: applied')
EOF_FMT_3901_OUTPUT_REDIRECT"""

_LINE_BUFFERING_REPAIR = r"""python3 - <<'EOF_FMT_3901_LINE_BUFFERING'
import os
import sys
import time
from pathlib import Path

path = Path('test/format-test.cc')
try:
    source = path.read_text()
except OSError:
    print('fmt-3901 line_buffering repair: format-test.cc not found; skipping', file=sys.stderr)
    raise SystemExit(0)

if 'NEXUS_FMT_3901_RACE_REPAIR' in source:
    raise SystemExit(0)

needle = '''  std::mutex mutex;
  std::condition_variable cv;
  auto read_end = pipe.read_end.fdopen("r");
  std::thread reader([&]() {
    int n = 0;
    int result = fscanf(read_end.get(), "%d", &n);
    (void)result;
    EXPECT_EQ(n, 42);
    cv.notify_one();
  });

  std::unique_lock<std::mutex> lock(mutex);
  ASSERT_EQ(cv.wait_for(lock, std::chrono::minutes(1)),
            std::cv_status::no_timeout);
  reader.join();
'''

replacement = '''  // NEXUS_FMT_3901_RACE_REPAIR: notify under the mutex and wait on the predicate overload so an
  // early notification is not lost, and join before asserting so a failed assertion cannot
  // destroy a joinable thread (which would std::terminate and abort the whole binary).
  std::mutex mutex;
  std::condition_variable cv;
  bool ready = false;
  auto read_end = pipe.read_end.fdopen("r");
  std::thread reader([&]() {
    int n = 0;
    int result = fscanf(read_end.get(), "%d", &n);
    (void)result;
    EXPECT_EQ(n, 42);
    {
      std::lock_guard<std::mutex> guard(mutex);
      ready = true;
    }
    cv.notify_one();
  });

  bool signaled = false;
  {
    std::unique_lock<std::mutex> lock(mutex);
    signaled = cv.wait_for(lock, std::chrono::minutes(1), [&] { return ready; });
  }
  reader.join();
  ASSERT_TRUE(signaled);
'''

if needle not in source:
    print('fmt-3901 line_buffering repair: test not in expected form; skipping', file=sys.stderr)
    raise SystemExit(0)

path.write_text(source.replace(needle, replacement, 1))
# Stamp forward so cmake cannot consider a pre-existing .o up to date (git checkout restores the
# original mtime, which may predate the object file).
now = time.time() + 1
os.utime(path, (now, now))
print('fmt-3901 line_buffering repair: applied')
EOF_FMT_3901_LINE_BUFFERING"""


def is_target_instance(instance: SWEbenchInstance) -> bool:
    """Return whether these exact-case repairs apply."""
    return (
        instance.get('instance_id') == INSTANCE_ID
        # Compare case-insensitively so a re-cased export cannot silently disable the repairs.
        and (instance.get('repo') or '').lower() == REPO
        and instance.get('dataset_type') == DATASET_TYPE
    )


def make_setup_commands() -> list[str]:
    """Return the repairs, to run after the test patch and before the build."""
    return [
        '# Nexus exact-case race repairs for fmt-3901',
        _LINE_BUFFERING_REPAIR,
        _OUTPUT_REDIRECT_REPAIR,
    ]


def make_cleanup_commands() -> list[str]:
    """Return commands restoring the harness files patched by :func:`make_setup_commands`.

    ``test/format-test.cc`` is already reset by the caller because it appears in ``test_patch``;
    these two do not, so grading would otherwise see a modified tree.
    """
    return [f'git checkout -- {" ".join(_PATCHED_HARNESS_FILES)} || true']
