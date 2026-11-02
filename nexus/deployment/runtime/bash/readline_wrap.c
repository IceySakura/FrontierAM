/**
 * @file readline_wrap.c
 * @brief Link-time wrapper for readline() that maintains capture batches.
 *
 * The patched bash binary is linked with --wrap=readline, so calls to
 * readline() enter __wrap_readline() first and can then call __real_readline().
 *
 * This wrapper has two responsibilities:
 * - initialize the capture session on the first interactive readline call;
 * - advance the shared batch counter when a later readline call appears to be
 *   the start of a new user input burst.
 *
 * Batch detection is intentionally small and conservative. Before calling the
 * real readline(), the wrapper polls stdin for a short time:
 * - timeout: stdin currently has no queued bytes, so this readline likely starts
 *   a new user submission and batch is incremented;
 * - readable: more bytes are already queued, so this readline likely continues
 *   the same submission and batch is kept unchanged;
 * - poll error: keep the current batch rather than accidentally splitting a
 *   single submitted burst.
 */
#define _GNU_SOURCE
#include <poll.h>
#include <unistd.h>

#include "bc_common.h"

/**
 * @brief Short stdin poll used to group quickly submitted lines into one batch.
 *
 * A value in the 1-5ms range avoids most split writes when the Python side feeds
 * multiple lines at once, without adding user-visible latency to interactive
 * reads.
 */
#define BC_POLL_TIMEOUT_MS 2

/**
 * @brief Real readline() symbol provided by the linker because bash is linked
 * with --wrap=readline.
 */
extern char *__real_readline(const char *prompt);

/** @brief Whether the interactive capture session has been initialized. */
static int bc_session_started = 0;

/**
 * @brief Wrapped readline() entry point.
 *
 * @param prompt Prompt string forwarded to the real readline().
 * @return The line returned by the real readline(), preserving readline's
 * ownership and NULL-on-EOF semantics.
 */
char *__wrap_readline(const char *prompt) {
    if (!bc_session_started) {
        bc_session_started = 1;
        bc_reset();
        bc_bump_batch();
        bc_write_status("idle", 0);
    } else {
        struct pollfd pfd = { .fd = STDIN_FILENO, .events = POLLIN };
        int pollret = poll(&pfd, 1, BC_POLL_TIMEOUT_MS);
        if (pollret == 0)
            bc_bump_batch();
    }

    return __real_readline(prompt);
}
