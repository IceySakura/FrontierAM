/**
 * @file cmd_wrap.c
 * @brief Link-time wrapper for Bash execute_command() with stdout/stderr capture.
 *
 * The patched bash binary is linked with --wrap=execute_command. This file
 * intercepts the outermost execute_command() call for each interactive command,
 * assigns a command_id, emits command metadata, and captures foreground command
 * stdout/stderr through pipes.
 *
 * Capture model:
 * - foreground commands: hijack fd 1 and fd 2 to pipes, run the real Bash
 *   command, stream reader chunks to stdout_<id>, stderr_<id>, and output_<id>,
 *   then emit exit metadata;
 * - background commands: emit command metadata and empty data files, execute
 *   without fd hijacking, then finalize immediately. This avoids waiting forever
 *   for async descendants that inherit capture pipe write ends.
 *
 * The skip-capture detector is best-effort and intentionally biased against
 * false positives. Misclassifying a foreground command as async would drop its
 * captured output, so the detector only matches high-confidence top-level async
 * Bash AST shapes.
 */
#include "config.h"
#include <sys/types.h>
#include "command.h"
#include "general.h"
#include "jobs.h"
#include "unwind_prot.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <errno.h>
#include <stdint.h>
#include <signal.h>

#include "bc_common.h"

/**
 * @brief Real execute_command() symbol provided by --wrap=execute_command.
 */
extern int __real_execute_command(COMMAND *command);

/**
 * @brief Bash printer used to reconstruct normalized command text.
 */
extern char *make_command_string(COMMAND *command);

/** @brief Bash's current command status, maintained by execute_cmd.c/sig.c. */
extern volatile int last_command_exit_value;

/**
 * @section bash_ast_dependency Bash AST dependency
 *
 * cmd_wrap.c is compiled inside Bash's configured source tree and includes
 * command.h directly. The code still depends on Bash's internal AST, but the
 * struct layouts, enum values, and flags come from Bash headers instead of
 * local ABI copies. Bash upgrades that change these internals should therefore
 * fail at compile time rather than silently reading the wrong layout.
 */

/**
 * @brief Best-effort check for commands that should not be fd-hijacked.
 *
 * The detector favors false negatives over false positives. A false positive
 * would treat a normal foreground command as background and drop captured
 * output. A false negative may hang until timeout/session cleanup if an async
 * descendant keeps a pipe write end open, but only in less common corner cases.
 *
 * High-confidence top-level matches:
 * - CMD_AMPERSAND on the submitted command node, for example sleep 1 &;
 * - a top-level connection connector '&' with no foreground successor;
 * - top-level cm_coproc.
 *
 * The detector intentionally does not recursively scan descendants. A command
 * list such as sleep 1 & echo done, or an unexecuted branch such as
 * if false; then sleep 1 & fi; echo done, contains an async child but also has
 * foreground work that must be captured. Returning skip-capture for those mixed
 * shapes would drop user-visible stdout/stderr, which is worse than a false
 * negative on rare async corner cases.
 *
 * Process substitution, for example cat <(cmd), is a foreground command from
 * the user's perspective and must remain captured. It is intentionally not a
 * skip condition even though Bash implements it with an asynchronous helper.
 *
 * @param command Bash COMMAND node, or NULL.
 * @return 1 to skip stdout/stderr capture; 0 to capture as foreground.
 */
static int bc_command_is_top_level_async(COMMAND *command) {
    COMMAND *c = command;
    if (!c) return 0;
    if (c->flags & CMD_AMPERSAND) return 1;     /* This node is `X &`. */
    switch (c->type) {
        case cm_connection: {
            CONNECTION *cn = c->value.Connection;
            if (!cn) return 0;
            return cn->connector == '&' && !cn->second; /* Top-level `X &`. */
        }
        case cm_coproc:
            return 1;                          /* coproc is always asynchronous. */
        default:
            return 0;
    }
}

/**
 * @brief Decide whether one top-level command should use the skip-capture path.
 *
 * @param command Bash COMMAND node for the top-level execute_command call.
 * @return 1 to avoid fd hijacking; 0 to capture stdout/stderr normally.
 */
static int bc_should_skip_capture(COMMAND *command) {
    return bc_command_is_top_level_async(command);
}

/**
 * @brief Arguments passed to one stdout/stderr reader thread.
 */
struct reader_arg {
    int         read_fd;     /* Pipe read end. */
    int         tee_fd;      /* Original fd used to tee output back to the user. */
    const char *type;        /* "out" / "err" */
    uint64_t    batch;       /* Batch id for this command. */
    uint64_t    command_id;  /* Command id shared by cmd/data/exit files. */
};

/**
 * @brief Drain one capture pipe, tee bytes to the original fd, and append files.
 *
 * @param p Pointer to struct reader_arg.
 * @return Always NULL.
 */
static void *reader_thread(void *p) {
    struct reader_arg *a = (struct reader_arg *)p;
    char buf[8192];
    ssize_t n;
    /* Continue until EOF. EINTR is retried; other read errors stop the reader. */
    for (;;) {
        n = read(a->read_fd, buf, sizeof buf);
        if (n > 0) {
            /* Tee bytes back to the original terminal fd so users see live output. */
            ssize_t off = 0;
            while (off < n) {
                ssize_t w = write(a->tee_fd, buf + off, n - off);
                if (w <= 0) break;
                off += w;
            }
            bc_emit_data(a->type, a->batch, a->command_id, buf, (size_t)n);
            continue;
        }
        if (n == 0) break;                 /* pipe EOF */
        if (errno == EINTR) continue;      /* Retry interrupted reads. */
        break;                             /* Stop on other read errors. */
    }
    return NULL;
}

/** @brief Reentrancy depth; only depth 0 performs capture setup. */
static int depth = 0;

/**
 * @brief Per-command foreground capture state registered with Bash unwind-protect.
 *
 * Bash can abort command execution with jump_to_top_level()/throw_to_top_level()
 * instead of returning from execute_command(). Registering this context on
 * Bash's unwind stack lets the capture layer restore fd 1/2 and finalize
 * cmd_<id>.json even on arithmetic expansion errors or SIGINT.
 *
 * Historical failure mode:
 * - arithmetic expansion errors such as `echo $((1/0))`;
 * - interactive interrupts such as Ctrl-C while a foreground command is running;
 *
 * can leave Bash through an internal longjmp path. When cleanup only lived
 * after __real_execute_command(), those paths skipped fd restoration, reader
 * joins, bc_emit_exit(), and depth--. The Python layer then observed a
 * permanently running command (`end_time == null` / status command_running) and
 * rejected later commands as "A command is already running".
 *
 * Keep the fix in C instead of force-finalizing from Python: cmd_<id>.json and
 * status.json are owned by the capture layer, so Bash's unwind-protect stack is
 * the right single source of truth for both normal returns and non-local exits.
 */
struct capture_ctx {
    uint64_t batch;
    uint64_t command_id;
    int saved_out;
    int saved_err;
    int out_read;
    int err_read;
    int hijacked;
    int finalized;
    int stopped;
    int exit_code;        /* -1 until execute_command returns normally. */
    int t_out_started;
    int t_err_started;
    pthread_t t_out;
    pthread_t t_err;
    struct reader_arg out_arg;
    struct reader_arg err_arg;
};

/** @brief Return whether Bash just returned from a genuinely stopped job. */
static int bc_command_was_stopped(int exit_code) {
#if defined (JOB_CONTROL)
    int sig = exit_code - 128;
    int is_stop_signal = sig == SIGSTOP || sig == SIGTSTP ||
                         sig == SIGTTIN || sig == SIGTTOU;
    int job = js.j_current;
    return is_stop_signal && !INVALID_JOB(job) && STOPPED(job) &&
           js.j_lastmade == jobs[job];
#else
    (void)exit_code;
    return 0;
#endif
}

/** @brief Join capture readers, close their fds, and release a heap context. */
static void bc_capture_release(struct capture_ctx *ctx) {
    if (ctx->t_out_started) pthread_join(ctx->t_out, NULL);
    if (ctx->t_err_started) pthread_join(ctx->t_err, NULL);
    if (ctx->out_read >= 0) close(ctx->out_read);
    if (ctx->err_read >= 0) close(ctx->err_read);
    if (ctx->saved_out >= 0) close(ctx->saved_out);
    if (ctx->saved_err >= 0) close(ctx->saved_err);
    free(ctx);
}

/** @brief Detached owner for readers whose stopped job still holds pipe writers. */
static void *bc_capture_reaper(void *p) {
    bc_capture_release((struct capture_ctx *)p);
    return NULL;
}

/**
 * @brief Finalize one foreground capture, safe to call from Bash unwind paths.
 *
 * @param p Pointer to struct capture_ctx.
 * @return Always 0 to satisfy Bash's Function callback type.
 */
static int bc_capture_cleanup(char *p) {
    struct capture_ctx *ctx = (struct capture_ctx *)p;
    if (!ctx || ctx->finalized) return 0;
    ctx->finalized = 1;

    if (ctx->hijacked) {
        /* Restoring fd 1/2 closes this process's pipe write ends. */
        if (ctx->saved_out >= 0) dup2(ctx->saved_out, 1);
        if (ctx->saved_err >= 0) dup2(ctx->saved_err, 2);
    }

    /*
     * On normal return, ctx->exit_code is set from execute_command(). On Bash
     * longjmp/unwind paths (for example SIGINT or arithmetic expansion errors),
     * execute_command() never returns to this wrapper, so use Bash's current
     * command status as the best available single source of truth.
     */
    int code = ctx->exit_code >= 0 ? ctx->exit_code : last_command_exit_value;

    if (ctx->stopped && ctx->hijacked) {
        pthread_t reaper;

        bc_emit_exit(ctx->batch, ctx->command_id, code, NULL);
        if (depth > 0) depth--;

        /* The stopped job owns the remaining pipe writers until bg/fg/kill. */
        if (pthread_create(&reaper, NULL, bc_capture_reaper, ctx) == 0) {
            pthread_detach(reaper);
            return 0;
        }
        /* Resource exhaustion fallback preserves correctness, though it blocks. */
        bc_capture_release(ctx);
        return 0;
    }

    uint64_t batch = ctx->batch;
    uint64_t command_id = ctx->command_id;
    bc_capture_release(ctx);
    bc_emit_exit(batch, command_id, code, NULL);
    if (depth > 0) depth--;
    return 0;
}

/**
 * @brief Wrapped Bash execute_command() entry point.
 *
 * @param command Bash COMMAND node to execute.
 * @return Exit status returned by the real execute_command().
 */
int __wrap_execute_command(COMMAND *command) {
    /* Preserve Bash's original execution path when capture is disabled. */
    if (!bc_capture_enabled())
        return __real_execute_command(command);

    /* Nested execute_command calls belong to the same top-level shell command. */
    if (depth > 0)
        return __real_execute_command(command);

    depth++;

    /* Allocate ids once for this outermost command. */
    uint64_t batch      = bc_cur_batch();
    uint64_t command_id = bc_next_command_id();

    /* Render normalized command text through Bash's own printer. */
    char *cs = make_command_string(command);

    /*
     * Skip-capture path: do not create pipes, do not change fd 1/2, and do not
     * start readers. Background children may inherit pipe write ends, so
     * hijacking here can make reader joins wait forever.
     */
    if (bc_should_skip_capture(command)) {
        bc_emit_cmd(batch, command_id, cs ? cs : "", cs ? strlen(cs) : 0,
                    NULL, /*is_background=*/1);
        /* Materialize empty data files instead of relying on FileNotFound fallback. */
        bc_begin_data(command_id);
        int ec = __real_execute_command(command);
        bc_emit_background(batch, command_id, NULL);
        depth--;
        return ec;
    }

    /* Foreground path: capture stdout/stderr through pipes. */
    bc_emit_cmd(batch, command_id, cs ? cs : "", cs ? strlen(cs) : 0,
                NULL, /*is_background=*/0);
    bc_begin_data(command_id);

    int saved_out = dup(1);
    int saved_err = dup(2);

    /* Pipe read ends feed reader threads; write ends temporarily replace fd 1/2. */
    int out_pipe[2] = {-1,-1}, err_pipe[2] = {-1,-1};
    int hijacked = 0;

    if (saved_out >= 0 && saved_err >= 0) {
        if (pipe(out_pipe) == 0 && pipe(err_pipe) == 0)
            hijacked = 1;
    }
    if (!hijacked) {
        if (out_pipe[1] >= 0) close(out_pipe[1]);
        if (err_pipe[1] >= 0) close(err_pipe[1]);
        out_pipe[1] = err_pipe[1] = -1;
    }

    struct capture_ctx *ctx = calloc(1, sizeof *ctx);
    if (!ctx) {
        if (out_pipe[0] >= 0) close(out_pipe[0]);
        if (out_pipe[1] >= 0) close(out_pipe[1]);
        if (err_pipe[0] >= 0) close(err_pipe[0]);
        if (err_pipe[1] >= 0) close(err_pipe[1]);
        if (saved_out >= 0) close(saved_out);
        if (saved_err >= 0) close(saved_err);
        int ec = __real_execute_command(command);
        bc_emit_exit(batch, command_id, ec, NULL);
        depth--;
        return ec;
    }

    ctx->batch = batch;
    ctx->command_id = command_id;
    ctx->saved_out = saved_out;
    ctx->saved_err = saved_err;
    ctx->out_read = out_pipe[0];
    ctx->err_read = err_pipe[0];
    ctx->hijacked = hijacked;
    ctx->exit_code = -1;
    ctx->out_arg = (struct reader_arg){out_pipe[0], saved_out, "out", batch, command_id};
    ctx->err_arg = (struct reader_arg){err_pipe[0], saved_err, "err", batch, command_id};

    if (hijacked) {
        /* Readers must run while the command executes to avoid pipe-buffer deadlock. */
        ctx->t_out_started = pthread_create(&ctx->t_out, NULL, reader_thread, &ctx->out_arg) == 0;
        ctx->t_err_started = pthread_create(&ctx->t_err, NULL, reader_thread, &ctx->err_arg) == 0;
        if (!ctx->t_out_started || !ctx->t_err_started) {
            close(out_pipe[1]); out_pipe[1] = -1;
            close(err_pipe[1]); err_pipe[1] = -1;
            if (ctx->t_out_started) pthread_join(ctx->t_out, NULL);
            if (ctx->t_err_started) pthread_join(ctx->t_err, NULL);
            ctx->t_out_started = ctx->t_err_started = 0;
            close(out_pipe[0]); ctx->out_read = out_pipe[0] = -1;
            close(err_pipe[0]); ctx->err_read = err_pipe[0] = -1;
            ctx->hijacked = hijacked = 0;
        }
    }

    if (hijacked) {
        /* Redirect stdout/stderr to capture pipe write ends. */
        dup2(out_pipe[1], 1);
        dup2(err_pipe[1], 2);
        /* Close extra write-end references; fd 1/2 now own the write sides. */
        close(out_pipe[1]); out_pipe[1] = -1;
        close(err_pipe[1]); err_pipe[1] = -1;
    }

    begin_unwind_frame("nexus-capture");
    add_unwind_protect((Function *)bc_capture_cleanup, (char *)ctx);

    /*
     * Execute the real Bash command.
     *
     * Normal path:
     *   __real_execute_command() returns, ctx.exit_code is set below, and
     *   run_unwind_frame() invokes bc_capture_cleanup().
     *
     * Non-local exit path:
     *   Bash may longjmp before returning here. Its top-level unwind handling
     *   runs the registered frame, so bc_capture_cleanup() still finalizes the
     *   capture record. Code after this call is intentionally not required for
     *   arithmetic expansion errors, SIGINT, and similar Bash unwind paths.
     */
    int ec = __real_execute_command(command);

    ctx->exit_code = ec;
    ctx->stopped = bc_command_was_stopped(ec);
    fflush(NULL);
    run_unwind_frame("nexus-capture");
    return ec;
}
