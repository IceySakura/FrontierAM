/**
 * @file bc_common.c
 * @brief Shared counters and Nexus multi-file output for bash-capture.
 *
 * This file is compiled as bc_common.o and linked into the patched bash binary
 * together with cmd_wrap.o and readline_wrap.o.
 *
 * It owns two kinds of state:
 * - process-local counters: readline_wrap.c advances batch, while cmd_wrap.c
 *   allocates command_id for each outermost execute_command;
 * - output files: status.json, cmd_<id>.json, cmd_content_<id>, stdout_<id>,
 *   stderr_<id>, and output_<id>.
 *
 * status.json and cmd_<id>.json finalization use temp-file plus rename so the
 * Python side does not observe partially written JSON. stdout/stderr/output are
 * append-only byte streams during command execution. output_<id> is protected by
 * a mutex because both reader threads can append to it.
 */
#define _GNU_SOURCE
#include "bc_common.h"

#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/** @brief Current readline batch, advanced only by readline_wrap.c. */
static uint64_t bc_batch = 0;

/** @brief Last allocated top-level command id, advanced only by cmd_wrap.c. */
static uint64_t bc_command_id = 0;

void bc_reset(void) {
    bc_batch = 0;
    bc_command_id = 0;
}

uint64_t bc_bump_batch(void) {
    return ++bc_batch;
}

uint64_t bc_cur_batch(void) {
    return bc_batch;
}

uint64_t bc_next_command_id(void) {
    return ++bc_command_id;
}

/**
 * @brief Cached capture directory.
 *
 * NULL means capture is disabled. The value is resolved lazily from
 * NEXUS_CAPTURE and NEXUS_CAPTURE_DIR on first use.
 */
static const char *bc_dir = NULL;

/** @brief Whether the capture environment has already been resolved. */
static int bc_dir_resolved = 0;

/**
 * @brief Return the capture directory if capture is enabled.
 *
 * Capture is enabled only when both NEXUS_CAPTURE and NEXUS_CAPTURE_DIR are set
 * and non-empty. The directory string is duplicated so later Bash environment
 * mutations cannot invalidate the cached pointer.
 *
 * @return Capture directory path, or NULL when capture is disabled.
 */
static const char *bc_capture_dir(void) {
    if (!bc_dir_resolved) {
        bc_dir_resolved = 1;
        const char *sw = getenv("NEXUS_CAPTURE");
        const char *dir = getenv("NEXUS_CAPTURE_DIR");
        if (sw && *sw && dir && *dir)
            bc_dir = strdup(dir);
    }
    return bc_dir;
}

int bc_capture_enabled(void) {
    return bc_capture_dir() != NULL;
}

/**
 * @brief Return current wall-clock time as Unix floating-point seconds.
 *
 * @return Seconds since Unix epoch, with sub-second precision.
 */
static double bc_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/**
 * @brief Write a complete byte range to dir/name_command_id.
 *
 * Handles short writes by retrying until the full buffer is written or write()
 * fails. Errors are intentionally swallowed: capture must never change normal
 * Bash command execution semantics.
 *
 * @param dir Capture directory.
 * @param name File prefix, such as "stdout", "stderr", or "output".
 * @param command_id Current command id.
 * @param flags open(2) flags supplied by the caller.
 * @param buf Bytes to write.
 * @param n Number of bytes to write.
 */
static void bc_write_blob(const char *dir, const char *name, uint64_t command_id,
                          int flags, const char *buf, size_t n) {
    char path[PATH_MAX];
    snprintf(path, sizeof path, "%s/%s_%llu", dir, name, (unsigned long long)command_id);
    int fd = open(path, flags, 0644);
    if (fd < 0) return;
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, buf + off, n - off);
        if (w <= 0) break;
        off += (size_t)w;
    }
    close(fd);
}

void bc_write_status(const char *status, uint64_t command_id) {
    const char *dir = bc_capture_dir();
    if (!dir) return;
    char path[PATH_MAX], tmp[PATH_MAX];
    snprintf(path, sizeof path, "%s/status.json", dir);
    snprintf(tmp, sizeof tmp, "%s/status.json.tmp", dir);
    FILE *fp = fopen(tmp, "w");
    if (!fp) return;
    fprintf(fp, "{\"status\":\"%s\",\"timestamp\":%.6f,\"command_id\":%llu}\n",
            status, bc_now_sec(), (unsigned long long)command_id);
    fclose(fp);
    rename(tmp, path);
}

/**
 * @brief Atomically append final fields to cmd_<id>.json.
 *
 * The start record is a single JSON object written by bc_emit_cmd(). Finalizing
 * reads that line, removes the trailing '}', appends @p extra, closes the JSON
 * object again, and replaces the file via tmp + rename.
 *
 * @param command_id Current command id.
 * @param extra JSON fragment beginning with a comma, for example
 * ,"end_time":...,"exit_code":0.
 */
static void bc_finalize_cmd(uint64_t command_id, const char *extra) {
    const char *dir = bc_capture_dir();
    if (!dir) return;
    char path[PATH_MAX], tmp[PATH_MAX], line[8192];
    snprintf(path, sizeof path, "%s/cmd_%llu.json", dir, (unsigned long long)command_id);
    snprintf(tmp, sizeof tmp, "%s/cmd_%llu.json.tmp", dir, (unsigned long long)command_id);
    FILE *in = fopen(path, "r"), *out = fopen(tmp, "w");
    if (in && out) {
        int ok = 0;
        if (fgets(line, sizeof line, in)) {
            size_t len = strlen(line);
            if (len > 0 && line[len - 1] == '\n') line[--len] = '\0';
            if (len > 0 && line[len - 1] == '}') line[--len] = '\0';
            fprintf(out, "%s%s}\n", line, extra);
            ok = 1;
        }
        fclose(in);
        fclose(out);
        if (ok) rename(tmp, path);
        else unlink(tmp);
    } else {
        if (in) fclose(in);
        if (out) fclose(out);
        unlink(tmp);
    }
    bc_write_status("command_finished", command_id);
}

void bc_emit_cmd(uint64_t batch, uint64_t command_id,
                 const char *val, size_t n,
                 const char *start_time, int is_background) {
    (void)start_time;
    (void)is_background;
    const char *dir = bc_capture_dir();
    if (!dir) return;
    char path[PATH_MAX];

    snprintf(path, sizeof path, "%s/cmd_content_%llu", dir, (unsigned long long)command_id);
    FILE *fp = fopen(path, "w");
    if (fp) {
        if (val && n) fwrite(val, 1, n, fp);
        fclose(fp);
    }

    snprintf(path, sizeof path, "%s/cmd_%llu.json", dir, (unsigned long long)command_id);
    fp = fopen(path, "w");
    if (fp) {
        fprintf(fp, "{\"command_id\":%llu,\"batch\":%llu,\"start_time\":%.6f}\n",
                (unsigned long long)command_id, (unsigned long long)batch, bc_now_sec());
        fclose(fp);
    }

    bc_write_status("command_running", command_id);
}

/**
 * @brief Mutex protecting merged output_<id> appends.
 *
 * stdout_<id> and stderr_<id> each have a single writer, but output_<id> is
 * written by both reader threads. The lock preserves chunk boundaries in the
 * merged stream.
 */
static pthread_mutex_t bc_output_mu = PTHREAD_MUTEX_INITIALIZER;

void bc_begin_data(uint64_t command_id) {
    const char *dir = bc_capture_dir();
    if (!dir) return;
    char path[PATH_MAX];
    const char *names[] = {"stdout", "stderr", "output"};
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++) {
        snprintf(path, sizeof path, "%s/%s_%llu", dir, names[i],
                 (unsigned long long)command_id);
        int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) close(fd);
    }
}

void bc_emit_data(const char *type, uint64_t batch, uint64_t command_id,
                  const char *buf_in, size_t n) {
    (void)batch;
    const char *dir = bc_capture_dir();
    if (!dir || !buf_in || n == 0) return;
    int is_err = (type && type[0] == 'e');

    bc_write_blob(dir, is_err ? "stderr" : "stdout", command_id,
                  O_WRONLY | O_CREAT | O_APPEND, buf_in, n);

    pthread_mutex_lock(&bc_output_mu);
    bc_write_blob(dir, "output", command_id, O_WRONLY | O_CREAT | O_APPEND, buf_in, n);
    pthread_mutex_unlock(&bc_output_mu);
}

void bc_emit_exit(uint64_t batch, uint64_t command_id, int code,
                  const char *end_time) {
    (void)batch;
    (void)end_time;
    char extra[64];
    snprintf(extra, sizeof extra, ",\"end_time\":%.6f,\"exit_code\":%d", bc_now_sec(), code);
    bc_finalize_cmd(command_id, extra);
}

void bc_emit_background(uint64_t batch, uint64_t command_id,
                        const char *start_time) {
    (void)batch;
    (void)start_time;
    char extra[64];
    snprintf(extra, sizeof extra, ",\"end_time\":%.6f,\"exit_code\":0", bc_now_sec());
    bc_finalize_cmd(command_id, extra);
}
