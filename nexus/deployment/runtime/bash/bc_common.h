/**
 * @file bc_common.h
 * @brief Shared state and file-output API for the Nexus bash-capture runtime.
 *
 * This module is compiled once as bc_common.o and linked into the patched bash
 * binary together with cmd_wrap.o and readline_wrap.o. Because all wrappers live
 * in the same process and link against the same object file, batch and
 * command_id are ordinary process-local counters. No memory-mapped file, shared memory, atomics, or environment-variable
 * handoff is required.
 *
 * Responsibilities:
 * - readline_wrap.c initializes the session, maintains the readline batch, and
 *   writes the initial ready/idle status.
 * - cmd_wrap.c allocates one command_id for each outermost execute_command,
 *   captures stdout/stderr for foreground commands, and emits command lifecycle
 *   records.
 * - bc_common.c owns the counters and writes the Nexus multi-file capture
 *   format.
 *
 * Typical call sequence:
 *
 * 1. First interactive readline call:
 *    @code
 *    readline_wrap -> bc_reset()
 *    readline_wrap -> bc_bump_batch()          // first batch is 1
 *    readline_wrap -> bc_write_status("idle", 0)
 *    @endcode
 *    TerminalSession uses status.json as the ready signal.
 *
 * 2. Later readline calls:
 *    @code
 *    readline_wrap -> bc_bump_batch()          // only when stdin looks idle
 *    @endcode
 *    Multiple commands submitted in one input burst share the same batch.
 *
 * 3. Foreground command path:
 *    @code
 *    cmd_wrap -> batch = bc_cur_batch()
 *    cmd_wrap -> command_id = bc_next_command_id()
 *    cmd_wrap -> bc_emit_cmd(...)
 *    cmd_wrap -> bc_begin_data(command_id)
 *    reader   -> bc_emit_data("out"/"err", ...)  // zero or more chunks
 *    cmd_wrap -> bc_emit_exit(...)
 *    @endcode
 *
 * 4. Background/async-risk command path:
 *    @code
 *    cmd_wrap -> batch = bc_cur_batch()
 *    cmd_wrap -> command_id = bc_next_command_id()
 *    cmd_wrap -> bc_emit_cmd(..., is_background=1)
 *    cmd_wrap -> bc_begin_data(command_id)      // materializes empty data files
 *    cmd_wrap -> __real_execute_command(command) // no fd hijack, no readers
 *    cmd_wrap -> bc_emit_background(...)
 *    @endcode
 *
 * Threading model:
 * - batch and command_id are updated only on bash's main command path.
 * - bc_emit_data() may be called concurrently by stdout and stderr reader
 *   threads; the implementation serializes writes to output_<id>.
 * - All APIs are no-ops unless both NEXUS_CAPTURE and NEXUS_CAPTURE_DIR are set.
 *
 * File contract consumed by nexus/runtime/executors/terminal/capture.py:
 * - status.json: current status, timestamp, and command_id.
 * - cmd_<id>.json: command_id, batch, start_time, end_time, and exit_code.
 * - cmd_content_<id>: command text.
 * - stdout_<id>: captured stdout bytes.
 * - stderr_<id>: captured stderr bytes.
 * - output_<id>: stdout/stderr merged bytes.
 */
#ifndef BC_COMMON_H
#define BC_COMMON_H

#include <stddef.h>
#include <stdint.h>

/**
 * @brief Reset process-local capture counters.
 *
 * Sets both batch and command_id to zero. readline_wrap.c calls bc_bump_batch()
 * immediately afterwards so the first user-visible batch is 1.
 *
 * @note Called once by readline_wrap.c when the interactive session first
 * enters readline().
 */
void bc_reset(void);

/**
 * @brief Increment and return the current readline batch.
 *
 * A batch represents one user input burst. Several top-level commands may share
 * one batch when they are submitted together; Python groups command records by
 * this value.
 *
 * @return The incremented batch id.
 *
 * @note Called by readline_wrap.c during session initialization and when a later
 * readline call appears to start a new input burst.
 */
uint64_t bc_bump_batch(void);

/**
 * @brief Return the current readline batch without mutating it.
 *
 * @return The current batch id written to cmd_<id>.json.
 *
 * @note Called by cmd_wrap.c when the outermost execute_command starts.
 */
uint64_t bc_cur_batch(void);

/**
 * @brief Allocate the next top-level command id.
 *
 * command_id is the stable key shared by cmd_<id>.json, cmd_content_<id>,
 * stdout_<id>, stderr_<id>, output_<id>, and status.json.
 *
 * @return The newly allocated command id, starting at 1.
 *
 * @note Called once by cmd_wrap.c for each outermost execute_command.
 * @warning Nested execute_command calls must be passed through by cmd_wrap.c and
 * must not allocate additional command ids, otherwise one shell command would be
 * split into multiple capture records.
 */
uint64_t bc_next_command_id(void);

/**
 * @brief Return whether Bash capture is enabled for this process.
 *
 * Capture is enabled only when both NEXUS_CAPTURE and NEXUS_CAPTURE_DIR are
 * set and non-empty.
 *
 * @return 1 when capture is enabled; otherwise 0.
 */
int bc_capture_enabled(void);

/**
 * @brief Atomically write status.json.
 *
 * @param status Status string, for example "idle", "command_running", or
 * "command_finished".
 * @param command_id Command id associated with the status, or 0 for the initial
 * idle/ready state.
 *
 * @note Implemented as write-to-temp-file plus rename so Python polling never
 * observes a partially written JSON file.
 */
void bc_write_status(const char *status, uint64_t command_id);

/**
 * @brief Emit the start record for one command.
 *
 * Creates cmd_content_<id>, writes the initial cmd_<id>.json, and updates
 * status.json to command_running.
 *
 * @param batch Current readline batch.
 * @param command_id Current command id.
 * @param val Command text buffer; may be NULL.
 * @param n Number of bytes to write from val.
 * @param start_time Reserved for the old call signature. The implementation
 * records the current Unix timestamp itself.
 * @param is_background Reserved for the old call signature. Background handling
 * is represented by the caller's use of bc_emit_background().
 *
 * @note Called by cmd_wrap.c before executing the real Bash command.
 */
void bc_emit_cmd(uint64_t batch, uint64_t command_id,
                 const char *val, size_t n,
                 const char *start_time, int is_background);

/**
 * @brief Create or truncate data files for one command.
 *
 * Materializes stdout_<id>, stderr_<id>, and output_<id> as empty files. This
 * gives readers a stable file contract even before output arrives, and prevents
 * stale content from a previous command id reuse in unusual filesystem states.
 *
 * @param command_id Current command id.
 *
 * @note Called by cmd_wrap.c before starting foreground reader threads. It is
 * also called for background/async-risk commands to explicitly represent "no
 * captured output" with empty files.
 */
void bc_begin_data(uint64_t command_id);

/**
 * @brief Append one stdout/stderr chunk to the capture files.
 *
 * The chunk is appended to the per-stream file (stdout_<id> or stderr_<id>) and
 * also to the merged output_<id> file.
 *
 * @param type Stream type: "out" for stdout, "err" for stderr.
 * @param batch Reserved for a uniform event signature; currently unused.
 * @param command_id Current command id.
 * @param buf_in Output bytes; ignored when NULL.
 * @param n Number of bytes to append; ignored when 0.
 *
 * @note Called by cmd_wrap.c reader threads. output_<id> appends are serialized
 * by the implementation so stdout/stderr chunks do not interleave at byte level.
 */
void bc_emit_data(const char *type, uint64_t batch, uint64_t command_id,
                  const char *buf_in, size_t n);

/**
 * @brief Finalize a foreground command record.
 *
 * Adds end_time and exit_code to cmd_<id>.json and updates status.json to
 * command_finished.
 *
 * @param batch Reserved for a uniform event signature; currently unused.
 * @param command_id Current command id.
 * @param code Exit code returned by Bash's execute_command.
 * @param end_time Reserved for the old call signature. The implementation
 * records the current Unix timestamp itself.
 *
 * @note Called by cmd_wrap.c after the real command returns, stdio fds are
 * restored, and reader threads have drained to EOF.
 */
void bc_emit_exit(uint64_t batch, uint64_t command_id, int code,
                  const char *end_time);

/**
 * @brief Finalize a background/async-risk command record.
 *
 * Background/async-risk commands are executed without stdout/stderr fd hijacking
 * and without waiting for their asynchronous descendants. This avoids blocking
 * reader threads on pipe write ends inherited by background processes.
 *
 * @param batch Reserved for a uniform event signature; currently unused.
 * @param command_id Current command id.
 * @param start_time Reserved for the old call signature. The implementation
 * records the current Unix timestamp itself.
 *
 * @note Called by cmd_wrap.c only on the skip-capture path. bc_begin_data() must
 * already have created empty stdout/stderr/output files.
 */
void bc_emit_background(uint64_t batch, uint64_t command_id,
                        const char *start_time);

#endif /* BC_COMMON_H */
