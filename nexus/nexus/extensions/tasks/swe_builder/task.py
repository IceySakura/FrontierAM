"""SWE Builder task implementation (In-Place Mode).

The task sets up the repository environment, runs an agent to write eval.sh and
validate fail2pass, then commits and uploads the image. Task is resolved when the
image is successfully uploaded.
"""

import asyncio
import logging

from typing import Any

from pydantic import Field

from nexus.core import (
    Registry,
    TaskRunInput,
    TaskRunOutput,
    WithAgentRunner,
    WithScore,
    require_context,
    task_registry,
)
from nexus.trace import get_current_span_id, update_span

# Import constants
from .constants import (
    DEFAULT_AGENT_NAME,
    DEFAULT_TESTBED_DIR,
    TaskPhase,
)


# Import agent to ensure tools are registered
Registry.import_module('nexus.extensions.agents.swe_setup_agent')

logger = logging.getLogger(__name__)

# Agent exit statuses that end the build outright. Kept next to the agent it mirrors:
# swe_setup_agent splits its giving-up path in two so a run stopped by an exhausted token
# budget is reported as TRUNCATION rather than FORMAT_ERROR, but both are failures here.
AGENT_FAILURE_EXIT_STATUSES = (
    'no_tool_calls_incomplete',
    'response_truncated',
    'submit_eval_failed',
    'llm_error',
)


# ============== Input/Output Models ==============


class SWEBuilderTaskInput(TaskRunInput, WithAgentRunner):
    """SWE Builder task input."""

    # Instance data
    instance_id: str = Field(description='Instance ID')
    repo: str = Field(description='Repository name (owner/repo)')
    base_commit: str = Field(description='Base commit hash')
    language: str = Field(description='Programming language')
    problem_statement: str = Field(description='Problem statement')
    test_patch: str = Field(default='', description='Test patch')
    golden_patch: str = Field(default='', description='Golden patch (the fix)')

    # Runtime config
    testbed_dir: str = Field(default=DEFAULT_TESTBED_DIR, description='Testbed directory')
    base_image: str = Field(default='ubuntu-git:22.04', description='Base image')

    # Agent config
    agent_name: str = Field(default=DEFAULT_AGENT_NAME, description='Agent name')
    setup_timeout: int = Field(default=3600, description='Agent timeout (seconds)')
    test_gen_timeout: int = Field(default=1800, description='Reserved for compatibility (unused)')

    # Submit limits
    max_submit_attempts: int = Field(default=30, description='Max submit_eval attempts (tool + agent level)')

    # Output config
    output_dir: str = Field(default='./outputs/swe_builder', description='Output directory')
    debug: bool = Field(default=False, description='Debug mode (skip upload)')


class SWEBuilderTaskOutput(TaskRunOutput, WithScore):
    """SWE Builder task output."""

    # Task success status
    resolved: bool = Field(description='Whether the task is resolved/successful')

    # Completion status
    phase_completed: int = Field(default=0, description='Highest phase reached')

    # Instance metadata
    instance_id: str = Field(default='', description='Instance ID')
    repo: str = Field(default='', description='Repository name')
    language: str = Field(default='', description='Programming language')

    # Outputs
    eval_script_path: str = Field(default='', description='Eval script path')
    eval_script_content: str = Field(default='', description='Eval script content')
    eval_script_actual_path: str = Field(
        default='',
        description='Actual path eval script was read from (may differ from eval_script_path if fallback search was used)',
    )
    eval_script_source: str = Field(
        default='',
        description='How eval script was found: "provided" (at given path) or "fallback_search" (found at alternative path)',
    )
    exported_image: str = Field(default='', description='Exported image name')
    image_uploaded: bool = Field(default=False, description='Whether image was successfully uploaded to registry')
    pass2pass_only: bool = Field(
        default=False,
        description='Image uploaded but only pass2pass (no fail2pass): likely refactoring/optimization issue',
    )

    # LLM validation results
    eval_llm_validation: dict[str, Any] = Field(
        default_factory=dict, description='LLM validation result for eval script'
    )

    # Runtime information
    final_base_image: str = Field(default='', description='Final base image used by runtime at task completion')

    # Agent iterations
    agent_iters: int = Field(default=0, description='Agent iterations')
    total_iters: int = Field(default=0, description='Total agent iterations')

    # Complete linear conversation trajectory (never trimmed)
    main_trajectory: list[dict] = Field(default_factory=list, description='Complete linear conversation trajectory')

    # Error information (if failed)
    error: str = Field(default='', description='Error message if task failed')
    error_details: str = Field(default='', description='Detailed error information including stack trace')
    agent_exit_status: str = Field(default='', description='Agent exit status')

    # Summary
    instance_with_results: dict[str, Any] = Field(default_factory=dict, description='Instance data with results')


# ============== Task Implementation ==============


@task_registry.register('swe_builder')
async def swe_builder(task_input: SWEBuilderTaskInput) -> SWEBuilderTaskOutput:
    """SWE Builder task - Configure environment and upload eval image (in-place mode).

    Step 0: Initialize testbed (clone repo, export patches)
    Step 1 (agent): Environment setup + write eval.sh + submit_eval → agent exits
    Task resolved when image is successfully uploaded.
    """
    ctx = require_context()

    # Initialize task state
    ctx.tool_state['task_phase'] = TaskPhase.INIT
    ctx.tool_state['instance_id'] = task_input.instance_id

    # Store task params in context for tools to access
    ctx.task_config.update({
        'instance_id': task_input.instance_id,
        'repo': task_input.repo,
        'base_commit': task_input.base_commit,
        'language': task_input.language,
        'problem_statement': task_input.problem_statement,
        'test_patch': task_input.test_patch,
        'golden_patch': task_input.golden_patch,
        'testbed_dir': task_input.testbed_dir,
        'base_image': task_input.base_image,
        'output_dir': task_input.output_dir,
        'debug': task_input.debug,
        'max_submit_attempts': task_input.max_submit_attempts,
        'max_base_image_switches': 5,
    })

    logger.debug('=' * 80)
    logger.debug(f'[SWE_BUILDER] Starting task for {task_input.instance_id}')
    logger.debug(f'[SWE_BUILDER] Repo: {task_input.repo}')
    logger.debug(f'[SWE_BUILDER] Language: {task_input.language}')
    logger.debug(f'[SWE_BUILDER] Base image: {task_input.base_image}')
    logger.debug('=' * 80)

    try:
        # Step 0: Initialize testbed
        logger.debug('[SWE_BUILDER] Initializing testbed')
        runtime = await ctx.get_runtime('main')

        # Record runtime_id for tracing (same pattern as swe task)
        runtime_id = ctx.get_runtime_id('main') or 'unknown'
        if runtime_id != 'unknown':
            logger.debug(f'[SWE_BUILDER] Recorded runtime_id: {runtime_id}')
            span_id = get_current_span_id()
            if span_id:
                update_span(span_id, attributes={'hytuner.runtime_id': runtime_id})

        await _initialize_testbed(ctx, runtime, task_input)
        await _write_patch_files(ctx, runtime, task_input)
        ctx.tool_state['task_phase'] = TaskPhase.SETUP

        # Cache copy-in (best-effort: warm container with host pip/npm/maven/gradle cache)
        try:
            from .cache_manager import copy_in_to_container

            _cname = getattr(ctx.runtime_provider, '_user_containers', {}).get('main')
            if _cname:
                await copy_in_to_container(_cname)
        except Exception as _e:
            logger.warning(f'[CACHE] copy-in skipped: {_e}')

        # Step 1: Run agent (env setup + submit_eval → exits on success)
        logger.debug('[SWE_BUILDER] Running agent (env setup + submit_eval)')
        agent_output = await _run_agent(ctx, task_input)

        # Check agent exit status
        exit_status = agent_output.exit_status
        logger.info(f'[SWE_BUILDER] Agent exit status: {exit_status}')

        # Check final phase
        final_phase = ctx.tool_state.get('task_phase', 0)

        # pass2pass_exhausted: test_patch has no fail2pass signal after 3 consecutive pass2pass results
        if exit_status == 'pass2pass_exhausted':
            logger.warning('[SWE_BUILDER] Agent gave up after 3 consecutive pass2pass results')
            ctx.tool_state['pass2pass_only'] = True
            return _build_output(
                ctx,
                task_input,
                phase_completed=final_phase,
                resolved=False,
                error='pass2pass: test_patch has no fail2pass signal',
                agent_exit_status=exit_status,
            )

        if exit_status in AGENT_FAILURE_EXIT_STATUSES:
            logger.error(f'[SWE_BUILDER] Agent failed with status: {exit_status}')
            return _build_output(
                ctx,
                task_input,
                phase_completed=final_phase,
                resolved=False,
                error=f'Agent failed: {exit_status}',
                agent_exit_status=exit_status,
            )

        if final_phase < TaskPhase.EVAL_SUBMITTED:
            logger.error(f'[SWE_BUILDER] submit_eval not completed (final_phase={TaskPhase.to_string(final_phase)})')
            return _build_output(
                ctx, task_input, phase_completed=final_phase, resolved=False, agent_exit_status=exit_status
            )

        # Task resolved logic:
        # - debug=False (upload enabled):  image must be uploaded AND fail2pass signal present
        # - debug=True  (upload skipped):  submit_eval reached EVAL_SUBMITTED phase AND fail2pass signal present
        #   In debug mode, docker commit/push is skipped but fail2pass validation still runs fully.
        phase_completed = ctx.tool_state.get('task_phase', 0)
        image_uploaded = ctx.tool_state.get('image_uploaded', False)
        pass2pass_only = ctx.tool_state.get('pass2pass_only', False)
        if task_input.debug:
            resolved = (phase_completed >= TaskPhase.EVAL_SUBMITTED) and not pass2pass_only
        else:
            resolved = image_uploaded and not pass2pass_only
        agent_iters = ctx.tool_state.get('agent_iters', 0)
        logger.debug('=' * 80)
        logger.info('[SWE_BUILDER] Task completed')
        logger.info(f'[SWE_BUILDER] Phase reached: {TaskPhase.to_string(phase_completed)}')
        logger.debug(f'[SWE_BUILDER] Agent iterations: {agent_iters}')
        logger.info(
            f'[SWE_BUILDER] image_uploaded: {image_uploaded}, pass2pass_only: {pass2pass_only}, resolved: {resolved}'
        )
        logger.debug('=' * 80)

        return _build_output(ctx, task_input, phase_completed=phase_completed, resolved=resolved)

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        logger.error(f'[SWE_BUILDER] Task failed: {e}', exc_info=True)
        actual_phase = ctx.tool_state.get('task_phase', TaskPhase.INIT)
        ctx.tool_state['task_phase'] = TaskPhase.FAILED
        ctx.tool_state['error_details'] = error_details
        return _build_output(
            ctx, task_input, phase_completed=actual_phase, resolved=False, error=str(e), error_details=error_details
        )

    finally:
        # Container cleanup is handled automatically by the framework's ExecutionContext.__aexit__
        # which calls runtime_provider.release_all() when the context exits.
        if task_input.debug:
            logger.debug('[SWE_BUILDER] Debug mode: container will be cleaned up by framework context manager')
        else:
            logger.debug('[SWE_BUILDER] Production mode: container will be cleaned up by framework context manager')

        # Safety cleanup: remove any leftover local Docker images for this instance.
        if not task_input.debug:
            _cleanup_instance_images(task_input.instance_id)


# ============== Image Cleanup ==============


def _cleanup_instance_images(instance_id: str) -> None:
    """Remove any local Docker images created for this instance.

    Covers two image names that may be left on disk:
    - swe_builder_eval_prs_{instance_id}:latest  (committed by submit_eval)
    - mirrors.tencent.com/.../swe_builder_eval_prs_{instance_id}:latest (tagged for push)

    The upload_handler already cleans up in its finally block, but this acts as a
    safety net for cases where upload was never reached (e.g. an earlier exception).
    """
    try:
        import docker as docker_lib

        from nexus.extensions.tasks.swe_builder.upload_handler import REGISTRY_PREFIX

        safe_id = instance_id.lower().replace('/', '_').replace('-', '_')
        local_name = f'swe_builder_eval_prs_{safe_id}:latest'
        mirror_name = f'{REGISTRY_PREFIX}.{local_name}'

        client = docker_lib.from_env(timeout=60)
        try:
            for name in (local_name, mirror_name):
                try:
                    client.images.remove(name, force=True)
                    logger.debug(f'[CLEANUP] Removed leftover image: {name}')
                except docker_lib.errors.ImageNotFound:
                    pass  # already gone, nothing to do
                except Exception as rm_err:
                    logger.warning(f'[CLEANUP] Could not remove image {name}: {rm_err}')
        finally:
            client.close()
    except Exception as e:
        # Never let cleanup failures surface as task errors
        logger.warning(f'[CLEANUP] Image cleanup skipped: {e}')


# ============== Helper Functions ==============


async def _initialize_testbed(ctx, runtime, task_input: SWEBuilderTaskInput):
    """Initialize testbed: clone repo, reset to base_commit."""
    from .runtime_utils import (
        git_clone,
        git_remove_remote,
        git_reset_and_clean,
    )

    repo_url = f'https://github.com/{task_input.repo}'
    testbed_dir = task_input.testbed_dir

    logger.debug(f'[INIT] Cloning {repo_url} to {testbed_dir}')

    # 1. Clone repository (retry up to 3 attempts for transient network errors)
    clone_result: dict[str, Any] = {}
    for attempt in range(1, 4):
        clone_result = await runtime.invoke(git_clone, **{'repo_url': repo_url, 'dest': testbed_dir})
        if clone_result['exit_code'] == 0:
            break
        logger.warning(f'[INIT] Git clone attempt {attempt}/3 failed: {clone_result["stderr"][:200]}')
        if attempt < 3:
            await asyncio.sleep(10)

    if clone_result.get('exit_code') != 0:
        raise RuntimeError(f'Git clone failed: {clone_result.get("stderr")}')

    # 2. Reset to base_commit
    reset_result = await runtime.invoke(
        git_reset_and_clean, **{'testbed_dir': testbed_dir, 'commit': task_input.base_commit}
    )

    if reset_result['exit_code'] != 0:
        raise RuntimeError(f'Git reset failed: {reset_result["stderr"]}')

    # 3. Remove remote
    await runtime.invoke(git_remove_remote, **{'testbed_dir': testbed_dir})

    logger.debug('[INIT] ✅ Testbed initialized')


async def _write_patch_files(ctx, runtime, task_input: SWEBuilderTaskInput):
    """Write golden_patch and test_patch to /tmp for agent reference during setup.

    These files let the agent use `git apply /tmp/golden_patch.patch` in Step 3
    verification without having to echo the full patch content via heredoc.
    submit_eval deletes these files before committing the image.
    """
    from .runtime_utils import write_file

    if task_input.golden_patch.strip():
        await runtime.invoke(write_file, **{'path': '/tmp/golden_patch.patch', 'content': task_input.golden_patch})
        logger.debug('[INIT] Wrote /tmp/golden_patch.patch')

    if task_input.test_patch.strip():
        await runtime.invoke(write_file, **{'path': '/tmp/test_patch.patch', 'content': task_input.test_patch})
        logger.debug('[INIT] Wrote /tmp/test_patch.patch')


async def _run_agent(ctx, task_input: SWEBuilderTaskInput):
    """Run agent (environment setup + submit_eval).

    Agent exits after submit_eval succeeds or on failure.
    """
    total_timeout = task_input.setup_timeout

    # max_iterations is intentionally NOT passed here so that agent_config from the
    # generator (e.g. {'max_iterations': 300}) takes effect via merged_kwargs.
    # agent_config is applied first, then task_kwargs override — omitting it here
    # lets the generator control the value.
    return await task_input.run_agent(
        repo=task_input.repo,
        base_commit=task_input.base_commit,
        language=task_input.language,
        problem_statement=task_input.problem_statement,
        test_patch=task_input.test_patch,
        golden_patch=task_input.golden_patch,
        testbed_dir=task_input.testbed_dir,
        base_image=task_input.base_image,
        timeout=total_timeout,
        user_prompt='',  # Ignored: agent generates prompt from task_config
    )


def _build_output(
    ctx,
    task_input: SWEBuilderTaskInput,
    phase_completed: int,
    resolved: bool,
    error: str = '',
    error_details: str = '',
    agent_exit_status: str = '',
) -> SWEBuilderTaskOutput:
    """Build task output."""

    # Get data from shared state
    eval_script_path = ctx.tool_state.get('eval_script_path', '')
    exported_image = ctx.tool_state.get('exported_image', '')
    agent_iters = ctx.tool_state.get('agent_iters', 0)
    image_uploaded = ctx.tool_state.get('image_uploaded', False)
    pass2pass_only = ctx.tool_state.get('pass2pass_only', False)
    main_trajectory = ctx.tool_state.get('main_trajectory', [])

    # Get current runtime base image (task_config['base_image'] is updated when switch_runtime_image is called)
    final_base_image = ctx.task_config.get('base_image', task_input.base_image)

    logger.info(f'[BUILD_OUTPUT] Final base image: {final_base_image}')

    # Get validation results
    eval_llm_validation = ctx.tool_state.get('eval_llm_validation', {})
    eval_fail2pass_validation = ctx.tool_state.get('eval_fail2pass_validation', {})

    eval_validation_combined = {
        'llm_check': eval_llm_validation,
        'fail2pass_check': eval_fail2pass_validation,
    }

    # Read eval script content
    eval_script_content = ''
    if eval_script_path:
        try:
            with open(eval_script_path) as f:
                eval_script_content = f.read()
        except Exception as e:
            logger.warning(f'Failed to read eval script: {e}')

    # Build instance_with_results
    instance_with_results = {
        'instance_id': task_input.instance_id,
        'repo': task_input.repo,
        'base_commit': task_input.base_commit,
        'language': task_input.language,
        'problem_statement': task_input.problem_statement,
        'patch': task_input.golden_patch,
        'test_patch': task_input.test_patch,
        'eval_script_content': eval_script_content,
        'test_results': {'fail2pass': [], 'pass2pass': []},
        'eval_validation': eval_validation_combined,
        'final_base_image': final_base_image,
        'agent_iters': agent_iters,
        'total_iters': agent_iters,
        'split': 't3',
    }

    # Calculate score
    score = 1.0 if resolved else 0.0

    return SWEBuilderTaskOutput(
        phase_completed=phase_completed,
        instance_id=task_input.instance_id,
        repo=task_input.repo,
        language=task_input.language,
        eval_script_path=eval_script_path,
        eval_script_content=eval_script_content,
        eval_script_actual_path=ctx.tool_state.get('eval_script_actual_path', eval_script_path),
        eval_script_source=ctx.tool_state.get('eval_script_source', 'provided'),
        exported_image=exported_image,
        image_uploaded=image_uploaded,
        pass2pass_only=pass2pass_only,
        eval_llm_validation=eval_validation_combined,
        final_base_image=final_base_image,
        agent_iters=agent_iters,
        total_iters=agent_iters,
        error=error,
        error_details=error_details if error_details else ctx.tool_state.get('error_details', ''),
        agent_exit_status=agent_exit_status,
        instance_with_results=instance_with_results,
        main_trajectory=main_trajectory,
        resolved=resolved,
        score=score,
    )
