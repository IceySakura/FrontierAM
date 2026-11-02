"""build_cc TASK-side finalize hook: upload the built image, then wipe DinD state.

Lives in the TASK package (nexus.extensions.projects.code_mt.task) on purpose: build_cc
runs `code_mt_task` (this package) with the native `claude_code` agent. The
claude_code agent is off-limits (must not be modified), and build_cc must NOT
pull in the code_mt *agent* package at all. So both the upload and the per-turn
DinD wipe live here, self-contained, importing nothing from nexus.extensions.agents.

Wired as the build_cc `eval_cleanup_strategy`: the MT loop imports and calls it
(via task.py's `_run_hook`) each turn after scoring, before the next retry.
Non-build_cc subsets leave eval_cleanup_strategy empty, so task.py short-circuits
and never imports this — zero impact on builder/tb/swe/repoqa.

Order per turn (in the finally, after compute_score, before the score>=1.0 break):
  1. UPLOAD cc_built:latest to the registry IF the turn passed (merged score
     >= 1.0) and an upload strategy is configured. cc_built lives inside the
     pod's Docker-in-Docker dockerd, so every docker command runs INSIDE the
     pod via runtime.run_command. Upload must run BEFORE the wipe (step 2).
  2. WIPE the inner docker state so turn N+1's agent starts from zero.

Anti-blocking contract (a legacy bug once let the upload block/abort the run):
the upload is entirely best-effort — every docker command is timeout-bounded and
the whole upload is wrapped in try/except that only logs. An upload failure NEVER
changes the score, NEVER raises, and NEVER prevents the wipe. task.py also wraps
this call in contextlib.suppress(Exception).

Uploaded image naming (MUST differ from swe_builder's hardcoded
swe_builder_eval_prs_<id>:latest, and MUST carry a tag):

    {registry_prefix}.{upload_image_template.format(safe_id)}:{upload_tag}

defaults upload_image_template='swe_builder_cc_prs_{}', upload_tag='v0'. safe_id
is the ORIGINAL raw instance id (NOT the build_cc__ prefixed one), lower-cased
with '/' and '-' -> '_'.

Credentials come from env (REGISTRY_USERNAME / REGISTRY_PASSWORD) first, with a
task_config [other] fallback. They are NEVER hardcoded and NEVER serialized into
task.toml; if neither source has them, docker login is skipped.

task_config ([other]) knobs:
  upload_strategy         : 'none' (default) | 'fail2pass' | 'all_built'
  upload_image_template   : default 'swe_builder_cc_prs_{}'
  upload_tag              : default 'v0'
  upload_instance_id      : raw instance id to embed (required for a real upload)
  upload_registry_prefix  : default the shared t3build mirror
  upload_registry_host    : default 'mirrors.tencent.com'
  upload_timeout_sec      : default 1800
  registry_username / registry_password : creds fallback (env wins)
  cc_built_image / cc_env_container : inner image/container names (for the wipe)
"""

from __future__ import annotations

import contextlib
import logging
import os

from typing import Any


logger = logging.getLogger(__name__)


DEFAULT_BUILT_IMAGE = 'cc_built:latest'
DEFAULT_ENV_CONTAINER = 'cc_env'
DEFAULT_REGISTRY_PREFIX = 'mirrors.tencent.com/hunyuan_yanguan/t3build_repolaunch_midtrain'
DEFAULT_REGISTRY_HOST = 'mirrors.tencent.com'
DEFAULT_IMAGE_TEMPLATE = 'swe_builder_cc_prs_{}'
DEFAULT_UPLOAD_TAG = 'v0'
DEFAULT_UPLOAD_TIMEOUT = 1800


async def finalize(
    *,
    ctx: Any,
    runtime: Any,
    task_input: Any,
    eval_result: dict[str, Any],
) -> None:
    """Upload the built image on success, then wipe inner docker state.

    Never raises. The upload is fully best-effort and gated on merged success;
    the wipe always runs."""
    try:
        await _maybe_upload(ctx=ctx, runtime=runtime, eval_result=eval_result)
    except Exception as e:  # noqa: BLE001 - upload must never abort the loop
        logger.warning('[build_cc_upload] upload step failed (non-fatal): %s', e)

    # ALWAYS wipe, even if upload raised/failed.
    await _wipe_inner_docker(ctx=ctx, runtime=runtime, eval_result=eval_result)


async def _maybe_upload(*, ctx: Any, runtime: Any, eval_result: dict[str, Any]) -> None:
    tc = _task_config(ctx)
    strategy = str(_cfg(tc, 'upload_strategy', 'none')).strip().lower()
    if strategy in ('', 'none'):
        return  # upload disabled — normal for every non-build_cc subset

    merged = eval_result.get('merged_score')
    if merged is None:
        merged = eval_result.get('score')
    passed = merged is not None and float(merged) >= 1.0

    if strategy == 'fail2pass' and not passed:
        logger.info('[build_cc_upload] skip upload (strategy=fail2pass, merged_score=%s)', merged)
        return
    # 'all_built' uploads regardless of score (whenever the image exists).

    raw_id = str(_cfg(tc, 'upload_instance_id', '') or '').strip()
    if not raw_id:
        logger.warning('[build_cc_upload] no upload_instance_id in task_config — skip upload')
        return
    safe_id = raw_id.lower().replace('/', '_').replace('-', '_')

    prefix = str(_cfg(tc, 'upload_registry_prefix', DEFAULT_REGISTRY_PREFIX)).strip().rstrip('/')
    host = str(_cfg(tc, 'upload_registry_host', DEFAULT_REGISTRY_HOST)).strip()
    template = str(_cfg(tc, 'upload_image_template', DEFAULT_IMAGE_TEMPLATE)).strip()
    tag = str(_cfg(tc, 'upload_tag', DEFAULT_UPLOAD_TAG)).strip() or DEFAULT_UPLOAD_TAG
    try:
        timeout = float(_cfg(tc, 'upload_timeout_sec', DEFAULT_UPLOAD_TIMEOUT))
    except (TypeError, ValueError):
        timeout = DEFAULT_UPLOAD_TIMEOUT

    image_name = template.format(safe_id)
    remote_tag = f'{prefix}.{image_name}:{tag}'
    built = _built_image(ctx)

    # Guard: only push if the built image actually exists in the pod's dockerd.
    inspect = await runtime.run_command(command=f'docker image inspect {built} >/dev/null 2>&1; echo rc=$?', timeout=60)
    if 'rc=0' not in (getattr(inspect, 'stdout', '') or ''):
        logger.warning('[build_cc_upload] built image %s not present — skip upload', built)
        return

    user = os.environ.get('REGISTRY_USERNAME') or str(_cfg(tc, 'registry_username', '') or '')
    pw = os.environ.get('REGISTRY_PASSWORD') or str(_cfg(tc, 'registry_password', '') or '')

    if user and pw:
        # Password via stdin so it never appears in the process listing / command
        # string / logs. --password-stdin reads the first stdin line.
        login = await runtime.run_command(
            command=f'docker login {host} --username {user} --password-stdin; echo rc=$?',
            stdin=pw,
            timeout=120,
        )
        if 'rc=0' not in (getattr(login, 'stdout', '') or ''):
            logger.warning('[build_cc_upload] docker login to %s failed — attempting push anyway', host)
    else:
        logger.info('[build_cc_upload] no registry creds (env/config) — attempting anonymous push')

    # Strip the test oracle (/eval.sh) and any leftover /tmp/*.patch from the
    # image BEFORE publishing, so the shipped stage-2 image cannot leak the test
    # to the fix-agent. The oracle's eval_script_content is preserved in the DB
    # (eval_result below), so stage-2 reconstructs tests/test_cmd.sh from there
    # (sftimg/t3build OMNIGRIL contract) instead of relying on a baked /eval.sh.
    # The removal is added as a new layer via `docker build` from a one-line
    # Dockerfile, which preserves the built image's ENTRYPOINT/CMD/ENV/WORKDIR.
    # If the strip build fails we do NOT push a (leaky) image.
    strip_push = (
        'set -e; STRIP_DIR=$(mktemp -d); '
        "printf 'FROM %s\\nRUN rm -f /eval.sh /tmp/*.patch 2>/dev/null || true\\n' "
        f'\'{built}\' > "$STRIP_DIR/Dockerfile"; '
        f'docker build -t {remote_tag} "$STRIP_DIR"; '
        f'docker push {remote_tag}; rm -rf "$STRIP_DIR"'
    )
    push_cmd = f'({strip_push}); echo rc=$?'
    push = await runtime.run_command(command=push_cmd, timeout=timeout)
    out = getattr(push, 'stdout', '') or ''
    if 'rc=0' in out:
        eval_result['uploaded_image'] = remote_tag
        # tool_state may be read-only/absent
        with contextlib.suppress(Exception):
            ctx.tool_state['uploaded_image'] = remote_tag
        logger.info('[build_cc_upload] pushed built image -> %s', remote_tag)
    else:
        logger.warning('[build_cc_upload] push of %s did not report success (non-fatal)', remote_tag)

    # Best-effort: drop the local remote-tagged copy so the wipe below leaves no
    # dangling registry tag behind.
    with contextlib.suppress(Exception):
        await runtime.run_command(command=f'docker rmi {remote_tag} 2>/dev/null; true', timeout=60)


async def _wipe_inner_docker(*, ctx: Any, runtime: Any, eval_result: dict[str, Any]) -> None:
    """Wipe inner docker state (built image + scratch container) between turns.

    Best-effort: guarded so a missing image/container or a transient dockerd
    hiccup can never abort the MT loop."""
    # Backfill feedback-template fields so the next turn's mt_feedback.md render
    # never KeyErrors when reward.json was missing or the judge crashed.
    if 'stage' not in eval_result:
        eval_result['stage'] = 'reward_missing' if not eval_result.get('test_output') else 'judge_crashed'
    eval_result.setdefault('built_image', _built_image(ctx))
    eval_result.setdefault('error', '')

    img = _built_image(ctx)
    container = _env_container(ctx)

    # Order: remove the scratch container first, then prune dangling layers.
    # NOTE: we intentionally do NOT `docker rmi -f {img}` here — the built image
    # (cc_built:latest) must survive across MT turns so heavy repos can build
    # incrementally instead of from scratch every turn. `image prune -f` only
    # reclaims dangling (untagged) layers and leaves the tagged image intact.
    cmd = (
        f'docker rm -f {container} 2>/dev/null; '
        'docker container prune -f 2>/dev/null; '
        'docker image prune -f 2>/dev/null; '
        'true'
    )
    try:
        await runtime.run_command(command=cmd, timeout=120)
    except Exception as e:  # noqa: BLE001 - cleanup must never abort the loop
        logger.warning('[build_cc_upload] docker wipe failed (non-fatal): %s', e)

    logger.debug(
        '[build_cc_upload] wiped inner docker (image=%s, container=%s, stage=%s, attempt=%s)',
        img,
        container,
        eval_result.get('stage', '?'),
        eval_result.get('attempt', '?'),
    )


def _task_config(ctx: Any) -> dict[str, Any]:
    return getattr(ctx, 'task_config', {}) or {}


def _cfg(tc: dict[str, Any], key: str, default: Any) -> Any:
    if key in tc:
        return tc[key]
    for nested in ('extra', 'other'):
        sub = tc.get(nested)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return default


def _built_image(ctx: Any) -> str:
    tc = _task_config(ctx)
    return _cfg(tc, 'cc_built_image', None) or DEFAULT_BUILT_IMAGE


def _env_container(ctx: Any) -> str:
    tc = _task_config(ctx)
    return _cfg(tc, 'cc_env_container', None) or DEFAULT_ENV_CONTAINER
