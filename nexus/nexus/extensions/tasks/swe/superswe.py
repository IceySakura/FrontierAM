"""Run SuperSWE's required evaluation components using one model patch."""

import hashlib
import json

from pathlib import PurePosixPath
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from nexus.core.context import require_context
from nexus.core.task import task_registry
from nexus.dataset.auth import WedataAuthenticator
from nexus.dataset.task_files import task_files_from_spec
from nexus.extensions.tasks.swe.task import SwebenchTaskInput, SwebenchTaskOutput, run_swe_task


COMPONENTS_KEY = '_superswe_evaluation_components'
TaskInputT = TypeVar('TaskInputT', bound=SwebenchTaskInput)


class SupersweTaskInput(SwebenchTaskInput):
    instance_file: str = ''
    instance_sha256: str = Field(default='', pattern=r'^(?:[a-f0-9]{64})?$')
    wedata_key: SecretStr | None = Field(default=None, exclude=True)


async def _load_task_files_instance(
    task_input: TaskInputT,
) -> tuple[TaskInputT, dict[str, Any] | None]:
    instance_file = getattr(task_input, 'instance_file', '')
    if not instance_file:
        return task_input, None
    if task_input.instance is not None or task_input.dataset_path or task_input.remote_dataset:
        raise ValueError('SuperSWE instance_file cannot be combined with another instance source')
    path = PurePosixPath(instance_file)
    if path.is_absolute() or '..' in path.parts or '\\' in instance_file or path == PurePosixPath('.'):
        raise ValueError('SuperSWE instance_file must be a relative task-files member')
    expected_sha256 = getattr(task_input, 'instance_sha256', '')
    if not expected_sha256:
        raise ValueError('SuperSWE instance_file requires instance_sha256')
    ctx = require_context()
    files = ctx.task_files
    remote = files.spec.remote if files is not None else None
    if files is None or remote is None or files.spec.type != 'zip':
        raise ValueError('SuperSWE instance_file requires remote ZIP task_files')
    if remote.version in {'latest', 'dev', ''} or not files.spec.sha256 or files.spec.size is None:
        raise ValueError('SuperSWE task_files requires a fixed version, SHA-256 and size')
    wedata_key = getattr(task_input, 'wedata_key', None)
    if wedata_key is not None:
        # Keep platform-injected credentials local to this request, not in process-wide environment variables.
        files = task_files_from_spec(
            files.spec,
            run_id=ctx.run_id,
            authenticator=WedataAuthenticator(raw_key=wedata_key.get_secret_value()),
        )
        if files is None:
            raise ValueError('SuperSWE remote task_files could not be initialized')
    # Read the instance on the host. Reference patches and grading files must not be copied into the agent sandbox.
    raw = await files.read_bytes(instance_file)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('SuperSWE task-files instance SHA-256 mismatch')
    instance = json.loads(raw)
    if not isinstance(instance, dict):
        raise ValueError('SuperSWE task-files instance must be a JSON object')
    source = {
        'kind': 'remote_task_files',
        'instance_file': instance_file,
        'instance_sha256': expected_sha256,
        'archive_sha256': files.spec.sha256,
        'remote': remote.model_dump(),
    }
    return task_input.model_copy(update={'instance': instance}), source


class EvaluationComponent(BaseModel):
    model_config = ConfigDict(extra='forbid')

    name: str = Field(pattern=r'^[a-z][a-z0-9_]*$')
    instance: dict[str, Any]
    eval_timeout_seconds: int = Field(gt=0)


def _read_components(task_input: SwebenchTaskInput) -> tuple[dict[str, Any], list[EvaluationComponent]]:
    if task_input.instance is None:
        raise ValueError('SuperSWE requires an inline instance, including its required evaluation components')
    instance = dict(task_input.instance)
    raw_components = instance.pop(COMPONENTS_KEY, [])
    if not isinstance(raw_components, list):
        raise ValueError(f'{COMPONENTS_KEY} must be a list')
    if instance.get('instance_id') != task_input.instance_id:
        raise ValueError('task_config.instance_id does not match the inline instance')
    if task_input.eval_max_attempts != 1:
        raise ValueError('SuperSWE requires exactly one grading attempt per component')
    if task_input.use_partial_score or task_input.skip_infer_patch_eval or task_input.skip_empty_patch_eval:
        raise ValueError('SuperSWE requires full binary grading of all evaluation components')
    components = [EvaluationComponent.model_validate(item) for item in raw_components]
    seen = {'main'}
    for component in components:
        if component.name in seen:
            raise ValueError(f'Duplicate evaluation component: {component.name}')
        seen.add(component.name)
        for field in ('instance_id', 'repo', 'base_commit', 'dataset_type'):
            if not instance.get(field) or component.instance.get(field) != instance[field]:
                raise ValueError(f'Component {component.name} has a different or missing {field}')
        if not component.instance.get('eval_script'):
            raise ValueError(f'Component {component.name} has no evaluation script')
        if COMPONENTS_KEY in component.instance:
            raise ValueError('Nested evaluation components are not supported')
    return instance, components


def _combine(
    instance_id: str,
    expected: list[str],
    outputs: list[tuple[str, SwebenchTaskOutput]],
    pending_reason: str | None,
) -> SwebenchTaskOutput:
    main = outputs[0][1]
    patch_sha256 = hashlib.sha256(main.patch_text.encode('utf-8')).hexdigest()
    results = {
        name: {
            **output.model_dump(exclude={'patch_text', 'extra_fields'}),
            'patch_sha256': hashlib.sha256(output.patch_text.encode('utf-8')).hexdigest(),
        }
        for name, output in outputs
    }
    pending = [name for name in expected if name not in results]
    completed = not pending and all(output.completed and not output.error for _, output in outputs)
    resolved = completed and all(output.resolved for _, output in outputs)
    first_error = next(((name, output) for name, output in outputs if output.error), None)
    error = f'{first_error[0]}: {first_error[1].error}' if first_error else pending_reason
    extra_fields = {
        **main.extra_fields,
        'superswe_components': {
            'required': expected,
            'pending': pending,
            'patch_sha256': patch_sha256,
            'results': results,
        },
    }
    report_json = main.report_json
    if len(expected) > 1:
        report_json = json.dumps({
            instance_id: {
                'resolved': resolved,
                'completed': completed,
                'component_reports': {name: output.report_json for name, output in outputs},
            }
        })
    return main.model_copy(
        update={
            'score': 1.0 if resolved else 0.0,
            'completed': completed,
            'resolved': resolved,
            'error': error,
            'error_code': first_error[1].error_code if first_error else main.error_code,
            'report_json': report_json,
            'extra_fields': extra_fields,
        }
    )


@task_registry.register('superswe_task')
async def run_superswe_task(task_input: SupersweTaskInput) -> SwebenchTaskOutput:
    """Evaluate all components without invoking the model again for supplements."""
    task_input, instance_source = await _load_task_files_instance(task_input)
    instance, components = _read_components(task_input)
    primary_input = task_input.model_copy(update={'instance': instance})
    main = await run_swe_task(primary_input)
    expected = ['main', *(component.name for component in components)]
    outputs = [('main', main)]
    pending_reason = None
    if components and (not main.completed or main.error):
        pending_reason = 'Primary evaluation did not complete; supplementary evaluation was not started'
    else:
        for component in components:
            component_input = primary_input.model_copy(
                update={
                    'instance': dict(component.instance),
                    'patch_text': main.patch_text,
                    'run_oracle': False,
                    'eval_runtime_name': f'{task_input.eval_runtime_name}_component_{component.name}',
                    'eval_timeout_seconds': component.eval_timeout_seconds,
                    'eval_max_attempts': 1,
                }
            )
            try:
                output = await run_swe_task(component_input)
            except Exception as exc:
                output = SwebenchTaskOutput(
                    score=0.0,
                    completed=False,
                    resolved=False,
                    patch_text=main.patch_text,
                    error=f'{type(exc).__name__}: {exc}',
                )
            if output.patch_text != main.patch_text:
                output = output.model_copy(
                    update={
                        'completed': False,
                        'resolved': False,
                        'score': 0.0,
                        'error': 'Supplementary evaluation did not retain the original model patch',
                    }
                )
            outputs.append((component.name, output))
    output = _combine(task_input.instance_id, expected, outputs, pending_reason)
    if instance_source is not None:
        output = output.model_copy(update={'extra_fields': {**output.extra_fields, 'superswe_input': instance_source}})
    return output
