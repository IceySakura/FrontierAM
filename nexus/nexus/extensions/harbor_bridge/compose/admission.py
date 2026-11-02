"""Compile and admit one prebuilt native Compose topology."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from nexus.extensions.harbor_bridge.compose.errors import ComposeAdmissionError, ComposeConfigError
from nexus.extensions.harbor_bridge.compose.models import ComposePlan, ComposeServiceSummary, validate_service_name
from nexus.extensions.harbor_bridge.compose.parser import (
    LoadedComposeDefinition,
    compose_config_from_task,
    load_compose_definition,
)


_HOST_NAMESPACE_FIELDS = ('network_mode', 'pid', 'ipc', 'uts')
_DOCKER_SOCKET_MARKERS = ('/var/run/docker.sock', '/run/docker.sock')
_RESERVED_RUNTIME_MOUNT_TARGETS = (PurePosixPath('/nix'), PurePosixPath('/opt/nexus-agent'))
_RESERVED_TMPFS_EXACT_TARGETS = (PurePosixPath('/nix'), PurePosixPath('/nix/runtime'))
_RESERVED_TMPFS_SUBTREES = (
    PurePosixPath('/nix/store'),
    PurePosixPath('/nix/runtime/nix-env'),
    PurePosixPath('/opt/nexus-agent'),
)
_ALLOWED_TOP_LEVEL_FIELDS = frozenset({'configs', 'name', 'services', 'networks', 'volumes', 'version'})
_ALLOWED_SERVICE_FIELDS = frozenset({
    'image',
    'pull_policy',
    'entrypoint',
    'command',
    'configs',
    'container_name',
    'cap_drop',
    'deploy',
    'env_file',
    'working_dir',
    'environment',
    'depends_on',
    'healthcheck',
    'networks',
    'ports',
    'expose',
    'volumes',
    'tmpfs',
    'user',
    'restart',
    'security_opt',
    'shm_size',
    'labels',
    'hostname',
    'domainname',
    'dns',
    'dns_search',
    'extra_hosts',
    'init',
    'stop_grace_period',
    'stop_signal',
    'read_only',
    'stdin_open',
    'tty',
    'ulimits',
    'platform',
})
_DEPENDS_ON_CONDITIONS = frozenset({'service_started', 'service_healthy', 'service_completed_successfully'})
_ALLOWED_HEALTHCHECK_FIELDS = frozenset({
    'test',
    'interval',
    'timeout',
    'retries',
    'start_period',
    'start_interval',
    'disable',
})


def compile_compose_plan(
    task_dir: Path,
    config: dict[str, Any],
    harbor_task: Any,
    *,
    controller_image: str,
    workdir: str,
    environment: dict[str, str],
    verifier_mode: str,
) -> ComposePlan | None:
    """Compile a strict ComposePlan, or return None for a single-container task."""
    compose = compose_config_from_task(config)
    if compose is None:
        return None

    raw_environment = config.get('environment', {}) or {}
    if not isinstance(raw_environment, dict):
        raise ComposeConfigError('[environment] must be a table')
    if raw_environment.get('enable_dind') is not True:
        raise ComposeConfigError('Native Compose requires environment.enable_dind = true')
    if raw_environment.get('browser') is not None:
        raise ComposeConfigError('Native Compose v1 does not support environment.browser')
    if verifier_mode not in {'shared', 'separate'}:
        raise ComposeConfigError(f'Unsupported native Compose verifier mode: {verifier_mode!r}')

    controller_image = _validate_image_reference('environment.docker_image Compose controller', controller_image)
    loaded = load_compose_definition(task_dir, compose.definition)
    _validate_top_level_resources(loaded.document)

    bind_paths: set[str] = set()
    declared_volumes = set((loaded.document.get('volumes') or {}).keys())
    declared_networks = set((loaded.document.get('networks') or {}).keys()) | {'default'}
    dependencies: dict[str, list[str]] = {}
    inferred_one_shots: set[str] = set()
    summaries: dict[str, ComposeServiceSummary] = {}
    for service_name, service in loaded.services.items():
        image = _validate_service_image(service_name, service)
        _validate_service_security(service_name, service)
        _validate_service_environment(service_name, service)
        _validate_service_tmpfs(service_name, service)
        _validate_extra_hosts(service_name, service)
        has_healthcheck = _validate_healthcheck(service_name, service)
        _validate_service_networks(service_name, service, declared_networks)
        dependencies[service_name] = _service_dependencies(service_name, service)
        inferred_one_shots.update(_completed_dependencies(service))
        bind_paths.update(
            _collect_service_bind_paths(
                task_dir,
                loaded,
                service_name,
                service,
                declared_volumes,
            )
        )
        summaries[service_name] = ComposeServiceSummary(
            image=image,
            depends_on=dependencies[service_name],
            has_healthcheck=has_healthcheck,
        )

    service_names = set(loaded.services)
    for service_name, referenced in dependencies.items():
        missing = sorted(set(referenced) - service_names)
        if missing:
            raise ComposeAdmissionError(f'Compose service {service_name!r} depends on unknown services: {missing}')
    _validate_dependency_graph(dependencies, loaded.services, summaries)

    one_shots = sorted(set(compose.one_shot_services) | inferred_one_shots)
    missing_one_shots = sorted(set(one_shots) - service_names)
    if missing_one_shots:
        raise ComposeAdmissionError(f'one_shot_services references unknown services: {missing_one_shots}')
    if compose.primary_service not in service_names:
        raise ComposeAdmissionError(
            f'primary_service {compose.primary_service!r} is not present in Compose services: {sorted(service_names)}'
        )
    if compose.primary_service in one_shots:
        raise ComposeAdmissionError(f'primary_service {compose.primary_service!r} cannot be a one-shot service')

    for service_name in one_shots:
        restart = str(loaded.services[service_name].get('restart') or '').strip().lower()
        if restart not in {'', 'no', 'none'}:
            raise ComposeAdmissionError(f'one-shot service {service_name!r} cannot use restart policy {restart!r}')
        summaries[service_name] = summaries[service_name].model_copy(update={'one_shot': True})

    primary = loaded.services[compose.primary_service]
    primary_image = summaries[compose.primary_service].image
    compose_workdir = primary.get('working_dir')
    if compose_workdir is not None and str(compose_workdir).strip() != workdir:
        raise ComposeAdmissionError(
            f'environment.workdir must match primary service {compose.primary_service!r} working_dir: '
            f'{workdir!r} != {compose_workdir!r}'
        )

    _validate_harbor_service_references(harbor_task, service_names, compose.primary_service)

    return ComposePlan(
        definition=loaded.relative_path,
        definition_sha256=loaded.sha256,
        controller_image=controller_image,
        primary_service=compose.primary_service,
        primary_image=primary_image,
        workdir=workdir,
        environment=environment,
        startup_timeout_sec=compose.startup_timeout_sec,
        startup_jitter_max=compose.startup_jitter_max,
        start_retry_attempts=compose.start_retry_attempts,
        start_retry_backoff_sec=compose.start_retry_backoff_sec,
        start_retry_jitter_max=compose.start_retry_jitter_max,
        shutdown_timeout_sec=compose.shutdown_timeout_sec,
        one_shot_services=one_shots,
        services=summaries,
        bind_paths=sorted(bind_paths),
    )


def compose_identity_paths(task_dir: Path, config: dict[str, Any]) -> tuple[Path, ...]:
    """Return the lock and runtime bind inputs that affect a Compose task."""
    compose = compose_config_from_task(config)
    if compose is None:
        return ()
    loaded = load_compose_definition(task_dir, compose.definition)
    declared_volumes = set((loaded.document.get('volumes') or {}).keys())
    paths = {loaded.path}
    for service_name, service in loaded.services.items():
        for relative in _collect_service_bind_paths(
            task_dir,
            loaded,
            service_name,
            service,
            declared_volumes,
        ):
            paths.add(task_dir.resolve() / relative)
    return tuple(sorted(paths, key=lambda path: path.relative_to(task_dir.resolve()).as_posix()))


def _validate_service_image(service_name: str, service: dict[str, Any]) -> str:
    if service.get('build') is not None:
        raise ComposeAdmissionError(f'Compose service {service_name!r} uses forbidden runtime build')
    if str(service.get('pull_policy') or '').strip().lower() == 'build':
        raise ComposeAdmissionError(f'Compose service {service_name!r} uses forbidden pull_policy=build')
    image = service.get('image')
    if not isinstance(image, str) or not image.strip():
        raise ComposeAdmissionError(f'Compose service {service_name!r} must define a prebuilt image')
    return _validate_image_reference(f'Compose service {service_name!r}', image)


def _validate_image_reference(label: str, image: str) -> str:
    image = image.strip()
    if not image or any(character.isspace() for character in image):
        raise ComposeAdmissionError(f'{label} image must be a non-empty reference without whitespace: {image!r}')
    image_name = image.split('@', 1)[0]
    first_component = image_name.split('/', 1)[0]
    if '/' not in image_name or not (
        first_component == 'localhost' or '.' in first_component or ':' in first_component
    ):
        raise ComposeAdmissionError(f'{label} image must include an explicit registry: {image!r}')
    return image


def _validate_service_security(service_name: str, service: dict[str, Any]) -> None:
    if service.get('privileged'):
        raise ComposeAdmissionError(f'Compose service {service_name!r} cannot be privileged')
    if service.get('use_api_socket'):
        raise ComposeAdmissionError(f'Compose service {service_name!r} cannot enable use_api_socket')
    if service.get('volumes_from'):
        raise ComposeAdmissionError(f'Compose service {service_name!r} cannot use volumes_from')
    security_opt = service.get('security_opt')
    if security_opt is not None:
        if not isinstance(security_opt, list) or not all(isinstance(item, str) for item in security_opt):
            raise ComposeAdmissionError(f'Compose service {service_name!r} security_opt must be a list of strings')
        allowed_security_opts = {'no-new-privileges:true', 'no-new-privileges=true'}
        unsupported_security_opts = sorted(
            item for item in security_opt if item.strip().lower() not in allowed_security_opts
        )
        if unsupported_security_opts:
            raise ComposeAdmissionError(
                f'Compose service {service_name!r} uses unsupported security_opt values: {unsupported_security_opts}'
            )
    for field in _HOST_NAMESPACE_FIELDS:
        value = str(service.get(field) or '').strip().lower()
        if value == 'host' or value.startswith(('container:', 'service:')):
            raise ComposeAdmissionError(f'Compose service {service_name!r} cannot use {field}={value}')
    unsupported = sorted(set(service) - _ALLOWED_SERVICE_FIELDS)
    if unsupported:
        raise ComposeAdmissionError(f'Compose service {service_name!r} contains unsupported fields: {unsupported}')


def _validate_service_environment(service_name: str, service: dict[str, Any]) -> None:
    environment = service.get('environment')
    if environment is None:
        return
    if isinstance(environment, dict):
        for key, value in environment.items():
            if not isinstance(key, str) or not key:
                raise ComposeAdmissionError(f'Compose service {service_name!r} has an invalid environment key')
            if value is None:
                raise ComposeAdmissionError(
                    f'Compose service {service_name!r} environment variable {key!r} must define a concrete value'
                )
        return
    if isinstance(environment, list):
        for item in environment:
            if not isinstance(item, str) or '=' not in item or not item.split('=', 1)[0]:
                raise ComposeAdmissionError(f'Compose service {service_name!r} has an invalid environment entry')
        return
    raise ComposeAdmissionError(f'Compose service {service_name!r} environment must be a mapping or list')


def _validate_service_tmpfs(service_name: str, service: dict[str, Any]) -> None:
    tmpfs = service.get('tmpfs')
    if tmpfs is None:
        return
    entries = [tmpfs] if isinstance(tmpfs, str) else tmpfs
    if not isinstance(entries, list) or not all(isinstance(entry, str) and entry.strip() for entry in entries):
        raise ComposeAdmissionError(f'Compose service {service_name!r} tmpfs must be a string or list of strings')
    for entry in entries:
        target = entry.split(':', 1)[0].strip()
        target_path = PurePosixPath(target)
        if not target or not target_path.is_absolute() or '..' in target_path.parts:
            raise ComposeAdmissionError(
                f'Compose service {service_name!r} tmpfs target must be an absolute container path: {target!r}'
            )
        if target_path in _RESERVED_TMPFS_EXACT_TARGETS or any(
            target_path == reserved or target_path.is_relative_to(reserved) for reserved in _RESERVED_TMPFS_SUBTREES
        ):
            raise ComposeAdmissionError(
                f'Compose service {service_name!r} tmpfs target is reserved for Nexus Runtime: {target!r}'
            )


def _validate_extra_hosts(service_name: str, service: dict[str, Any]) -> None:
    extra_hosts = service.get('extra_hosts')
    if extra_hosts is None:
        return
    if not isinstance(extra_hosts, (dict, list)):
        raise ComposeAdmissionError(f'Compose service {service_name!r} extra_hosts must be a mapping or list')
    if 'host-gateway' in str(extra_hosts).lower():
        raise ComposeAdmissionError(f'Compose service {service_name!r} cannot use host-gateway')


def _validate_healthcheck(service_name: str, service: dict[str, Any]) -> bool:
    healthcheck = service.get('healthcheck')
    if healthcheck is None:
        return False
    if not isinstance(healthcheck, dict):
        raise ComposeAdmissionError(f'Compose service {service_name!r} healthcheck must be a mapping')
    unsupported = sorted(set(healthcheck) - _ALLOWED_HEALTHCHECK_FIELDS)
    if unsupported:
        raise ComposeAdmissionError(
            f'Compose service {service_name!r} healthcheck contains unsupported fields: {unsupported}'
        )
    disabled = healthcheck.get('disable', False)
    if not isinstance(disabled, bool):
        raise ComposeAdmissionError(f'Compose service {service_name!r} healthcheck.disable must be a boolean')
    if disabled:
        return False
    test = healthcheck.get('test')
    if isinstance(test, list):
        if not test or str(test[0]).upper() == 'NONE':
            return False
    elif not isinstance(test, str) or not test.strip():
        raise ComposeAdmissionError(f'Compose service {service_name!r} healthcheck must define a non-empty test')
    retries = healthcheck.get('retries')
    if retries is not None and (not isinstance(retries, int) or isinstance(retries, bool) or retries <= 0):
        raise ComposeAdmissionError(f'Compose service {service_name!r} healthcheck.retries must be a positive integer')
    return True


def _validate_service_networks(
    service_name: str,
    service: dict[str, Any],
    declared_networks: set[str],
) -> None:
    networks = service.get('networks')
    if networks is None:
        return
    if isinstance(networks, list):
        referenced = networks
    elif isinstance(networks, dict):
        referenced = list(networks)
    else:
        raise ComposeAdmissionError(f'Compose service {service_name!r} networks must be a list or mapping')
    if not all(isinstance(item, str) and item for item in referenced):
        raise ComposeAdmissionError(f'Compose service {service_name!r} has invalid network references')
    missing = sorted(set(referenced) - declared_networks)
    if missing:
        raise ComposeAdmissionError(f'Compose service {service_name!r} references undeclared networks: {missing}')


def _validate_top_level_resources(document: dict[str, Any]) -> None:
    unsupported = sorted(set(document) - _ALLOWED_TOP_LEVEL_FIELDS)
    if unsupported:
        raise ComposeAdmissionError(f'Compose lock contains unsupported top-level fields: {unsupported}')
    for section_name in ('networks', 'volumes'):
        section = document.get(section_name) or {}
        if not isinstance(section, dict):
            raise ComposeAdmissionError(f'Compose top-level {section_name} must be a mapping')
        for name, item in section.items():
            if item is None:
                continue
            if not isinstance(item, dict):
                raise ComposeAdmissionError(f'Compose {section_name[:-1]} {name!r} must be a mapping')
            if item.get('driver_opts'):
                raise ComposeAdmissionError(f'Compose {section_name[:-1]} {name!r} cannot set driver_opts')
            driver = str(item.get('driver') or '').strip().lower()
            allowed_drivers = {'', 'bridge'} if section_name == 'networks' else {'', 'local'}
            if driver not in allowed_drivers:
                raise ComposeAdmissionError(f'Compose {section_name[:-1]} {name!r} uses unsupported driver {driver!r}')
    if document.get('secrets'):
        raise ComposeAdmissionError('Native Compose v1 does not support top-level secrets')


def _service_dependencies(service_name: str, service: dict[str, Any]) -> list[str]:
    depends_on = service.get('depends_on') or {}
    if isinstance(depends_on, list):
        dependencies = depends_on
    elif isinstance(depends_on, dict):
        dependencies = list(depends_on)
        for dependency, dependency_config in depends_on.items():
            if dependency_config is None:
                continue
            if not isinstance(dependency_config, dict):
                raise ComposeAdmissionError(
                    f'Compose service {service_name!r} dependency {dependency!r} must be a mapping'
                )
            unsupported = sorted(set(dependency_config) - {'condition', 'required', 'restart'})
            if unsupported:
                raise ComposeAdmissionError(
                    f'Compose service {service_name!r} dependency {dependency!r} '
                    f'contains unsupported fields: {unsupported}'
                )
            condition = str(dependency_config.get('condition') or 'service_started')
            if condition not in _DEPENDS_ON_CONDITIONS:
                raise ComposeAdmissionError(
                    f'Compose service {service_name!r} dependency {dependency!r} '
                    f'uses unsupported condition {condition!r}'
                )
            for boolean_field in ('required', 'restart'):
                if boolean_field in dependency_config and not isinstance(dependency_config[boolean_field], bool):
                    raise ComposeAdmissionError(
                        f'Compose service {service_name!r} dependency {dependency!r} '
                        f'field {boolean_field!r} must be a boolean'
                    )
    else:
        raise ComposeAdmissionError(f'Compose service {service_name!r} depends_on must be a list or mapping')
    if not all(isinstance(item, str) and item for item in dependencies):
        raise ComposeAdmissionError(f'Compose service {service_name!r} has invalid depends_on entries')
    return list(dict.fromkeys(dependencies))


def _validate_dependency_graph(
    dependencies: dict[str, list[str]],
    services: dict[str, dict[str, Any]],
    summaries: dict[str, ComposeServiceSummary],
) -> None:
    """Reject dependency cycles and invalid healthy conditions."""
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(service_name: str, path: tuple[str, ...]) -> None:
        if service_name in visiting:
            cycle = ' -> '.join((*path, service_name))
            raise ComposeAdmissionError(f'Compose depends_on contains a cycle: {cycle}')
        if service_name in visited:
            return
        visiting.add(service_name)
        for dependency in dependencies.get(service_name, []):
            visit(dependency, (*path, service_name))
        visiting.remove(service_name)
        visited.add(service_name)

    for service_name in dependencies:
        visit(service_name, ())

    for service_name, service in services.items():
        depends_on = service.get('depends_on') or {}
        if not isinstance(depends_on, dict):
            continue
        for dependency, dependency_config in depends_on.items():
            if (
                isinstance(dependency_config, dict)
                and dependency_config.get('condition') == 'service_healthy'
                and not summaries[dependency].has_healthcheck
            ):
                raise ComposeAdmissionError(
                    f'Compose service {service_name!r} requires {dependency!r} to be healthy, '
                    'but that service has no healthcheck'
                )


def _completed_dependencies(service: dict[str, Any]) -> set[str]:
    depends_on = service.get('depends_on') or {}
    if not isinstance(depends_on, dict):
        return set()
    return {
        name
        for name, config in depends_on.items()
        if isinstance(config, dict) and config.get('condition') == 'service_completed_successfully'
    }


def _collect_service_bind_paths(
    task_dir: Path,
    loaded: LoadedComposeDefinition,
    service_name: str,
    service: dict[str, Any],
    declared_volumes: set[str],
) -> set[str]:
    volumes = service.get('volumes') or []
    if not isinstance(volumes, list):
        raise ComposeAdmissionError(f'Compose service {service_name!r} volumes must be a list')
    paths: set[str] = set()
    for volume in volumes:
        source: str | None = None
        is_bind = False
        if isinstance(volume, str):
            parts = _split_short_volume_spec(volume)
            if len(parts) == 1:
                target = parts[0]
            else:
                source = parts[0]
                target = parts[1]
            _validate_volume_target(service_name, target)
            if any(marker in volume for marker in _DOCKER_SOCKET_MARKERS):
                raise ComposeAdmissionError(f'Compose service {service_name!r} cannot mount the Docker socket')
            if source:
                if '$' in source:
                    continue
                is_bind = source == '.' or source.startswith(('./', '../', '/', '~'))
                if not is_bind:
                    _validate_named_volume(service_name, source, declared_volumes)
        elif isinstance(volume, dict):
            unsupported = sorted(
                set(volume) - {'type', 'source', 'target', 'read_only', 'consistency', 'bind', 'volume', 'tmpfs'}
            )
            if unsupported:
                raise ComposeAdmissionError(
                    f'Compose service {service_name!r} volume contains unsupported fields: {unsupported}'
                )
            volume_type = str(volume.get('type') or 'volume').strip().lower()
            target = volume.get('target')
            if not isinstance(target, str):
                raise ComposeAdmissionError(f'Compose service {service_name!r} volume must define a target')
            _validate_volume_target(service_name, target)
            if any(marker in str(volume) for marker in _DOCKER_SOCKET_MARKERS):
                raise ComposeAdmissionError(f'Compose service {service_name!r} cannot mount the Docker socket')
            if volume_type == 'bind':
                source_value = volume.get('source')
                if not isinstance(source_value, str) or not source_value:
                    raise ComposeAdmissionError(f'Compose service {service_name!r} bind mount must define a source')
                bind_options = volume.get('bind')
                if bind_options is not None and (
                    not isinstance(bind_options, dict)
                    or set(bind_options) - {'propagation', 'create_host_path', 'selinux'}
                ):
                    raise ComposeAdmissionError(
                        f'Compose service {service_name!r} bind options are invalid or unsupported'
                    )
                source = source_value
                is_bind = True
            elif volume_type == 'volume':
                source_value = volume.get('source')
                if source_value is not None:
                    if not isinstance(source_value, str) or not source_value:
                        raise ComposeAdmissionError(
                            f'Compose service {service_name!r} named volume source must be a string'
                        )
                    _validate_named_volume(service_name, source_value, declared_volumes)
                volume_options = volume.get('volume')
                if volume_options is not None and (
                    not isinstance(volume_options, dict) or set(volume_options) - {'nocopy', 'subpath'}
                ):
                    raise ComposeAdmissionError(
                        f'Compose service {service_name!r} volume options are invalid or unsupported'
                    )
            elif volume_type == 'tmpfs':
                if volume.get('source') is not None:
                    raise ComposeAdmissionError(f'Compose service {service_name!r} tmpfs volume cannot define a source')
            else:
                raise ComposeAdmissionError(
                    f'Compose service {service_name!r} uses unsupported volume type {volume_type!r}'
                )
        else:
            raise ComposeAdmissionError(f'Compose service {service_name!r} has an invalid volume entry')

        if not is_bind or source is None:
            continue
        if any(marker in source for marker in _DOCKER_SOCKET_MARKERS):
            raise ComposeAdmissionError(f'Compose service {service_name!r} cannot mount the Docker socket')
        if '$' in source:
            continue
        source_path = PurePosixPath(source)
        if source_path.is_absolute():
            continue
        if source.startswith('~') or '..' in source_path.parts:
            raise ComposeAdmissionError(
                f'Compose service {service_name!r} bind source must be task-local and relative: {source!r}'
            )
        root = task_dir.resolve()
        candidate = loaded.path.parent / Path(*source_path.parts)
        _reject_bind_symlink_components(root, candidate)
        resolved = candidate.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ComposeAdmissionError(
                f'Compose service {service_name!r} bind source escapes the task directory: {source!r}'
            ) from exc
        if not resolved.exists():
            raise ComposeAdmissionError(f'Compose service {service_name!r} bind source does not exist: {source!r}')
        _reject_symlink_tree(resolved)
        paths.add(relative.as_posix())
    return paths


def _split_short_volume_spec(value: str) -> list[str]:
    """Split source:target:mode without treating ``${VAR:-x}`` colons as separators."""
    parts: list[str] = []
    start = 0
    template_depth = 0
    index = 0
    while index < len(value):
        if value.startswith('${', index):
            template_depth += 1
            index += 2
            continue
        if value[index] == '}' and template_depth:
            template_depth -= 1
        elif value[index] == ':' and template_depth == 0 and len(parts) < 2:
            parts.append(value[start:index])
            start = index + 1
        index += 1
    parts.append(value[start:])
    return parts


def _validate_volume_target(service_name: str, target: str) -> None:
    target_path = PurePosixPath(target)
    if not target or not target_path.is_absolute() or '..' in target_path.parts:
        raise ComposeAdmissionError(
            f'Compose service {service_name!r} volume target must be an absolute container path: {target!r}'
        )
    if any(
        target_path == reserved or target_path.is_relative_to(reserved) for reserved in _RESERVED_RUNTIME_MOUNT_TARGETS
    ):
        raise ComposeAdmissionError(
            f'Compose service {service_name!r} volume target is reserved for Nexus Runtime: {target!r}'
        )


def _validate_named_volume(service_name: str, source: str, declared_volumes: set[str]) -> None:
    try:
        validate_service_name(source)
    except ValueError as exc:
        raise ComposeAdmissionError(
            f'Compose service {service_name!r} has an invalid named volume reference: {source!r}'
        ) from exc
    if source not in declared_volumes:
        raise ComposeAdmissionError(f'Compose service {service_name!r} references undeclared volume {source!r}')


def _reject_bind_symlink_components(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ComposeAdmissionError(f'Compose bind source escapes the task directory: {path}') from exc
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ComposeAdmissionError(f'Compose bind source path must not contain symlinks: {current}')


def _reject_symlink_tree(path: Path) -> None:
    if path.is_symlink():
        raise ComposeAdmissionError(f'Compose bind source must not be a symlink: {path}')
    if path.is_dir():
        for child in path.rglob('*'):
            if child.is_symlink():
                raise ComposeAdmissionError(f'Compose bind source tree must not contain symlinks: {child}')


def _validate_harbor_service_references(
    harbor_task: Any,
    services: set[str],
    primary_service: str,
) -> None:
    def resolve(service: str | None) -> str:
        return primary_service if service in (None, 'main') else service

    for artifact in harbor_task.config.artifacts:
        service = getattr(artifact, 'service', None) if not isinstance(artifact, str) else None
        resolved = resolve(service)
        if resolved not in services:
            raise ComposeAdmissionError(f'Artifact references unknown Compose service {service!r}')
    for hook in harbor_task.config.verifier.collect:
        resolved = resolve(hook.service)
        if resolved not in services:
            raise ComposeAdmissionError(f'Verifier collect references unknown Compose service {hook.service!r}')


__all__ = ['compile_compose_plan', 'compose_identity_paths']
