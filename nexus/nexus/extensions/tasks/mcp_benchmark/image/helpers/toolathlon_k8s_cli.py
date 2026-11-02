#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys

from pathlib import Path
from typing import Any

import yaml


try:
    from .toolathlon_k8s_state import (
        ROOT_PATH_ENV,
        STORE_PATH_ENV,
        K8sStateError,
        add_helm_repo,
        apply_documents,
        case_root_from_cwd,
        cluster_info_text,
        create_cluster,
        delete_cluster,
        delete_resource,
        get_helm_values,
        get_resource,
        helm_install_or_upgrade,
        k8s_root_path,
        list_cluster_names,
        list_helm_releases,
        list_resources,
        mysql_exec_import,
        mysql_load_schema,
        parse_yaml_documents,
        patch_resource_json,
        pod_copy_in,
        read_kubeconfig_cluster_name,
        register_port_forward,
        rollout_ready,
        scale_workload,
        set_mysql_user,
        wait_pods_ready,
    )
except ImportError:
    from toolathlon_k8s_state import (
        ROOT_PATH_ENV,
        STORE_PATH_ENV,
        K8sStateError,
        add_helm_repo,
        apply_documents,
        case_root_from_cwd,
        cluster_info_text,
        create_cluster,
        delete_cluster,
        delete_resource,
        get_helm_values,
        get_resource,
        helm_install_or_upgrade,
        k8s_root_path,
        list_cluster_names,
        list_helm_releases,
        list_resources,
        mysql_exec_import,
        mysql_load_schema,
        parse_yaml_documents,
        patch_resource_json,
        pod_copy_in,
        read_kubeconfig_cluster_name,
        register_port_forward,
        rollout_ready,
        scale_workload,
        set_mysql_user,
        wait_pods_ready,
    )


def _error(message: str) -> int:
    print(message, file=sys.stderr)
    return 1


def _consume_common_flags(args: list[str]) -> tuple[dict[str, Any], list[str]]:
    options: dict[str, Any] = {'namespace': None, 'kubeconfig': None}
    remaining: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {'-n', '--namespace'} and index + 1 < len(args):
            options['namespace'] = args[index + 1]
            index += 2
            continue
        if token.startswith('--namespace='):
            options['namespace'] = token.split('=', 1)[1]
            index += 1
            continue
        if token == '--kubeconfig' and index + 1 < len(args):
            options['kubeconfig'] = args[index + 1]
            index += 2
            continue
        if token.startswith('--kubeconfig='):
            options['kubeconfig'] = token.split('=', 1)[1]
            index += 1
            continue
        remaining.append(token)
        index += 1
    if not options['kubeconfig'] and os.getenv('KUBECONFIG'):
        options['kubeconfig'] = os.getenv('KUBECONFIG')
    return options, remaining


def _cluster_name_from_options(options: dict[str, Any]) -> str | None:
    kubeconfig = options.get('kubeconfig')
    if kubeconfig:
        return read_kubeconfig_cluster_name(kubeconfig)
    return None


def _load_values_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if not isinstance(payload, dict):
        raise K8sStateError(f'Helm values file must contain a mapping: {path}')
    return payload


def _parse_set_literal(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == 'true':
        return True
    if lowered == 'false':
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _merge_set_overrides(values: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    payload = copy_json(values)
    for item in overrides:
        if '=' not in item:
            continue
        path_text, raw_value = item.split('=', 1)
        if (
            path_text.startswith('image.')
            or path_text.startswith('sentinel.image.')
            or path_text.startswith('metrics.image.')
            or path_text.startswith('volumePermissions.image.')
            or path_text.startswith('sysctl.image.')
        ):
            continue
        current = payload
        parts = [part for part in path_text.split('.') if part]
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = _parse_set_literal(raw_value)
    return payload


def copy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _list_object(items: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    return {'apiVersion': 'v1', 'kind': f'{kind.capitalize()}List', 'items': items}


def _maybe_output(payload: Any, output: str | None) -> None:
    if output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
        return
    if output == 'yaml':
        print(yaml.safe_dump(payload, sort_keys=False).rstrip())
        return
    print(_render_table(payload))


def _render_table(payload: Any) -> str:
    if isinstance(payload, dict) and 'items' in payload:
        items = payload['items']
        if not items:
            return 'No resources found'
        kind = str(items[0].get('kind') or '').lower()
    elif isinstance(payload, dict):
        items = [payload]
        kind = str(payload.get('kind') or '').lower()
    else:
        return str(payload)

    if kind == 'namespace':
        lines = ['NAME\tSTATUS']
        for item in items:
            lines.append(f'{item["metadata"]["name"]}\t{item.get("status", {}).get("phase", "Active")}')
        return '\n'.join(lines)

    if kind == 'node':
        lines = ['NAME\tSTATUS\tROLES\tVERSION']
        for item in items:
            lines.append(
                f'{item["metadata"]["name"]}\tReady\tcontrol-plane\t'
                f'{item.get("status", {}).get("nodeInfo", {}).get("kubeletVersion", "v1.30.0-mock")}'
            )
        return '\n'.join(lines)

    if kind == 'deployment':
        lines = ['NAME\tREADY\tUP-TO-DATE\tAVAILABLE']
        for item in items:
            status = item.get('status', {})
            replicas = int(item.get('spec', {}).get('replicas') or 0)
            ready = int(status.get('readyReplicas') or 0)
            lines.append(
                f'{item["metadata"]["name"]}\t{ready}/{replicas}\t{replicas}\t{int(status.get("availableReplicas") or 0)}'
            )
        return '\n'.join(lines)

    if kind == 'statefulset':
        lines = ['NAME\tREADY']
        for item in items:
            ready = int(item.get('status', {}).get('readyReplicas') or 0)
            replicas = int(item.get('spec', {}).get('replicas') or 0)
            lines.append(f'{item["metadata"]["name"]}\t{ready}/{replicas}')
        return '\n'.join(lines)

    if kind == 'pod':
        lines = ['NAME\tREADY\tSTATUS']
        for item in items:
            statuses = item.get('status', {}).get('containerStatuses') or []
            ready = sum(1 for status in statuses if status.get('ready'))
            lines.append(
                f'{item["metadata"]["name"]}\t{ready}/{len(statuses)}\t{item.get("status", {}).get("phase", "Running")}'
            )
        return '\n'.join(lines)

    if kind == 'service':
        lines = ['NAME\tTYPE\tCLUSTER-IP\tPORT(S)']
        for item in items:
            ports = []
            for port in item.get('spec', {}).get('ports', []) or []:
                rendered = f'{port.get("port")}/{port.get("protocol", "TCP")}'
                if 'nodePort' in port:
                    rendered += f':{port["nodePort"]}'
                ports.append(rendered)
            lines.append(
                f'{item["metadata"]["name"]}\t{item.get("spec", {}).get("type", "ClusterIP")}\t'
                f'{item.get("spec", {}).get("clusterIP", "")}\t{",".join(ports)}'
            )
        return '\n'.join(lines)

    if kind == 'endpoints':
        lines = ['NAME\tENDPOINTS']
        for item in items:
            addresses: list[str] = []
            for subset in item.get('subsets', []) or []:
                for address in subset.get('addresses', []) or []:
                    addresses.append(str(address.get('ip') or ''))
            lines.append(f'{item["metadata"]["name"]}\t{",".join(filter(None, addresses))}')
        return '\n'.join(lines)

    if kind in {'configmap', 'secret'}:
        lines = ['NAME']
        for item in items:
            lines.append(str(item['metadata']['name']))
        return '\n'.join(lines)

    return json.dumps(payload, ensure_ascii=False, indent=2)


def _jsonpath_values(current: Any, tokens: list[str]) -> list[Any]:
    if not tokens:
        return [current]
    token = tokens[0]
    rest = tokens[1:]
    if token == '*':
        values: list[Any] = []
        if isinstance(current, list):
            for item in current:
                values.extend(_jsonpath_values(item, rest))
        elif isinstance(current, dict):
            for item in current.values():
                values.extend(_jsonpath_values(item, rest))
        return values

    iterate = token.endswith('[*]')
    key = token[:-3] if iterate else token
    if isinstance(current, dict):
        next_value = current.get(key)
    else:
        return []
    if iterate:
        if not isinstance(next_value, list):
            return []
        values: list[Any] = []
        for item in next_value:
            values.extend(_jsonpath_values(item, rest))
        return values
    return _jsonpath_values(next_value, rest)


def _render_jsonpath(expression: str, payload: Any) -> str:
    import re

    matches = re.findall(r'\{([^{}]+)\}', expression)
    if not matches:
        matches = [expression]
    rendered_parts: list[str] = []
    for match in matches:
        path_text = match.strip()
        if path_text.startswith('.'):
            path_text = path_text[1:]
        tokens = [token for token in path_text.split('.') if token]
        values = _jsonpath_values(payload, tokens)
        scalars = []
        for value in values:
            if isinstance(value, (dict, list)):
                scalars.append(json.dumps(value, ensure_ascii=False))
            else:
                scalars.append(str(value))
        rendered_parts.append(' '.join(item for item in scalars if item))
    return ' '.join(part for part in rendered_parts if part).strip()


def _parse_output_flag(args: list[str]) -> tuple[str | None, list[str]]:
    output: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in {'-o', '--output'} and index + 1 < len(args):
            output = args[index + 1]
            index += 2
            continue
        if token.startswith('-o='):
            output = token.split('=', 1)[1]
            index += 1
            continue
        if token.startswith('--output='):
            output = token.split('=', 1)[1]
            index += 1
            continue
        remaining.append(token)
        index += 1
    return output, remaining


def _handle_kind(args: list[str]) -> int:
    if not args:
        return _error('kind: missing command')
    if args[:2] == ['get', 'clusters']:
        names = list_cluster_names()
        if names:
            print('\n'.join(names))
        return 0
    if args[:2] == ['delete', 'cluster']:
        name = None
        for index, token in enumerate(args[2:], start=2):
            if token == '--name' and index + 1 < len(args):
                name = args[index + 1]
                break
            if token.startswith('--name='):
                name = token.split('=', 1)[1]
                break
        if not name:
            return _error('kind delete cluster requires --name')
        delete_cluster(name)
        print(f'Deleted cluster "{name}"')
        return 0
    if args[:2] == ['create', 'cluster']:
        name = 'kind'
        kubeconfig = None
        config_from_stdin = False
        index = 2
        while index < len(args):
            token = args[index]
            if token == '--name' and index + 1 < len(args):
                name = args[index + 1]
                index += 2
                continue
            if token.startswith('--name='):
                name = token.split('=', 1)[1]
                index += 1
                continue
            if token == '--kubeconfig' and index + 1 < len(args):
                kubeconfig = args[index + 1]
                index += 2
                continue
            if token.startswith('--kubeconfig='):
                kubeconfig = token.split('=', 1)[1]
                index += 1
                continue
            if token == '--config=-':
                config_from_stdin = True
                index += 1
                continue
            index += 1
        if not kubeconfig:
            return _error('kind create cluster requires --kubeconfig')
        extra_port_mappings: list[dict[str, int]] = []
        if config_from_stdin and not sys.stdin.isatty():
            payload = yaml.safe_load(sys.stdin.read()) or {}
            for node in payload.get('nodes', []) or []:
                for mapping in node.get('extraPortMappings', []) or []:
                    extra_port_mappings.append({
                        'containerPort': int(mapping.get('containerPort') or 0),
                        'hostPort': int(mapping.get('hostPort') or 0),
                    })
        create_cluster(name, kubeconfig, extra_port_mappings=extra_port_mappings)
        print(f'Created cluster "{name}"')
        return 0
    return _error(f'kind: unsupported command: {" ".join(args)}')


def _kubectl_get(
    resource: str, name: str | None, *, namespace: str | None, kubeconfig: str | None, output: str | None
) -> int:
    cluster_name = read_kubeconfig_cluster_name(kubeconfig) if kubeconfig else None
    if '/' in resource and name is None:
        resource, name = resource.split('/', 1)
    if name is None:
        items = list_resources(resource, namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig)
        payload = _list_object(items, resource)
    else:
        payload = get_resource(
            resource, name, namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig
        )
    if output and output.startswith('jsonpath='):
        print(_render_jsonpath(output.split('=', 1)[1], payload))
        return 0
    _maybe_output(payload, output)
    return 0


def _kubectl_apply(args: list[str], *, kubeconfig: str | None) -> int:
    file_path = None
    index = 0
    while index < len(args):
        token = args[index]
        if token in {'-f', '--filename'} and index + 1 < len(args):
            file_path = args[index + 1]
            break
        if token.startswith('-f='):
            file_path = token.split('=', 1)[1]
            break
        if token.startswith('--filename='):
            file_path = token.split('=', 1)[1]
            break
        index += 1
    if not file_path:
        return _error('kubectl apply requires -f')
    content = sys.stdin.read() if file_path == '-' else Path(file_path).read_text(encoding='utf-8')
    refs = apply_documents(parse_yaml_documents(content), kubeconfig_path=kubeconfig)
    for ref in refs:
        print(f'{ref} configured')
    return 0


def _kubectl_wait(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    resource = None
    label_selector = None
    all_flag = False
    index = 0
    while index < len(args):
        token = args[index]
        if token == '--all':
            all_flag = True
            index += 1
            continue
        if token in {'-l', '--selector'} and index + 1 < len(args):
            label_selector = args[index + 1]
            index += 2
            continue
        if not token.startswith('-') and resource is None:
            resource = token
        index += 1
    cluster_name = read_kubeconfig_cluster_name(kubeconfig) if kubeconfig else None
    if all_flag:
        ok = wait_pods_ready(
            namespace=namespace, label_selector=label_selector, cluster_name=cluster_name, kubeconfig_path=kubeconfig
        )
    elif resource and '/' in resource:
        kind, name = resource.split('/', 1)
        if kind in {'pod', 'pods'}:
            ok = wait_pods_ready(namespace=namespace, name=name, cluster_name=cluster_name, kubeconfig_path=kubeconfig)
        else:
            ok = rollout_ready(kind, name, namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig)
    else:
        ok = wait_pods_ready(
            namespace=namespace, label_selector=label_selector, cluster_name=cluster_name, kubeconfig_path=kubeconfig
        )
    if not ok:
        return _error('resource did not become ready')
    print('condition met')
    return 0


def _kubectl_rollout(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    if args[:1] != ['status'] or len(args) < 2:
        return _error('kubectl rollout supports only "status"')
    target = args[1]
    if '/' not in target:
        return _error('kubectl rollout status requires resource/name')
    kind, name = target.split('/', 1)
    cluster_name = read_kubeconfig_cluster_name(kubeconfig) if kubeconfig else None
    if not rollout_ready(kind, name, namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig):
        return _error(f'{target} is not ready')
    print(f'{target} successfully rolled out')
    return 0


def _kubectl_patch(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    if len(args) < 2:
        return _error('kubectl patch requires kind and name')
    kind = args[0]
    name = args[1]
    patch_payload = None
    index = 2
    while index < len(args):
        token = args[index]
        if token == '-p' and index + 1 < len(args):
            patch_payload = args[index + 1]
            break
        if token.startswith('-p='):
            patch_payload = token.split('=', 1)[1]
            break
        index += 1
    if not patch_payload:
        return _error('kubectl patch requires -p')
    patch_ops = json.loads(patch_payload)
    cluster_name = read_kubeconfig_cluster_name(kubeconfig) if kubeconfig else None
    patch_resource_json(
        kind, name, patch_ops, namespace=namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig
    )
    print(f'{kind}/{name} patched')
    return 0


def _kubectl_create(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    if not args:
        return _error('kubectl create requires a resource kind')
    kind = args[0]
    output, remaining = _parse_output_flag(args[1:])
    dry_run = '--dry-run=client' in remaining or '--dry-run' in remaining
    remaining = [item for item in remaining if item not in {'--dry-run=client', '--dry-run'}]
    cluster_name = read_kubeconfig_cluster_name(kubeconfig) if kubeconfig else None

    if kind in {'namespace', 'ns'}:
        if not remaining:
            return _error('kubectl create namespace requires a name')
        name = remaining[0]
        payload = {'apiVersion': 'v1', 'kind': 'Namespace', 'metadata': {'name': name}}
        if output == 'yaml' or dry_run:
            print(yaml.safe_dump(payload, sort_keys=False).rstrip())
            return 0
        apply_documents([payload], cluster_name=cluster_name, kubeconfig_path=kubeconfig)
        print(f'namespace/{name} created')
        return 0

    if kind == 'configmap':
        if not remaining:
            return _error('kubectl create configmap requires a name')
        name = remaining[0]
        configmap_namespace = namespace or 'default'
        data: dict[str, str] = {}
        index = 1
        while index < len(remaining):
            token = remaining[index]
            if token.startswith('--from-literal='):
                literal = token.split('=', 1)[1]
                key, value = literal.split('=', 1)
                data[key] = value
                index += 1
                continue
            if token.startswith('--from-file='):
                raw = token.split('=', 1)[1]
                if '=' in raw:
                    key, file_path = raw.split('=', 1)
                else:
                    file_path = raw
                    key = Path(file_path).name
                data[key] = Path(file_path).read_text(encoding='utf-8')
                index += 1
                continue
            index += 1
        payload = {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {'name': name, 'namespace': configmap_namespace},
            'data': data,
        }
        if output == 'yaml' or dry_run:
            print(yaml.safe_dump(payload, sort_keys=False).rstrip())
            return 0
        apply_documents([payload], cluster_name=cluster_name, kubeconfig_path=kubeconfig)
        print(f'configmap/{name} created')
        return 0

    if kind == 'secret' and remaining[:1] == ['generic']:
        if len(remaining) < 2:
            return _error('kubectl create secret generic requires a name')
        name = remaining[1]
        secret_namespace = namespace or 'default'
        string_data: dict[str, str] = {}
        index = 2
        while index < len(remaining):
            token = remaining[index]
            if token.startswith('--from-literal='):
                literal = token.split('=', 1)[1]
                key, value = literal.split('=', 1)
                string_data[key] = value
            index += 1
        payload = {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {'name': name, 'namespace': secret_namespace},
            'type': 'Opaque',
            'stringData': string_data,
        }
        if output == 'yaml' or dry_run:
            print(yaml.safe_dump(payload, sort_keys=False).rstrip())
            return 0
        apply_documents([payload], cluster_name=cluster_name, kubeconfig_path=kubeconfig)
        print(f'secret/{name} created')
        return 0

    return _error(f'kubectl create: unsupported kind {kind}')


def _resolve_exec_pod(target: str) -> str:
    if target.startswith('statefulset/'):
        return f'{target.split("/", 1)[1]}-0'
    if target.startswith('pod/'):
        return target.split('/', 1)[1]
    return target


def _extract_mysql_sql(command_text: str) -> str | None:
    import re

    match = re.search(r"<<'?SQL'?\n(?P<body>.*)\nSQL", command_text, re.DOTALL)
    if match:
        return match.group('body')
    inline = re.search(r'-e\s+"(?P<body>.*)"', command_text, re.DOTALL)
    if inline:
        return inline.group('body')
    return None


def _kubectl_exec(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    interactive = '-i' in args
    filtered = [item for item in args if item != '-i']
    if not filtered:
        return _error('kubectl exec requires a target')
    target = filtered[0]
    command_args = filtered[1:]
    if command_args[:1] == ['--']:
        command_args = command_args[1:]
    command_text = ' '.join(command_args)
    stdin_text = ''
    if interactive and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
    cluster_name = read_kubeconfig_cluster_name(kubeconfig) if kubeconfig else None
    pod_name = _resolve_exec_pod(target)
    pod_namespace = namespace or 'default'

    if 'mysql' not in command_text:
        return _error('kubectl exec mock only supports mysql commands')

    if stdin_text.strip():
        mysql_load_schema(cluster_name=cluster_name, kubeconfig_path=kubeconfig, sql_text=stdin_text)
        return 0

    sql_text = _extract_mysql_sql(command_text)
    if sql_text and ('LOAD DATA LOCAL INFILE' in sql_text or 'TRUNCATE TABLE' in sql_text):
        csv_root = k8s_root_path() / 'clusters' / (cluster_name or '') / 'pods' / pod_namespace / pod_name / 'csv'
        mysql_exec_import(
            cluster_name=cluster_name,
            kubeconfig_path=kubeconfig,
            namespace=pod_namespace,
            pod_name=pod_name,
            sql_text=sql_text,
            csv_root=csv_root,
        )
        return 0

    if sql_text and 'CREATE USER' in sql_text:
        import re

        match = re.search(
            r"CREATE USER IF NOT EXISTS '([^']+)'@'%' IDENTIFIED BY '([^']+)'",
            sql_text,
            re.IGNORECASE,
        )
        if not match:
            return _error('Failed to parse CREATE USER statement')
        set_mysql_user(match.group(1), match.group(2), cluster_name=cluster_name, kubeconfig_path=kubeconfig)
        return 0

    return 0


def _kubectl_port_forward(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    if len(args) < 2:
        return _error('kubectl port-forward requires resource and port mapping')
    target = args[0]
    mapping = args[1]
    if ':' not in mapping:
        return _error('kubectl port-forward mapping must be LOCAL:REMOTE')
    local_port_text, remote_port_text = mapping.split(':', 1)
    local_port = int(local_port_text)
    remote_port = int(remote_port_text)
    target_kind, target_name = target.split('/', 1) if '/' in target else ('service', target)
    cluster_name = read_kubeconfig_cluster_name(kubeconfig) if kubeconfig else None
    service_namespace = namespace or 'default'
    protocol = 'http'
    if target_kind in {'service', 'svc'}:
        service = get_resource(
            'service', target_name, namespace=service_namespace, cluster_name=cluster_name, kubeconfig_path=kubeconfig
        )
        for port in service.get('spec', {}).get('ports', []) or []:
            if int(port.get('port') or 0) == remote_port or int(port.get('targetPort') or 0) == remote_port:
                if int(port.get('targetPort') or 0) == 3306 or int(port.get('port') or 0) == 3306:
                    protocol = 'mysql'
                break

    helper_path = Path(__file__).with_name('toolathlon_k8s_forwarder.py')
    env = dict(os.environ)
    env[STORE_PATH_ENV] = os.getenv(STORE_PATH_ENV) or str(case_root_from_cwd() / '.nexus_k8s_state.json')
    env[ROOT_PATH_ENV] = os.getenv(ROOT_PATH_ENV) or str(k8s_root_path())
    child = subprocess.Popen(
        [
            sys.executable,
            str(helper_path),
            '--state-path',
            env[STORE_PATH_ENV],
            '--root-path',
            env[ROOT_PATH_ENV],
            '--cluster',
            cluster_name or '',
            '--namespace',
            service_namespace,
            '--service',
            target_name,
            '--port',
            str(local_port),
            '--protocol',
            protocol,
        ],
        env=env,
        start_new_session=True,
    )
    register_port_forward(
        local_port=local_port,
        namespace=service_namespace,
        resource=target_name,
        pid=os.getpid(),
        kind=target_kind,
        cluster_name=cluster_name,
        kubeconfig_path=kubeconfig,
    )

    def _handler(signum, frame):  # type: ignore[unused-argument]
        try:
            child.terminate()
        finally:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)
    print(f'Forwarding from 127.0.0.1:{local_port} -> {target}:{remote_port}')
    return child.wait()


def _handle_kubectl(args: list[str]) -> int:
    options, remaining = _consume_common_flags(args)
    if not remaining:
        return _error('kubectl: missing command')
    command = remaining[0]
    namespace = options.get('namespace')
    kubeconfig = options.get('kubeconfig')
    output, command_args = _parse_output_flag(remaining[1:])

    if command == 'cluster-info':
        print(cluster_info_text(kubeconfig_path=kubeconfig))
        return 0
    if command == 'get':
        if not command_args:
            return _error('kubectl get requires a resource')
        resource = command_args[0]
        name = None
        if len(command_args) > 1 and not command_args[1].startswith('-'):
            name = command_args[1]
        return _kubectl_get(resource, name, namespace=namespace, kubeconfig=kubeconfig, output=output)
    if command == 'apply':
        return _kubectl_apply(command_args, kubeconfig=kubeconfig)
    if command == 'wait':
        return _kubectl_wait(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command == 'rollout':
        return _kubectl_rollout(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command == 'patch':
        return _kubectl_patch(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command == 'create':
        return _kubectl_create(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command == 'cp':
        if len(command_args) != 2:
            return _error('kubectl cp requires src and dest')
        pod_copy_in(command_args[0], command_args[1], namespace=namespace, kubeconfig_path=kubeconfig)
        return 0
    if command == 'exec':
        return _kubectl_exec(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command == 'port-forward':
        return _kubectl_port_forward(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command == 'delete':
        if not command_args:
            return _error('kubectl delete requires a resource')
        resource = command_args[0]
        name = command_args[1] if len(command_args) > 1 else None
        if '/' in resource and name is None:
            resource, name = resource.split('/', 1)
        if not name:
            return _error('kubectl delete requires a resource name')
        delete_resource(resource, name, namespace=namespace, kubeconfig_path=kubeconfig)
        print(f'{resource}/{name} deleted')
        return 0
    if command == 'scale':
        if len(command_args) < 2:
            return _error('kubectl scale requires resource/name and --replicas')
        resource = command_args[0]
        if '/' not in resource:
            return _error('kubectl scale requires resource/name')
        kind, name = resource.split('/', 1)
        replicas = None
        for index, token in enumerate(command_args[1:], start=1):
            if token == '--replicas' and index + 1 < len(command_args):
                replicas = int(command_args[index + 1])
                break
            if token.startswith('--replicas='):
                replicas = int(token.split('=', 1)[1])
                break
        if replicas is None:
            return _error('kubectl scale requires --replicas')
        scale_workload(kind, name, replicas, namespace=namespace, kubeconfig_path=kubeconfig)
        print(f'scaled {resource} to {replicas}')
        return 0
    return _error(f'kubectl: unsupported command {command}')


def _helm_list(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    output, _ = _parse_output_flag(args)
    releases = list_helm_releases(namespace=namespace, kubeconfig_path=kubeconfig)
    if output == 'json':
        print(json.dumps(releases, ensure_ascii=False))
        return 0
    if not releases:
        print('')
        return 0
    lines = ['NAME\tNAMESPACE\tREVISION\tSTATUS\tCHART']
    for release in releases:
        lines.append(
            f'{release["name"]}\t{release["namespace"]}\t{release["revision"]}\t{release["status"]}\t{release["chart"]}'
        )
    print('\n'.join(lines))
    return 0


def _helm_get(args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    if len(args) < 2 or args[0] != 'values':
        return _error('helm get only supports values')
    release_name = args[1]
    output, _ = _parse_output_flag(args[2:])
    payload = get_helm_values(release_name, namespace=namespace, kubeconfig_path=kubeconfig)
    if output == 'json':
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(yaml.safe_dump(payload, sort_keys=False).rstrip())
    return 0


def _helm_install_or_upgrade(command: str, args: list[str], *, namespace: str | None, kubeconfig: str | None) -> int:
    if len(args) < 2:
        return _error(f'helm {command} requires release and chart')
    release_name = args[0]
    chart_ref = args[1]
    values_file = None
    version = 'latest'
    reuse_values = '--reuse-values' in args
    set_overrides: list[str] = []
    index = 2
    while index < len(args):
        token = args[index]
        if token in {'-f', '--values'} and index + 1 < len(args):
            values_file = args[index + 1]
            index += 2
            continue
        if token.startswith('--values='):
            values_file = token.split('=', 1)[1]
            index += 1
            continue
        if token == '--version' and index + 1 < len(args):
            version = args[index + 1]
            index += 2
            continue
        if token.startswith('--version='):
            version = token.split('=', 1)[1]
            index += 1
            continue
        if token == '--set' and index + 1 < len(args):
            set_overrides.append(args[index + 1])
            index += 2
            continue
        if token.startswith('--set='):
            set_overrides.append(token.split('=', 1)[1])
            index += 1
            continue
        index += 1
    values = _load_values_file(values_file)
    values = _merge_set_overrides(values, set_overrides)
    helm_namespace = namespace or 'default'
    release = helm_install_or_upgrade(
        release_name,
        chart_ref,
        namespace=helm_namespace,
        version=version,
        values=values,
        reuse_values=reuse_values,
        kubeconfig_path=kubeconfig,
    )
    print(f'NAME: {release["name"]}')
    print(f'LAST DEPLOYED: {release["revision"]}')
    print(f'NAMESPACE: {release["namespace"]}')
    print(f'STATUS: {release["status"]}')
    print(f'CHART: {release["chart"]}')
    return 0


def _handle_helm(args: list[str]) -> int:
    options, remaining = _consume_common_flags(args)
    if not remaining:
        return _error('helm: missing command')
    command = remaining[0]
    namespace = options.get('namespace')
    kubeconfig = options.get('kubeconfig')
    command_args = remaining[1:]

    if command == 'repo' and command_args[:1] == ['add'] and len(command_args) >= 3:
        add_helm_repo(command_args[1], command_args[2])
        print(f'"{command_args[1]}" has been added to your repositories')
        return 0
    if command == 'repo' and command_args[:1] == ['update']:
        print('Hang tight while we grab the latest from your chart repositories...')
        print('...Successfully got an update')
        return 0
    if command == 'list':
        return _helm_list(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command == 'get':
        return _helm_get(command_args, namespace=namespace, kubeconfig=kubeconfig)
    if command in {'install', 'upgrade'}:
        return _helm_install_or_upgrade(command, command_args, namespace=namespace, kubeconfig=kubeconfig)
    return _error(f'helm: unsupported command {command}')


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('tool', choices=('kind', 'kubectl', 'helm'))
    parser.add_argument('args', nargs=argparse.REMAINDER)
    parsed = parser.parse_args()

    try:
        if parsed.tool == 'kind':
            raise SystemExit(_handle_kind(parsed.args))
        if parsed.tool == 'kubectl':
            raise SystemExit(_handle_kubectl(parsed.args))
        raise SystemExit(_handle_helm(parsed.args))
    except K8sStateError as exc:
        raise SystemExit(_error(str(exc))) from exc


if __name__ == '__main__':
    main()
