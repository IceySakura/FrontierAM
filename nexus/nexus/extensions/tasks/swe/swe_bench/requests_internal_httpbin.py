import hashlib
import json

from dataclasses import dataclass
from typing import Any

from nexus.extensions.tasks.swe.constants import SWEbenchInstance
from nexus.extensions.tasks.swe.swe_bench.constants import KEY_INSTANCE_ID
from nexus.extensions.tasks.swe.swe_bench.test_spec import get_test_directives


@dataclass(frozen=True)
class _RequestsInternalHttpbinCase:
    version: str
    base_commit: str
    test_patch_sha256: str
    fail_to_pass_count: int
    fail_to_pass_sha256: str
    pass_to_pass_count: int
    pass_to_pass_sha256: str
    post_test_file_sha256: str
    directives: tuple[str, ...]


REQUESTS_INTERNAL_HTTPBIN_TEST_FILE = 'test_requests.py'
REQUESTS_INTERNAL_HTTPBIN_TEST_COMMAND = (
    'env HTTPBIN_URL=http://hyrl-httpbin.woa.com/ REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt pytest -rA'
)

_REQUESTS_INTERNAL_HTTPBIN_CASES = {
    'psf__requests-1921': _RequestsInternalHttpbinCase(
        version='2.3',
        base_commit='3c88e520da24ae6f736929a750876e7654accc3d',
        test_patch_sha256='a9542a0223dcb934125f74642468fb69c5cae8d5fa1602279a3745862db6252d',
        fail_to_pass_count=6,
        fail_to_pass_sha256='62c407d559b8089a6cc3c3ee45e53ecede5d939683204f1a745d9a0263d6a168',
        pass_to_pass_count=107,
        pass_to_pass_sha256='36d6ebd9ae18d706c395c0b815c8e5ca0d8d69dad086121d3a3cf943ea9b3d6d',
        post_test_file_sha256='318da0521e0d056ccf97b64adee853e7d53244f7c82b4b8c7dd7c0ef72d50ae2',
        directives=(REQUESTS_INTERNAL_HTTPBIN_TEST_FILE,),
    ),
    'psf__requests-2317': _RequestsInternalHttpbinCase(
        version='2.4',
        base_commit='091991be0da19de9108dbe5e3752917fea3d7fdc',
        test_patch_sha256='ff283039874db19ea594bda54c2aa6812eb25fea9881e2517c4a143805cdb552',
        fail_to_pass_count=8,
        fail_to_pass_sha256='36ad2bde8f7bd06bef39d763c8710605b8072eb2969ad3e5c37fc9be439e2c44',
        pass_to_pass_count=133,
        pass_to_pass_sha256='cdce5ad360721a761c14706bd9038f4c0d3ec3b6619fc747099246ba2f851d10',
        post_test_file_sha256='fe816a0da3b4abd996014ef559a0ce7eeefcd0664d62a683924bd65840998b02',
        directives=(REQUESTS_INTERNAL_HTTPBIN_TEST_FILE,),
    ),
}


def _load_test_status(instance: SWEbenchInstance, instance_id: str, key: str) -> list[str]:
    value: Any = instance.get(key)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(f'{instance_id}: invalid {key}') from e

    if not isinstance(value, list) or not all(isinstance(test, str) for test in value):
        raise ValueError(f'{instance_id}: unexpected {key}')
    if len(value) != len(set(value)):
        raise ValueError(f'{instance_id}: duplicate {key}')
    return value


def _validate_test_status(
    instance: SWEbenchInstance,
    instance_id: str,
    key: str,
    expected_count: int,
    expected_sha256: str,
) -> None:
    tests = _load_test_status(instance, instance_id, key)
    canonical_json = json.dumps(tests, separators=(',', ':'))
    actual_sha256 = hashlib.sha256(canonical_json.encode()).hexdigest()
    if len(tests) != expected_count or actual_sha256 != expected_sha256:
        raise ValueError(f'{instance_id}: unexpected {key}')


def _validated_case(instance: SWEbenchInstance) -> tuple[str, _RequestsInternalHttpbinCase] | None:
    instance_id = instance.get(KEY_INSTANCE_ID) or ''
    case = _REQUESTS_INTERNAL_HTTPBIN_CASES.get(instance_id)
    if case is None:
        return None

    expected_fields = {
        'repo': 'psf/requests',
        'dataset_type': 'swe-bench',
        'version': case.version,
        'base_commit': case.base_commit,
    }
    for field, expected in expected_fields.items():
        if instance.get(field) != expected:
            raise ValueError(f'{instance_id}: unexpected {field}')

    test_patch = instance.get('test_patch')
    if not isinstance(test_patch, str):
        raise ValueError(f'{instance_id}: unexpected test_patch')
    if hashlib.sha256(test_patch.encode()).hexdigest() != case.test_patch_sha256:
        raise ValueError(f'{instance_id}: unexpected test_patch')

    _validate_test_status(
        instance,
        instance_id,
        'FAIL_TO_PASS',
        case.fail_to_pass_count,
        case.fail_to_pass_sha256,
    )
    _validate_test_status(
        instance,
        instance_id,
        'PASS_TO_PASS',
        case.pass_to_pass_count,
        case.pass_to_pass_sha256,
    )

    if get_test_directives(instance) != list(case.directives):
        raise ValueError(f'{instance_id}: unexpected test_patch directives')

    return instance_id, case


def _fixture_python(instance_id: str, case: _RequestsInternalHttpbinCase) -> str:
    rewrites = [
        (
            'https://httpbin.org/status/301',
            'https://hyrl-httpbin.woa.com/status/301',
        ),
    ]
    resolve_auth_target = instance_id == 'psf__requests-2317'
    if resolve_auth_target:
        rewrites.extend([
            (
                'https://httpbin.org/redirect/5',
                'https://hyrl-httpbin.woa.com/redirect/5',
            ),
            (
                'https://httpbin.org/delay/10',
                'https://hyrl-httpbin.woa.com/delay/10',
            ),
        ])

    lines = [
        'import hashlib',
        'import os',
        'import stat',
        'import sys',
        'import tempfile',
        'from pathlib import Path',
    ]
    if resolve_auth_target:
        lines.extend([
            'import ipaddress',
            'import socket',
        ])
    lines.extend([
        '',
        '',
        'def main():',
        "    ca_bundle = Path('/etc/ssl/certs/ca-certificates.crt')",
        '    try:',
        '        ca_stat = ca_bundle.stat()',
        '    except OSError as error:',
        "        raise RuntimeError(f'missing CA bundle: {ca_bundle}: {error}') from error",
        '    if not stat.S_ISREG(ca_stat.st_mode) or ca_stat.st_size <= 0:',
        "        raise RuntimeError(f'CA bundle is not a non-empty regular file: {ca_bundle}')",
        '',
        f"    test_file = Path('{REQUESTS_INTERNAL_HTTPBIN_TEST_FILE}')",
        '    payload = test_file.read_bytes()',
        '    actual_sha256 = hashlib.sha256(payload).hexdigest()',
        f"    expected_sha256 = '{case.post_test_file_sha256}'",
        '    if actual_sha256 != expected_sha256:',
        '        raise RuntimeError(',
        "            f'unexpected test_requests.py sha256: {actual_sha256}; expected {expected_sha256}'",
        '        )',
        '',
        "    source = payload.decode('utf-8')",
        '    rewrites = [',
    ])
    for old, new in rewrites:
        lines.append(f'        ({old!r}, {new!r}),')
    lines.append('    ]')
    if resolve_auth_target:
        lines.extend([
            '    address_info = socket.getaddrinfo(',
            "        'hyrl-httpbin.woa.com',",
            '        80,',
            '        family=socket.AF_INET,',
            '        type=socket.SOCK_STREAM,',
            '    )',
            '    if not address_info:',
            "        raise RuntimeError('hyrl-httpbin.woa.com did not resolve to an IPv4 address')",
            '    auth_target_ip = address_info[0][4][0]',
            '    try:',
            '        ipaddress.IPv4Address(auth_target_ip)',
            '    except ipaddress.AddressValueError as error:',
            "        raise RuntimeError(f'invalid auth redirect IPv4 address: {auth_target_ip}') from error",
            '    rewrites.append((',
            "        'http://www.google.co.uk',",
            "        f'http://{auth_target_ip}/get',",
            '    ))',
        ])
    lines.extend([
        '',
        '    for old, new in rewrites:',
        '        anchor_count = source.count(old)',
        '        if anchor_count != 1:',
        '            raise RuntimeError(',
        "                f'expected exactly one rewrite anchor {old!r}, found {anchor_count}'",
        '            )',
        '        if new in source:',
        "            raise RuntimeError(f'unexpected existing rewrite target {new!r}')",
        '',
        '    updated = source',
        '    for old, new in rewrites:',
        '        updated = updated.replace(old, new, 1)',
        '',
        '    test_stat = test_file.stat()',
        '    temporary_path = None',
        '    try:',
        '        with tempfile.NamedTemporaryFile(',
        "            mode='wb',",
        '            dir=test_file.parent,',
        "            prefix=f'.{test_file.name}.',",
        '            delete=False,',
        '        ) as temporary_file:',
        '            temporary_path = Path(temporary_file.name)',
        '            os.fchmod(temporary_file.fileno(), stat.S_IMODE(test_stat.st_mode))',
        "            temporary_file.write(updated.encode('utf-8'))",
        '            temporary_file.flush()',
        '            os.fsync(temporary_file.fileno())',
        '        os.replace(temporary_path, test_file)',
        '        temporary_path = None',
        '    finally:',
        '        if temporary_path is not None:',
        '            try:',
        '                temporary_path.unlink()',
        '            except FileNotFoundError:',
        '                pass',
        '',
        '',
        'try:',
        '    main()',
        'except Exception as error:',
        "    print(f'Failed to prepare requests internal HTTPBin fixture: {error}', file=sys.stderr)",
        '    raise SystemExit(1)',
    ])
    return '\n'.join(lines)


def get_requests_internal_httpbin_fixture_command(instance: SWEbenchInstance) -> str | None:
    validated = _validated_case(instance)
    if validated is None:
        return None

    instance_id, case = validated
    fixture_python = _fixture_python(instance_id, case)
    return '\n'.join([
        '# SWE-bench requests exact-case internal HTTPBin fixture',
        "if ! python - <<'EOF_SWE_BENCH_REQUESTS_INTERNAL_HTTPBIN_FIXTURE'",
        fixture_python,
        'EOF_SWE_BENCH_REQUESTS_INTERNAL_HTTPBIN_FIXTURE',
        'then',
        "    echo 'Failed to prepare requests exact-case internal HTTPBin fixture' >&2",
        '    exit 1',
        'fi',
        'if ! cp /etc/ssl/certs/ca-certificates.crt requests/cacert.pem',
        'then',
        "    echo 'Failed to stage requests exact-case CA bundle' >&2",
        '    exit 1',
        'fi',
    ])
