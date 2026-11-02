from __future__ import annotations

import base64
import importlib

import pytest


def _load_ids():
    spec = importlib.util.find_spec('nexus.utils.otel.ids')
    assert spec is not None, 'nexus.utils.otel.ids is not implemented'
    return importlib.import_module('nexus.utils.otel.ids')


def test_span_id_encoding_round_trip() -> None:
    raw = bytes.fromhex('0123456789abcdef')
    nexus_span_id = base64.urlsafe_b64encode(raw).rstrip(b'=').decode('ascii')
    ids = _load_ids()

    otel_hex = ids.nexus_span_id_to_otel_hex(nexus_span_id)

    assert otel_hex == '0123456789abcdef'
    assert ids.otel_hex_to_nexus_span_id(otel_hex) == nexus_span_id


@pytest.mark.parametrize(
    'value',
    ['', 'not-base64', base64.urlsafe_b64encode(b'short').decode('ascii'), 'ASNFZ4mrze8='],
)
def test_nexus_span_id_requires_eight_bytes(value: str) -> None:
    with pytest.raises(ValueError, match='8 bytes'):
        _load_ids().nexus_span_id_to_otel_hex(value)


@pytest.mark.parametrize('value', ['', '0' * 15, 'g' * 16, '00000000000000  '])
def test_otel_span_id_requires_sixteen_hex_characters(value: str) -> None:
    with pytest.raises(ValueError, match='16 hex chars'):
        _load_ids().otel_hex_to_nexus_span_id(value)
