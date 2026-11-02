from __future__ import annotations

from dataclasses import dataclass

import pytest

from nexus.distributed.config import DistributedConfig
from nexus.distributed.execution import trace_backends
from nexus.runner.trace.tdsql import TDSQLBoundlessTraceSink, TDSQLTraceSink, TDSQLTraceSource


class LegacySink(TDSQLTraceSink):
    def __init__(self, host, user, db, execution_id='', assignment_id='', attempt_id=''):
        super().__init__(
            host=host,
            user=user,
            db=db,
            execution_id=execution_id,
            assignment_id=assignment_id,
            attempt_id=attempt_id,
        )


@dataclass
class SupportingSink(TDSQLTraceSink):
    max_run_output_bytes: int = 0

    def __post_init__(self):
        if self.max_run_output_bytes < 0:
            raise ValueError('negative output limit')
        super().__post_init__()


class KwargsSink(TDSQLTraceSink):
    def __init__(self, **kwargs):
        limit = kwargs.pop('max_run_output_bytes')
        self.future_option = kwargs.pop('future_option')
        super().__init__(**kwargs)
        self.max_run_output_bytes = limit


@pytest.mark.parametrize(
    'sink_class', [LegacySink, TDSQLTraceSink, TDSQLBoundlessTraceSink, SupportingSink, KwargsSink]
)
@pytest.mark.parametrize('entrypoint', ['execution', 'scope'])
async def test_extra_config_respects_sink_capability(monkeypatch, caplog, sink_class, entrypoint):
    original_import = trace_backends._import_backend_class

    def import_class(name, role, *args):
        if name == 'probe':
            return sink_class if role == 'sink' else TDSQLTraceSource
        return original_import(name, role, *args)

    monkeypatch.setattr(trace_backends, '_import_backend_class', import_class)
    prepared = []

    async def prepare_scope(self, scope):
        prepared.append((self, scope))

    monkeypatch.setattr(sink_class, 'prepare_scope', prepare_scope)
    limit = 256 * 1024**2
    config = DistributedConfig(
        run_trace_sinks={
            'probe': {
                'sink_class': 'test::Sink',
                'source_class': 'test::Source',
                'config': {'host': 'db.invalid', 'user': 'test', 'db': 'runner'},
                'sink_config': {'max_run_output_bytes': limit, 'future_option': 'private-value-not-for-logs'},
            }
        }
    )
    if entrypoint == 'scope':
        await trace_backends.prepare_configured_run_trace_backend_scope(config, backend_name='probe', scope_id='scope')
        sink, scope = prepared[0]
        assert scope.scope_id == 'scope'
        assert sink.execution_id == 'scope'
    else:
        registry = trace_backends.create_run_trace_backend_registry(config)
        try:
            resolved = registry.resolve(
                run_kind='single',
                driver_config={'run_trace_backends': {'sinks': ['probe']}},
                request=trace_backends.RunTraceSinkRequest(execution_id='exec', assignment_id='asn', attempt_id='att'),
            )
            sink = resolved.backend_sinks['probe']
            assert sink.execution_id == 'exec'
        finally:
            await registry.close()
    if sink_class is not LegacySink:
        assert sink.max_run_output_bytes == limit
        assert 'max_run_output_bytes' not in caplog.text
    else:
        assert 'max_run_output_bytes' in caplog.text
        assert 'ignoring them' in caplog.text
    if sink_class is KwargsSink:
        assert sink.future_option == 'private-value-not-for-logs'
        assert 'ignoring them' not in caplog.text
    else:
        assert 'future_option' in caplog.text
    assert 'private-value-not-for-logs' not in caplog.text
    assert config.run_trace_sinks['probe'].sink_config == {
        'max_run_output_bytes': limit,
        'future_option': 'private-value-not-for-logs',
    }


def test_missing_required_parameter_still_fails(caplog):
    with pytest.raises(RuntimeError, match='failed to create run trace sink') as error:
        trace_backends._create_sink(
            'probe',
            TDSQLTraceSink,
            {
                'hots': 'db.invalid',
                'user': 'test',
                'db': 'runner',
            },
        )
    assert isinstance(error.value.__cause__, TypeError)
    assert 'host' in str(error.value.__cause__)
    assert 'hots' in caplog.text


def test_supported_parameter_validation_errors_are_not_swallowed():
    with pytest.raises(RuntimeError, match='failed to create run trace sink') as error:
        trace_backends._create_sink(
            'probe',
            SupportingSink,
            {
                'host': 'db.invalid',
                'user': 'test',
                'db': 'runner',
                'max_run_output_bytes': -1,
            },
        )
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == 'negative output limit'
