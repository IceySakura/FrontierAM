"""Destination view definitions derived from the instruments shipped in this package."""

from __future__ import annotations

from nexus.distributed.observability.telemetry import COUNTERS, COUNT_BUCKETS, GAUGES, HISTOGRAMS, SECONDS_BUCKETS


def metric_definitions() -> list[dict]:
    definitions = []
    for name in sorted(COUNTERS | GAUGES):
        duration = name.endswith(('_age', '_lag', '_timestamp'))
        unit = '字节' if name == 'resource_memory_bytes' else '秒' if duration else '个'
        calculation = 'plus' if name in COUNTERS else 'max' if duration else 'avg'
        definitions.append({
            'metric_enname': name,
            'metric_chname': name,
            'unit': unit,
            'type': 'delay' if duration else 'amount',
            'cal_op': calculation,
            'compound_type': 'simple',
            'status': 'enable',
        })
    histograms = [(name, SECONDS_BUCKETS, '秒', 'delay') for name in sorted(HISTOGRAMS)]
    histograms.append(('reconciler_batch_full_scan_items_distribution', COUNT_BUCKETS, '个', 'amount'))
    for name, bounds, unit, kind in histograms:
        source = name.removesuffix('_distribution')
        buckets = ','.join(str(value) for value in bounds)
        for suffix, operation in [('distribution', 'distribution'), ('p95', 'quantile'), ('sum', 'plus')]:
            definitions.append({
                'metric_enname': f'{source}_{suffix}',
                'metric_chname': f'{source}_{suffix}',
                'source_metric_enname': source,
                'unit': unit,
                'type': kind,
                'cal_op': operation,
                'compound_type': 'distribution',
                'distribution_interval': buckets,
                'quantile_buckets': buckets,
                **({'quantile': 9500} if operation == 'quantile' else {}),
                'status': 'enable',
            })
    return definitions
