"""Token accounting for HITL runs.

Three layers, so a change to one provider's key names does not touch the code that
totals or the code that hooks into the tracer:

* :mod:`.profiles` -- where each provider family puts each counter (a table);
* :mod:`.parse` -- reading one response's usage block into normalized counters;
* :mod:`.tracker` -- summing them per source;
* :mod:`.processor` -- the run-level span processor that attributes calls by label.
"""

from nexus.extensions.tasks.hitl.trace.usage.parse import UsageTrace, model_of, parse_usage
from nexus.extensions.tasks.hitl.trace.usage.processor import (
    CLIENT_ATTRIBUTE,
    COMPLETION_SPAN,
    LABEL_ATTRIBUTE,
    HITLUsageProcessor,
)
from nexus.extensions.tasks.hitl.trace.usage.profiles import (
    COUNTERS,
    DEFAULT_PROFILE,
    SPECS,
    Derive,
    KeyPaths,
    UsageProfile,
    UsageSpec,
    known_profiles,
    spec_for,
)
from nexus.extensions.tasks.hitl.trace.usage.tracker import (
    SERIES_FIELDS,
    SERIES_LIMIT,
    SUMMED_FIELDS,
    UsageTracker,
    combine,
)


__all__ = [
    'CLIENT_ATTRIBUTE',
    'COMPLETION_SPAN',
    'COUNTERS',
    'DEFAULT_PROFILE',
    'LABEL_ATTRIBUTE',
    'SERIES_FIELDS',
    'SERIES_LIMIT',
    'SPECS',
    'SUMMED_FIELDS',
    'Derive',
    'HITLUsageProcessor',
    'KeyPaths',
    'UsageProfile',
    'UsageSpec',
    'UsageTrace',
    'UsageTracker',
    'combine',
    'known_profiles',
    'model_of',
    'parse_usage',
    'spec_for',
]
