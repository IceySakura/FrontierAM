"""Span processor module.

Processor classes are registered lazily via PEP 723 metadata discovery
(see ``module_registry.discover_and_register()``).  The actual modules
are imported only when the corresponding name is first requested via
``span_processor_registry.get()``.
"""

from __future__ import annotations

from nexus.trace.processors.base import SpanProcessor as SpanProcessor
from nexus.trace.processors.base import span_processor_registry as span_processor_registry
from nexus.trace.processors.base import validate_span_processor as validate_span_processor
