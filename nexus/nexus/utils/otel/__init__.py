from __future__ import annotations

from nexus.utils.otel.auto import initialize_if_enabled as initialize_if_enabled
from nexus.utils.otel.auto import is_enabled as is_enabled
from nexus.utils.otel.environment import managed_otel_env as managed_otel_env
from nexus.utils.otel.lifecycle import async_force_flush_best_effort as async_force_flush_best_effort
from nexus.utils.otel.lifecycle import force_flush as force_flush
from nexus.utils.otel.lifecycle import shutdown as shutdown
from nexus.utils.otel.meters import get_meter as get_meter
