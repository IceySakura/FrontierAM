"""LibreOffice Calc getters."""

from __future__ import annotations

import csv

from typing import Any

from nexus.extensions.tasks.os_world.evaluation.registry import getter_registry


@getter_registry.register()
async def get_conference_city_in_order(runtime: Any, config: dict[str, Any]) -> list[str]:
    with open(str(config['csv_path']), encoding='utf-8') as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return [row[2] for row in reader if len(row) > 2]
