#!/usr/bin/env python3
from __future__ import annotations

import json

from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INVALID_PARAMS, ErrorData, TextContent, Tool


STATIONS = {
    '北京南': {'station_code': 'VNP', 'station_name': 'Beijing South Railway Station'},
    '北京南站': {'station_code': 'VNP', 'station_name': 'Beijing South Railway Station'},
    'Beijing South Railway Station': {'station_code': 'VNP', 'station_name': 'Beijing South Railway Station'},
    'Beijingnan Railway Station': {'station_code': 'VNP', 'station_name': 'Beijing South Railway Station'},
    '上海虹桥': {'station_code': 'AOH', 'station_name': 'Shanghai Hongqiao Railway Station'},
    '上海虹桥站': {'station_code': 'AOH', 'station_name': 'Shanghai Hongqiao Railway Station'},
    'Shanghai Hongqiao Railway Station': {'station_code': 'AOH', 'station_name': 'Shanghai Hongqiao Railway Station'},
    '曲阜东': {'station_code': 'QDK', 'station_name': 'Qufu East Railway Station'},
    '曲阜东站': {'station_code': 'QDK', 'station_name': 'Qufu East Railway Station'},
    'Qufu East Railway Station': {'station_code': 'QDK', 'station_name': 'Qufu East Railway Station'},
    '曲阜': {'station_code': 'QFK', 'station_name': 'Qufu Railway Station'},
    '曲阜站': {'station_code': 'QFK', 'station_name': 'Qufu Railway Station'},
    'Qufu Railway Station': {'station_code': 'QFK', 'station_name': 'Qufu Railway Station'},
    '曲阜南': {'station_code': 'QNK', 'station_name': 'Qufu South Railway Station'},
    '曲阜南站': {'station_code': 'QNK', 'station_name': 'Qufu South Railway Station'},
    'Qufu South Railway Station': {'station_code': 'QNK', 'station_name': 'Qufu South Railway Station'},
}

TICKET_SCHEDULES = {
    ('VNP', 'QDK'): [
        {'train_number': 'G171', 'departure_time': '17:28', 'arrival_time': '19:37', 'duration': '02:09', 'price': 309},
        {'train_number': 'G183', 'departure_time': '18:05', 'arrival_time': '20:14', 'duration': '02:09', 'price': 316},
    ],
    ('AOH', 'QDK'): [
        {'train_number': 'G138', 'departure_time': '17:42', 'arrival_time': '19:28', 'duration': '01:46', 'price': 259},
        {'train_number': 'G150', 'departure_time': '18:18', 'arrival_time': '20:03', 'duration': '01:45', 'price': 266},
    ],
    ('QDK', 'VNP'): [
        {'train_number': 'G186', 'departure_time': '14:36', 'arrival_time': '16:47', 'duration': '02:11', 'price': 316},
        {'train_number': 'G198', 'departure_time': '15:12', 'arrival_time': '17:22', 'duration': '02:10', 'price': 309},
    ],
    ('QDK', 'AOH'): [
        {'train_number': 'G149', 'departure_time': '14:22', 'arrival_time': '16:08', 'duration': '01:46', 'price': 266},
        {'train_number': 'G159', 'departure_time': '14:48', 'arrival_time': '16:35', 'duration': '01:47', 'price': 259},
    ],
}


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _station_name_from_code(code: str) -> str:
    for item in STATIONS.values():
        if item['station_code'] == code:
            return item['station_name']
    return code


def _render_ticket_line(from_code: str, to_code: str, ticket: dict[str, Any]) -> str:
    from_name = _station_name_from_code(from_code)
    to_name = _station_name_from_code(to_code)
    return (
        f'{ticket["train_number"]} (实际车次train_no: {ticket["train_number"]}) '
        f'{from_name} (telecode: {from_code}) -> {to_name} (telecode: {to_code}) '
        f'{ticket["departure_time"]} -> {ticket["arrival_time"]} 历时：{ticket["duration"]}'
    )


def _render_tickets(from_code: str, to_code: str) -> str:
    tickets = TICKET_SCHEDULES.get((from_code, to_code), [])
    lines = ['车次 | 出发站 -> 到达站 | 出发时间 -> 到达时间 | 历时']
    for ticket in tickets:
        lines.append(_render_ticket_line(from_code, to_code, ticket))
        lines.append(f'- 二等座: 有票 {ticket["price"]}元')
        lines.append(f'- 一等座: 有票 {ticket["price"] + 120}元')
    return '\n'.join(lines)


async def main() -> None:
    server = Server('toolathlon-12306-server')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name='get-station-code-by-names',
                description='Resolve one or more station names separated by | into 12306 station codes.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'stationNames': {'type': 'string', 'description': 'Station names separated by |.'},
                    },
                    'required': ['stationNames'],
                    'additionalProperties': False,
                },
            ),
            Tool(
                name='get-tickets',
                description='List direct rail schedules between two stations for a given date.',
                inputSchema={
                    'type': 'object',
                    'properties': {
                        'date': {'type': 'string'},
                        'fromStation': {'type': 'string'},
                        'toStation': {'type': 'string'},
                        'trainFilterFlags': {'type': 'string'},
                    },
                    'required': ['date', 'fromStation', 'toStation'],
                    'additionalProperties': False,
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        args = arguments or {}
        if name == 'get-station-code-by-names':
            raw = str(args.get('stationNames') or '').strip()
            if not raw:
                raise _error(INVALID_PARAMS, 'stationNames is required')
            response: dict[str, Any] = {}
            for station_name in [item.strip() for item in raw.split('|') if item.strip()]:
                station = STATIONS.get(station_name)
                if station is None:
                    raise _error(INVALID_PARAMS, f'unknown station: {station_name}')
                response[station_name] = station
            return _json_content(response)

        if name == 'get-tickets':
            from_station = str(args.get('fromStation') or '').strip()
            to_station = str(args.get('toStation') or '').strip()
            if not from_station or not to_station:
                raise _error(INVALID_PARAMS, 'fromStation and toStation are required')
            return [TextContent(type='text', text=_render_tickets(from_station, to_station))]

        raise _error(INVALID_PARAMS, f'Unknown tool: {name}')

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
