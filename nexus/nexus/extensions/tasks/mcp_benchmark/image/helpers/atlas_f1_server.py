#!/usr/bin/env python3
from __future__ import annotations

import json
import re

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool
from pydantic import BaseModel, Field


JOLPICA_BASE_URL = 'https://api.jolpi.ca/ergast/f1'
OPENF1_BASE_URL = 'https://api.openf1.org/v1'
REQUEST_TIMEOUT = 30

SESSION_ALIASES = {
    'race': {'race', 'grand prix', 'gp'},
    'qualifying': {'qualifying', 'quali', 'q'},
    'sprint': {'sprint', 'sprint shootout'},
    'fp1': {'fp1', 'practice 1', 'free practice 1'},
    'fp2': {'fp2', 'practice 2', 'free practice 2'},
    'fp3': {'fp3', 'practice 3', 'free practice 3'},
}

OPENF1_SESSION_NAMES = {
    'race': 'Race',
    'qualifying': 'Qualifying',
    'sprint': 'Sprint',
    'fp1': 'Practice 1',
    'fp2': 'Practice 2',
    'fp3': 'Practice 3',
}

JOLPICA_RESULT_ENDPOINTS = {
    'race': 'results',
    'qualifying': 'qualifying',
    'sprint': 'sprint',
}


class YearArgs(BaseModel):
    year: int = Field(ge=1950)


class EventArgs(YearArgs):
    event_identifier: str | int


class SessionArgs(EventArgs):
    session_name: str


class DriverArgs(BaseModel):
    driver_identifier: str
    year: int | None = Field(default=None, ge=1950)
    event_identifier: str | int | None = None
    session_name: str | None = None


class AnalyzeArgs(SessionArgs):
    driver_identifier: str


class CompareArgs(SessionArgs):
    drivers: str | list[str]


class TelemetryArgs(SessionArgs):
    driver_identifier: str | None = None
    driver_number: int | None = None
    lap_number: int | None = None


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _normalize_text(value: str | int | None) -> str:
    if value is None:
        return ''
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())


def _canonical_session_key(session_name: str) -> str:
    normalized = _normalize_text(session_name)
    for key, aliases in SESSION_ALIASES.items():
        if normalized == key:
            return key
        if any(normalized == _normalize_text(alias) for alias in aliases):
            return key
    raise McpError(ErrorData(code=INVALID_PARAMS, message=f'Unsupported session name: {session_name}'))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)
    except ValueError:
        return None


def _to_iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def _request_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    candidate_urls = [url]
    if url.startswith('https://'):
        candidate_urls.append(f'http://{url[len("https://") :]}')

    last_error: Exception | None = None
    for candidate in candidate_urls:
        try:
            response = requests.get(candidate, params=params, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_error = exc
            continue

        if response.status_code != 200:
            raise McpError(
                ErrorData(
                    code=INTERNAL_ERROR,
                    message=f'HTTP {response.status_code} for {response.url}: {response.text[:400]}',
                )
            )
        return response.json()

    try:
        raise last_error or RuntimeError('request failed without an exception')
    except Exception as exc:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f'Request failed: {exc}')) from exc


@lru_cache(maxsize=16)
def _season_schedule(year: int) -> list[dict[str, Any]]:
    payload = _request_json(f'{JOLPICA_BASE_URL}/{year}.json')
    return payload['MRData']['RaceTable']['Races']


@lru_cache(maxsize=16)
def _openf1_meetings(year: int) -> list[dict[str, Any]]:
    return _request_json(f'{OPENF1_BASE_URL}/meetings', params={'year': year})


@lru_cache(maxsize=64)
def _openf1_sessions(meeting_key: int) -> list[dict[str, Any]]:
    return _request_json(f'{OPENF1_BASE_URL}/sessions', params={'meeting_key': meeting_key})


def _event_candidates(event: dict[str, Any]) -> list[str]:
    circuit = event.get('Circuit') or {}
    location = circuit.get('Location') or {}
    candidates = [
        event.get('raceName'),
        circuit.get('circuitName'),
        location.get('locality'),
        location.get('country'),
        location.get('countryCode'),
    ]
    return [item for item in candidates if item]


def _meeting_candidates(meeting: dict[str, Any]) -> list[str]:
    candidates = [
        meeting.get('meeting_name'),
        meeting.get('meeting_official_name'),
        meeting.get('location'),
        meeting.get('country_name'),
        meeting.get('country_code'),
        meeting.get('circuit_short_name'),
    ]
    return [item for item in candidates if item]


def _best_match(items: list[dict[str, Any]], identifier: str | int, candidate_func) -> dict[str, Any]:
    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        target_round = str(identifier)
        for item in items:
            if str(item.get('round') or item.get('meeting_key')) == target_round:
                return item

    target = _normalize_text(identifier)
    exact: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for item in items:
        candidates = [_normalize_text(value) for value in candidate_func(item)]
        if target in candidates:
            exact.append(item)
            continue
        if any(target in candidate or candidate in target for candidate in candidates):
            partial.append(item)

    if exact:
        return exact[0]
    if partial:
        return partial[0]
    raise McpError(ErrorData(code=INVALID_PARAMS, message=f'Unable to resolve event identifier: {identifier}'))


def _resolve_event(year: int, event_identifier: str | int) -> dict[str, Any]:
    return _best_match(_season_schedule(year), event_identifier, _event_candidates)


def _resolve_openf1_meeting(year: int, event_identifier: str | int) -> dict[str, Any]:
    meetings = _openf1_meetings(year)
    try:
        return _best_match(meetings, event_identifier, _meeting_candidates)
    except McpError:
        event = _resolve_event(year, event_identifier)
        return _best_match(meetings, event.get('raceName') or event_identifier, _meeting_candidates)


def _resolve_openf1_session(year: int, event_identifier: str | int, session_name: str) -> dict[str, Any]:
    meeting = _resolve_openf1_meeting(year, event_identifier)
    target_name = OPENF1_SESSION_NAMES[_canonical_session_key(session_name)]
    for session in _openf1_sessions(int(meeting['meeting_key'])):
        if _normalize_text(session.get('session_name')) == _normalize_text(target_name):
            return session
    raise McpError(
        ErrorData(code=INVALID_PARAMS, message=f'Unable to resolve {session_name} for {event_identifier} in {year}')
    )


def _format_event_schedule(event: dict[str, Any]) -> dict[str, Any]:
    circuit = event.get('Circuit') or {}
    location = circuit.get('Location') or {}
    return {
        'season': event.get('season'),
        'round': event.get('round'),
        'race_name': event.get('raceName'),
        'date': event.get('date'),
        'time': event.get('time'),
        'circuit': circuit.get('circuitName'),
        'location': {
            'locality': location.get('locality'),
            'country': location.get('country'),
            'lat': location.get('lat'),
            'long': location.get('long'),
        },
        'sessions': {
            'fp1': event.get('FirstPractice'),
            'fp2': event.get('SecondPractice'),
            'fp3': event.get('ThirdPractice'),
            'qualifying': event.get('Qualifying'),
            'sprint': event.get('Sprint'),
            'race': {'date': event.get('date'), 'time': event.get('time')},
        },
    }


def _jolpica_session_rows(
    year: int, round_number: int | str, session_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    session_key = _canonical_session_key(session_name)
    endpoint = JOLPICA_RESULT_ENDPOINTS.get(session_key)
    if endpoint is None:
        return None

    payload = _request_json(f'{JOLPICA_BASE_URL}/{year}/{round_number}/{endpoint}.json')
    races = payload['MRData'].get('RaceTable', {}).get('Races', [])
    if not races:
        return None
    race = races[0]
    if session_key == 'qualifying':
        return race, race.get('QualifyingResults', [])
    key = 'SprintResults' if session_key == 'sprint' else 'Results'
    return race, race.get(key, [])


def _driver_candidates_from_jolpica(result: dict[str, Any]) -> list[str]:
    driver = result.get('Driver') or {}
    constructor = result.get('Constructor') or {}
    return [
        driver.get('code'),
        driver.get('driverId'),
        driver.get('givenName'),
        driver.get('familyName'),
        f'{driver.get("givenName", "")} {driver.get("familyName", "")}'.strip(),
        constructor.get('name'),
    ]


def _driver_candidates_from_openf1(driver: dict[str, Any]) -> list[str]:
    return [
        driver.get('name_acronym'),
        driver.get('full_name'),
        driver.get('broadcast_name'),
        driver.get('first_name'),
        driver.get('last_name'),
        driver.get('driver_number'),
        driver.get('team_name'),
    ]


def _match_driver(identifier: str, rows: list[dict[str, Any]], *, candidate_func) -> dict[str, Any]:
    target = _normalize_text(identifier)
    exact: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    for row in rows:
        candidates = [_normalize_text(value) for value in candidate_func(row)]
        if target in candidates:
            exact.append(row)
            continue
        if any(target in candidate or candidate in target for candidate in candidates):
            partial.append(row)
    if exact:
        return exact[0]
    if partial:
        return partial[0]
    raise McpError(ErrorData(code=INVALID_PARAMS, message=f'Unable to resolve driver: {identifier}'))


def _match_many_drivers(identifiers: list[str], rows: list[dict[str, Any]], *, candidate_func) -> list[dict[str, Any]]:
    return [_match_driver(identifier, rows, candidate_func=candidate_func) for identifier in identifiers]


def _openf1_driver_roster(session_key: int) -> list[dict[str, Any]]:
    return _request_json(f'{OPENF1_BASE_URL}/drivers', params={'session_key': session_key})


def _openf1_session_results(session_key: int) -> list[dict[str, Any]]:
    results = _request_json(f'{OPENF1_BASE_URL}/session_result', params={'session_key': session_key})
    drivers = {item['driver_number']: item for item in _openf1_driver_roster(session_key)}
    merged: list[dict[str, Any]] = []
    for item in results:
        driver = drivers.get(item.get('driver_number'), {})
        merged.append({**item, 'driver': driver})
    return merged


def _format_jolpica_session_result(session_name: str, result: dict[str, Any]) -> dict[str, Any]:
    driver = result.get('Driver') or {}
    constructor = result.get('Constructor') or {}
    payload = {
        'position': result.get('position'),
        'driver': {
            'code': driver.get('code'),
            'driver_id': driver.get('driverId'),
            'given_name': driver.get('givenName'),
            'family_name': driver.get('familyName'),
        },
        'constructor': constructor.get('name'),
    }
    session_key = _canonical_session_key(session_name)
    if session_key == 'qualifying':
        payload.update({'q1': result.get('Q1'), 'q2': result.get('Q2'), 'q3': result.get('Q3')})
    else:
        payload.update({
            'grid': result.get('grid'),
            'laps': result.get('laps'),
            'points': result.get('points'),
            'status': result.get('status'),
            'time': (result.get('Time') or {}).get('time'),
            'fastest_lap_rank': ((result.get('FastestLap') or {}).get('rank')),
            'fastest_lap_time': (((result.get('FastestLap') or {}).get('Time') or {}).get('time')),
        })
    return payload


def _format_openf1_session_result(result: dict[str, Any]) -> dict[str, Any]:
    driver = result.get('driver') or {}
    return {
        'position': result.get('position'),
        'driver_number': result.get('driver_number'),
        'number_of_laps': result.get('number_of_laps'),
        'dnf': result.get('dnf'),
        'dns': result.get('dns'),
        'dsq': result.get('dsq'),
        'gap_to_leader': result.get('gap_to_leader'),
        'duration': result.get('duration'),
        'driver': {
            'name_acronym': driver.get('name_acronym'),
            'full_name': driver.get('full_name'),
            'team_name': driver.get('team_name'),
        },
    }


def _lap_summary(session_key: int, driver_number: int, *, lap_number: int | None = None) -> dict[str, Any] | None:
    params: dict[str, Any] = {'session_key': session_key, 'driver_number': driver_number}
    if lap_number is not None:
        params['lap_number'] = lap_number
    laps = _request_json(f'{OPENF1_BASE_URL}/laps', params=params)
    if not laps:
        return None

    target_lap = None
    if lap_number is not None:
        target_lap = laps[0]
    else:
        timed_laps = [lap for lap in laps if isinstance(lap.get('lap_duration'), (int, float))]
        target_lap = min(timed_laps, key=lambda item: float(item['lap_duration'])) if timed_laps else laps[0]

    timed_laps = [lap for lap in laps if isinstance(lap.get('lap_duration'), (int, float))]
    average_lap_duration = (
        round(sum(float(lap['lap_duration']) for lap in timed_laps) / len(timed_laps), 3) if timed_laps else None
    )

    lap_payload: dict[str, Any] = {
        'lap_number': target_lap.get('lap_number'),
        'lap_duration': target_lap.get('lap_duration'),
        'sector_1': target_lap.get('duration_sector_1'),
        'sector_2': target_lap.get('duration_sector_2'),
        'sector_3': target_lap.get('duration_sector_3'),
        'speed_traps': {
            'i1': target_lap.get('i1_speed'),
            'i2': target_lap.get('i2_speed'),
            'finish': target_lap.get('st_speed'),
        },
        'average_lap_duration': average_lap_duration,
        'lap_count': len(laps),
    }

    start = _parse_datetime(target_lap.get('date_start'))
    duration = target_lap.get('lap_duration')
    if start is None or not isinstance(duration, (int, float)):
        return lap_payload

    end = start + timedelta(seconds=float(duration) + 1.0)
    car_data = _request_json(
        f'{OPENF1_BASE_URL}/car_data',
        params={
            'session_key': session_key,
            'driver_number': driver_number,
            'date>': _to_iso(start),
            'date<': _to_iso(end),
        },
    )
    if not car_data:
        return lap_payload

    speeds = [float(item['speed']) for item in car_data if item.get('speed') is not None]
    rpms = [float(item['rpm']) for item in car_data if item.get('rpm') is not None]
    throttles = [float(item['throttle']) for item in car_data if item.get('throttle') is not None]
    brakes = [item for item in car_data if item.get('brake') == 100]
    lap_payload['telemetry_summary'] = {
        'sample_count': len(car_data),
        'max_speed': max(speeds) if speeds else None,
        'avg_speed': round(sum(speeds) / len(speeds), 2) if speeds else None,
        'max_rpm': max(rpms) if rpms else None,
        'avg_throttle': round(sum(throttles) / len(throttles), 2) if throttles else None,
        'full_brake_samples': len(brakes),
    }
    return lap_payload


def _resolve_driver_number(session_key: int, driver_identifier: str | None, driver_number: int | None) -> int:
    if driver_number is not None:
        return driver_number
    if not driver_identifier:
        raise McpError(ErrorData(code=INVALID_PARAMS, message='Expected driver_identifier or driver_number'))
    roster = _openf1_driver_roster(session_key)
    row = _match_driver(driver_identifier, roster, candidate_func=_driver_candidates_from_openf1)
    return int(row['driver_number'])


def _driver_standings(year: int, round_number: int | None = None) -> list[dict[str, Any]]:
    if round_number is None:
        payload = _request_json(f'{JOLPICA_BASE_URL}/{year}/driverStandings.json')
    else:
        payload = _request_json(f'{JOLPICA_BASE_URL}/{year}/{round_number}/driverStandings.json')
    lists = payload['MRData']['StandingsTable'].get('StandingsLists', [])
    if not lists:
        return []
    return lists[0].get('DriverStandings', [])


def _format_driver_standing(standing: dict[str, Any]) -> dict[str, Any]:
    driver = standing.get('Driver') or {}
    constructors = standing.get('Constructors') or []
    return {
        'position': standing.get('position'),
        'points': standing.get('points'),
        'wins': standing.get('wins'),
        'driver': {
            'code': driver.get('code'),
            'driver_id': driver.get('driverId'),
            'given_name': driver.get('givenName'),
            'family_name': driver.get('familyName'),
            'date_of_birth': driver.get('dateOfBirth'),
            'nationality': driver.get('nationality'),
        },
        'constructors': [item.get('name') for item in constructors],
    }


async def main() -> None:
    server = Server('atlas-f1')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name='get_event_schedule',
                description='Get the F1 event schedule for a season.',
                inputSchema=YearArgs.model_json_schema(),
            ),
            Tool(
                name='get_event_info',
                description='Get details for a single F1 event.',
                inputSchema=EventArgs.model_json_schema(),
            ),
            Tool(
                name='get_session_results',
                description='Get session results for a race weekend session.',
                inputSchema=SessionArgs.model_json_schema(),
            ),
            Tool(
                name='get_driver_info',
                description='Get metadata for a driver using standings or session rosters.',
                inputSchema=DriverArgs.model_json_schema(),
            ),
            Tool(
                name='analyze_driver_performance',
                description='Summarize a driver performance for an event session.',
                inputSchema=AnalyzeArgs.model_json_schema(),
            ),
            Tool(
                name='compare_drivers',
                description='Compare multiple drivers in the same event session.',
                inputSchema=CompareArgs.model_json_schema(),
            ),
            Tool(
                name='get_telemetry',
                description='Return a summarized telemetry view for a driver lap.',
                inputSchema=TelemetryArgs.model_json_schema(),
            ),
            Tool(
                name='get_championship_standings',
                description='Get driver championship standings for a season.',
                inputSchema=YearArgs.model_json_schema(),
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        try:
            if name == 'get_event_schedule':
                args = YearArgs(**arguments)
                return _json_content([_format_event_schedule(event) for event in _season_schedule(args.year)])

            if name == 'get_event_info':
                args = EventArgs(**arguments)
                return _json_content(_format_event_schedule(_resolve_event(args.year, args.event_identifier)))

            if name == 'get_session_results':
                args = SessionArgs(**arguments)
                event = _resolve_event(args.year, args.event_identifier)
                jolpica = _jolpica_session_rows(args.year, event['round'], args.session_name)
                if jolpica is not None:
                    race, rows = jolpica
                    return _json_content({
                        'event': _format_event_schedule(race),
                        'session_name': args.session_name,
                        'results': [_format_jolpica_session_result(args.session_name, row) for row in rows],
                    })

                session = _resolve_openf1_session(args.year, args.event_identifier, args.session_name)
                return _json_content({
                    'event': {
                        'meeting_key': session.get('meeting_key'),
                        'session_key': session.get('session_key'),
                        'session_name': session.get('session_name'),
                        'date_start': session.get('date_start'),
                        'date_end': session.get('date_end'),
                    },
                    'results': [
                        _format_openf1_session_result(row)
                        for row in _openf1_session_results(int(session['session_key']))
                    ],
                })

            if name == 'get_driver_info':
                args = DriverArgs(**arguments)
                if args.year and args.event_identifier and args.session_name:
                    session = _resolve_openf1_session(args.year, args.event_identifier, args.session_name)
                    roster = _openf1_driver_roster(int(session['session_key']))
                    return _json_content(
                        _match_driver(args.driver_identifier, roster, candidate_func=_driver_candidates_from_openf1)
                    )
                if args.year:
                    standings = _driver_standings(args.year)
                    matched = _match_driver(
                        args.driver_identifier, standings, candidate_func=_driver_candidates_from_jolpica
                    )
                    return _json_content(_format_driver_standing(matched))
                raise McpError(
                    ErrorData(
                        code=INVALID_PARAMS,
                        message='get_driver_info expects either year or full session context (year, event_identifier, session_name)',
                    )
                )

            if name == 'get_championship_standings':
                args = YearArgs(**arguments)
                return _json_content([_format_driver_standing(item) for item in _driver_standings(args.year)])

            if name == 'analyze_driver_performance':
                args = AnalyzeArgs(**arguments)
                event = _resolve_event(args.year, args.event_identifier)
                session = _resolve_openf1_session(args.year, args.event_identifier, args.session_name)
                session_key = int(session['session_key'])
                driver_number = _resolve_driver_number(session_key, args.driver_identifier, None)
                telemetry = _lap_summary(session_key, driver_number)

                jolpica = _jolpica_session_rows(args.year, event['round'], args.session_name)
                result_payload: dict[str, Any] | None = None
                if jolpica is not None:
                    _, rows = jolpica
                    matched = _match_driver(
                        args.driver_identifier, rows, candidate_func=_driver_candidates_from_jolpica
                    )
                    result_payload = _format_jolpica_session_result(args.session_name, matched)
                else:
                    rows = _openf1_session_results(session_key)
                    matched = _match_driver(
                        args.driver_identifier,
                        rows,
                        candidate_func=lambda row: _driver_candidates_from_openf1(row.get('driver') or {}),
                    )
                    result_payload = _format_openf1_session_result(matched)

                return _json_content({
                    'event': _format_event_schedule(event),
                    'session': {
                        'session_key': session_key,
                        'session_name': session.get('session_name'),
                        'date_start': session.get('date_start'),
                    },
                    'driver_number': driver_number,
                    'result': result_payload,
                    'lap_summary': telemetry,
                })

            if name == 'compare_drivers':
                args = CompareArgs(**arguments)
                event = _resolve_event(args.year, args.event_identifier)
                identifiers = (
                    args.drivers
                    if isinstance(args.drivers, list)
                    else [item.strip() for item in args.drivers.split(',')]
                )
                identifiers = [item for item in identifiers if item]
                if len(identifiers) < 2:
                    raise McpError(ErrorData(code=INVALID_PARAMS, message='Expected at least two drivers'))

                session = _resolve_openf1_session(args.year, args.event_identifier, args.session_name)
                session_key = int(session['session_key'])
                jolpica = _jolpica_session_rows(args.year, event['round'], args.session_name)
                results: list[dict[str, Any]] = []

                if jolpica is not None:
                    _, rows = jolpica
                    matched_rows = _match_many_drivers(
                        identifiers, rows, candidate_func=_driver_candidates_from_jolpica
                    )
                    for row in matched_rows:
                        driver = row.get('Driver') or {}
                        results.append({
                            'driver_identifier': driver.get('code') or driver.get('driverId'),
                            'result': _format_jolpica_session_result(args.session_name, row),
                        })
                else:
                    rows = _openf1_session_results(session_key)
                    matched_rows = _match_many_drivers(
                        identifiers,
                        rows,
                        candidate_func=lambda row: _driver_candidates_from_openf1(row.get('driver') or {}),
                    )
                    for row in matched_rows:
                        driver = row.get('driver') or {}
                        results.append({
                            'driver_identifier': driver.get('name_acronym') or driver.get('full_name'),
                            'result': _format_openf1_session_result(row),
                        })

                return _json_content({
                    'event': _format_event_schedule(event),
                    'session': {'session_key': session_key, 'session_name': session.get('session_name')},
                    'drivers': results,
                })

            if name == 'get_telemetry':
                args = TelemetryArgs(**arguments)
                session = _resolve_openf1_session(args.year, args.event_identifier, args.session_name)
                session_key = int(session['session_key'])
                driver_number = _resolve_driver_number(session_key, args.driver_identifier, args.driver_number)
                telemetry = _lap_summary(session_key, driver_number, lap_number=args.lap_number)
                if telemetry is None:
                    raise McpError(
                        ErrorData(code=INTERNAL_ERROR, message='No telemetry data found for the requested driver/lap')
                    )
                return _json_content({
                    'session': {
                        'session_key': session_key,
                        'session_name': session.get('session_name'),
                        'date_start': session.get('date_start'),
                    },
                    'driver_number': driver_number,
                    'telemetry': telemetry,
                })

        except ValueError as exc:
            raise McpError(ErrorData(code=INVALID_PARAMS, message=str(exc))) from exc

        raise McpError(ErrorData(code=INVALID_PARAMS, message=f'Unknown tool: {name}'))

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
