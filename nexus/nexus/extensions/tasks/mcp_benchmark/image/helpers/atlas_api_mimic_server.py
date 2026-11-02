#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import subprocess
import tempfile
import urllib.parse

from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool


DATA_ROOT = Path('/data')
REQUEST_TIMEOUT = 30
BING_SEARCH_URL = 'https://www.bing.com/search'
COINGECKO_SEARCH_URL = 'https://api.coingecko.com/api/v3/search'
COINGECKO_MARKET_CHART_URL = 'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
MYMEMORY_URL = 'https://api.mymemory.translated.net/get'
ESPN_LEAGUES = {
    'NBA': ('basketball', 'nba'),
    'NFL': ('football', 'nfl'),
    'MLB': ('baseball', 'mlb'),
}
KNOWN_LANGUAGE_NAMES = {
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
}


def _error(code: int, message: str) -> McpError:
    return McpError(ErrorData(code=code, message=message))


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _request_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    try:
        response = requests.get(
            url, params=params, timeout=REQUEST_TIMEOUT, headers={'User-Agent': 'nexus-mcp-benchmark'}
        )
    except requests.RequestException as exc:
        raise _error(INTERNAL_ERROR, f'Request failed: {exc}') from exc
    if response.status_code != 200:
        raise _error(INTERNAL_ERROR, f'HTTP {response.status_code} for {response.url}: {response.text[:300]}')
    return response.json()


def _request_text(url: str, *, params: dict[str, Any] | None = None) -> str:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT, headers={'User-Agent': 'Mozilla/5.0'})
    except requests.RequestException as exc:
        raise _error(INTERNAL_ERROR, f'Request failed: {exc}') from exc
    if response.status_code != 200:
        raise _error(INTERNAL_ERROR, f'HTTP {response.status_code} for {response.url}: {response.text[:300]}')
    return response.text


def _strip_html(value: str) -> str:
    return re.sub(r'\s+', ' ', html.unescape(re.sub(r'<[^>]+>', ' ', value))).strip()


def _get_arg(arguments: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in arguments and arguments[name] not in {None, ''}:
            return arguments[name]
    return default


def _bing_search(query: str, *, limit: int = 8) -> list[dict[str, str]]:
    text = _request_text('https://html.duckduckgo.com/html/', params={'q': query})
    results: list[dict[str, str]] = []
    for match in re.finditer(
        r'(?s)<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        text,
    ):
        href = html.unescape(match.group(1))
        if href.startswith('//'):
            href = 'https:' + href
        parsed = urllib.parse.urlparse(href)
        params = urllib.parse.parse_qs(parsed.query)
        resolved_url = params.get('uddg', [href])[0]
        results.append({
            'title': _strip_html(match.group(2)),
            'url': urllib.parse.unquote(resolved_url),
            'snippet': _strip_html(match.group(3)),
        })
        if len(results) >= limit:
            break
    return results


def _fetch_page_summary(url: str) -> dict[str, str]:
    text = _request_text(url)
    title_match = re.search(r'<title>(.*?)</title>', text, re.S | re.I)
    meta_match = re.search(r'<meta[^>]+name="description"[^>]+content="([^"]+)"', text, re.I)
    og_match = re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]+)"', text, re.I)
    paragraphs = [_strip_html(item) for item in re.findall(r'<p[^>]*>(.*?)</p>', text, re.S | re.I)]
    body = ' '.join(item for item in paragraphs if item)[:2000]
    return {
        'title': _strip_html(title_match.group(1) if title_match else url),
        'description': _strip_html(meta_match.group(1) if meta_match else og_match.group(1) if og_match else body),
        'url': url,
    }


@lru_cache(maxsize=1)
def _airtable_tables() -> dict[str, dict[str, Any]]:
    tables: dict[str, dict[str, Any]] = {}
    for path in sorted(DATA_ROOT.glob('*.csv')):
        with path.open(encoding='utf-8', errors='ignore') as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        table_id = re.sub(r'[^A-Za-z0-9]+', '_', path.stem).strip('_').lower() or path.stem.lower()
        records = []
        for index, row in enumerate(rows, start=1):
            records.append({'id': f'rec_{table_id}_{index:04d}', 'fields': row})
        tables[table_id] = {'id': table_id, 'name': path.stem, 'fields': reader.fieldnames or [], 'records': records}
    return tables


def _airtable_list_bases(arguments: dict[str, Any]) -> dict[str, Any]:
    _ = arguments
    return {'bases': [{'id': 'app_local_atlas', 'name': 'Atlas Local Base'}]}


def _resolve_airtable_table(arguments: dict[str, Any]) -> dict[str, Any]:
    table_ref = str(_get_arg(arguments, 'table', 'table_name', 'table_id', default='')).strip().lower()
    tables = _airtable_tables()
    if not table_ref:
        return next(iter(tables.values()))
    for table in tables.values():
        if table_ref in {table['id'].lower(), table['name'].lower()}:
            return table
    raise _error(INVALID_PARAMS, f'Unknown Airtable table: {table_ref}')


def _airtable_list_tables(arguments: dict[str, Any]) -> dict[str, Any]:
    _ = arguments
    return {
        'tables': [
            {'id': table['id'], 'name': table['name'], 'fields': table['fields']}
            for table in _airtable_tables().values()
        ]
    }


def _airtable_list_records(arguments: dict[str, Any]) -> dict[str, Any]:
    table = _resolve_airtable_table(arguments)
    limit = int(_get_arg(arguments, 'max_records', 'limit', default=50))
    return {'records': table['records'][:limit]}


def _airtable_search_records(arguments: dict[str, Any]) -> dict[str, Any]:
    table = _resolve_airtable_table(arguments)
    query = str(_get_arg(arguments, 'query', 'search_query', default='')).strip().lower()
    if not query:
        return {'records': table['records'][:50]}
    records = [
        record for record in table['records'] if any(query in str(value).lower() for value in record['fields'].values())
    ]
    return {'records': records[:50]}


def _airtable_get_record(arguments: dict[str, Any]) -> dict[str, Any]:
    record_id = str(_get_arg(arguments, 'record_id', 'id', default='')).strip()
    if not record_id:
        raise _error(INVALID_PARAMS, 'record_id is required')
    table = _resolve_airtable_table(arguments)
    for record in table['records']:
        if record['id'] == record_id:
            return record
    raise _error(INVALID_PARAMS, f'Unknown Airtable record: {record_id}')


def _coin_id_for_symbol(symbol: str) -> str:
    normalized = symbol.strip().lower()
    static = {
        'btc': 'bitcoin',
        'eth': 'ethereum',
        'usdc': 'usd-coin',
        'usdt': 'tether',
        'link': 'chainlink',
        'aave': 'aave',
    }
    if normalized in static:
        return static[normalized]
    payload = _request_json(COINGECKO_SEARCH_URL, params={'query': symbol})
    coins = payload.get('coins') or []
    if not coins:
        raise _error(INVALID_PARAMS, f'Unable to resolve token symbol: {symbol}')
    return str(coins[0]['id'])


def _alchemy_fetch_token_price_history(arguments: dict[str, Any]) -> dict[str, Any]:
    symbol = str(_get_arg(arguments, 'symbol', default='')).strip()
    if not symbol:
        raise _error(INVALID_PARAMS, 'symbol is required')
    coin_id = _coin_id_for_symbol(symbol)
    days = 30
    payload = _request_json(
        COINGECKO_MARKET_CHART_URL.format(coin_id=coin_id),
        params={'vs_currency': 'usd', 'days': days, 'interval': 'daily'},
    )
    return {
        'symbol': symbol.upper(),
        'coin_id': coin_id,
        'currency': 'usd',
        'prices': [
            {'timestamp': datetime.fromtimestamp(item[0] / 1000, tz=UTC).isoformat(), 'value': item[1]}
            for item in payload.get('prices', [])
        ],
    }


def _e2b_run_code(arguments: dict[str, Any]) -> dict[str, Any]:
    code = str(_get_arg(arguments, 'code', default=''))
    if not code:
        raise _error(INVALID_PARAMS, 'code is required')
    with tempfile.TemporaryDirectory(prefix='atlas-e2b-') as temp_dir:
        script_path = Path(temp_dir) / 'snippet.py'
        script_path.write_text(code, encoding='utf-8')
        result = subprocess.run(
            ['python3', str(script_path)], capture_output=True, text=True, timeout=120, cwd=temp_dir
        )
    return {'stdout': result.stdout, 'stderr': result.stderr, 'return_code': result.returncode}


def _translate_segment(text: str, source: str, target: str) -> str:
    payload = _request_json(MYMEMORY_URL, params={'q': text, 'langpair': f'{source}|{target}'})
    return str((payload.get('responseData') or {}).get('translatedText') or '')


def _lara_translate(arguments: dict[str, Any]) -> dict[str, Any]:
    target = str(_get_arg(arguments, 'target', 'to', default='en'))
    source = str(_get_arg(arguments, 'source', 'from', default='auto'))
    raw_text = _get_arg(arguments, 'text', default='')
    segments: list[str]
    if isinstance(raw_text, list):
        segments = [str(item.get('text') if isinstance(item, dict) else item) for item in raw_text]
    else:
        segments = [str(raw_text)]
    translations = [_translate_segment(item, source, target) for item in segments]
    return {'translations': translations, 'target': target, 'source': source}


def _lara_list_languages(arguments: dict[str, Any]) -> dict[str, Any]:
    _ = arguments
    return {'languages': [{'code': code, 'name': name} for code, name in sorted(KNOWN_LANGUAGE_NAMES.items())]}


@lru_cache(maxsize=8)
def _espn_teams(league: str) -> list[dict[str, Any]]:
    sport, league_slug = ESPN_LEAGUES[league]
    payload = _request_json(f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{league_slug}/teams')
    teams = ((payload.get('sports') or [{}])[0].get('leagues') or [{}])[0].get('teams') or []
    rows: list[dict[str, Any]] = []
    for team_wrapper in teams:
        team = team_wrapper.get('team') or {}
        rows.append({
            'id': int(team.get('id') or 0),
            'abbreviation': team.get('abbreviation'),
            'full_name': team.get('displayName'),
            'name': team.get('name'),
            'conference': ((team.get('links') or [{}])[0].get('text') or ''),
        })
    return rows


def _balldontlie_get_teams(arguments: dict[str, Any]) -> dict[str, Any]:
    league = str(_get_arg(arguments, 'league', default='NBA')).upper()
    if league not in ESPN_LEAGUES:
        raise _error(INVALID_PARAMS, f'Unsupported league: {league}')
    return {'data': _espn_teams(league)}


def _balldontlie_get_games(arguments: dict[str, Any]) -> dict[str, Any]:
    league = str(_get_arg(arguments, 'league', default='NBA')).upper()
    if league not in ESPN_LEAGUES:
        raise _error(INVALID_PARAMS, f'Unsupported league: {league}')
    sport, league_slug = ESPN_LEAGUES[league]
    team_ids = [int(item) for item in (_get_arg(arguments, 'teamIds', default=[]) or [])]
    dates = list(_get_arg(arguments, 'dates', default=[]) or [])
    seasons = list(_get_arg(arguments, 'seasons', default=[]) or [])
    events: list[dict[str, Any]] = []

    if dates:
        for raw_date in dates[:5]:
            date_key = str(raw_date).replace('-', '')
            payload = _request_json(
                f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{league_slug}/scoreboard',
                params={'dates': date_key},
            )
            events.extend(payload.get('events') or [])
    elif team_ids and seasons:
        team_id = team_ids[0]
        season = seasons[0]
        payload = _request_json(
            f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{league_slug}/teams/{team_id}/schedule',
            params={'season': season},
        )
        events.extend(payload.get('events') or [])
    else:
        payload = _request_json(f'https://site.api.espn.com/apis/site/v2/sports/{sport}/{league_slug}/scoreboard')
        events.extend(payload.get('events') or [])

    rows: list[dict[str, Any]] = []
    for event in events[:50]:
        competitions = event.get('competitions') or []
        competition = competitions[0] if competitions else {}
        competitors = competition.get('competitors') or []
        team_ids_present = {int((item.get('team') or {}).get('id') or 0) for item in competitors}
        if team_ids and not team_ids_present.intersection(team_ids):
            continue
        rows.append({
            'id': event.get('id'),
            'date': event.get('date'),
            'status': ((competition.get('status') or {}).get('type') or {}).get('description'),
            'league': league,
            'home_team': (
                (next((item for item in competitors if item.get('homeAway') == 'home'), {}) or {}).get('team') or {}
            ).get('displayName'),
            'away_team': (
                (next((item for item in competitors if item.get('homeAway') == 'away'), {}) or {}).get('team') or {}
            ).get('displayName'),
            'home_score': next((item.get('score') for item in competitors if item.get('homeAway') == 'home'), None),
            'away_score': next((item.get('score') for item in competitors if item.get('homeAway') == 'away'), None),
        })
    return {'data': rows}


def _national_parks_find(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(_get_arg(arguments, 'query', 'q', 'parkCode', 'stateCode', default='national park')).strip()
    results = []
    for item in _bing_search(f'site:nps.gov {query} national park', limit=10):
        if 'nps.gov' not in item['url']:
            continue
        match = re.search(r'nps\.gov/([a-z0-9]{4})/', item['url'])
        results.append({
            'parkCode': match.group(1) if match else '',
            'fullName': item['title'],
            'description': item['snippet'],
            'url': item['url'],
        })
        if len(results) >= 6:
            break
    return {'data': results}


def _national_parks_details(arguments: dict[str, Any]) -> dict[str, Any]:
    park_code = str(_get_arg(arguments, 'parkCode', default='')).strip().lower()
    if not park_code:
        search = _national_parks_find(arguments).get('data') or []
        if not search:
            raise _error(INVALID_PARAMS, 'Unable to resolve park')
        url = search[0]['url']
    else:
        url = f'https://www.nps.gov/{park_code}/index.htm'
    summary = _fetch_page_summary(url)
    return {'data': summary}


def _national_parks_page(arguments: dict[str, Any], page: str) -> dict[str, Any]:
    park_code = str(_get_arg(arguments, 'parkCode', default='')).strip().lower()
    if not park_code:
        details = _national_parks_find(arguments).get('data') or []
        if not details or not details[0].get('parkCode'):
            raise _error(INVALID_PARAMS, 'parkCode is required')
        park_code = str(details[0]['parkCode'])
    return {'data': _fetch_page_summary(f'https://www.nps.gov/{park_code}/planyourvisit/{page}.htm')}


def _search_results_payload(query: str, *, limit: int = 8) -> dict[str, Any]:
    return {'results': _bing_search(query, limit=limit)}


def _brave_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(_get_arg(arguments, 'query', 'q', default='')).strip()
    if not query:
        raise _error(INVALID_PARAMS, 'query is required')
    return {'web': {'results': _bing_search(query, limit=8)}}


def _exa_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(_get_arg(arguments, 'query', 'q', default='')).strip()
    if not query:
        raise _error(INVALID_PARAMS, 'query is required')
    return _search_results_payload(query)


def _oxylabs_google_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(_get_arg(arguments, 'query', 'q', default='')).strip()
    if not query:
        raise _error(INVALID_PARAMS, 'query is required')
    return _search_results_payload(query)


def _oxylabs_amazon_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(_get_arg(arguments, 'query', 'q', 'keyword', default='')).strip()
    if not query:
        raise _error(INVALID_PARAMS, 'query is required')
    return _search_results_payload(f'site:amazon.com {query}')


def _oxylabs_universal_scraper(arguments: dict[str, Any]) -> dict[str, Any]:
    url = str(_get_arg(arguments, 'url', default='')).strip()
    if url:
        return _fetch_page_summary(url)
    query = str(_get_arg(arguments, 'query', 'q', default='')).strip()
    if query:
        return _search_results_payload(query)
    raise _error(INVALID_PARAMS, 'Expected url or query')


def _search_rijksmuseum(query: str) -> list[dict[str, Any]]:
    results = []
    for item in _bing_search(f'site:rijksmuseum.nl/en/collection {query}', limit=8):
        match = re.search(r'/collection/([A-Z0-9-]+)', item['url'])
        if not match:
            continue
        results.append({
            'objectNumber': match.group(1),
            'title': item['title'],
            'url': item['url'],
            'description': item['snippet'],
        })
    return results


def _rijksmuseum_search(arguments: dict[str, Any]) -> dict[str, Any]:
    query = str(_get_arg(arguments, 'q', 'query', default='')).strip()
    if not query:
        raise _error(INVALID_PARAMS, 'q is required')
    return {'results': _search_rijksmuseum(query)}


def _rijksmuseum_details(arguments: dict[str, Any]) -> dict[str, Any]:
    object_number = str(_get_arg(arguments, 'objectNumber', 'object_number', default='')).strip()
    if not object_number:
        raise _error(INVALID_PARAMS, 'objectNumber is required')
    summary = _fetch_page_summary(f'https://www.rijksmuseum.nl/en/collection/{object_number}')
    summary['objectNumber'] = object_number
    return summary


def _rijksmuseum_artist_timeline(arguments: dict[str, Any]) -> dict[str, Any]:
    artist = str(_get_arg(arguments, 'artist', 'name', 'query', default='')).strip()
    if not artist:
        raise _error(INVALID_PARAMS, 'artist is required')
    return {'results': _search_rijksmuseum(artist)}


def _normalize_symbol(symbol: str) -> str:
    text = symbol.strip().upper()
    if '/' in text and len(text) == 7:
        return text.replace('/', '') + '=X'
    if text.endswith('USD') and len(text) <= 6 and '-' not in text:
        return text[:-3] + '-USD'
    return text


def _history_frame(symbol: str, interval: str = '1d', period: str = '6mo'):
    import yfinance as yf  # lazy import for runtime image

    ticker = yf.Ticker(_normalize_symbol(symbol))
    frame = ticker.history(period=period, interval=interval, auto_adjust=False)
    if frame.empty:
        raise _error(INVALID_PARAMS, f'No market data returned for symbol: {symbol}')
    return frame.reset_index()


def _frame_to_values(frame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in frame.to_dict(orient='records'):
        dt = record.get('Date') or record.get('Datetime')
        if hasattr(dt, 'to_pydatetime'):
            dt = dt.to_pydatetime()
        dt_text = dt.isoformat() if isinstance(dt, datetime) else str(dt)
        rows.append({
            'datetime': dt_text,
            'open': record.get('Open'),
            'high': record.get('High'),
            'low': record.get('Low'),
            'close': record.get('Close'),
            'volume': record.get('Volume'),
            **{
                key: value
                for key, value in record.items()
                if key not in {'Date', 'Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'}
            },
        })
    return rows


def _twelvedata_time_series(arguments: dict[str, Any]) -> dict[str, Any]:
    symbol = str(_get_arg(arguments, 'symbol', default='')).strip()
    if not symbol:
        raise _error(INVALID_PARAMS, 'symbol is required')
    frame = _history_frame(symbol)
    return {'meta': {'symbol': symbol.upper(), 'interval': '1day'}, 'values': _frame_to_values(frame.tail(120))}


def _twelvedata_indicator(arguments: dict[str, Any], indicator: str) -> dict[str, Any]:
    import pandas as pd  # lazy import for runtime image

    symbol = str(_get_arg(arguments, 'symbol', default='')).strip()
    if not symbol:
        raise _error(INVALID_PARAMS, 'symbol is required')
    frame = _history_frame(symbol)
    close = frame['Close']
    high = frame['High']
    low = frame['Low']
    period = int(_get_arg(arguments, 'time_period', 'period', default=14))

    if indicator == 'rsi':
        delta = close.diff()
        gains = delta.clip(lower=0).rolling(period).mean()
        losses = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gains / losses.replace(0, math.nan)
        frame['rsi'] = 100 - (100 / (1 + rs))
    elif indicator == 'macd':
        fast = int(_get_arg(arguments, 'fast_period', default=12))
        slow = int(_get_arg(arguments, 'slow_period', default=26))
        signal = int(_get_arg(arguments, 'signal_period', default=9))
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        frame['macd'] = macd
        frame['macd_signal'] = macd.ewm(span=signal, adjust=False).mean()
        frame['macd_hist'] = frame['macd'] - frame['macd_signal']
    elif indicator == 'bbands':
        rolling = close.rolling(period)
        middle = rolling.mean()
        std = rolling.std()
        frame['middle_band'] = middle
        frame['upper_band'] = middle + (std * 2)
        frame['lower_band'] = middle - (std * 2)
    elif indicator == 'sma':
        frame['sma'] = close.rolling(period).mean()
    elif indicator == 'ema':
        frame['ema'] = close.ewm(span=period, adjust=False).mean()
    elif indicator == 'atr':
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        frame['atr'] = tr.rolling(period).mean()
    elif indicator == 'adx':
        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, math.nan)) * 100
        frame['adx'] = dx.rolling(period).mean()
    elif indicator == 'cross':
        fast = int(_get_arg(arguments, 'fast_period', default=5))
        slow = int(_get_arg(arguments, 'slow_period', default=20))
        fast_ma = close.rolling(fast).mean()
        slow_ma = close.rolling(slow).mean()
        frame['fast_ma'] = fast_ma
        frame['slow_ma'] = slow_ma
        frame['signal'] = [
            'bullish' if a > b else 'bearish' if a < b else 'neutral' for a, b in zip(fast_ma, slow_ma, strict=False)
        ]
    else:
        raise _error(INVALID_PARAMS, f'Unsupported indicator: {indicator}')

    return {'meta': {'symbol': symbol.upper(), 'indicator': indicator}, 'values': _frame_to_values(frame.tail(120))}


def _twelvedata_symbol_search(arguments: dict[str, Any]) -> dict[str, Any]:
    symbol = str(_get_arg(arguments, 'symbol', 'query', default='')).strip()
    if not symbol:
        raise _error(INVALID_PARAMS, 'symbol is required')
    return {
        'data': [
            {
                'symbol': symbol.upper(),
                'instrument_name': symbol.upper(),
                'exchange': '',
                'country': '',
                'currency': 'USD',
            }
        ]
    }


def _twelvedata_static_list(_arguments: dict[str, Any], *, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {'data': items}


SERVER_TOOLS: dict[str, dict[str, Any]] = {
    'airtable': {
        'airtable_list_bases': _airtable_list_bases,
        'airtable_list_tables': _airtable_list_tables,
        'airtable_list_records': _airtable_list_records,
        'airtable_search_records': _airtable_search_records,
        'airtable_get_record': _airtable_get_record,
    },
    'alchemy': {
        'alchemy_fetchTokenPriceHistoryBySymbol': _alchemy_fetch_token_price_history,
    },
    'balldontlie': {
        'balldontlie_get_teams': _balldontlie_get_teams,
        'balldontlie_get_games': _balldontlie_get_games,
    },
    'brave-search': {
        'brave-search_brave_web_search': _brave_search,
    },
    'e2b-server': {
        'e2b-server_run_code': _e2b_run_code,
    },
    'exa': {
        'exa_web_search_exa': _exa_search,
    },
    'lara-translate': {
        'lara-translate_translate': _lara_translate,
        'lara-translate_list_languages': _lara_list_languages,
    },
    'national-parks': {
        'national-parks_findParks': _national_parks_find,
        'national-parks_getParkDetails': _national_parks_details,
        'national-parks_getVisitorCenters': lambda arguments: _national_parks_page(arguments, 'visitorcenters'),
        'national-parks_getCampgrounds': lambda arguments: _national_parks_page(arguments, 'camping'),
    },
    'oxylabs': {
        'oxylabs_google_search_scraper': _oxylabs_google_search,
        'oxylabs_amazon_search_scraper': _oxylabs_amazon_search,
        'oxylabs_universal_scraper': _oxylabs_universal_scraper,
    },
    'rijksmuseum-server': {
        'rijksmuseum-server_search_artwork': _rijksmuseum_search,
        'rijksmuseum-server_get_artwork_details': _rijksmuseum_details,
        'rijksmuseum-server_get_artist_timeline': _rijksmuseum_artist_timeline,
    },
    'twelvedata': {
        'twelvedata_GetTimeSeries': _twelvedata_time_series,
        'twelvedata_GetTimeSeriesRsi': lambda arguments: _twelvedata_indicator(arguments, 'rsi'),
        'twelvedata_GetTimeSeriesMacd': lambda arguments: _twelvedata_indicator(arguments, 'macd'),
        'twelvedata_GetTimeSeriesBBands': lambda arguments: _twelvedata_indicator(arguments, 'bbands'),
        'twelvedata_GetTimeSeriesSma': lambda arguments: _twelvedata_indicator(arguments, 'sma'),
        'twelvedata_GetTimeSeriesEma': lambda arguments: _twelvedata_indicator(arguments, 'ema'),
        'twelvedata_GetTimeSeriesAtr': lambda arguments: _twelvedata_indicator(arguments, 'atr'),
        'twelvedata_GetTimeSeriesCross': lambda arguments: _twelvedata_indicator(arguments, 'cross'),
        'twelvedata_GetTimeSeriesAdx': lambda arguments: _twelvedata_indicator(arguments, 'adx'),
        'twelvedata_GetSymbolSearch': _twelvedata_symbol_search,
        'twelvedata_GetCryptocurrencies': lambda arguments: _twelvedata_static_list(
            arguments,
            items=[
                {'symbol': 'BTC/USD', 'currency_base': 'BTC', 'currency_quote': 'USD'},
                {'symbol': 'ETH/USD', 'currency_base': 'ETH', 'currency_quote': 'USD'},
            ],
        ),
        'twelvedata_GetExchanges': lambda arguments: _twelvedata_static_list(
            arguments,
            items=[{'name': 'NASDAQ', 'country': 'United States'}, {'name': 'NYSE', 'country': 'United States'}],
        ),
        'twelvedata_GetCryptocurrencyExchanges': lambda arguments: _twelvedata_static_list(
            arguments,
            items=[{'name': 'Coinbase', 'country': 'United States'}, {'name': 'Binance', 'country': 'Global'}],
        ),
        'twelvedata_GetForexPairs': lambda arguments: _twelvedata_static_list(
            arguments,
            items=[{'symbol': 'EUR/USD'}, {'symbol': 'USD/JPY'}, {'symbol': 'GBP/USD'}],
        ),
    },
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', required=True)
    args = parser.parse_args()

    handlers = SERVER_TOOLS.get(args.server)
    if handlers is None:
        raise SystemExit(f'Unsupported Atlas mimic server: {args.server}')

    server = Server(f'atlas-mimic-{args.server}')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=name,
                description=f'Atlas self-hosted mock for {args.server}: {name}.',
                inputSchema={'type': 'object', 'properties': {}, 'additionalProperties': True},
            )
            for name in handlers
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = handlers.get(name)
        if handler is None:
            raise _error(INVALID_PARAMS, f'Unknown tool: {name}')
        try:
            return _json_content(handler(arguments or {}))
        except McpError:
            raise
        except subprocess.TimeoutExpired as exc:
            raise _error(INTERNAL_ERROR, f'Command timed out: {exc}') from exc
        except Exception as exc:  # pragma: no cover - surfaced as MCP error
            raise _error(INTERNAL_ERROR, f'{args.server} mock failed: {exc}') from exc

    async with stdio_server() as streams:
        await server.run(streams[0], streams[1], server.create_initialization_options())


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
