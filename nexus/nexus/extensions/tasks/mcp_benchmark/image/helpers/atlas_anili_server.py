#!/usr/bin/env python3
from __future__ import annotations

import json

from typing import Any

import requests

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData, TextContent, Tool
from pydantic import BaseModel, Field


ANILIST_GRAPHQL_URL = 'https://graphql.anilist.co'


class EmptyArgs(BaseModel):
    pass


class SearchArgs(BaseModel):
    term: str
    page: int = 1
    amount: int = Field(default=5, ge=1, le=50)


class LookupArgs(BaseModel):
    id: int | None = None
    term: str | None = None
    search: str | None = None
    name: str | None = None


def _json_content(payload: Any) -> list[TextContent]:
    return [TextContent(type='text', text=json.dumps(payload, ensure_ascii=False, indent=2))]


def _graphql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.post(
            ANILIST_GRAPHQL_URL,
            json={'query': query, 'variables': variables or {}},
            headers={'Content-Type': 'application/json', 'Accept': 'application/json'},
            timeout=30,
        )
    except requests.RequestException as exc:
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=f'AniList request failed: {exc}')) from exc

    if response.status_code != 200:
        raise McpError(
            ErrorData(
                code=INTERNAL_ERROR, message=f'AniList returned status {response.status_code}: {response.text[:500]}'
            )
        )

    payload = response.json()
    if payload.get('errors'):
        raise McpError(ErrorData(code=INTERNAL_ERROR, message=json.dumps(payload['errors'], ensure_ascii=False)))
    return payload.get('data', {})


def _lookup_search_text(args: LookupArgs) -> str | None:
    return args.search or args.term or args.name


def _validate_lookup_args(arguments: dict[str, Any]) -> LookupArgs:
    try:
        args = LookupArgs(**arguments)
    except ValueError as exc:
        raise McpError(ErrorData(code=INVALID_PARAMS, message=str(exc))) from exc
    if args.id is None and _lookup_search_text(args) is None:
        raise McpError(ErrorData(code=INVALID_PARAMS, message='Expected one of: id, search, term, name'))
    return args


def _search_media(media_type: str, args: SearchArgs) -> dict[str, Any]:
    query = f"""
    query ($term: String, $page: Int, $amount: Int) {{
      Page(page: $page, perPage: $amount) {{
        pageInfo {{ total currentPage lastPage hasNextPage }}
        media(search: $term, type: {media_type}) {{
          id
          title {{ romaji english native }}
          type
          format
          status
          episodes
          chapters
          volumes
          startDate {{ year month day }}
          endDate {{ year month day }}
          genres
          averageScore
          popularity
          seasonYear
          siteUrl
          studios(isMain: true) {{ nodes {{ id name }} }}
        }}
      }}
    }}
    """
    return _graphql(query, {'term': args.term, 'page': args.page, 'amount': args.amount})


def _get_media(media_type: str, args: LookupArgs) -> dict[str, Any]:
    query = f"""
    query ($id: Int, $search: String) {{
      Media(id: $id, search: $search, type: {media_type}) {{
        id
        title {{ romaji english native }}
        synonyms
        description(asHtml: false)
        type
        format
        status
        episodes
        chapters
        volumes
        season
        seasonYear
        source
        startDate {{ year month day }}
        endDate {{ year month day }}
        genres
        averageScore
        meanScore
        popularity
        favourites
        siteUrl
        studios(isMain: true) {{ nodes {{ id name isAnimationStudio siteUrl }} }}
        tags {{ id name category rank isGeneralSpoiler isMediaSpoiler }}
      }}
    }}
    """
    return _graphql(query, {'id': args.id, 'search': _lookup_search_text(args)})


def _search_character(args: SearchArgs) -> dict[str, Any]:
    query = """
    query ($term: String, $page: Int, $amount: Int) {
      Page(page: $page, perPage: $amount) {
        pageInfo { total currentPage lastPage hasNextPage }
        characters(search: $term) {
          id
          name { full native alternative }
          description(asHtml: false)
          age
          gender
          image { large }
          siteUrl
        }
      }
    }
    """
    return _graphql(query, {'term': args.term, 'page': args.page, 'amount': args.amount})


def _get_character(args: LookupArgs) -> dict[str, Any]:
    query = """
    query ($id: Int, $search: String) {
      Character(id: $id, search: $search) {
        id
        name { full native alternative }
        description(asHtml: false)
        age
        gender
        bloodType
        dateOfBirth { year month day }
        image { large }
        siteUrl
      }
    }
    """
    return _graphql(query, {'id': args.id, 'search': _lookup_search_text(args)})


def _search_staff(args: SearchArgs) -> dict[str, Any]:
    query = """
    query ($term: String, $page: Int, $amount: Int) {
      Page(page: $page, perPage: $amount) {
        pageInfo { total currentPage lastPage hasNextPage }
        staff(search: $term) {
          id
          name { full native alternative }
          description(asHtml: false)
          primaryOccupations
          image { large }
          siteUrl
        }
      }
    }
    """
    return _graphql(query, {'term': args.term, 'page': args.page, 'amount': args.amount})


def _get_staff(args: LookupArgs) -> dict[str, Any]:
    query = """
    query ($id: Int, $search: String) {
      Staff(id: $id, search: $search) {
        id
        name { full native alternative }
        description(asHtml: false)
        primaryOccupations
        homeTown
        image { large }
        siteUrl
      }
    }
    """
    return _graphql(query, {'id': args.id, 'search': _lookup_search_text(args)})


def _search_studio(args: SearchArgs) -> dict[str, Any]:
    query = """
    query ($term: String, $page: Int, $amount: Int) {
      Page(page: $page, perPage: $amount) {
        pageInfo { total currentPage lastPage hasNextPage }
        studios(search: $term) {
          id
          name
          isAnimationStudio
          siteUrl
        }
      }
    }
    """
    return _graphql(query, {'term': args.term, 'page': args.page, 'amount': args.amount})


def _get_studio(args: LookupArgs) -> dict[str, Any]:
    search_text = _lookup_search_text(args)
    if args.id is None and search_text is not None:
        query = """
        query ($term: String) {
          Page(page: 1, perPage: 1) {
            studios(search: $term) {
              id
              name
              isAnimationStudio
              siteUrl
            }
          }
        }
        """
        return _graphql(query, {'term': search_text})
    query = """
    query ($id: Int, $search: String) {
      Studio(id: $id, search: $search) {
        id
        name
        isAnimationStudio
        siteUrl
      }
    }
    """
    return _graphql(query, {'id': args.id, 'search': search_text})


def _search_args_schema() -> dict[str, Any]:
    return SearchArgs.model_json_schema()


def _lookup_args_schema() -> dict[str, Any]:
    return LookupArgs.model_json_schema()


async def main() -> None:
    server = Server('atlas-anili')

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(name='search_anime', description='Search anime on AniList.', inputSchema=_search_args_schema()),
            Tool(name='search_manga', description='Search manga on AniList.', inputSchema=_search_args_schema()),
            Tool(
                name='search_character', description='Search characters on AniList.', inputSchema=_search_args_schema()
            ),
            Tool(name='search_staff', description='Search staff on AniList.', inputSchema=_search_args_schema()),
            Tool(name='search_studio', description='Search studios on AniList.', inputSchema=_search_args_schema()),
            Tool(name='get_genres', description='List all AniList genres.', inputSchema=EmptyArgs.model_json_schema()),
            Tool(
                name='get_media_tags', description='List AniList media tags.', inputSchema=EmptyArgs.model_json_schema()
            ),
            Tool(
                name='get_anime',
                description='Get anime details by id or search term.',
                inputSchema=_lookup_args_schema(),
            ),
            Tool(
                name='get_manga',
                description='Get manga details by id or search term.',
                inputSchema=_lookup_args_schema(),
            ),
            Tool(
                name='get_character',
                description='Get character details by id or search term.',
                inputSchema=_lookup_args_schema(),
            ),
            Tool(
                name='get_staff',
                description='Get staff details by id or search term.',
                inputSchema=_lookup_args_schema(),
            ),
            Tool(
                name='get_studio',
                description='Get studio details by id or search term.',
                inputSchema=_lookup_args_schema(),
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name in {'search_anime', 'search_manga', 'search_character', 'search_staff', 'search_studio'}:
            try:
                args = SearchArgs(**arguments)
            except ValueError as exc:
                raise McpError(ErrorData(code=INVALID_PARAMS, message=str(exc))) from exc

            if name == 'search_anime':
                return _json_content(_search_media('ANIME', args))
            if name == 'search_manga':
                return _json_content(_search_media('MANGA', args))
            if name == 'search_character':
                return _json_content(_search_character(args))
            if name == 'search_staff':
                return _json_content(_search_staff(args))
            return _json_content(_search_studio(args))

        if name == 'get_genres':
            return _json_content(_graphql('query { GenreCollection }'))
        if name == 'get_media_tags':
            return _json_content(
                _graphql(
                    """
                    query {
                      MediaTagCollection {
                        id
                        name
                        description
                        category
                        rank
                        isAdult
                      }
                    }
                    """
                )
            )

        if name in {'get_anime', 'get_manga', 'get_character', 'get_staff', 'get_studio'}:
            args = _validate_lookup_args(arguments)
            if name == 'get_anime':
                return _json_content(_get_media('ANIME', args))
            if name == 'get_manga':
                return _json_content(_get_media('MANGA', args))
            if name == 'get_character':
                return _json_content(_get_character(args))
            if name == 'get_staff':
                return _json_content(_get_staff(args))
            return _json_content(_get_studio(args))

        raise McpError(ErrorData(code=INVALID_PARAMS, message=f'Unknown tool: {name}'))

    options = server.create_initialization_options()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, options, raise_exceptions=True)


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
