"""Runtime registrations for the seven path-native general-context tools.

The implementations remain in :mod:`.registry` and :mod:`.tools`; these thin
adapters only validate Nexus tool inputs, recover the sandbox-side rollout
session, and convert the core string result to ``ToolResponse``.
"""

from __future__ import annotations

import copy

from typing import Any, ClassVar

from pydantic import BaseModel, field_validator

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry

from .registry import dispatch_tool
from .schemas import TOOL_SCHEMA_REGISTRY
from .sessions import cleanup_session, get_session
from .subprocess_env import execution_identity


TOOL_NAMESPACE = 'context_agent_general'
CLEANUP_TOOL_NAME = '_cleanup'


def registry_id(short_name: str) -> str:
    """Return the namespaced Nexus registry id for a model-visible tool name."""
    return f'{TOOL_NAMESPACE}/{short_name}'


def _text(result: str) -> ToolResponse:
    return ToolResponse(output=[{'type': 'text', 'text': result}])


class _ExactSchemaModel(BaseModel):
    """Pydantic validation whose generated schema is the existing contract."""

    _schema_name: ClassVar[str]

    @classmethod
    def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return copy.deepcopy(TOOL_SCHEMA_REGISTRY[cls._schema_name]['function']['parameters'])


class CurlInput(_ExactSchemaModel):
    _schema_name = 'curl'

    url: str
    output: str | None = None


class BashInput(_ExactSchemaModel):
    _schema_name = 'bash'

    command: str


class PythonInput(_ExactSchemaModel):
    _schema_name = 'python'

    code: str


class MsearchInput(_ExactSchemaModel):
    _schema_name = 'file_search_msearch'

    queries: list[str]
    top_k: int | None = None
    files: str | list[str] | None = None

    @field_validator('queries', mode='before')
    @classmethod
    def _coerce_queries(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class MclickInput(_ExactSchemaModel):
    _schema_name = 'file_search_mclick'

    pointers: list[str]

    @field_validator('pointers', mode='before')
    @classmethod
    def _coerce_pointers(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class GrepInput(_ExactSchemaModel):
    _schema_name = 'grep'

    pattern: str
    path: str | list[str] | None = None
    use_regex: bool = False
    context_lines: int = 1
    before: int | None = None
    after: int | None = None
    output_mode: str = 'content'
    head_limit: int | None = None
    offset: int = 0


class ReadLinesInput(_ExactSchemaModel):
    _schema_name = 'read_lines'

    path: str
    start: int
    end: int


class CleanupInput(BaseModel):
    pass


async def _dispatch_file_tool(name: str, arguments: dict[str, Any], context: ToolContext) -> ToolResponse:
    identity = context.tool_state.get('execution_identity')
    state = get_session(context)
    if identity is None:
        return _text(await dispatch_tool(name=name, arguments=arguments, state=state))
    from .isolated_reader import dispatch_reader

    with execution_identity(identity):
        return _text(await dispatch_reader(state, name, arguments))


def _description(name: str) -> str:
    return str(TOOL_SCHEMA_REGISTRY[name]['function']['description'])


@tool_registry.register(
    registry_id('curl'),
    meta={'name': 'curl', 'description': _description('curl'), 'tool_type': ToolType.RUNTIME},
)
async def curl_tool(params: CurlInput, context: ToolContext) -> ToolResponse:
    with execution_identity(context.tool_state.get('execution_identity')):
        return _text(
            await dispatch_tool(
                name='curl',
                arguments={'url': params.url, 'output': params.output},
                state=get_session(context),
            )
        )


@tool_registry.register(
    registry_id('bash'),
    meta={'name': 'bash', 'description': _description('bash'), 'tool_type': ToolType.RUNTIME},
)
async def bash_tool(params: BashInput, context: ToolContext) -> ToolResponse:
    with execution_identity(context.tool_state.get('execution_identity')):
        return _text(
            await dispatch_tool(name='bash', arguments={'command': params.command}, state=get_session(context))
        )


@tool_registry.register(
    registry_id('python'),
    meta={'name': 'python', 'description': _description('python'), 'tool_type': ToolType.RUNTIME},
)
async def python_tool(params: PythonInput, context: ToolContext) -> ToolResponse:
    with execution_identity(context.tool_state.get('execution_identity')):
        return _text(await dispatch_tool(name='python', arguments={'code': params.code}, state=get_session(context)))


@tool_registry.register(
    registry_id('file_search_msearch'),
    meta={
        'name': 'file_search_msearch',
        'description': _description('file_search_msearch'),
        'tool_type': ToolType.RUNTIME,
    },
)
async def msearch_tool(params: MsearchInput, context: ToolContext) -> ToolResponse:
    return await _dispatch_file_tool(
        'file_search_msearch', {'queries': params.queries, 'top_k': params.top_k, 'files': params.files}, context
    )


@tool_registry.register(
    registry_id('file_search_mclick'),
    meta={
        'name': 'file_search_mclick',
        'description': _description('file_search_mclick'),
        'tool_type': ToolType.RUNTIME,
    },
)
async def mclick_tool(params: MclickInput, context: ToolContext) -> ToolResponse:
    return await _dispatch_file_tool('file_search_mclick', {'pointers': params.pointers}, context)


@tool_registry.register(
    registry_id('grep'),
    meta={'name': 'grep', 'description': _description('grep'), 'tool_type': ToolType.RUNTIME},
)
async def grep_tool(params: GrepInput, context: ToolContext) -> ToolResponse:
    return await _dispatch_file_tool('grep', params.model_dump(), context)


@tool_registry.register(
    registry_id('read_lines'),
    meta={'name': 'read_lines', 'description': _description('read_lines'), 'tool_type': ToolType.RUNTIME},
)
async def read_lines_tool(params: ReadLinesInput, context: ToolContext) -> ToolResponse:
    return await _dispatch_file_tool('read_lines', params.model_dump(), context)


@tool_registry.register(
    registry_id(CLEANUP_TOOL_NAME),
    meta={
        'name': CLEANUP_TOOL_NAME,
        'description': 'Internal: close and remove one context_agent_general runtime session.',
        'tool_type': ToolType.RUNTIME,
    },
)
async def cleanup_tool(params: CleanupInput, context: ToolContext) -> ToolResponse:
    del params
    return _text('session closed' if cleanup_session(context) else 'no session')
