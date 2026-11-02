import os

import requests

from pydantic import BaseModel, Field

from nexus.core import ToolContext, ToolResponse, ToolType, tool_registry


class SearchR1Input(BaseModel):
    query: str = Field(description='The search query string.')


@tool_registry.register(
    'search_r1/search',
    meta={
        'name': 'search_r1',
        'description': 'Retrieve relevant documents for the given query.',
        'tool_type': ToolType.LOCAL,
    },
)
async def search_r1_tool(params: SearchR1Input, context: ToolContext) -> ToolResponse:
    """
    Retrieve relevant documents for the given query.

    Args:
        params: SearchR1Input containing the search query string.
        context: ToolContext for the current tool invocation.

    Environment Variables:
        SEARCH_R1_RETRIEVE_URL: The URL of the search API.
        SEARCH_R1_TOPK: The number of top results to return.
        SEARCH_R1_TIMEOUT_SECONDS: Request timeout in seconds.

    Returns:
        ToolResponse with formatted search results containing document titles and content.
    """
    retrieve_url = os.getenv('SEARCH_R1_RETRIEVE_URL', 'http://127.0.0.1:8000/retrieve')
    topk = int(os.getenv('SEARCH_R1_TOPK', '1'))
    timeout = float(os.getenv('SEARCH_R1_TIMEOUT_SECONDS', '10'))
    payload = {'query': params.query, 'topk': topk, 'return_scores': True}
    result = requests.post(retrieve_url, json=payload, timeout=timeout).json()

    format_reference = ''
    for idx, doc_item in enumerate(result['result']):
        content = doc_item['document']['contents']
        title = content.split('\n')[0]
        text = '\n'.join(content.split('\n')[1:])
        format_reference += f'Doc {idx + 1}(Title: {title}) {text}\n'

    return ToolResponse(
        output=[{'type': 'text', 'text': format_reference}],
        meta={'result_count': len(result['result'])},
    )
