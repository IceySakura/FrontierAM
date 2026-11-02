"""OpenAI function-call schemas for HyGeneralToolSystem.

Seven path-native tools. Everything addresses files by PATH, so there is no
``file`` argument naming a pre-loaded document — the model passes the path it
was given or the one it just wrote.

There is deliberately no ``list_documents`` or ``doc_outline`` here, unlike the
content-bearing design's document contract: with no pre-loaded document set to
enumerate, both are a couple of lines of ``bash`` or ``python``.

The order in :data:`GENERAL_TOOL_ORDER` mirrors the intended workflow
(acquire, inspect, convert, locate, read) because it is the order the model
sees the definitions in.

A single English set: the system prompt may be bilingual, but the tool schemas
the LLM sees are always English.
"""

from __future__ import annotations

from typing import Any


CURL_SCHEMA: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'curl',
        'description': (
            'Download an http(s) URL into ./workspace/ and report METADATA '
            'only: HTTP status, Content-Type, byte size, the saved path, and '
            'either the detected binary format or the line count if it is '
            'text. The response body is NEVER returned, so this is safe to '
            'call on a large or binary file. Use this before deciding how to '
            'read something. Local paths need no download — pass them straight '
            'to read_lines / grep / file_search_msearch.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'url': {
                    'type': 'string',
                    'description': 'The http(s) URL to download.',
                },
                'output': {
                    'type': 'string',
                    'description': (
                        'Optional filename to save as, inside ./workspace/. '
                        'Defaults to the last path segment of the URL. An '
                        'existing file with the same name is overwritten.'
                    ),
                },
            },
            'required': ['url'],
        },
    },
}

BASH_SCHEMA: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'bash',
        'description': (
            'Run a shell command in the working directory. General purpose, '
            'not limited to arithmetic: inspect files (ls -la, wc -l, wc -c, '
            'file, head -c), transform or convert data, move files around, '
            'batch-process, or anything the other tools do not cover. '
            'Downloads live at ./workspace/<name>. Output is truncated at '
            '200k chars, so do NOT cat whole documents — use read_lines for '
            'line ranges. Note shell tools like `grep -n` and `sed -n` number '
            "lines from 1, while this system's read_lines/grep number from 0."
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'command': {
                    'type': 'string',
                    'description': 'The shell command to execute.',
                },
            },
            'required': ['command'],
        },
    },
}

PYTHON_SCHEMA: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'python',
        'description': (
            'Execute Python in a persistent interpreter. The namespace '
            'survives across calls (Jupyter-like), and the working directory '
            'is the same one bash sees, so ./workspace/<name> resolves. This '
            'is how you convert a binary document to text: pypdf, pdfplumber, '
            'pdfminer and PyPDF2 are installed (there is NO pdftotext binary). '
            'Write the extracted text to a .txt file under ./workspace/ and '
            'then use the line-based tools on it — do not print a whole '
            'document to stdout.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'code': {
                    'type': 'string',
                    'description': 'Python code to execute.',
                },
            },
            'required': ['code'],
        },
    },
}

FILE_SEARCH_MSEARCH_SCHEMA: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'file_search_msearch',
        'description': (
            'BM25-search text files on disk and get back the most relevant '
            'CANDIDATE chunks — the primary way to locate information in a '
            'long document. Results are candidates, NOT the final answer, and '
            'each snippet may be partial. Each carries its Source path, a '
            'pointer ("Mclick Target", e.g. "0:c12" meaning chunk 12 of '
            'file index 0) and an [L#-L#] range local to that file. Issue '
            'several complementary queries in ONE call: at least one precision '
            'query plus one short recall query (1-2 keywords likely to appear '
            'in the answer text); for a non-English question, query in both the '
            'original language and English. Wrap an exact phrase in double '
            'quotes to boost literal matches. Only works on text files — '
            'extract a PDF to .txt first.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'queries': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        'Complementary search queries (aim for up to ~5). '
                        'Include a precise query AND a short keyword recall '
                        'query; matched independently and merged by relevance.'
                    ),
                },
                'top_k': {
                    'type': 'integer',
                    'description': 'Max candidate chunks to return per query. Default 5.',
                },
                'files': {
                    'anyOf': [
                        {'type': 'string'},
                        {'type': 'array', 'items': {'type': 'string'}},
                    ],
                    'description': (
                        'Optional path, or list of paths, to restrict the '
                        'search to. Omit to search every text file you have '
                        "(the task's own files plus anything under ./workspace/)."
                    ),
                },
            },
            'required': ['queries'],
        },
    },
}

FILE_SEARCH_MCLICK_SCHEMA: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'file_search_mclick',
        'description': (
            'Open and expand pointers returned by a prior file_search_msearch '
            'to read the full chunk plus its surrounding context (with [L#] '
            'markers) — msearch snippets are partial, mclick gives the fuller '
            'context. Run file_search_msearch FIRST, then pass its '
            '"Mclick Target" pointers here (format fidx:cNN). Up to 3 pointers.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pointers': {
                    'type': 'array',
                    'items': {'type': 'string'},
                    'description': (
                        '"Mclick Target" pointers from a prior '
                        'file_search_msearch result, e.g. ["0:c12", "1:c4"] '
                        '(max 3).'
                    ),
                },
            },
            'required': ['pointers'],
        },
    },
}

GREP_SCHEMA: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'grep',
        'description': (
            'Search a text file for lines matching a substring (or regex). '
            'Returns matching line numbers (0-based), a cluster summary '
            'grouping nearby matches, and a context block around every match. '
            "Matching is ALWAYS case-insensitive, so write 'supplier', not "
            "'[Ss]upplier'. Omit 'path' to search every text file you have — "
            "the task's own local files plus anything you downloaded or wrote. "
            "Set 'after' large to capture the text FOLLOWING each match in one "
            'call — e.g. grep(pattern=MARKER, after=25) locates an EXACT '
            'marker string / UUID and returns the following lines, which '
            'semantic file_search cannot do because random strings carry no '
            'meaning. Use read_lines for full content of a specific range.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'pattern': {
                    'type': 'string',
                    'description': (
                        'The search pattern. Substring match by default; set '
                        'use_regex=true for regular-expression matching. Use '
                        'the exact literal string for marker/UUID lookups.'
                    ),
                },
                'path': {
                    'anyOf': [
                        {'type': 'string'},
                        {'type': 'array', 'items': {'type': 'string'}},
                    ],
                    'description': (
                        'Optional file path, or list of paths, to search. '
                        "Omit to search every text file you have (the task's "
                        'own files plus anything under ./workspace/).'
                    ),
                },
                'use_regex': {
                    'type': 'boolean',
                    'description': (
                        'Treat pattern as a regex. Default false (literal). If '
                        'a literal search finds nothing and the pattern looks '
                        'like a regex, it is retried as one automatically and '
                        'the result says so — but set this to true when you '
                        'mean a regex, so the first attempt is the right one.'
                    ),
                    'default': False,
                },
                'context_lines': {
                    'type': 'integer',
                    'description': (
                        'Symmetric context lines above/below each match, used '
                        "as the default for both 'before' and 'after'. Default 1."
                    ),
                    'default': 1,
                },
                'before': {
                    'type': 'integer',
                    'description': ('Lines to show BEFORE each match (overrides context_lines for the leading side).'),
                },
                'after': {
                    'type': 'integer',
                    'description': (
                        'Lines to show AFTER each match (overrides '
                        'context_lines for the trailing side). Set large '
                        '(e.g. 25) to read what follows a marker string.'
                    ),
                },
                'output_mode': {
                    'type': 'string',
                    'enum': ['content', 'files_with_matches', 'count'],
                    'description': (
                        "'content' (default) shows matching lines with context. "
                        "'files_with_matches' lists which files matched and at "
                        'which line numbers WITHOUT printing content — use it '
                        'when searching many files at once, to learn WHERE to '
                        'look before spending context on what is there. '
                        "'count' gives per-file totals only."
                    ),
                    'default': 'content',
                },
                'head_limit': {
                    'type': 'integer',
                    'description': (
                        "Max results on this page: match blocks in 'content' "
                        'mode (default 10), file rows otherwise (default 100).'
                    ),
                },
                'offset': {
                    'type': 'integer',
                    'description': (
                        'Skip this many results before the page starts. A '
                        'truncated result reports the exact offset to pass '
                        'next, so a wide search can be walked through instead '
                        'of leaving most of its matches unreachable.'
                    ),
                    'default': 0,
                },
            },
            'required': ['pattern'],
        },
    },
}

READ_LINES_SCHEMA: dict[str, Any] = {
    'type': 'function',
    'function': {
        'name': 'read_lines',
        'description': (
            'Read an exact range of lines [start, end) from a text file by '
            'path (0-based, end exclusive; end=-1 reads to EOF). L0 is the '
            "file's first line, so a region located with shell tools can be "
            'read back here (shell numbering starts at 1, this starts at 0). '
            'Use it to verify the precise content of something found with '
            'file_search_msearch or grep. Output capped at ~80k chars / 500 '
            'lines — prefer this over `cat` for anything sizeable.'
        ),
        'parameters': {
            'type': 'object',
            'properties': {
                'path': {
                    'type': 'string',
                    'description': (
                        'Path of the file to read. Relative paths resolve '
                        'against the working directory, so a download is at '
                        './workspace/<name>.'
                    ),
                },
                'start': {
                    'type': 'integer',
                    'description': 'Start line number (0-based, inclusive).',
                },
                'end': {
                    'type': 'integer',
                    'description': ('End line number (exclusive). Use -1 to read to end of file.'),
                },
            },
            'required': ['path', 'start', 'end'],
        },
    },
}

TOOL_SCHEMA_REGISTRY: dict[str, dict[str, Any]] = {
    'curl': CURL_SCHEMA,
    'bash': BASH_SCHEMA,
    'python': PYTHON_SCHEMA,
    'file_search_msearch': FILE_SEARCH_MSEARCH_SCHEMA,
    'file_search_mclick': FILE_SEARCH_MCLICK_SCHEMA,
    'grep': GREP_SCHEMA,
    'read_lines': READ_LINES_SCHEMA,
}

GENERAL_TOOL_ORDER: tuple[str, ...] = (
    'curl',
    'bash',
    'python',
    'file_search_msearch',
    'file_search_mclick',
    'grep',
    'read_lines',
)


def build_tool_schemas(enabled_tools: list[str] | None = None) -> list[dict[str, Any]]:
    """Return OpenAI schemas for the enabled tools in canonical order."""
    if enabled_tools is None:
        enabled_tools = list(GENERAL_TOOL_ORDER)
    schemas: list[dict[str, Any]] = []
    for name in enabled_tools:
        schema = TOOL_SCHEMA_REGISTRY.get(name)
        if schema is not None:
            schemas.append(schema)
    return schemas
