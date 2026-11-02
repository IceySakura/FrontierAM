"""Shared protocol and runtime receipt patterns."""

import re


TOOL_TOKEN = re.compile(
    r'<\s*/?\s*(?:(?:tool_calls?|function_call|invoke|tool_sep|arg_key|arg_value|tool_response)(?::[A-Za-z0-9_-]+)?|[A-Za-z_][A-Za-z0-9_]*:[0-9a-f]{8})\s*>',
    re.I,
)
TOOL_CONTENT_TOKEN = re.compile(
    r'<\s*/?\s*(?:(?:tool_calls?|function_call|invoke|tool_sep|arg_key|arg_value|tool_response)(?::[A-Za-z0-9_-]+)?|(?!think:)[A-Za-z_][A-Za-z0-9_]*:[0-9a-f]{8})\s*>',
    re.I,
)
THINK_TAG = re.compile(r'<\s*/?\s*think(?:[:_][0-9A-Za-z_-]+)?\s*>', re.I)
HASHED_THINK_TAG = re.compile(r'<\s*/?\s*think[:_][0-9A-Za-z_-]+\s*>', re.I)
UNKNOWN_TOOL = re.compile(
    r"(?:No such tool available\s*:\s*(?P<name1>[A-Za-z_][\w:.-]*)|unknown tool\s*:?\s*['\"]?(?P<name2>[A-Za-z_][\w:.-]*)|tool\s+['\"]?(?P<name3>[A-Za-z_][\w:.-]*)['\"]?\s+(?:does not exist|is not available|is not enabled))",
    re.I,
)
HARD_REJECTION = re.compile(
    r'^\s*(?:error\s*:\s*)?(?:previous response had parsing errors|no valid json object found|missing required fields|invalid json)\b',
    re.I,
)
EDIT_FAILURE = re.compile(
    r'(?:string|text|old_string|old_str|pattern).{0,100}(?:not found|no match)|'
    r'(?:not found|no match).{0,100}(?:string|text|old_string|old_str|pattern)|'
    r'file has not been read yet|edit.{0,100}(?:failed|failure|error)',
    re.I | re.S,
)
SANDBOX_FAILURE = re.compile(
    r'sandbox was killed due to out of memory|OOMKilled|bash command failed to start|Command failed to start', re.I
)
ARGS_ERROR_RECEIPT = re.compile(
    r'(?:inputvalidationerror|validationerror|schema validation failed)|(?:arguments?|parameters?).{0,120}(?:missing required (?:field|property)|extra inputs? (?:are )?not permitted|not (?:one of|a valid enum)|expected type|type error)',
    re.I | re.S,
)
ARGS_PARSE_RECEIPT = re.compile(
    r'(?:failed|unable|could not) to parse.{0,80}(?:tool|function)?\s*arguments?|invalid json.{0,80}(?:tool|function)?\s*arguments?|function\.arguments.{0,80}(?:json|parse)',
    re.I | re.S,
)
