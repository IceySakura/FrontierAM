"""Error codes for local_one_agent tools."""

from enum import Enum


class LocalToolErrorCode(str, Enum):
    """Error codes for local tool operations. Used in ToolResponse.meta['error']['code']."""

    PERMISSION_DENIED = 'PERMISSION_DENIED'
    USER_REJECTED = 'USER_REJECTED'

    INVALID_ARGUMENT = 'INVALID_ARGUMENT'
    NOT_FOUND = 'NOT_FOUND'
    AMBIGUOUS_MATCH = 'AMBIGUOUS_MATCH'

    UNSUPPORTED_FILE_TYPE = 'UNSUPPORTED_FILE_TYPE'

    # edit：old_string 未找到 / 与 new_string 相同（对齐线上 for_model.go）。
    STRING_NOT_FOUND = 'STRING_NOT_FOUND'
    NO_CHANGES = 'NO_CHANGES'

    TIMEOUT = 'TIMEOUT'
    OUTPUT_TOO_LARGE = 'OUTPUT_TOO_LARGE'

    # load_skill：skill 未找到（对齐线上 SKILL_NOT_FOUND，而非通用 NOT_FOUND）。
    SKILL_NOT_FOUND = 'SKILL_NOT_FOUND'
