"""F5_json_format_error/_shared — F5-1 / F5-2 / F5-3 共用的 JSON / harness 工具。

3 个 detector 共享以下基础设施：

- ``_json_utils``    : ``_try_parse`` / ``_err_kind`` / ``_looks_like_json`` /
                       ``_NON_JSON_CONTENT_PREFIX_RE`` / 长度阈值常量
- ``_harness_utils`` : ``_find_prev_assistant_turn`` / ``_is_terminal_dump`` /
                       ``SHELL_PROMPT_RE`` / ``MAX_MSG_BYTES`` /
                       ``JSON_ERROR_PATTERN``（F5-3 主 regex）

不要把 detector-specific 的常量放在这里——这里只放 "改一处、多个 detector
都受益" 的通用工具。
"""
