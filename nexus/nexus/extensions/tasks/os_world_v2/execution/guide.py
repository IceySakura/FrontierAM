"""Harness-guide rendering shared by the V2 Claude Code and Codex profiles."""

CLIENT_PASSWORD_PLACEHOLDER = '{{client_password}}'


def render_guide(text: str, client_password: str) -> str:
    """Fill the guide's sudo password from task_config.client_password."""
    return text.replace(CLIENT_PASSWORD_PLACEHOLDER, client_password)
