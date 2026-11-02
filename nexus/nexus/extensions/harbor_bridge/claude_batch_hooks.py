"""Compatibility import for the renamed unattended interaction hooks."""

from nexus.extensions.harbor_bridge.claude_unattended_hooks import (
    install_unattended_interaction_hooks as install_claude_batch_hooks,
)


__all__ = ['install_claude_batch_hooks']
