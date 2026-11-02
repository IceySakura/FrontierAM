"""
Profiles module for SWE-smith.

This module contains repository profiles for different programming languages
and provides a global registry for accessing all profiles.
"""

# Auto-import all profile modules to populate the registry
from . import c, cpp, csharp, golang, java, javascript, php, python, rust
from .base import RepoProfile, registry


__all__ = ['RepoProfile', 'registry', 'c', 'cpp', 'csharp', 'golang', 'java', 'javascript', 'php', 'python', 'rust']
