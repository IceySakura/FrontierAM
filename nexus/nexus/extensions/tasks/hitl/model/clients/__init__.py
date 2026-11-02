"""HITL-private openai client variants.

Temporary workarounds that wrap upstream clients to normalize behavior for HITL
until the upstream is fixed. Each module self-registers in the openai client
registry on import.
"""
