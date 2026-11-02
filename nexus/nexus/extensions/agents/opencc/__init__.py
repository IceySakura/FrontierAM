"""OpenCC (OpenClaude) agent integration for Nexus.

Runs a modified Claude Code CLI (OpenCC) inside a runtime sandbox.
The CLI binary bundle is uploaded from the host, and LLM calls flow
through the Anthropic-to-OpenAI proxy via Socket.IO.
"""
