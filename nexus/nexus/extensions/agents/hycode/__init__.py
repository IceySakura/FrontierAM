"""HyCode agent integration for Nexus.

Runs HyCode (@tencent/hycode) inside a runtime sandbox via npm install.
The package bundles glibc and musl binaries; the installer auto-detects
the sandbox libc and picks the correct one. LLM calls flow through the
Anthropic-to-OpenAI proxy via Socket.IO.
"""
