"""WorkBuddy headless CLI (`wb`) agent.

Unlike OpenCC (a standalone bun binary), WorkBuddy ships as a platform npm
package that still needs Node >= 22 plus ``better-sqlite3``.  The runtime
installs it from a private npm registry or a pinned Generic artifact; host
tarball upload remains a development fallback. LLM calls route through the
local OpenAI proxy.
"""
