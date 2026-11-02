"""Allow running as `python -m nexus.extensions.tasks.hitl.mcp`."""

from nexus.extensions.tasks.hitl.mcp.search_server import main


if __name__ == '__main__':
    import asyncio

    asyncio.run(main())
