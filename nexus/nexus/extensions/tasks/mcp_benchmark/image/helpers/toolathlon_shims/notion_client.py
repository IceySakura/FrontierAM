from __future__ import annotations

import os

from typing import Any


# In API mode, delegate to the real notion-client SDK so that preprocess scripts
# (e.g. notion_page_duplicator) can call the live Notion API.  The shim store
# only contains eval-page data and lacks source-page entries, which causes
# "Unknown block or page" errors during the duplicate-from-source step.
_USE_REAL_SDK = os.getenv('NEXUS_TOOLATHLON_SERVER_MODE') != 'mock'

if _USE_REAL_SDK:
    # The real ``notion-client`` package is installed in the container venv.
    # Because this shim directory sits earlier on PYTHONPATH we must bypass it
    # and import the real package from its installed location.
    import importlib
    import sys

    _self_dir = os.path.dirname(os.path.abspath(__file__))
    _orig_path = list(sys.path)
    try:
        # Temporarily remove the shim directory so importlib finds the real package.
        sys.path = [p for p in sys.path if os.path.abspath(p) != _self_dir]
        # Drop any cached reference to *this* shim module so the reload picks up the real one.
        _cached = sys.modules.pop('notion_client', None)
        try:
            _real = importlib.import_module('notion_client')
        except ImportError:
            # Real SDK not installed – fall through to the shim implementation below.
            _USE_REAL_SDK = False
            if _cached is not None:
                sys.modules['notion_client'] = _cached
        else:
            # Re-export everything from the real SDK.
            Client = _real.Client  # type: ignore[assignment]
            APIResponseError = _real.APIResponseError  # type: ignore[assignment]
            # Restore module cache so subsequent ``import notion_client`` gets the real one.
            sys.modules['notion_client'] = _real
    finally:
        sys.path = _orig_path

if not _USE_REAL_SDK:
    try:
        from toolathlon_notion_store import (
            NotionStoreError,
            append_block_children,
            create_comment,
            create_database,
            create_page,
            delete_block,
            get_block,
            get_block_children,
            get_comment,
            get_database,
            get_page,
            list_comments,
            list_users,
            query_database,
            search_objects,
            update_block,
            update_database,
            update_page,
        )
    except ImportError:
        from ..toolathlon_notion_store import (
            NotionStoreError,
            append_block_children,
            create_comment,
            create_database,
            create_page,
            delete_block,
            get_block,
            get_block_children,
            get_comment,
            get_database,
            get_page,
            list_comments,
            list_users,
            query_database,
            search_objects,
            update_block,
            update_database,
            update_page,
        )

    class APIResponseError(RuntimeError):  # type: ignore[no-redef]
        def __init__(self, message: str, *, status: int | None = None) -> None:
            super().__init__(message)
            self.status = status

    class _ChildrenEndpoint:
        def list(self, *, block_id: str, **kwargs: Any) -> dict[str, Any]:
            return get_block_children(block_id, kwargs)

        def append(self, *, block_id: str, **kwargs: Any) -> dict[str, Any]:
            return append_block_children(block_id, kwargs)

    class _BlocksEndpoint:
        def __init__(self) -> None:
            self.children = _ChildrenEndpoint()

        def retrieve(self, *, block_id: str, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return get_block(block_id)

        def update(self, *, block_id: str, **kwargs: Any) -> dict[str, Any]:
            return update_block(block_id, kwargs)

        def delete(self, *, block_id: str, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return delete_block(block_id)

    class _PagesEndpoint:
        def retrieve(self, *, page_id: str, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return get_page(page_id)

        def update(self, *, page_id: str, **kwargs: Any) -> dict[str, Any]:
            return update_page(page_id, kwargs)

        def create(self, **kwargs: Any) -> dict[str, Any]:
            return create_page(kwargs)

    class _DatabasesEndpoint:
        def retrieve(self, *, database_id: str, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return get_database(database_id)

        def query(self, *, database_id: str, **kwargs: Any) -> dict[str, Any]:
            return query_database(database_id, kwargs)

        def create(self, **kwargs: Any) -> dict[str, Any]:
            return create_database(kwargs)

        def update(self, *, database_id: str, **kwargs: Any) -> dict[str, Any]:
            return update_database(database_id, kwargs)

    class _CommentsEndpoint:
        def create(self, **kwargs: Any) -> dict[str, Any]:
            return create_comment(kwargs)

        def list(self, *, block_id: str, **kwargs: Any) -> dict[str, Any]:
            return list_comments(block_id, kwargs)

        def retrieve(self, *, comment_id: str, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return get_comment(comment_id)

    class _UsersEndpoint:
        def list(self, **kwargs: Any) -> dict[str, Any]:
            return list_users(kwargs)

        def retrieve(self, *, user_id: str, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            users = list_users(None)['results']
            for user in users:
                if user.get('id') == user_id:
                    return user
            raise APIResponseError(f'Unknown user: {user_id}', status=404)

        def me(self, **kwargs: Any) -> dict[str, Any]:
            del kwargs
            return list_users(None)['results'][0]

    class Client:  # type: ignore[no-redef]
        def __init__(self, auth: str | None = None, **kwargs: Any) -> None:
            del kwargs
            self.auth = auth
            self.blocks = _BlocksEndpoint()
            self.pages = _PagesEndpoint()
            self.databases = _DatabasesEndpoint()
            self.comments = _CommentsEndpoint()
            self.users = _UsersEndpoint()

        def search(self, **kwargs: Any) -> dict[str, Any]:
            return search_objects(kwargs)

    __all__ = ['APIResponseError', 'Client', 'NotionStoreError']
