"""Sandbox-side proxy that labels each LLM request with its session.

Every Claude Code worker path -- PTY, stream, and print -- starts its own
``AnthropicProxy`` on its own free port, so a proxy instance corresponds to
exactly one session. This carries that identity to the host as a private field on
``openai_params``, which the host handler strips before the call reaches the
provider, following the convention the Anthropic converter uses for
``_client_has_thinking``. The envelope, the bridge, and the protocol are
unchanged.

The label is applied by wrapping ``self.bridge``: the proxy's hooks run either
before request validation or after the response, neither of which can reach the
assembled OpenAI params.

**This module runs in the sandbox.** ``AnthropicProxy`` needs FastAPI, which the
host does not have, so that import has to stay inside
:func:`build_labelled_proxy_class`; the rest of the module is importable host-side.
"""

import logging

from typing import Any

from nexus.extensions.tasks.hitl.trace.labels import LABEL_FIELD


logger = logging.getLogger(__name__)


class _LabellingBridge:
    """Delegates to a real bridge, stamping the session label on each request.

    Only ``request`` is intercepted; ``__getattr__`` passes everything else
    through, so the bridge can grow methods without changes here.
    """

    def __init__(self, inner: Any, label: str) -> None:
        self._inner = inner
        self._label = label

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    @property
    def label(self) -> str:
        return self._label

    async def request(self, openai_params: dict[str, Any], timeout: float = 300) -> dict[str, Any]:
        """Forward the request with the label attached.

        Builds a new dict rather than mutating the caller's, which the proxy may
        still hold a reference to.
        """
        return await self._inner.request({**openai_params, LABEL_FIELD: self._label}, timeout=timeout)


def attach_label(proxy: Any, session_label: str) -> Any:
    """Wrap ``proxy.bridge`` so its requests name their session.

    Call after ``start()``, which is what assigns ``proxy.bridge``. Idempotent:
    an already-wrapped bridge is left alone, since wrapping twice would stamp
    twice.
    """
    if not session_label:
        raise ValueError('a labelled proxy requires a non-empty session_label')
    if isinstance(proxy.bridge, _LabellingBridge):
        return proxy
    if proxy.bridge is None:
        raise RuntimeError('proxy has no bridge yet; call start() before attaching a label')
    proxy.bridge = _LabellingBridge(proxy.bridge, session_label)
    logger.info('proxy on %s labelling requests as %s', proxy.port, session_label)
    return proxy


def build_labelled_proxy_class() -> type:
    """Build the labelled proxy class, importing the FastAPI-dependent base lazily.

    Sandbox-only: raises ``ModuleNotFoundError`` on the host, where FastAPI is
    not installed.
    """
    from nexus.runtime.server.services.anthropic_proxy.server import AnthropicProxy

    class LabelledAnthropicProxy(AnthropicProxy):  # type: ignore[misc, valid-type]
        """Anthropic proxy whose requests name their session.

        ``session_label`` should come from
        :func:`~nexus.extensions.tasks.hitl.trace.labels.worker_label`, so it
        parses the way the trace filters expect.
        """

        def __init__(self, *args: Any, session_label: str, **kwargs: Any) -> None:
            if not session_label:
                raise ValueError('LabelledAnthropicProxy requires a non-empty session_label')
            super().__init__(*args, **kwargs)
            self._session_label = session_label

        @property
        def session_label(self) -> str:
            return self._session_label

        async def start(self) -> None:
            await super().start()
            attach_label(self, self._session_label)

    return LabelledAnthropicProxy


__all__ = ['LABEL_FIELD', '_LabellingBridge', 'attach_label', 'build_labelled_proxy_class']
