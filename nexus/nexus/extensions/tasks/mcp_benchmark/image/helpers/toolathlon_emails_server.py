#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys

from pathlib import Path


def _ensure_toolathlon_root() -> None:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / 'workspace' / 'mcp' / 'datasets' / 'toolathlon'
        if candidate.exists():
            sys.path.insert(0, str(candidate))
            return


def _patch_imaplib_compat() -> None:
    imaplib = importlib.import_module('imaplib')
    stdlib_imaplib = getattr(imaplib, '_STDLIB_IMAPLIB', None)
    if stdlib_imaplib is not None:
        for name in dir(stdlib_imaplib):
            if name.startswith('_') or hasattr(imaplib, name):
                continue
            setattr(imaplib, name, getattr(stdlib_imaplib, name))

    imap4 = getattr(imaplib, 'IMAP4', None)
    if imap4 is None:
        return

    if not hasattr(imap4, 'abort'):

        class Abort(imap4.error):
            pass

        imap4.abort = Abort
    if not hasattr(imap4, 'readonly'):

        class ReadOnly(imap4.error):
            pass

        imap4.readonly = ReadOnly

    imap4_ssl = getattr(imaplib, 'IMAP4_SSL', None)
    if imap4_ssl is not None:
        if not hasattr(imap4_ssl, 'abort'):
            imap4_ssl.abort = imap4.abort
        if not hasattr(imap4_ssl, 'readonly'):
            imap4_ssl.readonly = imap4.readonly


def _normalize_argv() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('legacy_attachment_download_path', nargs='?')
    parser.add_argument('--attachment_upload_path')
    parser.add_argument('--attachment_download_path')
    parser.add_argument('--email_export_path')
    parser.add_argument('--config_file')
    parser.add_argument('--debug', action='store_true')
    args, extras = parser.parse_known_args(sys.argv[1:])

    attachment_download_path = args.attachment_download_path or args.legacy_attachment_download_path
    normalized: list[str] = []
    if attachment_download_path:
        normalized.extend(['--attachment_download_path', attachment_download_path])
    if args.attachment_upload_path:
        normalized.extend(['--attachment_upload_path', args.attachment_upload_path])
    if args.email_export_path:
        normalized.extend(['--email_export_path', args.email_export_path])
    if args.config_file:
        normalized.extend(['--config_file', args.config_file])
    if args.debug:
        normalized.append('--debug')
    normalized.extend(extras)
    return args, normalized


def _patch_email_header_bug() -> None:
    """Fix emails-mcp bug: Header objects not converted to str before SMTP send.

    The emails-mcp package sets message headers like ``msg['From'] = Header(addr)``
    but smtplib expects strings, causing:
        TypeError: expected string or bytes-like object, got 'Header'

    We replace ``email.header.Header`` with a subclass of ``str`` that:
    1. IS a str (passes isinstance checks in smtplib/re)
    2. Preserves Header.encode() for email serialization (Generator calls
       header.encode(linesep=...) internally — plain str.encode() does not
       accept that keyword, leading to "'linesep' is an invalid keyword
       argument for encode()" or "'dict' object has no attribute 'append'"
       depending on the Python/email version).
    """
    import email.header

    _orig_header = email.header.Header

    class _StrHeader(str):
        """Drop-in replacement for email.header.Header that IS a str."""

        def __new__(cls, s='', charset=None, maxlinelen=None, header_name=None, continuation_ws=' ', errors='strict'):
            h = _orig_header(s, charset, maxlinelen, header_name, continuation_ws, errors)
            inst = str.__new__(cls, str(h))
            inst._header = h
            return inst

        def encode(self, *args, **kwargs):
            # email.generator calls Header.encode(splitchars=..., linesep=..., maxlinelen=...)
            # str.encode only accepts (encoding, errors) positional/keyword args.
            # Detect Header-style calls and delegate to the real Header.
            if (
                kwargs.get('linesep') is not None
                or kwargs.get('splitchars') is not None
                or kwargs.get('maxlinelen') is not None
            ):
                return self._header.encode(*args, **kwargs)
            return str.encode(self, *args, **kwargs)

    email.header.Header = _StrHeader  # type: ignore[misc]


def _skip_startup_check() -> None:
    """Disable emails-mcp's synchronous check_connection() at startup.

    check_connection() calls smtplib.SMTP(host, port) with no timeout.
    When the Poste.io SMTP port is slow to respond over gongfeng's
    external URL, the blocking TCP connect prevents the server from
    entering MCP stdio mode — the MCP client then times out (120s)
    waiting for the initialize response.

    The check is purely informational (logs warnings); each tool call
    independently establishes its own IMAP/SMTP connections.
    """
    from emails_mcp.services.email_service import EmailService

    EmailService.check_connection = lambda self: (True, True)


def main() -> None:
    args, normalized_argv = _normalize_argv()

    _ensure_toolathlon_root()

    # ensure_local_email_store_env MUST run before _patch_imaplib_compat().
    # Toolathlon ships a custom imaplib.py that decides at *class-definition
    # time* whether IMAP4 should be the local JSON mock (LocalIMAP4) or the
    # real stdlib IMAP4 — the decision is based on the env var
    # NEXUS_TOOLATHLON_EMAIL_STORE_PATH being set.  If _patch_imaplib_compat()
    # triggers ``import imaplib`` first, the custom module loads before the env
    # var exists and locks in the real stdlib IMAP4, causing import_emails to
    # attempt a real IMAP connection to gongfeng's HTTP proxy (which hangs).
    if args.config_file:
        try:
            from utils.app_specific.poste.local_store import ensure_local_email_store_env

            ensure_local_email_store_env(args.config_file)
        except ImportError:
            pass  # local_store not available; email service runs independently

    _patch_imaplib_compat()
    _patch_email_header_bug()
    _skip_startup_check()

    from emails_mcp.server import main as emails_mcp_main

    sys.argv = [sys.argv[0], *normalized_argv]
    emails_mcp_main()


if __name__ == '__main__':
    main()
