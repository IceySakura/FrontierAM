"""Utility functions for logging."""

from __future__ import annotations

import datetime


def overwrite_log(message, name='default', level='INFO'):
    """
    Log in the same line. Call finish_overwrite() to finish.
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'\r{" " * 100}\r{timestamp} | {level} | {name} | {message}', end='', flush=True)


def finish_overwrite():
    """Finish overwrite log by printing a newline."""
    print()
