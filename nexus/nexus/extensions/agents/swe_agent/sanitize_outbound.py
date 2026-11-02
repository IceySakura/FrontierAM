"""Load the bundled outbound request sanitization functions."""

from pathlib import Path


_source = Path(__file__).with_name('_sanitize_outbound_source.txt')
exec(compile(_source.read_bytes(), str(_source), 'exec'))
