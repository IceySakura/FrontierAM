"""Keep the existing FS codec import path compatible."""

from ...payload_codec import decode_blob, encode_blob


__all__ = ['decode_blob', 'encode_blob']
