"""Frozen bilinear-edge reference with the ordinary submission interface."""
from impl.full_frame import reconstruct as _reconstruct


def reconstruct(raw):
    return _reconstruct(raw, "bilinear-edge")
