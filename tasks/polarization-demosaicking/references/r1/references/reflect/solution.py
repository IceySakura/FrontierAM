"""Mirror-extended LEIC reference using the ordinary submission interface."""
from impl.full_frame import reconstruct as _reconstruct


def reconstruct(raw):
    return _reconstruct(raw, "reflect")
