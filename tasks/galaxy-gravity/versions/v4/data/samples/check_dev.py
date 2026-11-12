"""Check one fixed development snapshot against its full reference."""
import argparse
import math
from pathlib import Path
import struct

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("input", type=Path)
parser.add_argument("reference", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()

raw_input = args.input.read_bytes()
raw_ref = args.reference.read_bytes()
raw_out = args.output.read_bytes()
if len(raw_input) % 32 or len(raw_ref) % 24:
    parser.error("invalid input or reference length")
n = len(raw_input) // 32
if len(raw_ref) != 24 * n or len(raw_out) != 24 * n:
    parser.error(f"expected {24*n} output bytes")
errors = []
for i, (got, want) in enumerate(zip(struct.iter_unpack("<3d", raw_out),
                                     struct.iter_unpack("<3d", raw_ref))):
    if not all(map(math.isfinite, got)) or not all(map(math.isfinite, want)):
        parser.error(f"nonfinite vector in row {i}")
    norm = math.hypot(*want)
    if not norm:
        parser.error(f"zero reference vector in row {i}")
    errors.append(math.dist(got, want) / norm)
print(f"vectors={n}, RMS relative acceleration error={math.hypot(*errors)/math.sqrt(n):.12g}")
