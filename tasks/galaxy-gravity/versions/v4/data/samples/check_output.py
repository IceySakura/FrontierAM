"""Check a solver's output for the included smoke.in and report its RMS error."""
import argparse
import math
import struct
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
args = parser.parse_args()
reference = Path(__file__).with_name("smoke.ref").read_bytes()
actual = args.output.read_bytes()
if len(actual) != len(reference):
    parser.exit(1, f"invalid length: expected {len(reference)} bytes, got {len(actual)}\n")
errors = []
for i, (got, want) in enumerate(zip(struct.iter_unpack("<3d", actual),
                                      struct.iter_unpack("<3d", reference))):
    if not all(map(math.isfinite, got)):
        parser.exit(1, f"nonfinite acceleration in row {i}\n")
    errors.append(math.dist(got, want) / math.hypot(*want))
print(f"vectors={len(errors)}, RMS relative acceleration error={math.hypot(*errors)/math.sqrt(len(errors)):.12g}")
