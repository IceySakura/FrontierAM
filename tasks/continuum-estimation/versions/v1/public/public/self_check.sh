#!/bin/sh
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for input in "$HERE"/dev/*.input.json; do
  python3 - "$input" <<'PY2'
import json, math, sys
obj=json.load(open(sys.argv[1]))
assert set(obj) == {"x", "y"}
assert isinstance(obj["x"], list) and isinstance(obj["y"], list)
assert len(obj["x"]) == len(obj["y"]) >= 4
x=[float(v) for v in obj["x"]]
y=[float(v) for v in obj["y"]]
assert all(math.isfinite(v) and v > 0 for v in x) and len(set(x)) == len(x)
assert all(math.isfinite(v) for v in y)
PY2
  echo "ok $(basename "$input" .input.json)"
done
