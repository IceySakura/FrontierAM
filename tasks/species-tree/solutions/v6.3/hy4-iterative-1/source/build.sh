#!/bin/sh
# No native build is needed.  Syntax-check the source without writing files.
set -e
DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
python3 -c 'import ast,sys; ast.parse(open(sys.argv[1], encoding="utf-8").read())' "$DIR/main.py"
