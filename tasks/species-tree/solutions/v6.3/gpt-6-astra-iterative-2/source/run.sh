#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec ./infer "$1" "$2" --distance length --normalize 1 --method bionj --power 0.5 --nni 20 --qpower 2 --qexternal 0.5 --qlength 1 --qnormalize 1 --qmax 32 --qmixture 0.5
