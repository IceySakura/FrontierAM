#!/bin/sh
set -eu

usage() { echo 'Usage: run.sh --input /absolute/input.npz --output /absolute/result.npz' >&2; exit 2; }
input=''
output=''
while [ "$#" -gt 0 ]; do
    case "$1" in
        --input) [ "$#" -ge 2 ] && [ -z "$input" ] || usage; input=$2; shift 2 ;;
        --output) [ "$#" -ge 2 ] && [ -z "$output" ] || usage; output=$2; shift 2 ;;
        *) usage ;;
    esac
done
case "$input" in /*) ;; *) usage ;; esac
case "$output" in /*) ;; *) usage ;; esac

cd "$(dirname "$0")"
[ -f build_manifest.json ] || { echo 'Run build.sh successfully before run.sh.' >&2; exit 1; }
export PYTHONPATH="$PWD/.deps:$PWD"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
exec python3 -m conicbench.baseline --input "$input" --output "$output" --time-limit 1000 --tol 1e-9
