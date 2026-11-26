#!/bin/sh
# Entry point for the cubature submission: speaks the JSON Lines protocol on
# stdin/stdout.  Keep the interpreter quiet and deterministic.
exec python3 -u "$(dirname "$0")/main.py"
