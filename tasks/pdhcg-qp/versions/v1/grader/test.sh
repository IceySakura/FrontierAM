#!/bin/sh
set -eu
# Tests arrive only in the separate verifier. Keep data and calibration root-only.
chmod -R go-rwx /tests
chmod 700 /logs/verifier
# Temporary files live inside each solve's disposable working directory.
chmod 755 /tmp /var/tmp /dev/shm
if [ -d /workspace/submission ] && [ ! -L /workspace/submission ]; then
    chown 0:0 /workspace/submission
    chmod 700 /workspace/submission
fi
python3 -B /tests/grade.py --calibrate --submission /tests/reference \
    --output /logs/verifier/reference.json
exec python3 -B /tests/grade.py
