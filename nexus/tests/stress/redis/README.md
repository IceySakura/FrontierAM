# Redis benchmarks

Run from the Nexus root after `uv sync --extra redis` installs redis-py and ZMQ.
The default run also needs redis-server; Tensor comparisons need CPU torch:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python -m tests.stress.redis.bench_put_get --ops 200 --clients 4
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" uv run --no-sync python -m tests.stress.redis.bench_put_get --ops 200 --tensor-elements 16384 --tensor-count 5
```

By default, the script starts and stops isolated local Redis and FS services.
`--redis-url` borrows an existing Redis instance while still starting and stopping
local FS. Neither mode needs CHIEF_IP or Ray. Existing-instance runs retain benchmark
data under a random prefix (or `--redis-prefix`) and FS topology metadata.
The Redis paths are migrated from HyKVStore revision
`46b7d5aacae6818300a30fe1cfa9c64c5e156e34`; FS uses its default ZMQ settings.
