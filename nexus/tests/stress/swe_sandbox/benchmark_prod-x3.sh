#!/bin/bash

# 调用示例, 更多选项看 README.md/run_benchmark.py

export SWE_BENCH_TEST_SLEEP=360
exec bash tests/stress/swe_sandbox/benchmark_base.sh --env prod-x3 --total-count-per-round 10000