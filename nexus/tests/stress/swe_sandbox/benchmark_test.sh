#!/bin/bash

# 调用示例, 更多选项看 README.md/run_benchmark.py

export SWE_BENCH_TEST_SLEEP=180
# export SWE_RUNTIME_PROVIDER_CLUSTER=suanli # 指定后端pod在算力平台创建, 默认为"gongfeng"集群
# export SWE_RUNTIME_PROVIDER_TOKEN="your-api-token-here" # API token for authentication (optional)

# 常用选项
# --env prod-x3/prod-x8/test  指定sandbox调用k8s集群 
# --total-count-per-round 500  指定每轮测试的次数
# --batch-id <uuid>            指定批次ID，用于关联同一批start/stop的Runtime实例（默认自动生成UUID）
# export SWE_BATCH_ID="my-batch-123"  # 也可通过环境变量指定batch_id
exec bash tests/stress/swe_sandbox/benchmark_base.sh --env test --total-count-per-round 500