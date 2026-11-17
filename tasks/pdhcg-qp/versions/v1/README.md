# v1：构建与运行

题面、grader、Dockerfile、data 与 task.json 是源文件；Harbor 题包由仓库封存脚本生成，位于 `.runtime-ops/pdhcg-qp-v1/harbor/`。生成目录不进 Git，也不复制原题包的实验记录、历史校准、构造答案或 local-run 工具。

在仓库根目录执行：

```sh
python3 scripts/seal/prepare_pdhcg_qp.py --build-image
```

私有输入默认从 `var/raw/pdhcg-qp-v1/private/` 读取，也可用 `--private-data DIR` 指定。目录包含 `manifest.json` 及清单中的 `inputs/*.npz`，需由维护者提供。脚本核对文件哈希，并分别生成 `pdhcg-qp-public` 和 `pdhcg-qp-private` 两个 Harbor 题包。每个题包仅含该集合的输入。

封存目录已存在时，脚本拒绝覆盖；试验新配置时使用新的 `--output DIR`。`--image IMAGE` 可指定部署端能够读取的镜像名；远端后端运行前，维护者需把构建的镜像上传到其可访问的镜像仓库，并在封存时使用该镜像名。

## Harbor / Nexus

题包包含标准 `task.toml`、`instruction.md` 和 `tests/test.sh`。使用仓库现有 Harbor Trial 入口：

```sh
export FRONTIERAM_QP_HARBOR_TASK="$PWD/.runtime-ops/pdhcg-qp-v1/harbor/pdhcg-qp-private"
# 按现有 Nexus 环境配置 GONGFENG_RUNTIME_TOKEN、AIHUB_API_KEY；可选 FRONTIERAM_QP_MODEL。
cd nexus
uv run nexus run single ../scripts/rounds/pdhcg-qp-harbor.yaml \
  -o ../var/outputs/pdhcg-qp-harbor.json
```

切换到 public 时，将 `FRONTIERAM_QP_HARBOR_TASK` 指向 `pdhcg-qp-public`。公开输入和私有输入均只上传到独立 verifier；开发镜像只包含题面、四个样例和公开检查器。Codex CLI 已在镜像中预装，以匹配运行配置的固定版本；其他助手需在自己的开发镜像中预装。

`tests/test.sh` 先在本次 verifier 上运行参照三遍，确认每例全部通过且时限至少有 25% 余量，再评候选。每例时限为参照三遍最长时间的 1.5 倍，截断到 [1,60] 秒。评分使用参照三遍的**中位数**，与候选采用相同汇总。容器的单 CPU、4 GiB 与断网策略由 Harbor 配置执行；verifier 内的候选 UID 为 1000。参照、数据和日志仅 root 可读。

结果位于 `/logs/verifier/summary.json`，原始参照记录位于 `reference.json`，`reward.json` 包含 `reward`（0–1）、`score`（0–100）、`success_rate`、`time_ratio`、`g_ref`。检查器或基础设施失败不会产生 reward 文件；构建失败及求解错误才记为候选失败。

## 仓库 bench 评分

本题也注册为 `pdhcg-qp-v1`。使用仓库评测槽前，在同一物理核上封存和校准：

```sh
python3 scripts/seal/prepare_pdhcg_qp.py --build-image --calibrate --cpuset 8
./bench baseline --task pdhcg-qp-v1
./bench submit --task pdhcg-qp-v1 --source /absolute/path/to/submission
./bench grade-worker --task pdhcg-qp-v1 --drain
```

上面的 `--cpuset 8` 是示例，应使用目标机器的评测核。public 与 private 独立校准；`FRONTIERAM_QP_DATA=public` 选择 public，默认 private。两种集合的 bench baseline 应放在独立 store，禁止混用。模型开发实验使用上面的 Harbor 入口；此处接入 bench 的源码冻结、baseline 与评分队列。

## 验证

```sh
python3 -m pytest tasks/pdhcg-qp/versions/v1/grader/tests -q
```

纯数值与封装测试不需要 Docker。进程隔离测试需要 Linux root，会使用测试专用 UID；真实容器断网、资源限制与模型调用仍需在部署机器验收。
