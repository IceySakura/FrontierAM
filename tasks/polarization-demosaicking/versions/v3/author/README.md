# 偏振去马赛克 vNext 维护评测包

这是 vNext `1.3.0-dev` 的维护侧评测包。它包含完整的 Harbor 运行器、Qiu32 与 Wen100 固定评测数据、Wen5 开发数据、两条内部参考实现、评分器、校准脚本、Docker ARM64 运行镜像和校验工具。公开题包只应交给做题 agent，维护包不能挂载到 agent 的运行目录。

## 评测协议

- 正式集为 Qiu32 + Wen100，共 132 景。Wen5 只作开发诊断。
- 每景先取四个方向 RMSE 的算术平均，再对 132 景等权平均得到 `E`。
- 0 分、5 分和 10 分的全局锚点分别是经典参考、先进参考和零误差。
- 固定锚点为 `B=2.10798080792808`、`H5=1.5006853943125558`、完美输出为 `0`。
- 正式场景任一景因提交导致超时、超内存、异常、缺失或非法输出，整份分数为 0。
- 每景使用 1 个逻辑 CPU、8 GiB 内存、无 swap，墙钟上限为 1.000 秒。构建另限 300 秒。

## 目录和运行

`data/qiu32/`、`data/wen5/` 和 `data/wen100/` 分别包含固定数组。每个场景目录内有 `raw.npy` 和 `reference.npy`，参考通道顺序为 0、45、90、135。

先加载 `images/runtime-images.tar` 中的 ARM64 镜像，然后从本目录运行：

```sh
docker image load -i images/runtime-images.tar
python3 harbor/polarization-demosaicking/test_scoring.py
python3 harbor/polarization-demosaicking/grade_vnext.py \
  /absolute/path/to/submission output/formal --mode formal
python3 harbor/polarization-demosaicking/grade_vnext.py \
  /absolute/path/to/submission output/dev --mode dev
python3 tools/run_baselines.py --output output/baselines
python3 verify_package.py --runtime
```

`grade_vnext.py` 会先冻结提交并构建一次，然后逐景在 Docker 沙箱中运行。正式评分只从 `summary.json` 和 `per_scene.csv` 读取，不接受提交自行提供的参考或锚点。

## 来源和发布状态

Qiu32 数组来自已冻结的 v1.2 数据。Wen 数据按 `tasks/polarization-demosaicking/vnext/wen_config.json` 的 `wen-gray-polar-v0` 规则转换，固定开发集为 5 景，正式集为其余 100 景。原始来源、转换清单和切分清单保存在 `tasks/polarization-demosaicking/vnext/`。

本包是内部候选包。Wen 数据权利人的再分发、服务器评测和公开开发集许可仍需在对外发布前完成签收。
