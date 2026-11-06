# 开发数据与自检工具

本目录包含 **163 个开发实例**及接口示例，供开发、调试和自检。任务、输入输出及评测规则以 Harbor 提供的题面为准。开发实例不计入正式成绩；正式评测使用另行提供的 650 个实例。

## 文件

| 路径 | 内容 |
| --- | --- |
| `manifest.json` | 匿名 ID、文件路径、SHA-256、规模、锥类型和资源预算；路径相对于本目录 |
| `inputs/` | 163 个问题的 NPZ 输入 |
| `examples/` | 6 个示例输入、6 个合法结果、1 个故意错误的结果及预期检查结果 |
| `io/model.py` | 输入数据结构和读取函数 |
| `io/check.py`、`io/check_result.py` | 数值检查实现与命令行入口 |
| `requirements.txt` | 自检工具的固定依赖；算法实现语言不限 |

输入覆盖五类锥，规模各异；请根据清单和实际数组选择调试实例，不按压缩文件大小推断求解难度。

## 使用

自检工具使用 Python 3.12.14、NumPy 2.3.5、SciPy 1.16.3。在本目录执行：

```sh
python io/check_result.py --input examples/01-mixed-qp.input.npz --result examples/01-mixed-qp.result.npz
```

该示例预期返回 `outcome=verified`，退出码为 0。检查自己的结果时替换两个路径。`rejected` 或 `unresolved` 的退出码为 1，工具错误为 2；数值验收条件见题面“评测指标”。工具只验解，不运行求解算法或限制资源，自测时需自行控制耗时和内存。

将源码、可执行 `run.sh` 和 `README.md` 保存到 `/workspace/submission/`；需要安装或编译时提供 `build.sh`。评测调用为：

```text
./run.sh --input INPUT_PATH --output OUTPUT_PATH
```

实际路径由评测方传入。开发清单为每个实例公开一个 `time_limit_seconds`，预算分布用于暴露正式评测的时间压力；正式测试按自己的评测清单逐例限时，每次 1–300 秒，同一实例三次使用相同预算。求解使用单核、数值计算单线程；评测容器总内存预算为 8 GiB，包含评测开销。其他资源与文件要求见题面。依赖须预装或随提交提供，以支持离线构建与运行。
