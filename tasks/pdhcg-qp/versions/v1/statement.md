# 稀疏凸二次规划求解器

## 任务

实现一个可复用的 CPU 求解程序，读取一个稀疏凸二次规划，并返回可以独立验证的原始解和不等式乘子：

\[
\min_{x\in\mathbb R^n}\; \frac12x^TQx+c^Tx+c_0
\quad\text{s.t.}\quad Ax\le b,
\qquad Q=Q^T\succeq0.
\]

正式实例均可行且有有限最优值。程序不得依赖实例文件名、调用顺序、公开答案或未写入输入的生成器信息。算法、编程语言和公开数值库不限；实例相关的读取、预处理、参数选择、求解、精化和写回都计入逐例时间。

## 输入

输入是一个不含 pickle 的 NPZ，包含：

* `q_data,q_indices,q_indptr,q_shape`：Q 的 CSR 数组，完整存储对称矩阵；
* `a_data,a_indices,a_indptr,a_shape`：A 的 CSR 数组；
* 一维有限 `float64` 数组 `c,b`；
* 标量 Unicode 数组 `metadata_json`，JSON 至少含 `schema_version=1`、`n`、`m` 和 `objective_offset`。

公开开发输入可能额外包含诊断标签；正式留出输入只保证上述通用字段，不能假设存在实例编号、结构标签、随机数种子或其他生成器字段。索引从 0 开始，重复位置已在输入中合并；维度、有限性和对称性合法。`objective_offset` 是 `c_0`，不影响最优解。

维度范围为 `n=256–1024`；矩阵涵盖对角病态、稀疏低秩、网络和箱约束结构。开发目录包含四个样例，public 反馈集与最终 private 集各有 64 例，规模范围相同、输入互不重合。

公开开发输入位于 `public-dev/inputs/`，清单在 `public-dev/manifest.json`。这些输入只用于调试，最终留出输入不会出现在开发目录。

## 输出

入口必须是可执行的 `run.sh`，接受两个绝对路径参数：

```sh
./run.sh --input INPUT.npz --output OUTPUT.npz
```

输出 NPZ 必须含：

* 标量 Unicode `metadata_json`，其中 `{"schema_version":1,"status":"solved"}`；
* 一维有限 `float64` `x`（长度 n）；
* 一维有限 `float64` `y`（长度 m），对应 `Ax<=b` 的非负拉格朗日乘子。

可额外写入有限的 `objective`，但评测器会从原始输入重新计算。任何异常、未收敛或无法证实的结果都应写 `status="unresolved"` 并退出 0；这类结果不会通过正确性检查。不得写后台进程或依赖网络。

## 正确性和资源

独立检查要求（`eps=1e-7`）通过以下四项归一化 KKT 残差：

\[
\begin{aligned}
\frac{\max_i(Ax-b)_i^+}{1+\max(\|Ax\|_\infty,\|b\|_\infty)}&\le\varepsilon,\\
\frac{\max_i(-y_i)^+}{1+\|y\|_\infty}&\le\varepsilon,\\
\frac{\|Qx+c+A^Ty\|_\infty}
 {1+\max(\|Qx\|_\infty,\|c\|_\infty,\|A^Ty\|_\infty)}&\le\varepsilon,\\
\frac{\max_i|y_i(b_i-A_ix)|}{1+\|b\|_\infty}&\le\varepsilon.
\end{aligned}
\]

评测环境为 Linux AMD64、单逻辑 CPU、数值库单线程、4 GiB 内存。镜像预装 Python 3.12、NumPy 2.3.5、SciPy 1.16.3 和 C/C++ 编译器，运行时断网。构建阶段最多 900 秒；每个留出实例的时限由评测侧冻结的 CPU 参考运行确定，并限制在 1 到 60 秒。每个实例运行三个全新进程；计时包含启动、读写、预处理和求解。超时、格式错误、非有限输出、残差失败、非零退出或留下后台进程均算失败。

## 分数

一个实例只有三次全部通过才算成功。设 `p` 为成功实例比例；失败实例的时间按该实例时限计入，成功实例用三次中位数，所有实例的时间取几何平均 `g`。评测侧在同一环境、同一集合上重跑参照，参照必须全部通过，其三次中位数按相同方式汇总为 `g_ref`。先汇总，再统一映射到满分 100：

```text
score = 100 * p / (1 + g / g_ref)
Harbor reward = score / 100
```

同参照持平为 50 分；全部正确且耗时趋近于零时趋近 100 分；全部失败或构建失败为 0 分。错误用例同时降低成功率并按时限惩罚用时。原始指标、参照和评分版本会写入评测日志；基础设施故障不给分数。开发期间只能看到 samples，public 仅用于反馈，private 用于最终评分。

## 交付

提交目录为 `/workspace/submission/`，至少包含可执行 `run.sh` 和说明依赖/方法的 `README.md`；需要编译时另提供可执行 `build.sh`。构建不可下载依赖或读取留出数据。运行时只允许从 `--input` 指定文件读取问题，并把结果写到 `--output` 指定路径。

输出文件与展开后的 NPZ 内容均不得超过 4 MiB；单次 stdout/stderr 各不超过 4 MiB。提交及构建后目录均最多 64 MiB、10000 个目录项，不允许符号链接或特殊文件。评测会在新副本中运行一次 build.sh，然后以非 root 用户执行求解；每例三次均为新进程和独立工作目录，只带入冻结的构建产物，禁止跨调用缓存实例或结果。

样例自检：`python3 /workspace/public-dev/check_result.py --input INPUT.npz --output OUTPUT.npz`。
