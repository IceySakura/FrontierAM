# 题目名称

固定资源下的高精度凸锥优化求解

# 问题描述

实现一个通用求解程序，处理以下凸优化问题：

\[
\min_{x,s}\ \tfrac12x^\top Px+q^\top x+c_0,
\qquad Ax+s=b,\quad s\in\mathcal K.
\]

| 输入 | 含义 |
| --- | --- |
| \(P\) | \(n\times n\) 对称半正定稀疏矩阵 |
| \(q\) | 长度 \(n\) 的向量 |
| \(A,b\) | \(m\times n\) 稀疏矩阵与长度 \(m\) 的向量 |
| \(\mathcal K\) | 按约束行排列的锥块乘积，由 `cones` 列表描述；支持零锥、非负锥、二阶锥、指数锥和半正定锥 |
| \(c_0\) | 目标函数常数项，对应 `objective_offset`，无常数项时取 0 |

\(\mathcal K\) 是各锥块的笛卡尔积。零锥为 \(\{0\}^d\)，非负锥为 \(\mathbb R_+^d\)，二阶锥为 \(\{(t,v):t\ge\|v\|_2\}\)，半正定锥为实对称半正定矩阵的集合。指数锥定义为
\(K_{\exp}=\{(u,v,w):v>0,\ v e^{u/v}\le w\}\cup\{(u,0,w):u\le0,w\ge0\}\)。各锥块的向量编码见“交付要求”。\(\mathcal K^*\) 表示对偶锥，即与 \(\mathcal K\) 中任意向量的内积均非负的向量集合。

程序返回下列一种状态及对应向量：

| `status` | 含义 | 必需向量 |
| --- | --- | --- |
| `solved` | 找到满足精度要求的最优解 | `x,s,z` |
| `primal_infeasible` | 判断不存在满足全部约束的解 | `z`，用于验证这一判断 |
| `dual_infeasible` | 找到满足规定条件的下降方向；不单独表示原问题无界 | 方向向量 `x,s` |
| `unresolved` | 未得到可验证的结果 | 无 |

`x` 长度为 \(n\)，`s,z` 长度为 \(m\)，均为一维有限实数数组。`solved` 中 `x` 是所求解，`s=b-Ax`，`z` 用于验证最优性；目标值由评测方根据 `x` 计算。其余状态的验证条件见“评测指标”。

# 算法目标

在规定精度和逐例资源预算内，提高正确求解的实例比例。算法路线和实现语言不限。

请自行检索相关论文与开源实现，调研当前先进方法，评估其在本题精度要求和资源限制下的适用性，并据此选择、实现或改进算法。可以采用、组合或改进已有方法，也可以设计新方法。调研说明不单独计分。

# 算法约束

开发工作目录为 `/workspace`，开发阶段不限时。正式构建和求解须遵守以下限制：

| 项目 | 要求 |
| --- | --- |
| 运行环境 | Debian 12、Linux ARM64；1 个逻辑 CPU，数值计算单线程 |
| 求解预算 | 每例每次按评测清单的 `time_limit_seconds` 限时，范围为 1–300 秒；评测容器总内存上限 8 GiB，包含评测程序开销 |
| 构建预算 | 最多 1,800 秒；同样使用单核和 8 GiB 总内存预算 |
| 文件与日志 | 临时目录、输出目录各限 1 GiB；单个普通文件限 1 GiB；stdout、stderr 各限 4 MiB |
| 进程与文件描述符 | 提交程序及其后代最多 128 个进程／线程；每进程最多 256 个打开的文件描述符；退出时不得留下后台计算进程 |

逐例时限在评测前固定，对所有提交一致：参考实现在某例三次均成功时，取三次最大耗时的 1.2 倍，最低 1 秒、最高 300 秒；其余实例为 18 秒。三次重复使用同一时限。

构建和评测运行均离线、以普通权限执行；求解时代码和输入只读。根据本次输入计算，不依赖实例名称、调用顺序、跨实例状态或预存答案。允许预处理和等价变换，返回向量须恢复到原问题的坐标和约束块顺序。

# 交付要求

## 提交目录与调用方式

将全部源码和离线依赖保存到 **`/workspace/submission/`**，提供：

| 文件 | 要求 |
| --- | --- |
| `run.sh` | 可执行运行入口，接收 `--input` 和 `--output` 参数 |
| `README.md` | 说明运行方式、依赖及固定版本；简要列出主要参考文献或实现，说明采用的方法、改进内容及实验依据 |
| `build.sh` | 需要安装或编译时提供；无参数、可执行，否则可省略 |

提交目录只接收普通文件和目录，不接收符号链接或特殊文件。评测方将提交复制到独立环境，以提交目录为工作目录，先执行一次 `./build.sh`（若提供），再逐例调用：

```sh
./run.sh --input INPUT_PATH --output OUTPUT_PATH
```

两个参数均由评测方指定为绝对路径，输入和输出均为 NPZ 文件。脚本须声明解释器；`run.sh` 用 `exec` 启动求解进程并传递参数。构建成功、求解完成后均以退出码 0 结束；返回 `unresolved` 时也须写出结果。

依赖须由环境预装或随提交提供，不依赖开发会话中的临时安装。预装工具及版本见 `/workspace/resources.yaml`。构建只能做与测试实例无关的准备，依赖和持久构建产物保存在提交目录；实例相关准备计入逐例运行时间。不得写死输入、输出或工作目录路径；临时文件使用 `TMPDIR`，临时目录和输出目录在每次调用前重新创建。

## 输入文件格式

NPZ 包含一维 `float64` 数组 `q,b`，以及 `P_data,P_indices,P_indptr,P_shape`、`A_data,A_indices,A_indptr,A_shape` 两组标准 CSC 数组。索引从 0 开始，重复位置求和，\(P\) 存储完整对称矩阵。所有系数有限，索引和维度合法，不使用对象数组或 pickle。输入对称性容许 \(\max_{ij}|P_{ij}-P_{ji}|/\max(1,\max_{ij}|P_{ij}|)\le10^{-12}\)。

`metadata_json` 为标量 Unicode 数组，内容为 JSON 对象，以下键均必需：

```json
{"schema_version":1,"name":"instance_id","cones":[{"type":"nonnegative","dim":2}],"objective_offset":0.0,"metadata":{}}
```

`cones` 按约束行排列，恰好覆盖 \(m\) 行；`dim` 为正整数：

| `type` | `dim` 与编码 |
| --- | --- |
| `zero`、`nonnegative` | 向量长度 |
| `soc` | 向量长度，至少为 2；首项为标量，其余为向量部分 |
| `exponential` | 固定为 3，坐标顺序为 \((u,v,w)\) |
| `psd` | 矩阵阶数 \(d\)，占 \(d(d+1)/2\) 行；上三角逐列排列，非对角项乘 \(\sqrt2\) |

## 输出文件格式

NPZ 的 `metadata_json` 同样为标量 Unicode 数组，例如 `{"schema_version":1,"status":"solved"}`；`schema_version` 必须是整数 `1`。按状态提供 `x,s,z`；必需向量长度为零时仍须保存空数组，额外提供的向量也须满足对应维度。

`solved` 可在 JSON 中提供包含 \(c_0\) 的有限目标值 `objective_primal`、`objective_dual`，省略或 `null` 表示未报告。`primal_infeasible`、`dual_infeasible` 的目标字段须省略或为 `null`。JSON 不得含 `NaN` 或 `Infinity`。

# 评测指标

## 成功条件与计分指标

在固定的 650 个留出实例上评测，每例等权、运行三次；5 个样例实例供开发，不计入最终指标。各类别另报指标，不先按类别打分再平均。

唯一计分指标为成功率 \(p=k/N\)：\(k\) 是三次均在该例时限内通过独立检查且状态一致的实例数，\(N=650\)。成功率越高越好。运行时间用于判定是否超时，并保留实际耗时记录；更快、更低内存或更小数值误差不另加分。

每次以新进程顺序运行。计时从启动提交程序前开始，到检测到退出为止，包含加载、数据读写、实例相关准备、进程内编译、求解和恢复原坐标；不含通用构建、容器启停及独立检查。不逐例预热或清空系统缓存。

`unresolved`、超时、超内存、异常退出、输出缺失或格式错误、未通过检查或三次返回状态不一致，均使该实例计失败。失败不另设时间罚分。缺少必需提交文件、构建失败或运行入口无效时，整次提交计 0 分。评测程序或检查器故障不计为算法失败；评测记录缺失或无效时不产生有效总分。

## 0–10 分映射

令 \(p_B\) 为参考记录按本题时限和验收条件核算的成功率，\(p_W\) 为弱参照（只求解零锥与非负锥的 HiGHS，其余锥返回 `unresolved`）按同样条件核算的成功率，满足 \(0<p_W<p_B<1\)。这两个参数由评测方预先固定，对所有提交一致使用，不公开具体值。

\[
S(p)=\begin{cases}
0,&0\le p\le p_W,\\[3pt]
5\dfrac{p-p_W}{p_B-p_W},&p_W<p\le p_B,\\[6pt]
5+5\dfrac{p-p_B}{1-p_B},&p_B<p\le1.
\end{cases}
\]

参考成功率对应 5 分，弱参照的成功率及以下对应 0 分，100% 成功率对应 10 分（2026-11-26 起；此前 0 分点为参考成功率的 90%）。评分参数不根据提交成绩调整。

计算使用完整精度，展示总分保留两位小数；若平台以 0–1 的 `reward` 返回成绩，其值为 \(S/10\)。先汇总每例的三次结果，再计算一次 \(p,S\)，不分别计算三轮总分后平均。

## 正确性条件

评测器使用原始输入及 `float64` 独立验解。未标下标的范数均为无穷范数，空向量范数为 0，\(\varepsilon=10^{-8}\)；所有检查计算须有限。

`solved` 须满足

\[
\begin{gathered}
r_p=\frac{\|Ax+s-b\|}{1+\max(\|Ax\|,\|s\|,\|b\|)}\le\varepsilon,\qquad
r_d=\frac{\|Px+q+A^\top z\|}{1+\max(\|Px\|,\|q\|,\|A^\top z\|)}\le\varepsilon,\\
C_{\mathcal K}(s)\le\varepsilon,\qquad C_{\mathcal K^*}(z)\le\varepsilon,\\
p_0=\tfrac12x^\top Px+q^\top x,\quad d_0=-\tfrac12x^\top Px-b^\top z,\quad
|p_0-d_0|\le\varepsilon\ \ \text{或}\ \ \frac{|p_0-d_0|}{\max(|p_0|,|d_0|)}\le\varepsilon.
\end{gathered}
\]

两目标均为 0 时相对间隙取 0。若报告目标值，另要求
\(|p_{\rm reported}-(p_0+c_0)|/(1+|p_0|)\le\varepsilon\)，对偶目标同理。\(|s^\top z|\) 仅作诊断，无独立通过阈值，但须有限。

**锥缺陷。** 对每块令 \(u=v_{\rm block}/\max(1,\|v_{\rm block}\|)\)，\(C_{\mathcal K}(v)\) 取所有块缺陷的最大值（空列表取 0）。记 \(r_+=\max(r,0)\)，各块缺陷为：

| 锥 | 块缺陷 |
| --- | --- |
| 零锥 | \(\lVert u\rVert_\infty\) |
| 非负锥 | \((-\min_j u_j)_+\) |
| 二阶锥 | \((\lVert u_{2:}\rVert_2-u_1)_+\) |
| 半正定锥 | \((-\lambda_{\min}(\operatorname{smat}(u)))_+\)，`smat` 为“交付要求”所定义 PSD 编码的逆变换 |
| 指数锥 | \(E(u_1,u_2,u_3)\) |

\[
B(a,b,c)=\max(|b|,a_+,(-c)_+),\qquad
E(a,b,c)=\begin{cases}
\min\{B(a,b,c),[a-b(\log c-\log b)]_+\},&b>0,c>0,\\
B(a,b,c),&\text{其他情况}.
\end{cases}
\]

对偶锥使用相同规则，但零锥的对偶缺陷为 0，指数锥的对偶缺陷为 \(E(-u_2,-u_1,e u_3)\)；变换后不再归一化。PSD 直接计算最小特征值。

**不可行证据。** 令 \(a=\max_{ij}|A_{ij}|\)、\(h=\max_{ij}|P_{ij}|\)，全零或空矩阵的尺度取 1。

- `primal_infeasible`：要求 \(\|z\|,\|b\|>0\)。令 \(\hat z=z/\|z\|\)、\(\beta=-(b/\|b\|)^\top\hat z\)，须满足

  \[
  \beta>\varepsilon,\qquad
  \frac{\|(A/a)^\top\hat z\|}{\beta}\le\varepsilon,\qquad
  \frac{C_{\mathcal K^*}(\hat z)}{\beta}\le\varepsilon.
  \]

- `dual_infeasible`：要求 \(\|x\|,\|q\|>0\)。令 \(\hat x=x/\|x\|\)、\(\hat s=s/\|x\|\)、\(\alpha=-(q/\|q\|)^\top\hat x\)，须满足

  \[
  \alpha>\varepsilon,\qquad
  \frac{\|(P/h)\hat x\|}{\alpha}\le\varepsilon,\qquad
  \frac{\|(A/a)\hat x+\hat s/a\|}{\alpha}\le\varepsilon,\qquad
  \frac{C_{\mathcal K}(\hat s/a)}{\alpha}\le\varepsilon.
  \]

对偶不可行证据不自动表示原始问题无界。

# 开发材料与数据入口

- 资源与预装工具：`/workspace/resources.yaml`。
- 开发数据：[samples/manifest.json](samples/manifest.json) 列出 5 个样例输入的路径、校验值、规模、类别和逐例开发预算，路径相对于 `/workspace/samples/`。开发清单公开每例 `time_limit_seconds`，用于暴露正式评测的时间压力；正式评测采用自己的固定逐例时限。
- 读取与自检：[输入读取器](samples/io/model.py)、[结果检查工具](samples/io/check_result.py)、[接口示例](samples/examples/README.md)。检查已有结果可运行：

  ```sh
  python /workspace/samples/io/check_result.py --input INPUT_PATH --result RESULT_PATH
  ```

- 留出数据由评测方通过 `--input` 传入，与开发数据使用同一文件格式和正确性条件。公开自检工具只检查结果，不执行求解，也不限制时间和内存。
