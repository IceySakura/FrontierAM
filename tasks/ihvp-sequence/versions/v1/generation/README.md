# 数据生成

本目录提供同一显式强凸二次 bilevel 族的通用构造。代码不包含反馈池的完整种子计划，也不包含最终私有集的种子或答案。

对每个状态，

\[
g_s(w)=\tfrac12w^\top H(s)w-c(s)^\top w,
\qquad w^*(s)=H(s)^{-1}c(s),
\]

验证目标是固定的二次函数，右端由 \(b(s)=\nabla f(w^*(s))\) 产生。相同的外生控制同时决定矩阵、内层线性项和右端；控制轨迹不是优化算法产生的真实梯度轨迹。

矩阵通过带随机符号和置换的正交 DCT、宽谱对角因子及 rank-12 修正构造。算子不分配完整的 \(d\times d\) 矩阵。`model.py` 提供 `build_operator_data`、`read_case`、`case_matvec` 和教师数据生成；`common.py` 提供数学辅助函数。教师参考使用 Woodbury，并由独立的低条件数坐标系求解和完整残差检查复核。

## 生成给模型的样例

在仓库根目录、已安装 NumPy/SciPy 的解释器中运行：

```sh
python tasks/ihvp-sequence/versions/v1/generation/generate.py \
  --samples --out var/outputs/ihvp-sequence-v1/samples
```

输出包含一个 4 维协议示例和两个完整的 `d=16384, T=50` 序列，来源与公开种子记录在 `data/samples/recipes.json`。生成后的 NPZ 包含教师构造和参考解，属于宿主侧样例材料；求解进程只通过题目接口取得右端和受控算子回复。

样例记录源环境下的数学数组哈希。它们不含变动的生成耗时与源码位置等元数据。同一源环境可以逐位核对；不同 BLAS/FFT 实现的舍入可能不同，所以默认将此哈希作为诊断，另行复算参考和残差。`--strict-source-match` 可要求完全匹配。

## 创建新的本地私有数据

```sh
python tasks/ihvp-sequence/versions/v1/generation/generate.py \
  --private --out var/raw/ihvp-sequence-new-version/private
```

此命令用 `secrets.randbits(64)` 采样一次种子，并在输出目录冻结本地计划；不使用公开样例种子，也不会打印私有种子。生成内容必须留在忽略路径或仓库外。

这条命令用于维护者创建新版本或新实验，不重建本次独立保管的 v1 私有集。改变正式测试集需要按仓库流程建立新版本，不能以重新生成的数据替换已封存成绩。

这里的生成校验不能代替目标机计时、参照校准或 amd64 容器验收。
