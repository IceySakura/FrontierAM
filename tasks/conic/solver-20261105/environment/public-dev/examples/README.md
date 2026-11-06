# 接口示例

这 6 个小问题用于检查输入、输出及独立验收接口，均可直接验证；不包含正式测试数据或答案。输入为 `Ax+s=b, s∈K`，矩阵 `P` 保存完整对称矩阵，目标为 `xᵀPx/2+qᵀx+c0`。

| 前缀 | 问题 | 合法结果 |
| --- | --- | --- |
| `01-mixed-qp` | `P=[[2,1],[1,2]], q=(-1,0), c0=3`；`x₂=0, x₁≥1`；覆盖零锥和非负锥 | `x=(1,0), s=(0,0), z=(-1,1)`，目标值 3 |
| `02-soc` | 最小化 `t`，使 `(t,1)` 属于二阶锥 | `x=(1), s=(1,1), z=(1,-1)` |
| `03-exponential` | 最小化 `t`，使 `(0,1,t)` 属于指数锥 | `x=(1), s=(0,1,1), z=(-1,-1,1)` |
| `04-psd` | 最小化 `t`，使 `[[t,1],[1,t]]` 半正定 | `x=(1), s=(1,√2,1), z=(1/2,-√2/2,1/2)` |
| `05-primal-infeasible` | 同时要求 `x≥1` 和 `x≤0` | `primal_infeasible`，证据 `z=(1,1)` |
| `06-dual-infeasible` | 最小化 `-x`，要求 `x≥0` | `dual_infeasible`，方向 `x=(1), s=(1)` |

PSD 向量按上三角逐列排列，非对角元乘 `√2`。不可行证据文件不填写有限目标值；`dual_infeasible` 表示通过对偶不可行证据检查，接口不会把它自动改写为原问题无界。

每例提供 `<前缀>.input.npz` 与 `<前缀>.result.npz`。结果只包含 `metadata_json` 及所需的一维向量；元数据含 `schema_version=1`、`status`，最优解示例另含两个目标值。没有算法名称或来源标签。NPZ 使用 `allow_pickle=False` 读取。

从公开开发包 `public-dev` 目录运行（使用已安装其 `requirements.txt` 的 Python）：

```sh
python io/check_result.py --input examples/01-mixed-qp.input.npz --result examples/01-mixed-qp.result.npz
python io/check_result.py --input examples/04-psd.input.npz --result examples/04-psd.result.npz
python io/check_result.py --input examples/05-primal-infeasible.input.npz --result examples/05-primal-infeasible.result.npz
python io/check_result.py --input examples/06-dual-infeasible.input.npz --result examples/06-dual-infeasible.result.npz
```

`expected_checks.json` 列出全部 7 项检查及预期 `accepted` / `rejected`；前 6 项应通过固定 `1e-8` 验收。

反例只把第一个问题的候选 `x` 改为 `(0,0)`，保留其余字段。该点与正确解具有相同目标值 3，却违反 `x₁≥1`，所以比较目标值不能代替完整检查：

```sh
python io/check_result.py --input examples/01-mixed-qp.input.npz --result examples/01-mixed-qp.rejected.npz
```

反例应被拒绝；这是预期结果，不是接口故障。合法示例不调用任何求解器生成。
