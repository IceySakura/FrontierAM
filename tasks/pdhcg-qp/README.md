# pdhcg-qp

在固定 CPU 资源下求解稀疏凸二次规划，输出原始解和不等式乘子，由独立 KKT 检查器验证；在同机重测参照后比较正确率与耗时。

## 目录结构

| 目录 | 内容 |
| --- | --- |
| `versions/` | 题目的各个版本；当前版本 `v1/`，题面 `versions/v1/statement.md` |
| `references/` | 参照实现（`r1/` 为现行参照；`candidates/` 为复核中的候选） |
| `solutions/<协议>/<模型>-<设置>-<序号>/` | 模型冻结提交的源码（`source/`）与成绩（`result.json`、`score.json`） |
| `analysis/` | 这道题的专题分析与复核记录 |
| `meta.json` | 本页与看板的数据源（题目说明、版本、成绩） |

## 版本

| 版本 | 评测系统中的名字 | 说明 |
| --- | --- | --- |
| `v1` | `pdhcg-qp-v1` | 从独立题包 scale-aligned v3 接入；统一 AMD64、samples/public/private、独立 Harbor verifier 和参照 50 分映射；等待目标机验收 |

## 题目背景

稀疏凸二次规划是投资组合、网络优化与约束估计的基础问题。输入为 Q、A 的 CSR 数组以及 c、b，求解 0.5*xᵀQx+cᵀx+c₀，约束 Ax≤b。方法与编程语言不限；PDHCG 只是评测侧的参照。

## 题面、数据、评测器与参照在哪里

| 内容 | 位置 |
| --- | --- |
| 题面 | [versions/v1/statement.md](versions/v1/statement.md) |
| 公开开发材料 | [versions/v1/data/](versions/v1/data/)；模型仅看到 samples，public 留在评测侧 |
| 私有测试集 | 维护端 `var/raw/pdhcg-qp-v1/private/`；不进 Git |
| 评测器 | [versions/v1/grader/](versions/v1/grader/)；封存脚本生成 Harbor tests/test.sh |
| 参照实现 | [references/r1/](references/r1/) |

## 数据划分（协议 v6.3）

| 层 | 内容 |
| --- | --- |
| samples（给模型） | 对角病态、稀疏低秩、网络、箱约束各一例 |
| public（不给，反馈用） | 64 个反馈输入，不复制到开发镜像 |
| 反馈分类 | diag_condition、low_rank、network、portfolio_box |

## 评测依据

- 每个集合 64 例，涵盖四类结构与 n=256–1024 的相同规模范围。
- 四项归一化 KKT 残差均须 ≤ 1e-7。每例三个新进程，全部通过才成功；失败例时间按时限计，成功例取中位数，再对所有例取几何平均 g，成功比例记 p。
- 单 CPU、数值库单线程、4 GiB、断网；构建 900 秒，每例时限从本机参照冻结为最长参考时间的 1.5 倍，限于 1–60 秒。
- 最终分数 = 100*p/(1+g/g_ref)，参照全部正确时为 50；Harbor reward 为最终分数除以 100。基础设施失败记未测。

## Baseline

`references/r1/` 保留出题人的重启 PDHCG、内层 CG、活动集精化参照。参照必须在当前评测环境、当前集合全部通过，public 与 private 分别标定，不使用原包历史计时。当前环境没有 Docker，尚未完成目标机容器验收和正式 baseline，不发布成绩。

## 观察与说明

构建镜像、封存题包和启动 Harbor 实验的完整命令见 [versions/v1/README.md](versions/v1/README.md)。新增实现不包含 local-run、旧实验提交、旧成绩、重复 solver bundle 或构造答案。原题包 reward 将参考与最快结果同时截断到 1，接入时为满足仓库参照 = 50 的约定，单独定义了上述映射；原有结果不作跨协议换算。

---

本文件由 `scripts/site/build_task_pages.py` 从 `meta.json` 生成。
