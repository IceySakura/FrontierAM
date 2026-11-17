# 数据与来源

输入来自导入题包的 scale-aligned v3：四类结构，各 8 个规模档 × 2 个变体，共 64 public 和 64 private。两个集合均覆盖 n=256–1024。文件按原始 manifest 的 SHA-256 原样迁入，未重生成数据，也未带入生成种子。

- `samples/`：从 public 选取四类各一例，附清单和检查器入口；只给模型这些输入。
- `public/`：64 个反馈输入；仅维护端评测可见，不打进开发镜像。
- private：单独保管在 `var/raw/pdhcg-qp-v1/private/`，不进 Git。重新部署需提供 manifest 与对应输入。

构造出的 KKT 答案不参与评分。评分只依据输入和候选输出重新计算 KKT 残差；原包的 calibration.json 和历史成绩未作为锚点导入。样例检查器的 qpbench 模块由 Dockerfile / 封存脚本从 grader 的同一份源码复制，避免两套数值协议漂移。

导入时的完整原包保存在本机 `var/raw/pdhcg-qp-import/`，仅用于来源追溯，不是新任务的依赖，也不进入 Git 或 Harbor 题包。
