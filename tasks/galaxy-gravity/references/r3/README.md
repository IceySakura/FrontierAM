# galaxy-gravity 参考材料 r3（不作锚点）

> 11-11 用户确定参照用出题人自己做或深度参与的版本（`results/BASELINE_POLICY.md`），本实现由 agent 编写，只作参考材料；galaxy-gravity-v3 的锚点是出题方的 Treecode2（r1）。

- **来源**：按 Dehnen 2000/2002/2014 独立实现的高阶笛卡尔 FMM（`../candidates/fmm-dehnen/`，实现者未接触任何模型提交）。`source/` 只含冻结配置所需的源码；论文复现、标定和消融见候选目录的 `VALIDATION.md`、`CALIBRATION.md`、`ABLATION.md`。
- **冻结配置**（编译进默认值，按事先声明的规则在 10 张公开快照上选出）：p = 8，常张角 θ = 0.48，(N_crit, N_cc, N_cs) = (96, 96, 256)，不用生成核、不做核拆分。配置固定，不随机器速度调整，锚点可复现。
- **为什么替换 r2**：r2 是 claude-opus-5-5 的冻结提交。按 `results/BASELINE_POLICY.md`（11-11 修订）参照只用人类方法的人类实现，不用被评模型的提交。
- **计分（对数刻度）**：\(E < W\) 时 \(50 + 50\,\log(W/E)/\log(W/10^{-15})\)；\(W \le E < U\) 时 \(50\,\log(U/E)/\log(U/W)\)，\(U\) 为弱参照（经典树算法）的误差，\(W\) 为本参照的误差，均在评测槽私有集上测量。
