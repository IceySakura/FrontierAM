# scripts/：脚本索引

入口命令是仓库根目录的 `./bench`（运行 `arena/cli.py`），子命令包括 `prepare`、`baseline`、`develop`、`grade`、`suite`、`public-eval` 等；下面的脚本围绕它完成封存、批量评分与分析，按用途放在子目录里（`scripts/<子目录>/<脚本>`）。

| 类别 | 脚本 |
| --- | --- |
| `setup/` 环境与镜像 | `build-images.sh`、`build-operator-images.sh`、`load-grade-images.sh`、`switch-layout.sh`、`isolation-experiment.sh`、`uid-sampler.py`、`shim-audit.py` |
| `seal/` 题目封存 | `prepare.py`（conic 运行副本，含公开反馈与评分提速版）、`prepare_exponential_v2.py`、`exp2-freeze.py`、`prepare_operator_tasks.py`（出题方新题）、`prepare_galaxy_v2.py`、`prepare_species_v2.py`、`relock.py`、`migrate_task_layout.py`（11-10 的目录迁移记录） |
| `references/` 参照与基线 | `new-task-baselines.sh`、`exp2-panel-suite.sh`、`exp2-public.sh`、`exp2-private.sh`、`exp2-freeze-and-private.sh`、`species-v2-public-baseline.sh`、`traj-ref-screen.py`（trajectory 人类候选参照筛选） |
| `rounds/` 尝试（模型开发） | `campaign.sh`、`serial-attempts.sh`、`concurrent-round.py`、`v62-smoke.py`、`v62-chain.sh`、`v62-final.sh`、`v62-followup.sh`、`v62-after.sh`、`round5-singles.sh`、`round5-sol-retry.sh`、`round5-pipeline.sh` |
| `grade/` 评分与计分 | `exp2-suite-score.py`、`exp2-sol-suite.sh`、`species-v2-suite-score.py`、`conic-recal.py`、`conic-recalibrate.py`、`conic-grade-remaining.py`、`conic2-sol-suite.sh`、`conic-skip-check.sh`、`grade-sol-operator.sh`、`paired-validation.sh`、`exp2-container-check.py` |
| `analyze/` 分析 | `conic-final-report.py`、`conic-per-case-export.py`、`conic-case-times.py`、`conic-substantive.py`、`conic-suite-compare.py`、`trajectory-compare.py`、`trajectory-expected.py`、`polar-first-scene.py`、`polar-warmup-check.py` |
| `site/` 看板与题目页 | `build_dashboard.py`、`build_task_pages.py` |
| `ops/` 运维 | `run_leased.py`（在租用的开发 slot 上跑容器任务）、`data-maintenance.sh`、`overnight.sh`、`finish-orphan.py`、`release-after.py`、`release-when-free.py` |

只含 GitHub 公开材料的 checkout 新增单题时，可用 `python scripts/site/build_task_pages.py --task ihvp-sequence` 生成该题 README 并更新索引和看板，其他题目的封存材料与历史页面保持不动。`python scripts/site/build_dashboard.py --task ihvp-sequence --out var/dist/ihvp-preview` 可生成单题预览，无需读取其他题目的维护端材料；正式全站构建仍使用原有默认命令。
