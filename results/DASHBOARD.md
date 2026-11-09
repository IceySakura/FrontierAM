# FrontierAM 任务看板

**访问地址：http://9.135.91.91:18765/**（公司办公网内可访问）

一个在线评测（OJ）风格的静态网站：首页有题目卡片和总榜（题目 × 模型，满分 100，本机参照实现 = 50）；每道题一页，包括题目背景、给模型的公开题面原文、评测依据（输入、规模、时限、判定规则、指标和换算公式）、baseline 的做法和本机实测结果、各模型的成绩（原始指标、换算方式、得分）、按问题类型或结构族的分组表现，以及各模型的做法（我们的分析和模型提交中自带的 README）；另有评测协议页。

## 组成

| 部分 | 位置 |
| --- | --- |
| 人工撰写的内容（背景、评测依据、baseline、成绩、说明） | `dashboard/site.json`、`dashboard/tasks/*.json` |
| 样式与公式渲染（KaTeX 0.16.11，随站打包，不依赖外部资源） | `dashboard/assets/` |
| 构建脚本（读取上述内容、公开题面、提交中的 README 和结果文件中的汇总数字） | `scripts/build_dashboard.py` |
| 网站输出（仓库外，只读） | `/srv/frontieram-dashboard/` |
| 服务（`python3 -m http.server`，用户 nobody，只读、固定在管理核） | `frontieram-dashboard.service`，端口 18765 |
| 定时重建（每 15 分钟） | `frontieram-dashboard-build.timer` |

更新成绩：修改 `dashboard/tasks/<题目>.json` 后，等定时器重建，或立即运行 `.venv-ops/bin/python scripts/build_dashboard.py --out /srv/frontieram-dashboard`。

## 内容审核规则

- 不出现任何隐藏测试数据：私有输入、真值、种子；分组表现只给汇总数字。
- 不出现会帮助未来参赛 agent 的私有信息，例如私有测试集取自哪个公开数据集、用了哪些具体矩阵或场景。已据此删去 exponential-v2 私有母矩阵的名称和 polarization 测试场景的来源。
- 题面只展示给模型的公开版本。
- 凭据只在宿主机的私有文件里，构建脚本不读取。

## 公网访问（待决定）

这台机器是 AnyDev 云开发环境，只有内网地址（9.135.91.91）；出公网经 NAT，外部无法连入。AnyDev 的环境域名也只在办公网内可用，开发机不支持直接对公网开放服务，按公司网络安全要求也不应使用内网穿透等方式绕过。

看板是完全静态、自包含的网站，若需要公网访问，可走正规流程申请公网托管（例如腾讯云 COS 静态网站或其他经过安全审批的公网主机），把 `/srv/frontieram-dashboard/` 整个目录上传即可。上线前需要再审一遍：公网上的内容会被未来的模型看到，模型做法说明（README）是否公开需要决定。
