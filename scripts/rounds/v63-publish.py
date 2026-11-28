"""Publish the v6.3 round (gpt-6.1-sol, claude-opus-5-5) into tasks/<family>/meta.json as a "协议 v6.3" result group.

Reads tasks/<family>/solutions/v6.3/<model>-<setting>-<n>/score.json (written by scripts/rounds/v63-grade.sh) and the
attempt logs for the submission IDs; writes raw metric, mapping and score per cell. Reviews (results/BASELINE_REVIEW.md)
are kept from an earlier publish; method notes come from NOTES. Cells without a score.json are left out.

Usage: v63-publish.py [TASK ...]   (default: every task version below)
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = {'conic': 'conic', 'exponential-v3': 'exponential', 'galaxy-gravity-v3': 'galaxy-gravity',
         'polarization-demosaicking-v2': 'polarization-demosaicking', 'species-tree-v3': 'species-tree',
         'trajectory-covariance-v2': 'trajectory-covariance', 'ihvp-sequence': 'ihvp-sequence',
         'ihvp-sequence-v2': 'ihvp-sequence', 'ihvp-sequence-v3': 'ihvp-sequence',
         'galaxy-gravity-v4': 'galaxy-gravity', 'ihvp-sequence-v4': 'ihvp-sequence',
         'polarization-demosaicking-v3': 'polarization-demosaicking', 'blackbox-cubature-v2': 'blackbox-cubature',
         'continuum-estimation': 'continuum-estimation', 'fluorescence-deconvolution': 'fluorescence-deconvolution',
         'pdhcg-qp-v1': 'pdhcg-qp', 'ihvp-sequence-v5': 'ihvp-sequence'}
SUBFOLDER = {'ihvp-sequence-v2': 'v2', 'ihvp-sequence-v3': 'v3', 'galaxy-gravity-v4': 'v4', 'ihvp-sequence-v4': 'v4',
             'polarization-demosaicking-v3': 'v3', 'blackbox-cubature-v2': 'v2', 'ihvp-sequence-v5': 'v5'}
# Candidate versions are listed after the task's current group (the board row) instead of replacing it.
CANDIDATE = {'ihvp-sequence-v5': '真实训练轨迹候选回放（锚点在看过结果后确定，单列，不替代 v4）'}  # later task versions within protocol v6.3 keep their own solutions subfolder
SETTING = {'single': '单轮', 'iterative': '迭代'}
NOTES = {
    ('blackbox-cubature-v2', 'iterative', 'claude-opus-5-5'): '先用查询检验结构再精确积分：可分离函数做一维自适应 Gauss–Kronrod，脊函数 g(w·x) 估方向后做一维 Chebyshev 积分，分段线性函数定位折面后用有理数精确积分，否则退回通用求积。fourier_ridge 是单方向脊函数、product_peak 可分离，两族因此接近精确（+1.56、+2.00）；16 次测试，public 39.7 → 83.0。超过 80，题目设计弱点已记录待出题人决定（results/OPEN_ITEMS.md）。',
    ('conic', 'single'): 'HiGHS 处理线性与二次规划并做 KKT 精修；一般锥用 Ruiz 均衡加齐次原始对偶内点法（Nesterov–Todd 缩放，指数锥用解析障碍函数）。',
    ('conic', 'iterative'): '结构捷径（零最优、可分盒约束 QP、代数不可行证书）后：LP 与小 QP 用 HiGHS，其余多面体 QP 用固定分解的算子分裂加有效集 KKT 精修；一般锥用齐次嵌入预估校正内点法（Ruiz 均衡、Nesterov–Todd 缩放），二阶锥用稀疏秩二展开，指数锥用解析障碍函数。重跑（原尝试受测试排队故障影响）。有一个回合被 ATR 本地代理当时默认的 600 秒连接空闲超时切断，损失约 10 分钟（基础设施问题，11-13 已修复）；原 OpenAI 通道现拒绝 Codex 请求，待限制解除后重跑。',
    ('exponential-v3', 'single'): 'C++ 实现：Chebyshev 展开把指数与 φ₁ 合成一条稳定 Clenshaw 递推，正向量加权 Gershgorin 收紧谱区间；小连通块扩展精度对角化。',
    ('exponential-v3', 'iterative'): '刚度小的会话直接用自适应 Taylor；对角和小连通块扩展精度谱分解；一般情形用合并的 Chebyshev 展开（行和与 Collatz 界给谱区间，Reinsch 形式的 Clenshaw 递推）。重跑（原尝试受测试排队故障影响）。',
    ('galaxy-gravity-v3', 'single'): '笛卡尔与调和多极展开结合的 FMM，远场也保留 Plummer 软化修正；按相互作用选展开阶数，按进程 CPU 时间控制工作量。',
    ('galaxy-gravity-v3', 'iterative'): '核拆分：牛顿部分用调和多极展开，软化差项用笛卡尔修正；按调用规划精度，剩余 CPU 时间对部分粒子做完整直接求和。',
    ('polarization-demosaicking-v2', 'single'): '低通估计通道差、把马赛克细节转移到四个通道，再用约 7.7k 参数的残差卷积网络修正（只用 2 张公开样例加增强训练），最后做以强度为引导的局部仿射回归。',
    ('polarization-demosaicking-v2', 'iterative'): '逐格双线性插值加通道差的偏振估计，用全部测量恢复共同细节引导，再用小型残差卷积网络修正，带结构门控与残差校准。',
    ('species-tree-v3', 'single'): '支持度加权的内部节点距离 NJ 起点、加权四元组 NNI；最多 60 个基因座自助树，再用动态规划求兼容分裂频率最大的共识树。',
    ('species-tree-v3', 'iterative'): '支持度加权的内部节点距离 NJ 起点、加权四元组 NNI；四元组按悬挂枝长降权（加权方式与 Weighted ASTRAL 相同）。有一个回合被 ATR 本地代理当时默认的 600 秒连接空闲超时切断，损失约 10 分钟（基础设施问题，11-13 已修复）；原 OpenAI 通道现拒绝 Codex 请求，待限制解除后重跑。',
    ('trajectory-covariance-v2', 'single'): '白化后自适应窗宽的重叠批均值；README 写明把估计除以了轨迹长度，与题目要求的尺度不符，误差约为强参照的 10 倍。会话只用了 2 分钟。',
    ('trajectory-covariance-v2', 'iterative'): '白化后：有限状态链拟合正则化转移矩阵并解 Markov 泊松方程；其余拟合至多 32 阶的条件向量自回归，按信息准则加权平均零频谱估计；可逆性诊断通过时与共同动力学、分组动力学拟合比较。重跑（原尝试受测试排队故障影响）。',
    ('ihvp-sequence', 'single'): '用少量 HVP 查询识别生成结构（带符号 DCT 对角化、对角项与 rank-12 修正），再精确求解；误差到机器精度，每条序列约 1.7 秒。题目设计缺陷，已触发请出题人修改（`tasks/ihvp-sequence/analysis/INTEGRATION.md`）。',
    ('ihvp-sequence', 'iterative'): '同一思路：从查询恢复 DCT 坐标符号与低秩部分，构造精确预条件后求解；1 次测试即得 public 满分并提交。题目设计缺陷，已触发请出题人修改。',
    ('ihvp-sequence-v2', 'single'): '工作区不含生成代码；Galerkin 回收 CG：在历史解张成的子空间上投影热启动，再用 C 写的 CG 内核修正。没有识别矩阵结构；困难序列被“解序列沿时间平滑”绕开，误差为参照的约 1/180。',
    ('ihvp-sequence-v3', 'single'): '通用回收 CG：最近 4 个解与搜索方向的 Galerkin 投影热启动，加右端项历史预测，C 内核做投影 CG，阶段切换时多分时间，最后做能量最优缩放；没有识别矩阵结构。原有 15 条 98.9 分，新增 15 条 65.1 分（五个族 63–69 分）（`tasks/ihvp-sequence/analysis/V3_SOL_ANALYSIS.md`）。',
    ('ihvp-sequence-v3', 'iterative'): '回收 CG 加三层预条件选择：对角加低秩精确拟合、对角探测、从 HVP 逐层学出 14 层蝶形正交基后做对角预条件。原有 15 条 98.1 分，新增 15 条 74.4 分：学习基只在两个 butterfly 族成立（87–88 分），DCT 三族退回无预条件（66–71 分）。超过 80，已请出题人决定（`tasks/ihvp-sequence/analysis/V3_SOL_ANALYSIS.md`）。',
    ('ihvp-sequence-v4', 'single'): '通用回收 CG（历史子空间投影加 C 内核 CG），没有识别矩阵结构。',
    ('ihvp-sequence-v4', 'iterative'): '按截止时间分配的延续 CG（8 次测试，public 50.9 → 63.2），没有识别矩阵结构。',
    ('ihvp-sequence-v4', 'single', 'claude-opus-5-5'): '热启动 CG 加小规模回收紧缩子空间，没有识别矩阵结构。',
    ('ihvp-sequence-v4', 'iterative', 'claude-opus-5-5'): '前一步解与修正量的 Galerkin 投影加 CG（6 次测试，public 50.1 → 62.5），没有识别矩阵结构。',
    ('ihvp-sequence-v4', 'single', 'intern-s2'): '带 L-BFGS 预条件器的 PCG 加子空间投影、按剩余时间分配迭代，没有识别矩阵结构；误差只略好于冷启动 CG。',
    ('conic', 'single', 'qwen3.8-max-0902'): '原生 C 求解器，但提交缺少题面要求的 README.md，提交检查未通过，650 例均未运行。',
    ('ihvp-sequence-v2', 'iterative'): '工作区不含生成代码；最近 16 个解做 QR，在子空间上精确投影得到初始解，再跑 CG，按剩余步数分配时间（阶段突变时多给），Gauss–Radau 上界停止。没有识别矩阵结构；public 分数 83.7 → 99.8。',
}
# Infrastructure failures make an attempt unmeasured; its replacement is published once it has a result. Causes:
# the 1100 s idle cut on the path to the model gateway (fixed by streaming upstream requests), a cut upstream stream
# that made the client retry without streaming (fixed by the transport retrying cut streams), and Claude Code
# silently re-sending a request after about 13 minutes of pings (fixed by its stream timeouts set to the window).
# Cell -> (attempt to publish, note).
CUT_1 = '第 1 次尝试中有回合被模型网关链路的 1100 秒空闲超时切断'
RESEND = '客户端约 13 分钟收不到内容时静默重发、原请求作废'
RERUN = {
    **{cell: (2, CUT_1 + '（基础设施故障，记未测）；本格为上游改用流式后的第 2 次尝试。') for cell in (
        ('galaxy-gravity-v4', 'single', 'qwen3.8-max-0902'), ('conic', 'single', 'qwen3.8-max-0902'),
        ('conic', 'single', 'hy4'), ('galaxy-gravity-v4', 'single', 'hy4'),
        ('ihvp-sequence-v4', 'iterative', 'qwen3.8-max-0902'))},
    **{cell: (2, f'第 1 次尝试中有回合因{RESEND}（基础设施故障，记未测）；本格为修复后的第 2 次尝试。') for cell in (
        ('conic', 'iterative', 'qwen3.8-max-0902'), ('exponential-v3', 'iterative', 'qwen3.8-max-0902'),
        ('exponential-v3', 'single', 'qwen3.8-max-0902'), ('conic', 'iterative', 'hy4'),
        ('galaxy-gravity-v4', 'iterative', 'hy4'))},
    **{cell: (2, '第 1 次尝试中有回合因客户端约 300 秒等不到响应头而重发、原请求作废（基础设施故障，记未测）；本格为修复后的第 2 次尝试。')
       for cell in (('conic', 'iterative', 'claude-opus-5-5'), ('exponential-v3', 'iterative', 'claude-opus-5-5'),
                    ('galaxy-gravity-v4', 'single', 'claude-opus-5-5'), ('ihvp-sequence-v4', 'iterative', 'claude-opus-5-5'))},
    ('species-tree-v3', 'single', 'intern-s2'): (2, '第 1 次尝试中一次上游流中断，传输层自动重试成功，但该回合多耗约 22 分钟'
        '（基础设施问题，记未测）；本格为第 2 次尝试。'),
    ('species-tree-v3', 'single', 'qwen3.8-max-0902'): (3, CUT_1 + f'；第 2 次尝试中有回合因{RESEND}'
        '（均为基础设施故障，记未测）；本格为修复后的第 3 次尝试。'),
    **{cell: (4, CUT_1 + f'；第 2、3 次尝试中有回合因上游流中断后改发非流式请求被切断，或因{RESEND}'
              '（均为基础设施故障，记未测）；本格为修复后的第 4 次尝试。') for cell in (
        ('species-tree-v3', 'iterative', 'qwen3.8-max-0902'), ('galaxy-gravity-v4', 'iterative', 'qwen3.8-max-0902'))},
}
AZURE_2 = ('第 1 次尝试中有回合被 ATR 本地代理当时默认的 600 秒连接空闲超时切断（基础设施问题，记未测）；'
           '本格为 sol 改走 AI Hub azure 通道后的第 2 次尝试。')
RERUN.update({('conic', 'iterative', 'gpt-6.1-sol'): (2, AZURE_2), ('species-tree-v3', 'iterative', 'gpt-6.1-sol'): (2, AZURE_2)})
# qwen iteratives: both attempts lost time to upstream (Alibaba Bailian) streams that sent nothing for about 600 s
# (AIHub records them as successful with zero tokens); user decision 2026-11-18: the attempt that lost less time stands.
QWEN_STALL = ('两次尝试都有回合遇到上游（阿里百炼）流约 600 秒没有任何输出后断开（AIHub 记为成功、0 token，无其他供应商可换），'
              '第 1 次损失约 {0} 分钟、第 2 次约 {1} 分钟；按用户决定（11-18）取损失较少的第 {2} 次为正式格。')
RERUN.update({('continuum-estimation', 'iterative', 'qwen3.8-max-0902'): (2, QWEN_STALL.format(70, 30, 2)),
              ('fluorescence-deconvolution', 'iterative', 'qwen3.8-max-0902'): (2, QWEN_STALL.format(50, 40, 2))})
NOTES[('blackbox-cubature-v2', 'iterative', 'qwen3.8-max-0902')] = QWEN_STALL.format(23, 49, 1)
# Intern S2: its first round stands as the formal result (user decision 2026-11-17); the 11-17 second singles are noted only.
INTERN_NOTE = ('书生 S2 端点生成约每秒 16–25 个 token（opus 约 90、hy4 约 50、qwen 约 37），单轮时间多耗在等生成上；迭代 7 次均未调用'
               '测试反馈、3 次缺 README 或 run.sh。11-17 以最多 2 个并发、模型卡采样参数复核 4 个单轮（polarization、trajectory 无提交，'
               'galaxy 提交不完整，ihvp 发散），均为 0 分，不替换正式结果（results/OPEN_ITEMS.md）。')
for _task in ('conic', 'exponential-v3', 'galaxy-gravity-v4', 'polarization-demosaicking-v2', 'species-tree-v3',
              'trajectory-covariance-v2', 'ihvp-sequence-v4'):
    for _setting in ('single', 'iterative'):
        NOTES[(_task, _setting, 'intern-s2')] = ' '.join(filter(None, (NOTES.get((_task, _setting, 'intern-s2')), INTERN_NOTE)))
_cell = ('species-tree-v3', 'single', 'intern-s2')
RERUN[_cell] = (RERUN[_cell][0], RERUN[_cell][1] + INTERN_NOTE)
INTERN_FILL = ('11-18 补跑：同一把 key 两个并发（裸测单请求 32.7 token/秒、两并发每路约 31），会话内生成约 17–23 token/秒，'
               '无上游错误、断流或重试（results/OPEN_ITEMS.md）。')
INTERN_ITER = ('11-18 前两次迭代都在约 3 小时时因书生端点按自己的粗估判定上下文超限（HTTP 400，约 910 KB 请求、端点报告约 12 万 token）'
               '而退出，Claude Code 在这种单轮长会话中没有压缩；基础设施问题，两次均记未测。11-19 起 Claude Code 自动压缩阈值设为 100000，'
               '并由传输层在请求超过 800 KB 时清理最早的工具输出（最近 8 条消息不动），本格为此后的尝试。')
for _task in ('polarization-demosaicking-v3', 'blackbox-cubature-v2', 'continuum-estimation', 'fluorescence-deconvolution'):
    for _setting in ('single', 'iterative'):
        NOTES[(_task, _setting, 'intern-s2')] = INTERN_FILL
for _task in ('polarization-demosaicking-v3', 'blackbox-cubature-v2', 'continuum-estimation'):
    NOTES[(_task, 'iterative', 'intern-s2')] = ' '.join((INTERN_FILL, INTERN_ITER))
NOTES[('species-tree-v3', 'iterative', 'intern-s2')] = ' '.join((INTERN_FILL, '11-16 的第 1 次在 11-17 停书生时中止，未测；', INTERN_ITER))
SAMPLE_FEEDBACK = ('出题人只提供带真值的公开样例和 private 集，没有留出的 public 集：迭代时的 public 反馈就是这些样例上的成绩，'
                   '模型本地也能算出，不含留出信息（results/OPEN_ITEMS.md）。')
for _task in ('continuum-estimation', 'fluorescence-deconvolution'):
    for _key in ((_task, 'iterative'), *((_task, 'iterative', _model) for _model in ('claude-opus-5-5', 'qwen3.8-max-0902', 'hy4', 'intern-s2',
                                                                                   'gpt-6.2-sol', 'gpt-6-astra', 'qwen4-max',
                                                                                   'minimax-m3.1-flash'))):
        NOTES[_key] = ' '.join(filter(None, (NOTES.get(_key), SAMPLE_FEEDBACK)))
# continuum iteratives (user decision 2026-11-26): the feedback became 15 held-out primary cases, scored on the other 45;
# every iterative cell is rerun. PENDING: shown on the earlier attempt (regraded on the 45 cases) until the rerun has a result.
CONTINUUM_SPLIT = ('此前尝试的测试反馈用的是工作区里带真值的 8 个样例（迭代中最后一次 public 82–97 分，私有 23–73 分）；11-26 起 60 个 '
                   'primary 例按真值排序每 4 个取 1 作为不发给模型的反馈集（15 例），正式计分用其余 45 例，本格为此后的第 {0} 次尝试'
                   '（results/OPEN_ITEMS.md）。')
PENDING = {}
for _model in ('gpt-6.2-sol', 'gpt-6-astra', 'gpt-6.1-sol', 'claude-opus-5-5', 'hy4', 'qwen4-max', 'qwen3.8-max-0902', 'minimax-m3.1-flash'):
    _n = 3 if _model == 'qwen3.8-max-0902' else 2
    RERUN[('continuum-estimation', 'iterative', _model)] = (_n, CONTINUUM_SPLIT.format(_n))
    PENDING[('continuum-estimation', 'iterative', _model)] = (
        f'反馈集改为 15 个留出例后的第 {_n} 次尝试进行中；此处暂列此前尝试在 45 个计分例上的成绩（其测试反馈是带真值的样例）。')
# polarization iteratives (user decision 2026-11-26): the feedback was the five shipped Wen5 scenes; it became 16 held-out
# formal scenes (Qiu 4, Wen 12), scored on the other 116; every iterative cell is rerun.
POL_SPLIT = ('此前尝试的测试反馈用的是发给模型的 5 个 Wen5 样例，没有 Qiu 场景（qwen4 在样例上拟合的回归器反馈 43 分、私有 0 分）；'
             '11-26 起 132 个正式场景中按强参照误差分层取出 16 景（Qiu 4、Wen 12）作为不发给模型的反馈集，正式计分用其余 116 景，'
             '本格为此后的第 2 次尝试（results/OPEN_ITEMS.md）。')
for _model in ('gpt-6.2-sol', 'gpt-6-astra', 'gpt-6.1-sol', 'claude-opus-5-5', 'hy4', 'qwen4-max', 'qwen3.8-max-0902', 'minimax-m3.1-flash'):
    RERUN[('polarization-demosaicking-v3', 'iterative', _model)] = (2, POL_SPLIT)
    PENDING[('polarization-demosaicking-v3', 'iterative', _model)] = (
        '反馈集改为 16 个留出场景后的第 2 次尝试进行中；此处暂列此前尝试在 116 个计分场景上的成绩（其测试反馈是 Wen5 样例）。')
# gpt-6-astra runs on the user's ChatGPT Pro account through the Codex backend instead of AI Hub (user decision 2026-11-21).
ASTRA_ROUTE = ('经用户的 ChatGPT Pro 账号额度（Codex 后端）运行，不走 AI Hub：同一模型、推理 max；上下文窗口取该后端上限 872k token'
               '（其他 OpenAI 行经 AI Hub 为 1.05M，6.1-sol 与 6.2-sol 单次请求最大约 62 万 token）。该后端生成约慢 30%：同一提示的'
               '成对实测为每秒 47、51 个 token，AI Hub azure 为 69、74，限时内可做的工作相应少一些（用户 11-21 选择此通道）。')
for _task in TASKS:
    for _setting in ('single', 'iterative'):
        NOTES[(_task, _setting, 'gpt-6-astra')] = ' '.join(filter(None, (NOTES.get((_task, _setting, 'gpt-6-astra')), ASTRA_ROUTE)))
# gpt-6.2-sol conic iterative: tests 2-8 of attempt 1 failed to build on the evaluator (build.sh used the zlib headers
# that Node.js puts in the dev container, absent from the verifier image), and the feedback reported that as an
# evaluator-side failure, not a verdict (fixed 2026-11-19 in arena/public_v63.py). Attempt 1 is unmeasured; it shows as
# such until the rerun has a result.
SOL62_FEEDBACK = ('第 1 次尝试的第 2–8 次测试在评测方构建失败（`build.sh` 用了开发容器里 Node.js 自带的 zlib 头文件，评测镜像没有），'
                  '反馈流程却把它报成“未测（评测方原因，不是对提交的判定）”，模型因此没有收到构建失败的信息；评测方问题，记未测')
RERUN[('conic', 'iterative', 'gpt-6.2-sol')] = (2, SOL62_FEEDBACK + '；本格为修复反馈分类后的第 2 次尝试。')
UNMEASURED = {('conic', 'iterative', 'gpt-6.2-sol'): SOL62_FEEDBACK + '；修复反馈分类后的第 2 次尝试进行中。'}
# gpt-6.2-sol galaxy-gravity-v4 iterative: the v6 launcher offered Codex's provider-side web_search until 2026-11-26,
# and attempt 1 ran one search (completed upstream, query not kept). User decision 2026-11-26: offline runs no longer
# offer provider-side web tools and this cell is rerun; attempt 1 shows as unmeasured until the rerun has a result.
SOL62_WEB = ('第 1 次尝试中模型用 Codex 自带的 web_search 做过一次联网搜索（在厂商侧执行，不经过断网的容器；查询内容未保留）；'
             '自 11-26 起离线运行不再提供厂商侧联网工具（用户决定）')
RERUN[('galaxy-gravity-v4', 'iterative', 'gpt-6.2-sol')] = (2, SOL62_WEB + '；本格为关闭联网工具后的第 2 次尝试。')
UNMEASURED[('galaxy-gravity-v4', 'iterative', 'gpt-6.2-sol')] = SOL62_WEB + '；关闭联网工具后的第 2 次尝试进行中。'
# species, exponential and galaxy iteratives (user decision 2026-11-27): their feedback gaps were fixed (species 11-26:
# 20 held-out cases instead of 5 including the 2 samples; exponential 11-26: 43 cases with p7 and p10, scored per case;
# galaxy 11-27: a 4,194,304-particle snapshot), and every iterative cell is rerun (scripts/rounds/feedback-fix-reruns.sh).
FEEDBACK_FIX = {
    'species-tree-v3': ('此前尝试的测试反馈是 5 个用例，其中 2 个就是工作区里带真值的样例；11-26 起反馈改为 20 个不发给模型的用例'
                        '（与私有集同一组模拟参数、不同种子），本格为此后的第 {0} 次尝试（results/OPEN_ITEMS.md）。',
                        '反馈改为 20 个留出用例后的第 {0} 次尝试进行中；此处暂列此前尝试的成绩（其反馈含工作区样例）。'),
    'exponential-v3': ('此前尝试的测试反馈只有 p4 精度档的 16 例，p7、p10 档的失败在反馈里看不到，且反馈按旧的全有或全无规则计分；'
                       '11-26 起反馈为 43 例（每个结构族第一个公开母矩阵的 p4、p7、p10 全部会话加 2 个对照例），按逐例规则计分，'
                       '本格为此后的第 {0} 次尝试（results/OPEN_ITEMS.md）。',
                       '反馈扩到 43 例（含 p7、p10）后的第 {0} 次尝试进行中；此处暂列此前尝试的成绩（其反馈只有 p4）。'),
    'galaxy-gravity-v4': ('此前尝试的测试反馈最大只到 1,048,576 粒子，4,194,304 粒子档的内存与 CPU 问题在反馈里看不到；11-27 起反馈加入'
                          '1 张 4,194,304 粒子快照（每例跑 1 次），本格为此后的第 {0} 次尝试（results/OPEN_ITEMS.md）。',
                          '反馈加入 4,194,304 粒子快照后的第 {0} 次尝试进行中；此处暂列此前尝试的成绩（其反馈最大到 1,048,576 粒子）。'),
}
FEEDBACK_FIX_ATTEMPT = {
    'species-tree-v3': {'gpt-6.2-sol': 2, 'gpt-6-astra': 2, 'gpt-6.1-sol': 3, 'claude-opus-5-5': 2, 'hy4': 2, 'qwen4-max': 2,
                        'qwen3.8-max-0902': 5, 'minimax-m3.1-flash': 2},
    'exponential-v3': {'gpt-6.2-sol': 2, 'gpt-6-astra': 2, 'gpt-6.1-sol': 2, 'claude-opus-5-5': 3, 'hy4': 2, 'qwen4-max': 2,
                       'qwen3.8-max-0902': 3, 'minimax-m3.1-flash': 2},
    'galaxy-gravity-v4': {'gpt-6.2-sol': 3, 'gpt-6-astra': 2, 'gpt-6.1-sol': 2, 'claude-opus-5-5': 2, 'hy4': 3, 'qwen4-max': 2,
                          'qwen3.8-max-0902': 5, 'minimax-m3.1-flash': 2},
}
for _task, _attempts in FEEDBACK_FIX_ATTEMPT.items():
    for _model, _n in _attempts.items():
        _note = FEEDBACK_FIX[_task][0].format(_n)
        if (_task, _model) == ('galaxy-gravity-v4', 'gpt-6.2-sol'):
            _note = SOL62_WEB + '；第 2 次尝试关闭了联网工具；' + _note
        RERUN[(_task, 'iterative', _model)] = (_n, _note)
        PENDING[(_task, 'iterative', _model)] = FEEDBACK_FIX[_task][1].format(_n)
# gpt-6.2-sol polarization v3: the first gradings ran beside 6 development attempts and other gradings, and scenes
# timed out at the 1 s limit. As in the 11-14 and 11-17 cases: unmeasured, regraded once no development slot is leased
# (scripts/rounds/regrade-dev-idle.sh keeps the first result as score.first.json); the regrade stands once it exists.
POL_LOAD = ('第一次评分时机器上同时有 6 个开发会话和 conic、galaxy、species 等评分，{0} 景在 1.0 秒单景时限超时（通过的景中位约 0.7 秒）；'
            'polarization 的计时受开发负载影响（11-12 实测最坏 2.0–2.8 倍），按 11-14、11-17 的做法记未测，开发槽全空时对同一冻结提交重评，')
REGRADE = {('polarization-demosaicking-v3', 'single', 'gpt-6.2-sol'): POL_LOAD.format(3),
           ('polarization-demosaicking-v3', 'iterative', 'gpt-6.2-sol'): POL_LOAD.format(20),
           # first scene only, with the development slots idle: the cold start of 11-06 and 11-10, not load
           # every scene within the limit in at least two gradings, per-scene RMSE identical across them (user, 11-23)
           ('polarization-demosaicking-v3', 'iterative', 'gpt-6-astra'):
               '第一次评分只有第一景 qiu/ball 超时（1.0007 秒，首景冷启动）；第一次重评时同机有其他项目的进程启动，31 景成段超时；'
               '第二次重评（机器安静）只有 qiu/lock 一景因偶发停顿超时（1.0009 秒，其余最长 0.755 秒）。各次评分逐景 RMSE 完全相同'
               '（输出是确定的），每景都至少两次在时限内跑完；按用户确认（11-23），取第二次重评，qiu/lock 用第一次评分中的结果'
               '（0.737 秒）补齐，合成完整评分。评测环境约 2.5% 的场景有 0.2–0.3 秒的停顿，已记入 results/OPEN_ITEMS.md 待反馈出题人；',
           # first grading beside a 7-shard exponential regrade; a quiet regrade hit an environment stall; the third
           # grading ran every scene within the limit (results/OPEN_ITEMS.md, 11-26)
           ('polarization-demosaicking-v3', 'iterative', 'qwen4-max'):
               '第一次评分（开发槽全空）与 exponential 的 7 分片重评同时进行，前 9 景连续超时（约 1.0005 秒，其余约 0.85 秒），'
               '记未测；机器空闲时重评又有 4 景因环境停顿超时（约 1.001 秒）；第三次评分 132 景都在时限内（最长 0.911 秒），',
           # two shards flagged slow_slot; a two-case remeasure passed k1 at 10.3 s CPU against the 15 s cap
           ('exponential-v3', 'single', 'qwen4-max'):
               '第一次评分有两个分片负载报警（探针 1.38、1.42），判无效的两例都是 CPU 上限；只重测这两例时 '
               '`syn_geometric_p10_k1` 以 10.3 秒 CPU 通过（原评分三次触 15 秒上限），故记未测，开发槽全空时整套重评：'
               'k1 通过，`syn_geometric_p10_k16` 复用仍超 150 秒 CPU 上限，按逐例计分该例记 0，'}
NOTES[('conic', 'iterative', 'qwen4-max')] = (
    '第 1 次运行在第 75 分钟中断：模型为清理自己卡住的 `python -` 进程执行 `pkill -f \'python -\'`，一并杀掉了容器内命令行为 '
    '`/opt/atr-venv/bin/python -c …` 的 harness 启动器（harness 隔离缺陷，记未测）；本格为同编号重跑。')
NOTES[('galaxy-gravity-v4', 'iterative', 'qwen4-max')] = (
    '提交含 985 MB 预计算表，私有评分时 7 个分片同时解包，评测器 30 秒的 `docker create` 超时（评测设施问题，两次记未测）；'
    '分片错开 90 秒启动后评完：419 万粒子各例超出 2 GiB 内存上限，100 万粒子 jaffe 一例超 CPU 上限，这 5 组按经典树误差计入。')
NOTES[('galaxy-gravity-v4', 'iterative', 'minimax-m3.1-flash')] = (
    '419 万粒子 disc 组 3 例三次重复都超 CPU 上限（100–110 秒；所在分片负载探针 1.06–1.10），该组按经典树误差计入；'
    '100 万粒子各组和 419 万粒子其余组的相对误差约 1（disc 组 22），总误差高于 \\(E_0\\)。')
NOTES[('galaxy-gravity-v4', 'single', 'minimax-m3.1-flash')] = (
    '原评分构建失败：`build.sh` 把 CPU 检测小程序编译到 `/tmp` 再执行，评测构建容器的 `/tmp` 不可执行（Docker tmpfs 默认 noexec），'
    '开发容器可执行，题面未写明，属评测设施与开发环境不一致；11-26 起构建容器的 `/tmp` 可执行，同一冻结提交重评，以重评为准'
    '（原结果留作 `score.tmp-noexec.json`，results/OPEN_ITEMS.md）。60 分钟到时截停。')
TITLE = '协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）'
LINEAR = '\\(E \\ge U\\) 记 0；\\(W \\le E < U\\) 为 \\(50\\,(U-E)/(U-W)\\)；\\(E < W\\) 为 \\(100 - 50\\,E/W\\)'


# (model, attempt-log tag), roughly in board order (task-page rows follow it); NOTES keys add the model for opus
MODELS = [('gpt-6.2-sol', 'sol62'), ('gpt-6-astra', 'astra'), ('gpt-6.1-sol', 'sol'), ('claude-opus-5-5', 'opus'), ('hy4', 'hy4'),
          ('qwen4-max', 'qwen4'), ('qwen3.8-max-0902', 'qwen'), ('minimax-m3.1-flash', 'minimax'), ('glm-5.4', 'glm'),
          ('intern-s2', 'intern')]


def submission_of(task: str, setting: str, n: int, tag: str = 'sol') -> str | None:
    log = ROOT / f'var/stores/store/reruns/v63-{tag}-{setting}-{task}-{n}.log'
    rows = [json.loads(line) for line in log.read_text().splitlines() if 'develop_finished' in line] if log.exists() else []
    return rows[-1]['submission'] if rows else None


def baseline_anchors(task: str) -> dict:
    root = ROOT / 'var/stores/store-v63/baselines' / task
    active = json.loads((root / (root / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())
    return {w: active['metrics']['parts'][w]['raw_error'] for w in ('weak', 'strong')}


CUBATURE_FAMILY = {'fourier_ridge': 'fourier', 'kinked_ridge_sum': '折点 ridge', 'product_peak': '乘积峰', 'polynomial_ridge': '多项式'}


def cell(task: str, score: dict) -> tuple[str, str]:
    family = TASKS[task]
    if family == 'pdhcg-qp':
        scored = score.get('scored') or {}
        if scored.get('status') == 'invalid_submission':
            return f'无效提交（{scored.get("reason") or "构建失败"}）', '构建失败记 0'
        return (f'正确率 \\(p = {scored["p"]:.3f}\\)，罚时几何均值 \\(g = {scored["g"]:.3g}\\) 秒'
                f'（参照 PDHCG：\\(g_{{ref}} = {scored["g_ref"]:.3g}\\) 秒）',
                '\\(100\\,p/(1+g/g_{ref})\\)，参照全部正确时为 50；失败例按时限计时')
    if family == 'blackbox-cubature':
        if score.get('status') == 'submission_failure':
            return f'无效提交（{(score.get("reasons") or ["有用例失败"])[0]}）', '任一用例失败、超时或违反协议记 0'
        parts = '，'.join(f'{CUBATURE_FAMILY.get(f, f)} {v:+.2f}' for f, v in sorted(score['families'].items()))
        return (f'增益 \\(G = {score["gain"]:.3f}\\)（{parts}），{score["cases"]} 例，单例最长 {score["max_wall_s"]:.1f} 秒'
                f'（普通蒙特卡洛 \\(G_0 = {score["mc_gain"]:.3f}\\)，HumanPortfolio-v1 为 0）',
                '与参照同例配对的 log10 相对误差增益（每例截到 ±2），按族×维度平均：\\(G \\ge 0\\) 为 \\(50 + 25\\,G\\)；'
                '\\(G < 0\\) 为 \\(\\max(0,\\,50\\,(G - G_0)/(-G_0))\\)')
    if family == 'conic':
        if score.get('p0') is not None:
            return (f'{score["successes"]} / {score["cases"]}（\\(p = {score["p"]:.3f}\\)；弱参照 HiGHS \\(p_0 = {score["p0"]:.3f}\\)，'
                    f'参照 \\(p_B = {score["p_B"]:.3f}\\)）',
                    '\\(p \\le p_0\\) 记 0；\\(p_0 < p \\le p_B\\) 为 \\(50\\,(p-p_0)/(p_B-p_0)\\)；高于 \\(p_B\\) 为 \\(50 + 50\\,(p-p_B)/(1-p_B)\\)')
        return (f'{score["successes"]} / {score["cases"]}（\\(p = {score["p"]:.3f}\\)，参照 \\(p_B = {score["p_B"]:.3f}\\)）',
                '\\(p \\le 0.9\\,p_B\\) 记 0；否则 \\(50\\,(p-0.9p_B)/(0.1p_B)\\)，高于 \\(p_B\\) 为 \\(50 + 50\\,(p-p_B)/(1-p_B)\\)')
    if family == 'exponential':
        failed = score.get('failed_cases') or {}
        reasons = '，'.join(f'`{r}` {n}' for r, n in Counter(failed.values()).most_common())
        if score.get('rule') == 'per-case':
            if score.get('status') != 'ok':
                return f'无效提交（对照例不合格；{len(failed)} 例不合格：{reasons}）', '对照例不合格记 0'
            valid, scored = score['valid_scored_cases'], score['scored_cases']
            whole = (f'，总体 \\(R = {score["relative_time"]:.3f}\\)' if score.get('relative_time') is not None
                     else f'；不合格：{reasons}')
            return (f'{valid}/{scored} 个计分例合格{whole}（\\(r\\)：逐例用时 ÷ 7 方法参照组该例最快成员）',
                    '逐例 \\(100/(1+r)\\)，不合格例记 0；按精度、场景、族、母矩阵逐层等权平均')
        if score.get('status') != 'ok':
            return f'无效提交（{len(failed)} 例不合格）', '任一用例不合格记 0'
        return (f'74/74 合格，\\(R = {score["relative_time"]:.3f}\\)（对 7 方法参照组逐例最快成员）',
                '\\(100/(1+R)\\)')
    candidate = score.get('candidate') or {}
    raw = candidate.get('raw_error')
    if task == 'galaxy-gravity-v4':
        log_map = ('对数刻度：\\(E < E_{50}\\) 为 \\(50 + 50\\,\\log(E_{50}/E)/\\log(E_{50}/10^{-15})\\)；'
                   '\\(E_{50} \\le E < E_0\\) 为 \\(50\\,\\log(E_0/E)/\\log(E_0/E_{50})\\)')
        anchors = (f'经典树 theta 0.8：\\(E_0 = {score["weak"]["raw_error"]:.3g}\\)，'
                   f'改进树 theta 0.7：\\(E_{{50}} = {score["strong"]["raw_error"]:.3g}\\)')
        failed = score.get('failed_cells') or []
        if score.get('rule') == 'per-group' and len(failed) < len(score.get('per_group_cells') or {}):
            raw = score['per_group_raw_error']
            named = f'（{"、".join(failed)}）' if len(failed) <= 5 else ''
            lost = f'；{len(failed)} 组失败{named}，按经典树该组误差计入' if failed else ''
            return (f'\\(E = {raw:.3g}\\)，20 组（4 档规模 × 5 类）等权{lost}；CPU 用量最多到逐例时限的 '
                    f'{100 * (candidate.get("max_cpu_fraction") or 0):.0f}%（{anchors}）',
                    log_map + '；失败或超限的组按经典树参照该组的误差计')
        if score.get('score') == 0.0 and raw is None:
            reasons = '；'.join(dict.fromkeys(filter(None, candidate.get('reasons') or [])))
            if score.get('rule') == 'per-group':
                return f'无效提交（{reasons or "20 组全部失败"}）', '构建失败或全部组失败记 0'
            return f'无效提交（{reasons or "有调用失败"}）', '任一调用失败记 0'
        return (f'\\(E = {raw:.3g}\\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 {100 * (candidate.get("max_cpu_fraction") or 0):.0f}%'
                f'（{anchors}）', log_map)
    if family == 'galaxy-gravity':
        anchors = baseline_anchors(task)
        return (f'\\(E = {raw:.3g}\\)，单次最多 {candidate.get("max_cpu_s", 0):.2f} CPU 秒（出题方 Treecode2：\\(U = {anchors["weak"]:.3g}\\)，\\(W = {anchors["strong"]:.3g}\\)）',
                '对数刻度：\\(E < W\\) 为 \\(50 + 50\\,\\log(W/E)/\\log(W/10^{-15})\\)；\\(W \\le E < U\\) 为 \\(50\\,\\log(U/E)/\\log(U/W)\\)')
    if family == 'ihvp-sequence':
        # v4 anchors come with the score (same private measurement for every model: scripts/analyze/ihvp-v4-runs.py)
        anchors = score.get('anchors') or baseline_anchors(task)
        if raw is None:
            return f'无效提交（{candidate.get("reason") or "有序列失败"}）', '任一序列失败记 0'
        weak, strong = ('Warm CG', 'Cold CG') if task == 'ihvp-sequence-v5' else ('CG', 'L-BFGS-PCG')
        return (f'\\(E = {raw:.4g}\\)，最长序列 {candidate.get("max_wall_s") or 0:.1f} 秒（{weak}：\\(U = {anchors["weak"]:.4f}\\)，'
                f'{strong}：\\(W = {anchors["strong"]:.4f}\\)）', LINEAR)
    if family == 'continuum-estimation':
        anchors = baseline_anchors(task)
        if candidate.get('status') == 'submission_failure':
            return f'无效提交（{candidate.get("reason") or "有用例失败"}）', '任一用例无效、缺失、超时或失败记 0'
        return (f'损失 \\(E = {raw:.6g}\\)，{candidate.get("cases")} 例，单例最长 {candidate.get("max_wall_s") or 0:.1f} 秒'
                f'（一阶 Richardson：\\(U = {anchors["weak"]:.6g}\\)，GRE：\\(W = {anchors["strong"]:.6g}\\)）', LINEAR)
    if family == 'fluorescence-deconvolution':
        anchors = baseline_anchors(task)
        if raw is None:
            return f'无效提交（{candidate.get("reason") or "有用例失败"}）', '任一用例失败或超时记 0'
        return (f'nMSE \\(E = {raw:.5f}\\)，{candidate.get("cases")} 例，单例最长 {candidate.get("max_wall_s") or 0:.2f} 秒'
                f'（RL20：\\(U = {anchors["weak"]:.5f}\\)，SHB10：\\(W = {anchors["strong"]:.5f}\\)）', LINEAR)
    if family == 'polarization-demosaicking':
        anchors = baseline_anchors(task)
        if raw is None:
            return f'无效提交（{candidate.get("reason") or "有场景失败"}）', '任一计分场景失败记 0'
        scenes = f'，{candidate["expected_cases"]} 景' if candidate.get('expected_cases') else ''
        return (f'RMSE \\(E = {raw:.4f}\\){scenes}（\\(U = {anchors["weak"]:.4f}\\)，\\(W = {anchors["strong"]:.4f}\\)）',
                LINEAR)
    weak, strong = score['weak']['raw_error'], score['strong']['raw_error']
    name = 'nRF' if family == 'species-tree' else '白化误差'
    return f'{name} \\(E = {raw:.4f}\\)（\\(U = {weak:.4f}\\)，\\(W = {strong:.4f}\\)）', LINEAR


def main():
    import sys
    chosen = sys.argv[1:] or list(TASKS)
    for task, family in TASKS.items():
        if task not in chosen:
            continue
        path = ROOT / 'tasks' / family / 'meta.json'
        meta = json.loads(path.read_text())
        version = {'polarization-demosaicking': 'v2', 'trajectory-covariance': 'v2', 'ihvp-sequence': 'v1',
                   'continuum-estimation': 'v1', 'fluorescence-deconvolution': 'v1', 'pdhcg-qp': 'v1'}.get(family, 'v3')
        version = SUBFOLDER.get(task, version)
        old = next((g for g in meta['result_groups'] if g.get('protocol') == 'v6.3' and g.get('version') == version), None)
        reviews = {(r['model'], r['setting']): {k: r[k] for k in ('review',) if k in r}
                   for r in (old or {}).get('results', [])}
        results = []
        for model, tag in MODELS:
            for setting in ('single', 'iterative'):
                for n in range(RERUN.get((task, setting, model), (1, ''))[0], 0, -1):
                    name = '/'.join(filter(None, (SUBFOLDER.get(task), f'{model}-{setting}-{n}')))
                    folder = ROOT / 'tasks' / family / 'solutions/v6.3' / name
                    if (folder / 'score.json').exists() or (folder / 'develop_result.json').exists():
                        break
                if (folder / 'score.json').exists():
                    score = json.loads((folder / 'score.json').read_text())
                elif (folder / 'develop_result.json').exists():  # the attempt froze no valid submission
                    score = {'score': 0.0, 'invalid': '开发结束时没有冻结出有效提交'}
                else:
                    continue
                try:
                    if score.get('invalid'):
                        raw, mapping = f'无效提交（{score["invalid"]}）', '无效提交记 0'
                    elif ((score.get('candidate') or {}).get('status') == 'submission_failure'
                          and (score.get('candidate') or {}).get('raw_error') is None and score.get('rule') != 'per-group'):
                        reasons = score['candidate'].get('reasons') or [score['candidate'].get('reason')]
                        if isinstance(reasons, str):  # some scorers store the list as its (truncated) text
                            reasons = re.findall(r"'([^']+)'", reasons) or [reasons]
                        reasons = [r for r in reasons if r]
                        score['score'] = 0.0
                        raw, mapping = f'无效提交（{reasons[0] if reasons else "提交失败"}）', '任一用例失败或提交检查未通过记 0'
                    elif TASKS[task] == 'conic' and score.get('p') is None:
                        score['score'] = 0.0
                        raw, mapping = '无效提交（提交检查未通过，650 例均未运行）', '无效提交记 0'
                    else:
                        raw, mapping = cell(task, score)
                except Exception as error:  # one malformed cell must not stop the others from being published
                    print(json.dumps({'task': task, 'cell': name, 'error': repr(error)}), file=sys.stderr)
                    continue
                value = score.get('score')
                rerun_note = RERUN[(task, setting, model)][1] if n > 1 else ''
                if (task, setting, model) in PENDING and n < RERUN[(task, setting, model)][0]:
                    rerun_note = PENDING[(task, setting, model)]
                elif n > 1 and n < RERUN[(task, setting, model)][0]:   # the replacing attempt has no result yet
                    rerun_note = CUT_1 + '（基础设施故障，记未测）；本格为上游改用流式后的第 2 次尝试，替代它的下一次尝试仍在进行。'
                note = rerun_note or NOTES.get((task, setting) if tag == 'sol' else (task, setting, model), '')
                if (task, setting, model) in UNMEASURED and n == 1 and n < RERUN[(task, setting, model)][0]:
                    raw, mapping, value, note = '未测', '评测方问题，记未测（不记 0）', None, UNMEASURED[(task, setting, model)]
                if (task, setting, model) in REGRADE and n == 1:   # every regrade concerns a first attempt
                    regraded = (folder / 'score.first.json').exists()
                    if not regraded:
                        raw, mapping, value = '未测', '评测方问题，记未测（不记 0）', None
                    note = REGRADE[(task, setting, model)] + ('以重评为准。' if regraded else '重评待开发槽全空后进行。')
                    if (task, setting, model) in PENDING and n < RERUN[(task, setting, model)][0]:
                        note += PENDING[(task, setting, model)]
                results.append({'model': model, 'protocol': 'v6.3', 'setting': SETTING[setting], 'attempt': n,
                                'submission': submission_of(task, setting, n, tag), 'store': 'var/stores/store-v63', 'raw': raw,
                                'mapping': mapping, 'score': None if value is None or (isinstance(value, float) and math.isnan(value)) else round(value, 3),
                                'notes_md': note, **({'folder': name} if task in SUBFOLDER else {}), **reviews.get((model, SETTING[setting]), {})})
        if not results:
            continue
        group = {'title': f'{TITLE} · {CANDIDATE[task]}' if task in CANDIDATE else TITLE, 'version': version, 'protocol': 'v6.3',
                 'results': results, **({'candidate': True} if task in CANDIDATE else {})}
        others = [g for g in meta['result_groups'] if not (g.get('protocol') == 'v6.3' and g.get('version') == version)]
        meta['result_groups'] = others[:1] + [group] + others[1:] if task in CANDIDATE else [group] + others
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'task': family, 'cells': [(r['setting'], r['score']) for r in results]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
