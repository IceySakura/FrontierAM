"""Publish the v6.3 round (gpt-6.1-sol, claude-opus-5-5) into tasks/<family>/meta.json as a "协议 v6.3" result group.

Reads tasks/<family>/solutions/v6.3/<model>-<setting>-<n>/score.json (written by scripts/rounds/v63-grade.sh) and the
attempt logs for the submission IDs; writes raw metric, mapping and score per cell. Reviews (results/BASELINE_REVIEW.md)
are kept from an earlier publish; method notes come from NOTES. Cells without a score.json are left out.

Usage: v63-publish.py [TASK ...]   (default: every task version below)
"""

import json
import math

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = {'conic': 'conic', 'exponential-v3': 'exponential', 'galaxy-gravity-v3': 'galaxy-gravity',
         'polarization-demosaicking-v2': 'polarization-demosaicking', 'species-tree-v3': 'species-tree',
         'trajectory-covariance-v2': 'trajectory-covariance', 'ihvp-sequence': 'ihvp-sequence',
         'ihvp-sequence-v2': 'ihvp-sequence', 'ihvp-sequence-v3': 'ihvp-sequence',
         'galaxy-gravity-v4': 'galaxy-gravity', 'ihvp-sequence-v4': 'ihvp-sequence'}
SUBFOLDER = {'ihvp-sequence-v2': 'v2', 'ihvp-sequence-v3': 'v3', 'galaxy-gravity-v4': 'v4', 'ihvp-sequence-v4': 'v4'}  # later task versions within protocol v6.3 keep their own solutions subfolder
SETTING = {'single': '单轮', 'iterative': '迭代'}
NOTES = {
    ('conic', 'single'): 'HiGHS 处理线性与二次规划并做 KKT 精修；一般锥用 Ruiz 均衡加齐次原始对偶内点法（Nesterov–Todd 缩放，指数锥用解析障碍函数）。',
    ('conic', 'iterative'): '结构捷径（零最优、可分盒约束 QP、代数不可行证书）后：LP 与小 QP 用 HiGHS，其余多面体 QP 用固定分解的算子分裂加有效集 KKT 精修；一般锥用齐次嵌入预估校正内点法（Ruiz 均衡、Nesterov–Todd 缩放），二阶锥用稀疏秩二展开，指数锥用解析障碍函数。重跑（原尝试受测试排队故障影响）。',
    ('exponential-v3', 'single'): 'C++ 实现：Chebyshev 展开把指数与 φ₁ 合成一条稳定 Clenshaw 递推，正向量加权 Gershgorin 收紧谱区间；小连通块扩展精度对角化。',
    ('exponential-v3', 'iterative'): '刚度小的会话直接用自适应 Taylor；对角和小连通块扩展精度谱分解；一般情形用合并的 Chebyshev 展开（行和与 Collatz 界给谱区间，Reinsch 形式的 Clenshaw 递推）。重跑（原尝试受测试排队故障影响）。',
    ('galaxy-gravity-v3', 'single'): '笛卡尔与调和多极展开结合的 FMM，远场也保留 Plummer 软化修正；按相互作用选展开阶数，按进程 CPU 时间控制工作量。',
    ('galaxy-gravity-v3', 'iterative'): '核拆分：牛顿部分用调和多极展开，软化差项用笛卡尔修正；按调用规划精度，剩余 CPU 时间对部分粒子做完整直接求和。',
    ('polarization-demosaicking-v2', 'single'): '低通估计通道差、把马赛克细节转移到四个通道，再用约 7.7k 参数的残差卷积网络修正（只用 2 张公开样例加增强训练），最后做以强度为引导的局部仿射回归。',
    ('polarization-demosaicking-v2', 'iterative'): '逐格双线性插值加通道差的偏振估计，用全部测量恢复共同细节引导，再用小型残差卷积网络修正，带结构门控与残差校准。',
    ('species-tree-v3', 'single'): '支持度加权的内部节点距离 NJ 起点、加权四元组 NNI；最多 60 个基因座自助树，再用动态规划求兼容分裂频率最大的共识树。',
    ('species-tree-v3', 'iterative'): '支持度加权的内部节点距离 NJ 起点、加权四元组 NNI；四元组按悬挂枝长降权（加权方式与 Weighted ASTRAL 相同）。',
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
    ('ihvp-sequence-v2', 'iterative'): '工作区不含生成代码；最近 16 个解做 QR，在子空间上精确投影得到初始解，再跑 CG，按剩余步数分配时间（阶段突变时多给），Gauss–Radau 上界停止。没有识别矩阵结构；public 分数 83.7 → 99.8。',
}
TITLE = '协议 v6.3（单轮 1 小时 / 迭代 4 小时；2026-11-10 起）'
LINEAR = '\\(E \\ge U\\) 记 0；\\(W \\le E < U\\) 为 \\(50\\,(U-E)/(U-W)\\)；\\(E < W\\) 为 \\(100 - 50\\,E/W\\)'


MODELS = [('gpt-6.1-sol', 'sol'), ('claude-opus-5-5', 'opus'), ('qwen3.8-max-0902', 'qwen'), ('hy4', 'hy4')]  # (model, attempt-log tag); NOTES keys add the model for opus


def submission_of(task: str, setting: str, n: int, tag: str = 'sol') -> str | None:
    log = ROOT / f'var/stores/store/reruns/v63-{tag}-{setting}-{task}-{n}.log'
    rows = [json.loads(line) for line in log.read_text().splitlines() if 'develop_finished' in line] if log.exists() else []
    return rows[-1]['submission'] if rows else None


def baseline_anchors(task: str) -> dict:
    root = ROOT / 'var/stores/store-v63/baselines' / task
    active = json.loads((root / (root / 'ACTIVE').read_text().strip() / 'baseline.json').read_text())
    return {w: active['metrics']['parts'][w]['raw_error'] for w in ('weak', 'strong')}


def cell(task: str, score: dict) -> tuple[str, str]:
    family = TASKS[task]
    if family == 'conic':
        return (f'{score["successes"]} / {score["cases"]}（\\(p = {score["p"]:.3f}\\)，参照 \\(p_B = {score["p_B"]:.3f}\\)）',
                '\\(p \\le 0.9\\,p_B\\) 记 0；否则 \\(50\\,(p-0.9p_B)/(0.1p_B)\\)，高于 \\(p_B\\) 为 \\(50 + 50\\,(p-p_B)/(1-p_B)\\)')
    if family == 'exponential':
        if score.get('status') != 'ok':
            return f'无效提交（{len(score.get("failed_cases") or {})} 例不合格）', '任一用例不合格记 0'
        return (f'74/74 合格，\\(R = {score["relative_time"]:.3f}\\)（对 7 方法参照组逐例最快成员）',
                '\\(100/(1+R)\\)')
    candidate = score.get('candidate') or {}
    raw = candidate.get('raw_error')
    if task == 'galaxy-gravity-v4':
        if score.get('score') == 0.0 and raw is None:
            return f'无效提交（{"；".join(filter(None, candidate.get("reasons") or [])) or "有调用失败"}）', '任一调用失败记 0'
        return (f'\\(E = {raw:.3g}\\)，20 组（4 档规模 × 5 类）等权；CPU 用量最多到逐例时限的 {100 * (candidate.get("max_cpu_fraction") or 0):.0f}%'
                f'（经典树 theta 0.8：\\(E_0 = {score["weak"]["raw_error"]:.3g}\\)，改进树 theta 0.7：\\(E_{{50}} = {score["strong"]["raw_error"]:.3g}\\)）',
                '对数刻度：\\(E < E_{50}\\) 为 \\(50 + 50\\,\\log(E_{50}/E)/\\log(E_{50}/10^{-15})\\)；\\(E_{50} \\le E < E_0\\) 为 \\(50\\,\\log(E_0/E)/\\log(E_0/E_{50})\\)')
    if family == 'galaxy-gravity':
        anchors = baseline_anchors(task)
        return (f'\\(E = {raw:.3g}\\)，单次最多 {candidate.get("max_cpu_s", 0):.2f} CPU 秒（出题方 Treecode2：\\(U = {anchors["weak"]:.3g}\\)，\\(W = {anchors["strong"]:.3g}\\)）',
                '对数刻度：\\(E < W\\) 为 \\(50 + 50\\,\\log(W/E)/\\log(W/10^{-15})\\)；\\(W \\le E < U\\) 为 \\(50\\,\\log(U/E)/\\log(U/W)\\)')
    if family == 'ihvp-sequence':
        # v4 anchors come with the score (same private measurement for every model: scripts/analyze/ihvp-v4-runs.py)
        anchors = score.get('anchors') or baseline_anchors(task)
        if raw is None:
            return f'无效提交（{candidate.get("reason") or "有序列失败"}）', '任一序列失败记 0'
        return (f'\\(E = {raw:.4g}\\)，最长序列 {candidate.get("max_wall_s") or 0:.1f} 秒（CG：\\(U = {anchors["weak"]:.4f}\\)，'
                f'L-BFGS-PCG：\\(W = {anchors["strong"]:.4f}\\)）', LINEAR)
    if family == 'polarization-demosaicking':
        anchors = baseline_anchors(task)
        if raw is None:
            return f'无效提交（{candidate.get("reason") or "有场景失败"}）', '任一场景失败记 0'
        return f'RMSE \\(E = {raw:.4f}\\)（\\(U = {anchors["weak"]:.4f}\\)，\\(W = {anchors["strong"]:.4f}\\)）', LINEAR
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
        version = {'polarization-demosaicking': 'v2', 'trajectory-covariance': 'v2', 'ihvp-sequence': 'v1'}.get(family, 'v3')
        version = SUBFOLDER.get(task, version)
        old = next((g for g in meta['result_groups'] if g.get('protocol') == 'v6.3' and g.get('version') == version), None)
        reviews = {(r['model'], r['setting']): {k: r[k] for k in ('review',) if k in r}
                   for r in (old or {}).get('results', [])}
        results = []
        for model, tag in MODELS:
            for setting in ('single', 'iterative'):
                name = '/'.join(filter(None, (SUBFOLDER.get(task), f'{model}-{setting}-1')))
                folder = ROOT / 'tasks' / family / 'solutions/v6.3' / name
                if not (folder / 'score.json').exists():
                    continue
                score = json.loads((folder / 'score.json').read_text())
                raw, mapping = cell(task, score)
                value = score.get('score')
                note = NOTES.get((task, setting) if tag == 'sol' else (task, setting, model), '')
                results.append({'model': model, 'protocol': 'v6.3', 'setting': SETTING[setting], 'attempt': 1,
                                'submission': submission_of(task, setting, 1, tag), 'store': 'var/stores/store-v63', 'raw': raw,
                                'mapping': mapping, 'score': None if value is None or (isinstance(value, float) and math.isnan(value)) else round(value, 3),
                                'notes_md': note, **({'folder': name} if task in SUBFOLDER else {}), **reviews.get((model, SETTING[setting]), {})})
        if not results:
            continue
        group = {'title': TITLE, 'version': version, 'protocol': 'v6.3', 'results': results}
        meta['result_groups'] = [group] + [g for g in meta['result_groups'] if not (g.get('protocol') == 'v6.3' and g.get('version') == version)]
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'task': family, 'cells': [(r['setting'], r['score']) for r in results]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
