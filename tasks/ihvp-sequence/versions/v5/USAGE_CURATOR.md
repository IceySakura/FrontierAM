> 当前使用balanced-split-20261117完整包：先读START_BALANCED_EVALUATION.md，再进入balanced-evaluator/，所有replay/calibrate命令传--split-manifest split_manifest.json。参见[README](README.md)与[重划报告](reports/BALANCED_SPLIT_20261117.md)。下文旧pool说明与旧命令仅作归档，不可替代新清单。

# 评测维护者使用指南

**本文件与评测数据包只给评测维护者，不给被测模型。**

这批数据是候选实验集：15条public反馈轨迹、30条private正式轨迹，每条50个16384维SPD系统。完整数学定义和构造见PROTOCOL.md；最终实测结果见reports/final/REPORT.md与baseline_summary.csv。

## 模型能看什么

模型工作目录只放通用题面、worker接口说明、自己的提交代码，以及明确公开的小型接口样例。模型每一步仅取得当前b、HVP查询回复、全局谱界、阶段标识和截止时间。public反馈只给整池汇总loss、汇总时间/内存及是否合法。

不要把这个维护者包解压到模型的工作目录。模型不能读取code/、runtime/backend.py、var/、data/、reference.npz、step_*.npz、plans.json、生成种子、教师轨迹日志、参考解、其他agent源码、本维护者指南和private统计。private仅在提交源码、构建与参数冻结后运行，不反馈逐例结果，不据此重新挑选或修改提交。

评测端和模型端应使用不同目录和文件挂载；本地开发脚本的进程协议与UID保护不能代替完整的隔离环境。

## 本批数据实际来自哪里

A负责生成：/mnt/sdb/henryzhang/ihvp_realtrace_20261116/var/data。
B负责CPU评测：/mnt/sdb/ihvp_realtrace_20261116/var/data_A。

使用官方CIFAR-10、CIFAR-100、Fashion-MNIST训练集，下载后先核对发布方校验和。每条轨迹使用独立模型初始化、随机批次和预先封存的种子，两池共享数据集来源，不共享具体系统或序列前缀。这里没有“private图像全部未出现在public”这一承诺；划分单位是独立训练轨迹。

模型为无偏置两层ReLU分类器，完整下层变量为(B,C)。CIFAR-100用hidden128、rank28、100类；另外两个数据集用hidden256、rank54、10类，联合维数均为16384。训练是float32，冻结HVP是float64。它们是完整小网络的真实训练点，不是从原671744维网络随机投影得到的系统。

H为真实联合交叉熵Hessian加可证明的正定阻尼；b为独立query批次的下层梯度。support4096、query128。固定教师采用32次Jacobi-PCG、水平隐式梯度、polar retraction和固定四次标量回溯。候选解不驱动后续状态。原始神经Hessian不保证正定；正定假设针对保存的阻尼系统。

五种预先声明的时序设置包含连续固定批次、连续换批次、中等长度重置、短子问题重置、较快P更新。连续长子问题预算510步，每10步一次P查询，得到50个系统；短子问题保持50步形态，拼接其查询点直到50个系统。所有阶段重置均通知求解器。

## CPU评测

目标环境是B的Linux amd64、Python3.12、NumPy2.3.5、SciPy1.16.3。C++ agent二进制为该环境构建；更换Python ABI或CPU时请从附带源码重新离线构建。

每条序列从空进程开始，求解器单进程单线程，30秒墙钟和256MiB峰值RSS。计时含启动、导入、通信、全部HVP和预条件/历史构建；私有数据装载、参考解构造和独立验分不计时。内存只计求解进程，评测端缓存不计入。Linux使用内核VmHWM及独立RSS监控，避免把fork前评测器高水位误计为求解器内存。

在B运行现有冻结方法可使用：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 0 \
  .runtime/eval/bin/python code/benchmark.py \
  --data var/data_A --methods cold_cg warm_cg lbfgs_pcg \
  --out var/new_diagnostic_run
```

如果从ZIP在新目录复现，先创建权限为700的评测端专用解压目录，解压后创建var/并将audit/freeze.json复制到var/freeze.json；安装上述固定版本依赖，命令的数据目录改为--data data。这个目录不挂载给模型。

使用新输出目录保留既有结果。完整比较由analysis/parallel_evaluate.py负责，不同序列分配到互不共享末级缓存的物理核，同一序列内各方法串行且固定使用同一个核。参数与源码在private评测前固定。

已有10种固定方案：Cold/Warm CG、L-BFGS-PCG、Ritz/POD/Spectral回收改写，以及GPT-6.1 Sol与Opus5.5的单轮/迭代方案。agent原始提交固定于60269c3fdcc471e916e62163229a9fdd56242eaf。三个文献方法是有界历史的HVP改写，不是完整作者代码复现。

## 验分与解释

loss是各步归一化平方能量误差，再对每条序列和整池等权平均。任一序列非法或超限，该提交不进入有效loss排名。不要把平均有效子集的loss当成全池有效成绩。

先用15条public，从预先固定的五个人类候选中选择参考；选择过程不读取private loss。30条private用于独立对照。原CG=0、L-BFGS-PCG=50的分数不能自动沿用，必须先确认本机新数据上E_CG>E_LBFGS>0；报告中的换算分数仅作诊断。

维数、步数、非零右端项、线性/对称、正定下界、谱上界与参考解误差均验证。相邻相关性和历史解的近似维度属于实测诊断，不是低秩保证。旧题面列举的宽谱/多谱带/均值回归等合成类别，不等于本真实数据逐项覆盖；正式采用时应修改这一段数据分布描述。

## 故障与版本

B上早期CPU生成尝试在切换A时暂停，未混入最终数据。内存计量错误的初期试跑保留，但不计最终结果。public_14的参考求解器重启间隔由16改为128后重新生成；该更改只影响高精度参考解计算，目标证书仍为1e-18。前27个算子缓存逐位一致，构造H/b、教师驱动器和候选方法未改动，此修复发生在private生成前。

生成程序按整条轨迹重启失败尝试，不宣称从中途训练状态精确续算。失败目录、源代码/配置哈希和修复对照均保留。
