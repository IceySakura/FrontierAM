# Feasibility probes

日期：2026-11-04。所有实际运行均使用 default Docker daemon、`frontieram-exponential:port-v1`、`--network none --cpuset-cpus 2,3`，宿主命令由 `nice -n 19 taskset -c 2,3` 启动。只运行解包后的 29 个 PUBLIC dev case，`run_local.py --repeats 1`；没有读取 private NPZ。一次重复只用于可行性和粗略量级，不是正式统计。

## 结果

| implementation | 29/29 correct | max scaled error | 29 case measured-time sum | per-case median | geometric mean time / Expokit |
|---|---:|---:|---:|---:|---:|
| Expokit DSPHIV frozen generic binary | yes | 0.0024721 | 0.7690 s | 0.01105 s | 1.00× |
| KIOPS author MATLAB code, Octave adapter | yes | 0.0158344 | 3.9231 s | 0.05730 s | 11.08× |
| SciPy 1.18.1 `expm_multiply` | yes | 0.0005776 | 8.6859 s | 0.11306 s | 21.48× |

逐 case 的 SciPy/Expokit 比为 6.57–99.18×，KIOPS/Expokit 为 3.63–51.59×。SciPy 的 one-shot/reuse 几何平均比约为 32.20×/13.05×。这些小 case 反复支付 Python/Octave 启动、对象构造和协议开销，不能外推到 `n=120000`、`nnz=10^6`、`t||L||=10^5`。尤其不能据此删除 Chebyshev、rational Krylov 或 reuse-oriented 实现。

### 包装方式

* Expokit：使用仓库中已冻结的 unmodified Sidje Fortran + `stream_expokit.c`，`m=30`、public-only 预声明规则得到 `tol=1000*rtol`。
* KIOPS：使用仓库中 commit `9414984…` 的 unmodified `kiops.m`/`expm_kiops.m`，现有 thin Octave stream adapter，`tol=0.01*rtol`。
* SciPy：新增 `vendor/scipy_expm_multiply/stream_scipy.py`。每个 RHS 构造
  `t*[[-L,b1],[0,0]]`，调用未修改的 `scipy.sparse.linalg.expm_multiply` 作用于 `[b0;1]`。这直接得到目标组合；adapter 只处理 wire/CSR/augmentation。

## 获取与构建核验

`vendor/PROVENANCE.json` 记录 URL、版本/commit 和 SHA-256。已实际下载且保持 archive 原样：Expokit mirror、KIOPS、Higham `expmv`、ExponentialUtilities.jl、SLEPc 3.18.2。Tencent PyPI 的 SciPy simple index和 GitHub codeload 均可达。task image 有 GCC/G++/GFortran 12.2、Make、Octave、Python/SciPy 1.18.1、MPI runtime 与 PETSc 3.18 shared library；没有 Julia、CMake，也没有可用的 SLEPc shared library/header。

未做的重型工作是有意的：

* SLEPc/PETSc：源码可取，但当前镜像不是可直接链接的完整 SLEPc dev 环境；按约束未编译。
* ExponentialUtilities.jl：源码可取但镜像无 Julia；按约束未安装/预编译。
* Higham `expmv`：源码可取（MATLAB），但本轮没有再写一个 Octave adapter；SciPy 已探测同一 Al-Mohy–Higham truncated-Taylor family，正式 panel 仍应分别测试 author MATLAB 与 SciPy implementation。
* `phipm`：官方 Leeds host DNS 解析失败；论文确认作者 MATLAB code 的获取页，但尚未找到权威 license，故不能冻结进可再分发 panel。
* rational Krylov：van den Eshof–Hochbruck (2006) 是方法依据；未发现可直接冻结且 license 清楚、适配本协议的原作者实现。应实现独立的 documented baseline，或选用 license 清楚的 PSFUN/近期实现后另行审计，不能冒称“原作者代码”。

## 可复现命令形态

探针均采用以下形态（省略 bind mount 的公共路径）：

```sh
python3 /workspace/tasks/exponential/public/tools/run_local.py \
  --manifest /public/task/dev/manifest.json --repeats 1 \
  --command '<frozen stream command>' --output /hosttmp/report.json
```

原始临时 report 没有放入交付目录，因为它们包含冗长的逐进程 PID/RSS；上面的数字由其中 `aggregates` 与 `runs[*].errors` 直接汇总。
