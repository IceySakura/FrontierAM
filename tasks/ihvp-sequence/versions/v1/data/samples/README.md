# 给模型的样例

`smoke.json` 是一个可直接阅读的 `d=4, T=3` 协议示例，不参与正式评分。前两步的矩阵平滑变化，第三步切换阶段。第一个系统的参考解为

```text
[1/11, 7/11, -7/11, 6/11]
```

对向量 `[1,-1,2,0.5]`，第一个算子的回复是 `[3,-2,4.25,1.75]`。

`recipes.json` 另给出两个完整规模样例的公开生成配方：

| 样例 | 来源反馈池编号 | 规模 | 类型 | 基础谱条件数 |
|---|---|---|---|---:|
| `sample_01` | `case_04` | d=16384，T=50 | 宽谱平滑变化 | 10³ |
| `sample_02` | `case_14` | d=16384，T=50 | 第 21、36 步控制重启 | 10⁵ |

这两例取自反馈池，不是额外随机创建的样例。完整数组约 44 MB，因此 GitHub 保存配方、生成脚本及独立自检工具；开发镜像中必须实际包含全部三个样例及其 manifest。

## 在开发容器中自检

镜像已将完整样例放在 `/workspace/samples/`，不需要在容器里寻找仓库的 `scripts/` 目录。先检查样例包是否完整：

```sh
python /workspace/samples/check.py --verify-package
```

随后用自己的提交运行一个完整序列：

```sh
python /workspace/samples/check.py \
  --submission /workspace/submission \
  --case /workspace/samples/cases/sample_01/teacher.npz \
  --out /tmp/ihvp-sample_01.json
```

第二个完整序列的路径为 `/workspace/samples/cases/sample_02/teacher.npz`；小型示例的路径为 `/workspace/samples/smoke.json`。自检报告接口有效性、原始误差与运行资源，不给出正式成绩。

## 在仓库内准备样例

维护者或本地开发者在仓库根目录运行：

```sh
python scripts/seal/prepare_ihvp_sequence.py --samples-only
```

默认输出为忽略目录 `var/outputs/ihvp-sequence-v1/samples/`，含三个样例的 `manifest.json`、两个 `cases/<id>/teacher.npz`、`check.py` 及 `support/`。已有完整数组时先按 SHA-256 核验并复用，只刷新说明和独立自检资产，不重新生成大样例。首次生成时复核每步参考解，另报告与源环境数组是否逐位相同。

维护者预先取得 Linux amd64 wheelhouse 后，可准备离线镜像构建目录：

```sh
python scripts/seal/prepare_ihvp_sequence.py \
  --image-context var/dist/ihvp-sequence-v1 \
  --wheelhouse /path/to/pre-staged-linux-amd64-wheels
```

该目录包含 Dockerfile、锁定依赖、完整 `data/samples/` 和 wheelhouse。脚本只准备并校验构建上下文，不下载依赖、不构建镜像，也不代表 amd64 运行验收已通过。

教师文件便于公开自检，但正式接口不会向求解进程传递矩阵构造或参考答案。反馈池的其余用例不随样例包交给模型；私有测试集独立保管。
