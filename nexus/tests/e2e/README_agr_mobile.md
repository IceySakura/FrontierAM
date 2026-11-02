# AGR Mobile E2E 运行说明

这份 README 只描述 `tests/e2e/test_e2e_agr_mobile.py` 这条 API 级 E2E 测试链路。
如果你要跑 `run-batch` + `agr_mobile_batch_generator.py` 的 Agent 驱动批量流程，请看 `docs/how-to/use-agr-mobile/`。

## 配置文件

E2E 使用的配置文件放在：

`tests/e2e/config/agr_mobile_e2e.yaml`

如果是第一次使用，可以先参考：

`tests/e2e/config/agr_mobile_e2e_example.yaml`

建议做法：

1. 复制一份 example 文件为 `tests/e2e/config/agr_mobile_e2e.yaml`
2. 填入自己的 `api_key`、`domain`、`sandbox_template`
3. 如果要跑已有沙箱并发场景，把 `sandbox_ids` 列表改成自己的沙箱 ID

说明：

- `tests/e2e/config/agr_mobile_e2e.yaml` 已加入 `.gitignore`
- `tests/e2e/config/agr_mobile_e2e_example.yaml` 会保留在仓库里作为模板

## 直接运行

直接跑模板建沙箱场景：

```bash
python tests/e2e/test_e2e_agr_mobile.py template_parallel_e2e
```

直接跑已有 `sandbox_id` 并发场景：

```bash
python tests/e2e/test_e2e_agr_mobile.py existing_sandbox_parallel_e2e
```

如果两个场景都要跑：

```bash
python tests/e2e/test_e2e_agr_mobile.py all
```

## 输出目录

输出根目录由 `tests/e2e/config/agr_mobile_e2e.yaml` 中每个场景的 `output_dir` 控制。

实际落盘时，会在该目录下面再追加一层时间戳目录，例如：

```text
tests/e2e/agr_mobile_output/template_parallel_e2e/20260410_153012/
tests/e2e/agr_mobile_output/existing_sandbox_parallel_e2e/20260410_153012/
```

每次执行都会生成新的时间戳目录，避免覆盖历史结果。

其中会包含：

- `*.db`：batch 执行数据库
- `*_generator.py`：本次运行生成的批量配置
- `<run_id>/device_info.json`：设备信息
- `<run_id>/sandbox_info.json`：本次 run 实际连接到的 sandbox_id
- `<run_id>/app_store_after_launch.png`：安装并启动应用宝后的截图
- `<run_id>/logcat.txt`：本次任务抓取的 logcat
- `<run_id>/upload_result.json` / `install_result.json` / `launch_result.json` / `reset_result.json`：关键步骤结果
