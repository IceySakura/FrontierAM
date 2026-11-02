# Runner hooks

在 `RunInput.runner.hooks` 中配置任务生命周期扩展。和 processor 一样，每个配置键标识一个实例，
`name` 是注册名称，`args` 直接传给 hook 类构造函数。每次 run 都创建独立实例。

```yaml
runner:
  hooks:
    recording:
      name: gui_recording
      required: false
      args:
        runtime_name: main
        fps: 10
        max_duration_seconds: 3600
```

`required` 与 `name`、`args` 同级，默认 `true`。
强依赖 hook 抛出普通异常时，`ComposedHook` 记录后原样抛出；设为 `false` 则记录后继续执行。
该配置覆盖三个生命周期回调和 `get_result`，不吞掉任务取消信号。
录屏通常设为 `required: false`，避免启动录屏失败阻止主任务执行。

`max_duration_seconds` 为正整数，默认 3600 秒。后台录制到期后自动正常结束并保留成片，
仍需调用 `gui_stop_recording` 获取结果；hook 会在任务结束时自动调用并上传。也可以提前调用 stop 结束录制。

`gui_recording` 的 `artifact_name` 默认是 `recording-<run_id>.mp4`。
可在 `args` 中设置 `artifact_folder: /gui-recording`，按路径查找已有目录；
`artifact_name` 仅指定文件名，通常无需配置。
目录必须预先创建（例如 `nexus artifact mkdir /gui-recording`），hook 不会自动创建；
路径解析失败会在启动录屏和执行任务前报错。`artifact_folder` 不填或为 `/` 时上传到根目录。
原 `parent_folder_id` 参数已替换为 `artifact_folder`，配置时应使用目录路径。

上传脚本在 COS 返回成功响应及 ETag 后直接删除原始 MP4，分片上传仅在最后一片成功后删除，
不再单独调用删除脚本。hook 结果中的 `source_deleted: true` 表示原文件已删除，`output_path` 仅为原始路径。
上传失败时保留原文件。Artifact 的完成确认在删除后由 host 执行，确认重试复用已返回的上传信息，
不再读取或上传原文件。

## 实现和注册

接口与 registry 位于 `nexus/runner/hooks/base.py`，内置实现也放在这个包下。
例如自定义模块 `my_project.hooks`：

```python
from nexus.runner.hooks import RunHook, run_hook_registry


@run_hook_registry.register('my_hook')
class MyHook(RunHook):
    def __init__(self, label: str):
        self.label = label

    async def on_context_ready(self, ctx):
        pass

    async def on_run_output(self, ctx, output):
        pass

    async def on_context_closing(self, ctx):
        pass

    def get_result(self):
        return {'label': self.label}
```

外部模块通过顶层 `imports: [my_project.hooks]` 加载。内置单文件模块使用 PEP 723
元数据声明 `[tool.nexus]` 下的 `run_hooks = { my_hook = "模块路径" }`，由模块发现机制延迟注册。

## 生命周期

1. `on_context_ready(ctx)`：ExecutionContext 可用后、Task 执行前，按配置顺序调用。
   位于 processor scope 内、cache scope 外，可以通过 `ctx.get_runtime()` 创建或获取沙盒；
   不在 Task 的 `runner.timeout` 计时范围内。
2. `on_run_output(ctx, output)`：Task 成功、异常或超时后，在外部 RunHooks 输出回调、
   early output 写入以及 `hold_after_task` 之前调用。
3. `on_context_closing(ctx)`：runtime 释放之前，按相反顺序清理；Task 被取消时也会调用。
   仅对进入过 ready 回调的实例调用，因此实现需要允许初始化只完成一部分。

强依赖 ready 失败时，异常向调用方传播，跳过 Task、output 和后续实例的初始化，
已进入 ready 的实例仍会收到 closing 回调。弱依赖失败仅记录异常，后续 hooks 和任务继续执行。
强依赖的 output/closing 异常由 ComposedHook 抛给原有 runner 调用点，记入 `RunHooks.hook_failures`，
调用方可通过 `raise_if_failed()` 检查；不改写任务的 `success`、`error`。
配置解析、hook 构造阶段的错误仍按 runner 初始化失败处理，不受 `required` 控制。
取消会继续向调用方传播；进程被强制杀死、机器断开或重复取消时无法保证异步清理完成。

`get_result()` 必须返回可 JSON 序列化的数据，写入 `RunOutput.hook_results[配置键]`。
产生 RunOutput 时，配置 hook 异常记录在 `hook_errors["配置键.回调方法"]`。
output、closing 或 get_result 的异常不改变任务的 `success` 和 `error`。
closing 后会刷新 hook 结果；closing 阶段新增的信息不会回写已经输出的
early output。

YAML hooks 在初始化阶段与 Python API `run(run_input, hooks=RunHooks(...))` 合并为
`nexus/runner/hooks/composed.py` 中的 `ComposedHook`（继承 RunHooks），
复用原有三个生命周期调用点，不增加独立 scope。外部 ready 回调先执行，
配置 hook 的 output 回调先于外部 output 回调，配置 hook 的 closing 回调也先于外部 closing 回调。
外部 RunHooks 的错误处理约定保持不变。
