# 求解接口

提交目录提供solver.py：

```python
def make_solver(dimension, seed, options):
    return Solver(...)

class Solver:
    def solve(self, hvp, b, context):
        return x  # 或 (x, diagnostics)
```

工厂每序列调用一次。b和x形状为(dimension,)，均为float64；x必须全部有限。hvp(v)计算当前H_t v，允许(dimension,)或(dimension,k)、1≤k≤64，每列计一次HVP，所有工作按实际墙钟计时，不另设查询数硬上限。返回当前解后不能修改，不能查询过去或未来系统。

context包含step（从0开始）、steps、phase、phase_changed、mu、L、deadline、sequence_deadline和hard_deadline，时间值为time.monotonic()同一时钟的绝对截止时间。序列预算共享；清理余量不是新的逐步限额。预算耗尽可能抛出ihvp_api.BudgetExceeded；应保留已得到的有限近似并按时输出。target_rtol是可选诊断目标，不是输出有效性的门槛。日志写标准错误，不使用标准输出传日志。
