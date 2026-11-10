# 会话二进制接口

所有整数采用无符号little-endian，所有浮点采用IEEE754 little-endian float64。流中无padding。标准输入由控制器写入，标准输出由提交写入；日志只写stderr。辅助实现位于`sdk/wire.py`。

1. 提交完成通用初始化后，输出恰好6字节`READY\n`并flush。此时没有会话输入；初始化墙钟最多15秒。
2. 控制器开始数值会话计时，发送8字节magic `FAMEXP2\n`、4字节uint32 JSON长度（≤4096），再发送该长度UTF8 JSON。
3. JSON字段为version=2、n、nnz、requests（1或16）、scenario（one_shot或reuse）、t、rtol、atol。n≥1。三者L/t/精度在会话内固定。
4. 紧随其后依次发送CSR的indptr（n+1个uint64）、indices（nnz个uint32）、data（nnz个float64）。shape=(n,n)。indptr从0开始、单调不降、末项等于nnz；每行列号严格递增且无重复。对称位置均显式存储。不会隐藏只存上三角的约定。
5. 控制器发送一个请求：先n个float64 b0，再n个float64 b1。提交返回恰好n个float64 y并flush，不附长度、编号、换行或JSON。矩阵行序就是向量分量顺序。
6. 控制器收到完整y后，才发送下一请求；保持同一进程。共requests次。三次统计重复是三个独立会话，不能保留上一重复的状态。
7. 最后一个y发送后，提交必须以退出码0结束，不能继续输出其他字节。控制器在完整输出和正常退出后停止计时。精度检查在计时后进行，不把裁判验解时间插入相邻请求。

例如n=2、requests=16时，每个请求32字节，每次回答16字节，共16轮。缓冲但不flush会导致控制器等待，计入墙钟并可能超时。漏字节、多字节、NaN/Inf、错误退出或不结束均无效。

JSON不含family、mother、种子、文件名、参考方法或计时。合法输入由评测器保证；不要求提交处理恶意输入。只有输入的实际数值及公开字段可用于选择算法。
