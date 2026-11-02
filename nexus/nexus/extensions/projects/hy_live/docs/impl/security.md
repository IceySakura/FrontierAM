# Hy-live 沙箱网络出口安全

## 1. 目标与范围

当前方案的目标只有一个：

> 防止用户把沙箱作为跳板访问公司内网、云元数据服务或其他沙箱。

它由部署侧（在线服务创建沙箱的地方）落地，分三部分：

1. **沙箱不复用** ：一通对话独占一个预热沙箱；归还时销毁，再补一个全新沙箱。
2. **登录加强** ：由部署侧负责，不属于 Nexus 改动。
3. **沙箱出口限制** ：创建 `GongfengRuntimeProvider` 时传入 `network_policy`，在平台网络层限制出口。

Nexus 核心代码不需要为当前方案增加 SIO 鉴权、UID 隔离或新的 invoke 协议。

## 2. 明确不解决的问题

当前方案 **不是恶意沙箱到 Host 的完整安全隔离** 。 下列问题不在范围内：

- 用户在自己沙箱里执行恶意命令、修改文件或启动后台进程；
- 沙箱内进程冒充 runtime server 或 Socket.IO worker；
- Host 与沙箱之间使用 pickle、Host API 或 shared-value 的信任问题；
- 模型额度滥用、日志刷量、超大消息等拒绝服务；
- 提示词注入、工具输出污染等模型安全问题；
- 公网滥用。 当前策略保留公网访问，以保证 `WebFetch` 可用。

因此本文只能说明 **内网穿透已受限** ，不能用来宣称系统能够抵御恶意攻击。

## 3. 实现

### 3.1 沙箱一次性使用

沙箱池的归还逻辑不再清理后复用，而是销毁原沙箱、再预热一个新沙箱补位。 这样上一位用户留下的进程、文件或环境不会影响下一位用户。 预热池只优化启动延迟，不改变「每个沙箱最多服务一通对话」的约束。

### 3.2 网络策略

`network_policy` 使用黑名单模式（`default_action = allow`），规则按顺序匹配，第一条命中生效：

1. 先放行 CodeBuddy 正常工作需要的目标：平台显式代理、CodeBuddy `WebSearch` 服务、npm 镜像。
2. 再拒绝：云元数据服务（链路本地地址段 **和** 元数据服务域名）、私有与内网地址段、公司内网域名后缀。

只封链路本地地址段不够：元数据服务域名在未加 policy 时可以直接访问，必须按域名再封一次。

`network_policy` 由沙箱平台在网络层执行，不依赖程序自觉读取 `HTTP_PROXY`。 用户即使绕过代理直接开 socket，也仍受策略限制。 传入 `network_policy` 时平台会自己选择调度后端，不要再显式指定 `runtime_provider`，否则平台返回 `InvalidParam`。

### 3.3 显式代理与 CodeBuddy 出网分流

平台仍向沙箱注入 `HTTP(S)_PROXY`。 它是 CodeBuddy 的联网通道之一，不是内网限制本身。 带 `network_policy` 的沙箱上：

- 公网可以直连，经平台代理访问公网被拒，所以 `WebFetch` 必须直连；
- CodeBuddy `WebSearch` 服务直连不可靠，经代理可用，所以 `WebSearch` 必须走代理。

因此给 CodeBuddy 配：

- `NO_PROXY=*`：关掉 CodeBuddy 自己的「凡有代理就全走代理」wrap；
- `NEXUS_PROXY_HOSTS=<WebSearch 服务域名>`：hy_live 的 Node 代理预加载（`node_proxy_preload.py`）只把这些域名送进代理，其余直连。 细节见 [`codebuddy_web_search.md`](codebuddy_web_search.md)。

policy 放行了平台代理，所以还要确认代理不能替用户向内网建立 CONNECT 隧道。 这不能用普通请求验证：`NO_PROXY` 可能让普通请求根本不经过代理，必须在沙箱里忽略 `NO_PROXY`、直接向代理发 CONNECT。

## 4. 验证结论

在真实沙箱上用同一份 `network_policy` 验证过（测试后已销毁沙箱）：

- 内网域名与元数据服务域名直连解析失败；另一个沙箱的 runtime 端口不可达。
- 强制经代理 CONNECT 内网主机或元数据服务，代理关闭连接，不建立隧道。
- npm 镜像和公网直连可用；`WebSearch` 服务经代理可用。

结论：没有发现通过显式代理绕过 policy 访问内网或元数据服务的路径；CodeBuddy 必需的目标在「代理只给 `WebSearch`、其余直连」的分流下可用。

## 5. 变更后的检查

修改 policy、代理配置、runtime provider 或沙箱平台后，至少重新验证：

1. 直连任一内网主机失败；
2. 忽略 `NO_PROXY`、强制经代理 CONNECT 内网主机不返回 200；
3. 直连和经代理访问元数据服务都失败；
4. 访问另一个沙箱的 runtime 端口失败；
5. `WebSearch`、npm 镜像和公网 `WebFetch` 正常；
6. 用完的沙箱确实销毁，不重新放回空闲池。
