# 沙箱里 CodeBuddy 原生联网搜索

CodeBuddy 自带的 `WebSearch` / `WebFetch` 在 Gongfeng 沙箱里的行为。 CodeBuddy 的配置只涉及 Host 侧，不需要 rebuild 镜像；WorkBuddy ACP 的启动适配见第 5 节。

## 1. 默认行为

- **网络**：沙箱用平台默认的显式代理（平台注入 `HTTP(S)_PROXY` / `NO_PROXY` 与 CA）。 runtime 的 `environment` 不要覆盖这些变量；不开 `need_transparent_proxy`。
- **Node 代理预加载**：backend 在每个沙箱的第一轮 CodeBuddy 调用前装一次（`undici@6` + `/tmp/nexus-node-proxy/preload.js`，约 1–2 秒），并给 CodeBuddy 的 `NODE_OPTIONS` 前面加 `--require <preload>`。 安装失败时这一轮按 infrastructure 失败返回，下一轮重试。 实现：[`../../node_proxy_preload.py`](../../node_proxy_preload.py)。
- **key**：默认是 Nexus 的占位 key。 `WebFetch` 可用；`WebSearch` 返回 401。

## 2. 打开 WebSearch：数据里配有效 key

用现有的 `extra_env` 覆盖占位 key：

```json
"work_agent_configs": {
  "codebuddy": {
    "extra_env": {
      "CODEBUDDY_API_KEY__env": "CODEBUDDY_API_KEY",
      "CODEBUDDY_INTERNET_ENVIRONMENT": "internal"
    }
  }
}
```

- `CODEBUDDY_API_KEY__env`：runner 运行时从该环境变量取 key，数据和产物里只留变量名。 直接写 `CODEBUDDY_API_KEY` 也行，但 key 会进数据。
- `CODEBUDDY_INTERNET_ENVIRONMENT`：中国版 key 填 `internal`（`copilot.tencent.com`），iOA 版填 `ioa`，海外版不填（`www.codebuddy.ai`）。 填错返回 401。

## 3. 为什么这样

| 工具 | 实现 | 需要 |
|---|---|---|
| `WebSearch` | POST `${endpoint}/agenttool/v1/search`，CodeBuddy 服务端代搜 | 有效 key + 沙箱出网 |
| `WebFetch` | 沙箱内直接 `fetch(url)`，再交给模型总结 | 沙箱出网 |

- 两个工具都用 Node 全局 `fetch`。 runtime 镜像自带的 Node 是 v22.20，`fetch` **不读** `HTTP(S)_PROXY`（`NODE_USE_ENV_PROXY` 从 v22.21 起才支持），所以用预加载换全局 dispatcher。 Node 版本由 runtime 镜像决定；升到 v22.21+ 后可改用 `NODE_USE_ENV_PROXY=1`，去掉预加载。
- 预加载默认用 `EnvHttpProxyAgent`（按 `HTTP(S)_PROXY` / `NO_PROXY`）。 若设置 `NEXUS_PROXY_HOSTS`（逗号分隔域名），则只把这些域名送进代理，其余直连——带 `network_policy` 的沙箱需要这样：平台代理拒绝公网，而 `copilot.tencent.com` 只能经代理访问（见 [`security.md`](security.md)）。
- 预加载只在 runtime 自带的 node 里生效，agent 在 Bash 里跑的其他 node 不受影响。
- 透明代理下 `copilot.tencent.com` 连不上（TLS 超时），中国版 key 只能走显式代理。
- CodeBuddy 自己也有一层 fetch wrap：只要环境里有 `HTTP(S)_PROXY` 且目标不在 `NO_PROXY`，就会给每个请求挂上 `ProxyAgent`。 带 `network_policy` 的沙箱上要配合 `NO_PROXY=*`，否则 WebFetch 会被它拖进代理。

## 4. 注意

- **密钥**：CodeBuddy 的 Bash 能看到 `CODEBUDDY_API_KEY`，agent 跑 `env` 就会写进轨迹。 example CLI 已把它加入脱敏；`result.json` 照旧不脱敏。
- **限流**：所有沙箱共用一个 key，大规模 RL 可能触发 `WebSearchRateLimit`（错误码 15001）。
- **自查**：沙箱里 `env | grep -i proxy` 确认代理已注入；带预加载用 Node `fetch` POST `https://copilot.tencent.com/agenttool/v1/search`（`Authorization: Bearer <key>`），应返回 200。

## 5. WorkBuddy ACP

WorkBuddy 5.11.0 的联网工具在内部 agent 进程中执行，不在 `wb --acp` 主进程中。 它会清理子进程的代理、CA 和 `NODE_OPTIONS`；hy-live 的预加载因此只对 WorkBuddy 的 sidecar 和 agent 恢复这些变量，覆盖普通 spawn 和 Linux 的 `node-pty.spawn`。 agent 在 Bash 里启动的其他程序不受这项适配影响。

搜索凭据通过同一个环境变量引用传入：

```json
"work_agent_configs": {
  "workbuddy": {
    "extra_env": {
      "CODEBUDDY_API_KEY__env": "CODEBUDDY_API_KEY"
    }
  }
}
```

WorkBuddy 使用包内的产品配置，不使用 CodeBuddy 的 `CODEBUDDY_INTERNET_ENVIRONMENT`。 5.11.0 的搜索 endpoint 为 `https://www.workbuddy.cn/agenttool/v1/search`；本次实测该 key 可正常返回搜索结果，无需重定向 endpoint。 模型仍由 `models.json` 的本地 URL 和占位 key 走 Nexus proxy，搜索使用真实 key，两者分开。

这项凭据修复包含 runtime 的 session 启动代码。 使用镜像内代码时需要更新镜像；使用 worker venv 时可直接加载新 wheel，不必重建镜像。 联网验收 case 见 [`../../example/backend_web_research.md`](../../example/backend_web_research.md)。
