# API Proxy (Luna API Proxy)

> 代码位置: `/home/ubuntu/api-proxy/server.py`
> 端口: **8180** (代理层) → **8080** (Antigravity 上游)
> 状态: ✅ 生产运行中

## 架构

```
OpenClaw → api-proxy (8180) → Antigravity (8080) → Claude/Gemini API
```

API Proxy 是一个 FastAPI 应用，在 Antigravity（Google Cloud Code 代理）前面加了一层，提供：
- **API Key 鉴权**：每个用户一个 Key，独立统计用量
- **智能 Fallback**：主动感知各模型额度，到限额自动切换
- **用量统计**：按 Key/日/小时/模型 多维度统计
- **格式转换**：Anthropic ↔ OpenAI 格式互转（支持流式）
- **OAuth 回调**：处理 Lark 日历授权 + 卡片按钮回调

## 常用命令

### 启动/重启

```bash
# 查看进程
ps aux | grep api-proxy

# 重启
kill $(pgrep -f "api-proxy/server.py")
nohup python3 /home/ubuntu/api-proxy/server.py > /home/ubuntu/api-proxy/server.log 2>&1 &
```

### 健康检查

```bash
# 代理层健康
curl http://localhost:8180/health

# 上游 Antigravity 健康（含各模型额度详情）
curl http://localhost:8080/health
```

### 管理接口（需要 Admin Key）

```bash
ADMIN_KEY="sk-admin-luna2026"

# 查看 Fallback 状态（各模型额度 + 可用性）
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/fallback

# 查看所有 API Key
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys

# 查看总用量
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/usage

# 按日统计（含模型维度）
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/usage/daily

# 按小时统计
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/usage/hourly

# 某个 Key 的详细用量
curl -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys/<api_key>/usage

# 创建新 Key
curl -X POST -H "x-api-key: $ADMIN_KEY" -H "Content-Type: application/json" \
  -d '{"name": "NewUser"}' http://localhost:8180/admin/keys

# 停用/启用 Key
curl -X POST -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys/<api_key>/disable
curl -X POST -H "x-api-key: $ADMIN_KEY" http://localhost:8180/admin/keys/<api_key>/enable
```

## 配置文件

| 文件 | 说明 |
|------|------|
| `server.py` | 主服务代码 |
| `keys.json` | API Key 数据（含用量统计） |
| `fallback.json` | Fallback 链配置 |
| `server.log` | 运行日志 |

## Fallback 逻辑

1. 后台每 30s 轮询 Antigravity `/health`，缓存各模型额度
2. 请求进来时检查目标模型额度，< 5% 则自动切换到下一个可用 tier
3. 额度聚合：遍历所有 Antigravity 账户，取每个模型的最大剩余额度
4. 响应式 Fallback：即使主动检查通过，如果上游返回 429/503/exhausted，也会尝试切换

## 已知问题 & 修复记录

### 2026-02-11: 多账户额度聚合 Bug
- **问题**: `HealthCache.poll()` 只读 `accounts[0]`，忽略后续账户
- **修复**: 改为遍历所有账户，取每个模型的最大 `remainingFraction`
- **影响**: 新加的账户（carlnoah6）额度未被识别，导致 Claude 一直误判为 0% 触发 fallback

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `UPSTREAM_URL` | `http://localhost:8080` | Antigravity 上游地址 |
| `PROXY_PORT` | `8180` | 代理监听端口 |
| `KEYS_FILE` | `/home/ubuntu/api-proxy/keys.json` | Key 数据文件 |
| `FALLBACK_CONFIG` | `/home/ubuntu/api-proxy/fallback.json` | Fallback 配置文件 |
