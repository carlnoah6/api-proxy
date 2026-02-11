# API Proxy 部署与上线流程

## 架构概览

- **代理端口**: 8180 (API Key 鉴权层)
- **上游端口**: 8080 (Antigravity/OpenClaw)
- **服务管理**: systemd (`api-proxy.service`)
- **代码位置**: `/home/ubuntu/api-proxy/server.py`

---

## 修改前备份

```bash
# 备份当前代码
cp /home/ubuntu/api-proxy/server.py /home/ubuntu/api-proxy/server.py.bak

# 备份配置（keys 等）
cp /home/ubuntu/api-proxy/keys.json /home/ubuntu/api-proxy/keys.json.bak

# 带时间戳备份（推荐）
cp /home/ubuntu/api-proxy/server.py "/home/ubuntu/api-proxy/server.py.bak.$(date +%Y%m%d_%H%M%S)"
```

---

## 测试方法

### Health Check（上游连通性）

```bash
curl http://localhost:8080/health
```

预期：返回 JSON，`status: "ok"`，所有账号 available。

### 模型列表（鉴权 + 代理功能）

```bash
curl http://localhost:8180/v1/models -H "x-api-key: <YOUR_API_KEY>"
```

预期：返回模型列表 JSON。

### Admin 接口

```bash
# Fallback 状态
curl -H "x-api-key: sk-admin-luna2026" http://localhost:8180/admin/fallback

# 日用量
curl -H "x-api-key: sk-admin-luna2026" http://localhost:8180/admin/usage/daily
```

### 完整功能测试（发送请求）

```bash
curl -X POST http://localhost:8180/v1/messages \
  -H "x-api-key: <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Say hello"}]
  }'
```

---

## 上线步骤

### 常规重启（代码修改后）

```bash
sudo systemctl restart api-proxy
```

### 验证

```bash
# 1. 检查服务状态
systemctl status api-proxy

# 2. Health check
curl http://localhost:8080/health

# 3. 功能验证
curl http://localhost:8180/v1/models -H "x-api-key: <YOUR_API_KEY>"
```

### 查看启动日志

```bash
journalctl -u api-proxy --no-pager -n 30
```

---

## 回滚方案

### 方法 1: 恢复备份文件

```bash
# 恢复代码
cp /home/ubuntu/api-proxy/server.py.bak /home/ubuntu/api-proxy/server.py

# 重启服务
sudo systemctl restart api-proxy

# 验证
systemctl status api-proxy
curl http://localhost:8080/health
```

### 方法 2: Git 回滚（如果使用 Git）

```bash
cd /home/ubuntu/api-proxy
git log --oneline -5          # 查看最近提交
git checkout <commit> -- server.py  # 恢复特定版本
sudo systemctl restart api-proxy
```

---

## 日志查看

```bash
# 实时跟踪日志
journalctl -u api-proxy -f

# 查看最近 N 行
journalctl -u api-proxy --no-pager -n 50

# 查看今天的日志
journalctl -u api-proxy --since today

# 查看特定时间段
journalctl -u api-proxy --since "2026-02-11 00:00" --until "2026-02-11 23:59"

# 只看错误
journalctl -u api-proxy -p err --no-pager
```

---

## 服务管理命令速查

| 操作 | 命令 |
|------|------|
| 启动 | `sudo systemctl start api-proxy` |
| 停止 | `sudo systemctl stop api-proxy` |
| 重启 | `sudo systemctl restart api-proxy` |
| 状态 | `systemctl status api-proxy` |
| 开机自启 | `sudo systemctl enable api-proxy` |
| 禁用自启 | `sudo systemctl disable api-proxy` |
| 查看日志 | `journalctl -u api-proxy -f` |
| 重载配置 | `sudo systemctl daemon-reload` |

---

## 注意事项

- 服务配置了 `Restart=on-failure`，崩溃后 5 秒自动重启
- 服务配置了 `PYTHONUNBUFFERED=1`，日志实时输出不缓冲
- 修改 service 文件后需要 `sudo systemctl daemon-reload`
- 旧的 nohup 启动方式已废弃，统一使用 systemctl 管理
