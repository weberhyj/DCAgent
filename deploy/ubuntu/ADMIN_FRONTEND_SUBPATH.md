# 管理端子路径部署

管理端默认构建到 `/admin/`，后端 API 保持在根路径 `/api/`。子路径是构建时配置，修改路径后必须重新构建管理端。

## 构建

在仓库根目录执行：

```bash
corepack enable
pnpm install --frozen-lockfile
VITE_ADMIN_BASE_PATH=/admin/ pnpm --filter dc-agent-admin-frontend build
```

如果需要使用其他子路径，例如 `/operations/`：

```bash
VITE_ADMIN_BASE_PATH=/operations/ pnpm --filter dc-agent-admin-frontend build
```

构建路径必须与 Nginx 的 `location` 完全一致。没有设置 `VITE_ADMIN_BASE_PATH` 时，默认值为 `/admin/`。

## 发布静态文件

以下示例将构建结果发布到 `/var/www/dcagent/admin/`。请使用实际 Nginx 运行用户和受控的发布目录：

```bash
sudo install -d -m 0755 /var/www/dcagent/admin.staging
sudo cp -a admin-frontend/dist/. /var/www/dcagent/admin.staging/
release_id=$(date +%Y%m%d%H%M%S)
sudo mv /var/www/dcagent/admin "/var/www/dcagent/admin.previous.$release_id" 2>/dev/null || true
sudo mv /var/www/dcagent/admin.staging /var/www/dcagent/admin
```

生产环境建议在维护窗口内完成目录切换，并保留 `admin.previous` 作为回滚目录。

## Nginx 配置

将 [`admin-frontend-subpath.nginx.conf.example`](./admin-frontend-subpath.nginx.conf.example) 合并到实际 server 块。

关键行为：

- `/admin` 规范化重定向到 `/admin/`；
- `/admin/` 下的静态文件从 `/var/www/dcagent/admin/` 提供；
- history-mode 路由刷新时回退到 `/admin/index.html`；
- `/api/` 继续反向代理到宿主机 FastAPI `127.0.0.1:8000`；
- API 请求不会变成 `/admin/api/`。

检查并重载：

```bash
sudo nginx -t
sudo systemctl reload nginx
```

## 验收

```bash
curl --fail --silent --show-error -I http://127.0.0.1/admin/
curl --fail --silent --show-error -I http://127.0.0.1/admin/knowledge
curl --fail --silent --show-error -I http://127.0.0.1/admin/favicon-logo.svg
curl --fail --silent --show-error http://127.0.0.1/api/version
```

浏览器中直接打开并刷新以下地址：

```text
/admin/
/admin/knowledge
/admin/quality/reports
```

检查构建产物中的资源前缀：

```bash
rg '/admin/assets/|/admin/favicon-logo.svg' admin-frontend/dist/index.html
```

如果构建使用了 `/operations/`，Nginx location、验收 URL 和资源前缀也必须全部改为 `/operations/`。

## 回滚

恢复上一版静态目录后重新加载 Nginx。将下面的时间戳替换为实际上一版目录：

```bash
sudo mv /var/www/dcagent/admin /var/www/dcagent/admin.failed.$(date +%Y%m%d%H%M%S)
sudo mv /var/www/dcagent/admin.previous.YYYYMMDDHHMMSS /var/www/dcagent/admin
sudo nginx -t
sudo systemctl reload nginx
```

此回滚只影响管理端静态文件，不需要重启 API、Worker 或 ClickHouse。
