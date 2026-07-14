# DC-Agent

DC-Agent 是一个公司内部只读知识 Agent。管理员把制度、合同、会议纪要、经营数据等资料上传到知识库后，DCAgent 会围绕用户问题执行有界的多步检索、资料深挖和证据对比，再基于已索引资料生成回答。

## 项目结构

- `frontend`：用户检索端，面向普通用户提问和查看 DCAgent 生成的答案。
- `admin-frontend`：知识库管理端，面向管理员上传、筛选、重新解析、删除资料源，查看解析片段和 Agent 执行审计。
- `backend`：Python + FastAPI + LangGraph 服务，提供只读 Agent、问答、资料上传、解析、知识库索引和审计接口。

## 本地环境

建议环境：

- Python 3.12 或更高版本
- Node.js 20 或更高版本
- PostgreSQL，本地默认库名为 `dc_agent`

后端默认数据库连接：

```text
postgresql+psycopg://postgres:123456@127.0.0.1:5432/dc_agent
```

手动创建数据库示例：

```powershell
$env:PGPASSWORD="123456"
D:\PostgreSQL\18\bin\createdb.exe -h 127.0.0.1 -p 5432 -U postgres dc_agent
Remove-Item Env:PGPASSWORD
```

也可以用 `DATABASE_URL` 覆盖默认连接串。管理员上传的文件默认保存在 `backend/uploads/knowledge`，该目录只用于本地运行数据，不进入版本管理。

## 环境变量

后端启动时会自动读取项目根目录 `.env` 和 `backend/.env`。读取顺序为根目录 `.env` 后读取 `backend/.env`，但系统环境变量优先级最高，不会被文件覆盖。

复制示例文件：

```powershell
Copy-Item .env.example .env
Copy-Item backend\.env.example backend\.env
```

本地兜底模式：

```text
LLM_PROVIDER=template
```

真实模型模式使用 OpenAI-compatible Chat Completions 接口：

```text
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://your-llm-host.example/v1
LLM_API_KEY=replace-with-your-api-key
LLM_MODEL=your-model-name
```

当知识库没有命中资料时，DCAgent 会返回“未检索到足够依据”，不会调用真实模型编造答案。

两个前端项目都支持通过 `VITE_API_PROXY_TARGET` 覆盖本地 API 代理目标，默认代理到 `http://127.0.0.1:8000`。

## 启动

后端：

```powershell
cd backend
py -m pip install -r requirements.txt
py -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

用户检索端：

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

默认访问地址：`http://127.0.0.1:5173`

知识库管理端：

```powershell
cd admin-frontend
npm.cmd install
npm.cmd run dev
```

默认访问地址：`http://127.0.0.1:5174`

管理端按功能拆分为独立路由：

- `/overview`：管理概览与最近活动。
- `/knowledge`：资料上传、筛选、重建索引和删除。
- `/knowledge/{sourceId}`：指定资料的解析详情与片段预览。
- `/agent-runs`：DCAgent 只读执行审计。

## 本地验证

后端测试：

```powershell
cd backend
py -m unittest discover -s tests -p "test_*.py" -v
```

用户检索端：

```powershell
cd frontend
npm.cmd run test:run
npm.cmd run build
```

知识库管理端：

```powershell
cd admin-frontend
npm.cmd run test:run
npm.cmd run build
```

页面级冒烟（可选，需先确保已安装 Playwright Chromium）：

```powershell
# 终端 1
tools\start_smoke_backend.cmd

# 终端 2
tools\start_smoke_frontend.cmd

# 终端 3
tools\start_smoke_admin.cmd

# 终端 4
py tools\ui_smoke.py
```

冒烟脚本会使用临时后端 `8015`、用户端 `5177`、管理端 `5178`，截图输出到 `qa-screenshots`。

完整冒烟建议：

1. 启动后端。
2. 启动知识库管理端，上传一份 `.txt`、`.md`、`.pdf`、`.docx`、`.xlsx` 或 `.csv` 文档。
3. 等待资料源状态从 `解析中` 变为 `已索引`。
4. 启动用户检索端，询问文档中的制度、合同或业务问题。
5. 确认 DCAgent 回答只基于知识库内容，不在用户侧暴露资料原文管理入口。
6. 回到知识库管理端，确认“Agent 执行审计”中出现本次检索、资料检查、证据对比和回答生成步骤。

## 当前能力

- 用户侧一次性知识检索问答，不展示历史会话侧栏。
- 首次从小输入框进入大聊天时带启动动画和 loading。
- DCAgent 回答支持等待态和逐步显现。
- 用户侧不暴露管理员资料原文，只在回答文本中保留必要引用。
- LangGraph 只读 Agent 最多执行两轮检索、深入检查三个资料来源，并将最多五条证据交给模型。
- Agent 会在证据不足时自动扩展检索词，命中多个来源时执行证据对比；没有证据时拒绝调用模型编造答案。
- 管理端采用路由级模块拆分，支持管理概览、知识库维护、资料解析详情和 Agent 执行审计。
- 知识库模块支持多文件上传、解析状态轮询、失败原因展示、重新解析、单条删除、批量删除和列表筛选。
- 后端支持 PostgreSQL 持久化、上传文件解析、知识片段索引、基础语义扩展检索、Agent 审计持久化和 OpenAI-compatible LLM 接入。
