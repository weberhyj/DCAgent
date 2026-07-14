# DC-Agent API Contract

本文档记录用户端、知识库管理端和 FastAPI 后端的当前接口契约。用户端只消费问题与回答；Agent 步骤、资料库索引状态和解析片段仅供管理端使用。

## Database

本地默认使用 PostgreSQL：

```text
postgresql+psycopg://postgres:123456@127.0.0.1:5432/dc_agent
```

可以通过 `DATABASE_URL` 覆盖。后端启动时会自动创建包括 `agent_runs`、`agent_steps` 在内的缺失表。

## LLM Provider

离线兜底模式：

```text
LLM_PROVIDER=template
```

OpenAI-compatible 模式：

```text
LLM_PROVIDER=openai_compatible
LLM_API_BASE=https://your-llm-host.example/v1
LLM_API_KEY=replace-with-your-api-key
LLM_MODEL=your-model-name
```

知识库没有可用证据时，DCAgent 会直接返回“未检索到足够依据”，不会调用外部模型自由生成答案。

## Agent Workflow

`POST /api/conversations/{conversationId}/messages` 会启动 LangGraph 只读 Agent：

1. `plan_retrieval`：根据问题和检索模式生成有界检索策略。
2. `search_knowledge`：检索已索引知识片段；证据不足时最多扩展并执行第二轮检索。
3. `inspect_document`：深入检查最多三个资料来源的相关片段。
4. `compare_evidence`：汇总多来源证据并标记可能需要核对的约束措辞。
5. `compose_answer`：将最多五条证据交给 LLM 生成回答。

所有工具均为只读调用。每次运行和步骤都会持久化到 Agent 审计表。

## Core Types

```ts
interface ConversationBundle {
  conversations: Conversation[]
  activeConversationId: string
  messages: ChatMessage[]
}

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  time: string
  content?: string | null
  paragraphs: ResponseParagraph[]
  artifacts: Artifact[]
}

interface KnowledgeSource {
  id: string
  name: string
  sourceType: string
  records: number
  status: '已索引' | '解析中' | '待复核' | '解析失败'
  updatedAt: string
  classification: string
  fileSize?: number | null
  mimeType?: string | null
  errorMessage?: string | null
}

interface KnowledgeChunk {
  id: string
  sourceId: string
  chunkIndex: number
  text: string
  tokenCount: number
}

interface AgentRunAudit {
  id: string
  conversationId: string
  query: string
  mode: 'quick' | 'deep' | 'source'
  status: 'completed' | 'failed'
  startedAt: string
  completedAt: string
  answerMessageId: string
  evidenceCount: number
  sourceCount: number
  steps: AgentStepAudit[]
}

interface AgentStepAudit {
  id: string
  stepIndex: number
  toolName: string
  status: 'completed' | 'failed'
  inputSummary: string
  outputSummary: string
  sourceIds: string[]
  readOnly: boolean
  startedAt: string
  completedAt: string
}
```

用户接口会移除结构化来源详情，不暴露管理员资料原文、`sourceId`、`chunkId` 或解析片段。管理端审计接口保留执行来源 ID，但不返回资料原文。

## Endpoints

### Health and Q&A

- `GET /api/health`：服务健康检查。
- `GET /api/conversations`：返回当前会话和消息。
- `POST /api/conversations`：创建空会话。
- `DELETE /api/conversations/{conversationId}`：删除会话。
- `GET /api/conversations/{conversationId}/messages`：读取消息。
- `POST /api/conversations/{conversationId}/messages`：执行 Agent 并返回更新后的 `ConversationBundle`。

发送问题示例：

```json
{
  "content": "对比差旅制度和财务票据要求",
  "mode": "deep"
}
```

### Knowledge Management

- `GET /api/knowledge/sources`：返回资料来源列表并推进待处理解析任务。
- `POST /api/knowledge/sources`：注册资料来源。
- `POST /api/knowledge/uploads`：批量上传 `.pdf`、`.docx`、`.xlsx`、`.csv`、`.txt`、`.md` 等文件。
- `POST /api/knowledge/sources/{sourceId}/reindex`：重新解析失败资料。
- `DELETE /api/knowledge/sources/{sourceId}`：删除资料及其索引片段。
- `GET /api/knowledge/sources/{sourceId}/chunks`：管理端读取解析片段。

### Agent Audit

- `GET /api/admin/agent/runs`：返回最近 50 次 Agent 执行记录及其只读步骤。

时间字段统一使用 `YYYY-MM-DD hh:mm:ss`。
