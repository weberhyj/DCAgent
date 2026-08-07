# 结构化发布任务外键写入顺序修复设计

## 目标

修复管理员确认 Excel 字段后点击“Publish data”时，PostgreSQL 报错：

```text
insert or update on table "structured_ingestion_jobs"
violates foreign key constraint "structured_ingestion_jobs_publication_id_fkey"
```

修复后，结构化发布记录必须先持久化，再创建引用该记录的发布任务；现有发布状态机、工作进程和数据库表结构保持不变。

## 根因

`StructuredRepository.enqueue_publication()` 在同一个 SQLAlchemy Session 中依次调用 `session.add(publication)` 和 `session.add(job)`，但 `StructuredIngestionJobRecord` 与 `StructuredPublicationRecord` 之间只有数据库外键，没有 ORM `relationship()`。

SQLAlchemy ORM 的 flush 顺序没有获得这两个对象之间的对象级依赖关系。在 PostgreSQL 上，任务记录可能先于发布记录执行 INSERT；任务的 `publication_id` 此时找不到父记录，数据库立即拒绝写入。

现有结构化发布测试主要使用 SQLite。SQLite 连接默认没有启用 `PRAGMA foreign_keys=ON`，因此即使 INSERT 顺序错误，测试也不会触发外键约束，导致问题只在实际 PostgreSQL 环境暴露。

## 方案选择

采用“显式父记录 flush + 外键启用回归测试”。

在添加 `StructuredPublicationRecord` 后立即执行一次 `session.flush()`，确保发布记录已经写入数据库；随后再添加 `StructuredIngestionJobRecord`。最终事务仍由现有 Session 上下文统一提交，因此如果任务写入失败，发布记录也会随事务整体回滚，不会产生孤立记录。

没有选择新增 ORM relationship，因为本次只需要保证创建顺序。新增双向关系会扩大模型状态、级联和删除语义的改动范围。没有选择移除或延迟数据库外键，因为外键是防止孤立任务的重要数据完整性约束。

## 事务数据流

```text
锁定知识源和已确认数据集
  -> 检查是否已有可复用任务
  -> 创建 StructuredPublicationRecord
  -> session.add(publication)
  -> session.flush()，父记录先写入
  -> 创建并添加 StructuredIngestionJobRecord
  -> 刷新知识源状态
  -> Session 统一提交
```

任何后续步骤失败时，整个事务回滚，父记录和任务记录都不会保留半成品。

## 错误处理

- 发布 ID 冲突继续返回现有 `StructuredConflictError`。
- 数据集未确认继续返回现有 409 响应。
- 父记录 flush 发生数据库错误时，不创建任务记录，并由事务回滚。
- 不捕获或隐藏未知数据库完整性错误，避免把真实数据问题错误地转换为可重试任务。
- 前端错误文案本次不修改；修复后正常入队不会进入通用异常提示。

## 测试设计

增加一个启用 SQLite 外键约束的结构化发布仓库测试：

1. 创建知识源、预览和已确认数据集。
2. 对测试数据库连接执行 `PRAGMA foreign_keys=ON`。
3. 调用 `enqueue_publication()`。
4. 断言任务成功创建，且任务引用的发布记录存在。
5. 断言任务状态与发布状态均为 `queued`。

测试在修复前应因外键约束失败，在添加显式 flush 后通过。现有结构化 API、Worker、完整后端测试、Ruff 和 ty 必须继续通过。

## 版本规则

本次只修改后端。完成实现和验证后，后端版本从 `0.1.11` 升级到 `0.1.12`；用户端与管理端版本保持 `0.1.1`。
