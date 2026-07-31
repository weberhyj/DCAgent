# Ubuntu 内网部署事务与启动门禁设计

## 背景

Ubuntu Bash 部署入口已经具备环境校验、secret 生成和 Compose 拓扑预检能力，但现有环境准备流程仍把目录权限、secret 发布和 `.env` 写入拆成多个局部事务。连续故障注入和代码审查证明，继续增加局部回滚分支无法可靠覆盖 PostgreSQL 初始化竞态、跨阶段失败和强制终止恢复。

本设计将 Ubuntu 生产路线重构为一套共享部署状态协议：环境准备和 Compose 执行共用排他锁；首次请求启动 Compose 后写入持久标记；环境准备使用带日志的完整事务。目标是让所有支持的操作满足“完整成功、完整回滚，或明确进入人工恢复状态”之一，不允许静默留下半升级配置。

## 决策

采用“完整事务 + 部署启动标记”方案。

- `prepare_offline_env.sh` 和 `invoke_offline_compose.sh` 共用 Linux 排他文件锁。
- 首次执行任何受支持的 `docker compose ... up`、`exec` 或 `cp` 前，原子写入部署启动标记。
- 部署启动标记或 PostgreSQL `PG_VERSION` 存在后，`--rotate-secrets` 对普通命令永久 fail closed，只能通过受审计的人工恢复流程解除标记。
- `.env`、目录、权限和 secret 由同一个环境准备事务统一提交或恢复。
- `${HOST_DATA_ROOT}` 和 `${HOST_MODEL_ROOT}` 只从清理后的进程环境展开，不从 `.env` 映射中递归查找。
- Windows PowerShell 入口继续作为开发机兼容路线，不依赖 Linux 文件锁实现。

## 范围

本次设计包含：

- 新增共享部署状态组件；
- 新增受审计的 Ubuntu 恢复入口；
- 重构 `tools/offline_env.py` 的环境准备事务；
- 扩展 `tools/offline_compose.py` 的锁、启动标记和主机变量展开；
- 更新 Ubuntu 生产文档中的轮换、故障恢复和目标机验收说明；
- 增加跨阶段故障、竞态、强制终止遗留状态和 Compose 标记测试。

本次设计不包含：

- 自动修改 PostgreSQL role 密码；
- 初始化完成后的 secret 在线轮换；
- 自动删除部署启动标记；
- 放宽直接执行 `docker compose`、远程 Docker、非默认 context 或 rootless Docker 的限制；
- 改写 PowerShell 兼容入口的内部实现。

## 组件边界

### 共享部署状态组件

新增 `tools/offline_deployment_state.py`，只负责部署状态协议，不解析 `.env`，也不执行 Docker：

- 从 `.env` 中读取固定的绝对 `DEPLOYMENT_STATE_ROOT`；首次初始化或旧部署接管时才允许根据规范化 `DATA_ROOT` 生成默认值；
- 获取和释放 Linux `fcntl.flock` 排他锁；
- 以 30 秒为固定上限等待锁；
- 原子创建和读取部署启动标记；
- 创建权限为 `0700` 的事务目录；
- 原子写入不含敏感值的事务阶段记录；
- 检测未完成事务并 fail closed。

部署状态不再放在仓库本地 `artifacts` 中。首次初始化或旧部署接管会把绝对 `DEPLOYMENT_STATE_ROOT=<canonical DATA_ROOT>/.dcagent-deployment-state` 写入 `.env`；之后普通命令只认该固定路径，不会因为 `DATA_ROOT` 被修改而在新位置隐式建立第二套状态。两个 checkout 或 worktree 只要指向同一数据根并使用同一个 state root，就会共享同一个锁、启动标记和事务目录。

固定路径：

```text
${DEPLOYMENT_STATE_ROOT}/deployment.lock
${DEPLOYMENT_STATE_ROOT}/deployment-started.json
${DEPLOYMENT_STATE_ROOT}/deployment-identity.json
${DEPLOYMENT_STATE_ROOT}/transactions/<transaction-id>/
${DEPLOYMENT_STATE_ROOT}/control-transactions/<transaction-id>/
${DEPLOYMENT_STATE_ROOT}/history/<transaction-id>.json
${DEPLOYMENT_STATE_ROOT}/quarantine/<transaction-id>/
```

状态目录是完整业务预检前唯一允许创建的持久对象，必须为当前部署账号所有且模式为 `0700`。普通准备/Compose 命令要求 `DEPLOYMENT_STATE_ROOT` 和 identity 已存在，不得隐式初始化。只有显式 `prepare_offline_env.sh --initialize-state` 或恢复执行器的 `adopt-existing` 能在取得锁后以 control WAL 和 exclusive-create 创建 identity，并绑定规范化后的 state/data/model/secret roots。后续任一绑定值变化都 fail closed；本设计不提供在线 data-root 迁移，更换数据根必须按新部署处理。

状态引导顺序固定为：只读解析 `.env` → 使用两个主机变量 allowlist 展开 data/model roots → 读取已有 `DEPLOYMENT_STATE_ROOT`，或仅在显式初始化/接管模式下推导默认 state root → 对现有路径组件执行绝对路径、owner、mode 和无符号链接检查 → 竞态安全地只创建 state root 与 lock file → 获取共享锁 → 在锁内 exclusive-create 或验证 identity → 执行完整业务预检。identity 文件和父目录都必须 `fsync`。两个进程即使同时首次启动，也会在同一规范化 state root 下竞争同一锁；无法安全规范化时不得创建状态目录。

事务 ID 使用 UUIDv4 小写十六进制字符串。启动标记只记录固定 JSON schema：`schema_version`、`created_at`、`operation`、`deployment_identity_hash`，不记录环境变量、主机路径、命令参数或 secret。成功事务会把不含敏感值的 receipt 移入 `history`，至少记录事务 ID、UTC 时间、最终阶段、变更对象类别和部署身份哈希；staging 与 backup 不进入历史记录。

固定 schema 约束：

- `schema_version` 当前只能为整数 `1`；
- `operation` 只能为 `up`、`exec`、`cp` 或 `legacy_adoption`；
- `transaction_id` 为 32 位 UUIDv4 小写十六进制；
- 阶段记录只能包含 transaction ID、阶段枚举、UTC 时间、部署身份哈希和对象类别集合；
- history receipt 默认保留，不自动过期，由公司的审计归档策略统一管理。

部署身份使用 canonical JSON（UTF-8、字段名排序、无多余空格）序列化，哈希为 SHA-256。哈希字段固定为 `schema_version`、`deployment_uuid`、`state_root`、`data_root`、`model_root` 和 `secret_root`；仓库 checkout 路径不进入哈希。不同 checkout 会先共享同一锁，再因 secret root 或 identity 不匹配 fail closed。所有 UTC 时间统一为 RFC 3339、微秒精度、`Z` 后缀，禁止本地时区。

state root、transactions、control-transactions、history、quarantine 目录固定为 `0700`；lock、identity、marker、阶段记录、undo manifest、receipt 固定为 `0600`。

### 环境准备规划器

`tools/offline_env.py` 保留 CLI、`.env` 解析和业务规则，并新增不可变的准备计划。计划阶段只读取状态，必须完成：

- 当前账号、UID/GID、Docker endpoint/context 契约检查；
- `.env` 重复键、ClickHouse 旧配置升级规则和固定 secret 路径检查；
- 所有主机路径、符号链接、文件类型、owner 和现有 secret 内容检查；
- 需要创建的全部父子目录计算；
- 需要保留、生成或轮换的 PostgreSQL/ClickHouse secret pair 计算；
- 原有 `.env` 内容、文件模式、现有目录模式和 owner 快照；
- 部署启动标记与 `PG_VERSION` 的 fail-closed 检查。

规划器不得创建目录、修改权限、生成正式 secret、改写 `.env` 或移动现有文件。

### 环境准备事务

环境准备事务在持有共享排他锁期间执行：

1. 检测未完成事务，存在时停止并报告恢复目录。
2. 完成无副作用预检并生成准备计划。
3. 创建事务目录，持久化原 `.env` 备份或“原文件不存在”记录，并写入 write-ahead 阶段日志。
4. 将新 secret 写入事务 staging，权限固定为 `0600`，并在发布前验证完整集合。
5. 再次使用 `lstat()` 检查部署启动标记和 `PG_VERSION`。
6. 逐级创建目录，记录本次创建的每个祖先，并记录所有将被修改的原 mode。事务不对已有对象执行 `chown`；owner 不匹配必须在预检阶段失败，新对象由当前账号创建后复验 owner。
7. 把旧 secret 移入事务 backup，再发布新 secret。
8. 设置并复验目录 `0700`、secret `0600`、owner UID/GID 和 secret 内容。
9. 最后原子写入 `.env`，再重新读取验证所需键、路径和 UID/GID。
10. 将事务阶段原子更新为 `committed`，随后清理 staging、backup 和事务目录，并写入脱敏 history receipt。

write-ahead 状态机固定为：

```text
planned
staging
staged
backing_up
backup_complete
publishing
published
verifying
verified
env_committing
env_committed
committed
committed_cleanup_required
rollback_in_progress
rollback_failed
```

阶段文件只记录总体状态；确定性恢复由独立 `undo-manifest.json` 和 `operations/<sequence>.json` 提供。undo manifest 在任何业务 mutation 前持久化，逐对象记录：规范化路径、对象类型、原存在性、原 mode、对应 backup 名称、预期动作，以及不含 secret 内容的 before/after 判定信息。owner 只记录用于复验，因为工具不执行 `chown`。每个对象操作都有单独的 intent/done 记录：先写入并 `fsync` intent，再执行操作，再原子更新为 done 并 `fsync`。注释、内存快照或仅有对象类别集合不能作为恢复依据。

恢复不能仅依赖 done 标志。对“intent 已持久化、mutation 可能发生、done 未持久化”的记录，恢复器必须观察磁盘状态并按操作类型判定：

- `mkdir`：路径不存在表示未执行；路径存在且由本事务创建、owner/mode/空目录符合预期表示已执行；其它状态为冲突；
- `chmod`：当前 mode 等于 before 表示未执行，等于 after 表示已执行，其它 mode 为冲突；
- `active_to_backup`：active 存在且 backup 不存在表示未执行，active 不存在且 backup 存在表示已执行，其它组合为冲突；
- `staging_to_active`：staging 存在且 active 不存在表示未执行，staging 不存在且 active 存在并通过目标 secret 验证表示已执行，其它组合为冲突；
- `env_replace`：active `.env` 的 SHA-256 等于 before digest 表示未执行，等于 after digest 表示已执行，其它内容为冲突；原 `.env` 不存在时使用显式 absent 标志；
- `unlink`/cleanup：对象存在表示未执行，不存在表示已执行；对象类型改变为冲突。

判定为未执行时跳过补偿；判定为已执行时执行幂等逆操作；冲突状态立即进入 `rollback_failed` 并保留材料。secret 不把内容 digest 写入日志，使用 active/staging/backup 的位置组合、pair 语法校验和 metadata 判断。

任何具有副作用的总体阶段都必须先持久化 `*_in_progress` 或对应意图阶段，再执行逐对象操作，全部对象完成并持久化后才进入下一阶段。阶段文件、undo manifest 和 operation record 都采用“同目录临时文件 → 文件 `fsync` → 原子 `os.replace` → 父目录 `fsync`”；`.env` 使用其自身目录中的临时文件执行同样流程。旧 `.env` 内容必须复制到事务 backup 并 `fsync`，原文件不存在则在 undo manifest 中记录。

secret staging/backup 固定在 `${SECRET_ROOT}/.dcagent-transactions/<transaction-id>/{staging,backup}`，目录 `0700`、文件 `0600`，因此与 active secret 天然位于同一文件系统。state transaction journal 记录该固定 companion 路径；启动扫描 state transactions 时也检查对应 secret companion，扫描 secret companions 时反向要求存在 state journal。孤立、类型异常或 identity 不匹配的 companion 一律 fail closed，只能通过恢复执行器 quarantine。数据目录不依赖跨文件系统 rename，只记录逐项创建和权限变化。

恢复规则固定如下：

| 最后持久阶段 | 恢复动作 |
| --- | --- |
| `planned`、`staging`、`staged` | active 状态未改变；删除已完成的 staging 操作 |
| `backing_up`、`backup_complete` | 根据 done 记录把已移动旧文件恢复到 active |
| `publishing`、`published`、`verifying`、`verified` | 逆序删除已发布新文件并恢复旧 secret、mode 和新建目录 |
| `env_committing`、`env_committed` | 先恢复持久化旧 `.env`，再执行 secret/目录逆序恢复 |
| `rollback_in_progress` | 幂等重放仍未完成的逆序恢复 |
| `rollback_failed` | 停止自动处理，保留全部材料并要求显式恢复 |
| `committed`、`committed_cleanup_required` | 保留新状态，只执行幂等 cleanup reconciliation |

commit 后的可重入顺序固定为：先原子写入 `history/<id>.json`，状态为 `committed_cleanup_pending`；再删除 staging/backup 并 `fsync` 相关父目录；随后把 receipt 更新为 `complete`；最后删除事务目录并 `fsync` transactions 父目录。进程在任一点终止时，下次运行依据 committed transaction 或 pending receipt 幂等续做清理。receipt 已存在不得视为冲突，内容与 deployment identity 不一致时才 fail closed。

提交点之前的任何异常都执行逆序恢复：恢复 `.env`、恢复旧 secret、恢复原 mode、删除本次创建的文件和目录。目录删除从最深层开始，只删除本事务创建且仍为空的目录。

如果恢复失败，事务目录、持久化的旧 `.env` 和可用 backup 必须保留，错误只报告恢复路径和失败阶段，不输出 secret 内容。如果状态已经验证并标记为 `committed`，但清理失败，则不尝试把已提交状态回滚到不确定的中间状态；保留已提交状态和事务目录，报告 `committed_cleanup_required`，下次操作先进入 cleanup reconciliation，完成前其它准备或 Compose 命令 fail closed。

### Compose 执行器

`tools/offline_compose.py` 在渲染和执行 Compose 时使用共享排他锁：

- `config`、`build` 和 `down` 不创建部署启动标记；
- 任何 profile 下的 `up`、`exec` 或 `cp` 都在实际调用 Docker 前原子创建标记，因为 `exec` 可运行写命令，`cp` 也可向容器写入文件；
- `up` 失败后标记仍保留；
- 已存在的普通标记、符号链接、损坏链接或不可读取对象均按“部署已启动”处理；
- 禁止直接删除标记的 CLI 参数。

Compose 执行器在任何命令前都必须检测未完成事务、`rollback_failed` 和 `committed_cleanup_required`，发现后 fail closed。共享锁从状态检查前一直持有到 Docker Compose 子进程退出；因此长时间 `build`、前台 `up` 或 `exec` 会让另一个管理操作在等待 30 秒后明确失败，这是预期的串行化行为。

Compose 环境拆成两个明确对象：

- `.env` 配置映射：提供 `DATA_ROOT`、`MODEL_ROOT` 等配置值；
- 清理后的进程环境：从 `os.environ` 复制 `PATH`、`HOME`、`DOCKER_CONFIG` 等 Docker 运行所需键，删除全部 `COMPOSE_*`、`DOCKER_HOST`、`DOCKER_CONTEXT` 和 `.env` 中出现的全部键，再单独保留获准的 `HOST_DATA_ROOT`、`HOST_MODEL_ROOT`。

路径解析只允许下列形式：

```text
DATA_ROOT=/absolute/path
DATA_ROOT=${HOST_DATA_ROOT}
MODEL_ROOT=/absolute/path
MODEL_ROOT=${HOST_MODEL_ROOT}
```

路径展开器只接收由 `HOST_DATA_ROOT`、`HOST_MODEL_ROOT` 构成的 allowlist 映射；引用其它变量、变量未定义、引号、相对路径、符号链接或重定向到非批准位置时必须 fail closed。Compose 子进程使用同一个清理环境，确保 Python 预检和 Docker Compose 获得一致的主机变量。

批准位置定义为部署身份中绑定的规范化 `DATA_ROOT` 和 `MODEL_ROOT`。首次绑定时，工具对每个已存在路径组件执行 `lstat()`，拒绝符号链接，要求最近的已存在祖先由部署账号所有且不允许 group/other 写入；随后在 mutation 前再次检查。威胁模型假设部署账号本身可信，且运维不在受支持工具持锁期间用同一账号并发替换路径。其他账号无法写入受保护祖先，因此不能利用检查后的路径替换。

路径规范化算法固定为：输入必须是绝对路径或完整的单一 `${HOST_DATA_ROOT}`/`${HOST_MODEL_ROOT}` token；先拒绝引号、NUL、空白包裹、嵌入变量、任何原始 `..` 路径段和 `//` 双前导斜杠，再执行 Linux `normpath`，除根目录外移除尾部 `/`。对从根到最后一个已存在组件逐级 `lstat()`；不存在的尾部组件保持规范化拼接，不能通过 `resolve()` 隐式跟随链接。

Compose verb allowlist 固定为：`config`、`build`、`up`、`down`、`exec`、`cp`。`run`、`create`、`start`、`restart`、`scale`、`pull`、`push`、`logs`、`ps` 以及任何未列出的 verb 一律 fail closed。解析器先处理 wrapper 允许的全局 `--profile <approved-profile>`，然后把第一个非全局参数作为 verb；`exec`/`cp` 的 `--` 及其后参数只作为子命令参数，不再参与 Compose verb 识别，但标记已在执行前创建。

Compose 子进程固定在仓库根目录运行，命令前缀固定为：

```text
docker --context default compose --project-name dcagent-offline --env-file <absolute deploy/offline/.env> -f <absolute deploy/offline/compose.yaml>
```

环境构造从空映射开始，只复制 `PATH`、`HOME`、`USER`、`LOGNAME`、`TMPDIR`、`XDG_RUNTIME_DIR`、`DOCKER_CONFIG`、`LANG`、`LC_ALL`、`SSL_CERT_FILE`、`SSL_CERT_DIR`；检测到 `DOCKER_HOST`、`DOCKER_CONTEXT`、`DOCKER_TLS_VERIFY` 或任意 `COMPOSE_*` 时先拒绝，再从子进程环境移除。`.env` 不得定义 `HOST_DATA_ROOT` 或 `HOST_MODEL_ROOT`；两个 HOST 值只能来自调用进程，并在清理后重新加入。所有 `.env` key 都从子进程环境移除，避免进程环境高优先级覆盖显式 `--env-file`。不使用隐式项目 `.env` 发现。

### 人工恢复执行器

新增最小 Bash 入口 `tools/recover_offline_deployment.sh` 和 Python 核心 `tools/offline_recovery.py`。它们只接受以下显式动作：

- `inspect --state-root <absolute-path> --transaction <id>`：只读输出脱敏阶段、对象类别和可选恢复动作；
- `resume-rollback --state-root <absolute-path> --transaction <id>`：依据 undo records 幂等继续既定 rollback，不允许临时选择另一套状态；
- `finalize-cleanup --state-root <absolute-path> --transaction <id>`：只处理 `committed`/`committed_cleanup_required` 的幂等 cleanup；
- `clear-start-marker --state-root <absolute-path>`：执行容器、权威数据根、`PG_VERSION` 和未完成事务四项检查后清除标记；
- `adopt-existing --state-root <absolute-path>`：显式、幂等地接管旧部署；
- `acknowledge-repaired --state-root <absolute-path> --transaction <id> --evidence <absolute-path>`：在运维已人工修复损坏日志或冲突状态后，重新验证 active `.env`、secret、mode、owner 和部署身份，把原事务移入 quarantine 并写入恢复 receipt；不直接删除损坏材料。

恢复命令必须显式提供 state root，因此即使 `.env` 缺失或损坏也能定位权威 identity。state root 使用与普通路径相同的绝对化、`//`/`..`/符号链接拒绝规则，且必须包含有效 identity，只有 adopt 允许 identity 尚不存在。全新部署初始化由 prepare 的显式模式完成。

恢复执行器从动作开始到审计 receipt 持久化结束始终持有同一排他锁。所有改变状态的动作本身也使用 `control-transactions/<id>` 下的 WAL/undo 协议：adopt 只有在 runtime 检查和必要 marker durable 后才发布 identity；clear marker 先把 marker 移入 control backup，审计 receipt durable 后才提交删除；中断后普通命令会检测未完成 control transaction 并 fail closed，恢复命令可幂等续做。所有动作把脱敏结果写入 `${DEPLOYMENT_STATE_ROOT}/history/recovery-<uuid>.json`，schema 包含动作、事务 ID、开始/结束 UTC、检查结果、exit code 和部署身份哈希。普通 prepare/Compose 命令不能隐式执行这些动作。

## PostgreSQL 初始化保护

`--rotate-secrets` 只允许在首次 Compose `up`、`exec` 或 `cp` 之前使用。

环境准备在计划阶段和 secret 发布临界区各执行一次检查：

- 使用 `lstat()` 检查 `deployment-started.json`；
- 使用 `lstat()` 检查解析后 `DATA_ROOT/postgres/PG_VERSION`；
- 普通文件、目录、符号链接、损坏链接和无法检查的对象均拒绝轮换。

共享锁阻止受支持的环境准备和 Compose 命令并发。部署启动标记在 Docker 调用前写入，因此即使 `up -d` 很快返回、PostgreSQL 尚未创建 `PG_VERSION`，后续轮换也会被标记阻止。该禁止对普通命令不可逆；只有满足恢复检查的人工流程可以解除标记。直接绕过 wrapper 的 Docker 操作不属于支持范围，并继续在文档中明确禁止。

## 故障恢复

### 可自动恢复

事务提交前的普通异常、校验失败和可捕获中断会依据 write-ahead 阶段恢复持久化的旧 `.env`、旧 secret 和原 mode。工具从不改变已有对象 owner，因此无需恢复 owner。自动恢复成功后删除事务目录，并返回非零退出码。

### 必须人工处理

以下状态必须 fail closed，并在错误中报告事务目录：

- rollback 过程中任一文件恢复失败；
- 进程被强制终止后遗留非终态事务；
- 已提交状态的 backup/staging 清理失败；
- 事务阶段记录损坏、类型异常或无法读取。

工具不自动猜测应该保留新 secret 还是旧 secret。运维必须根据事务阶段、事务中的旧 `.env`、当前 `.env` 和 backup 内容完成恢复，再删除事务目录。

`committed_cleanup_required` 只允许执行清理，不允许选择旧状态。运维必须在持锁状态下确认：阶段为 `committed` 或 `committed_cleanup_required`、当前 `.env` 可重新解析、四个 active secret 形成两个完整 pair、数据库 URL 可由当前 PostgreSQL 密码重新派生、所有 active secret owner/mode 合规。确认后才能删除 staging/backup，`fsync` 父目录并写入 history receipt。任一验证失败时不得删除恢复材料。

### 部署启动标记恢复

标记不会因 `up`、`exec` 或 `cp` 失败而自动删除。只有同时确认以下条件后，运维才可按 runbook 人工删除：

- 没有 DC-Agent Compose 容器正在运行或创建；
- `DATA_ROOT/postgres/PG_VERSION` 不存在；
- PostgreSQL 数据目录不存在或确认从未初始化；
- 不存在未完成环境准备事务。

恢复时以状态目录中的 `deployment-identity.json` 为权威身份；如果 `.env` 缺失或损坏，运维不能自行猜测数据根。人工清理必须在持有同一排他锁时进行，并把检查结果、容器列表、权威数据根、`PG_VERSION` 检查和事务列表写入审计记录。文档不提供单行删除命令。

## 旧部署接管与身份迁移

已有部署不能由普通 prepare 自动接管，必须通过恢复执行器的 `adopt-existing` 显式执行。接管状态机为 `adoption_planned → identity_created → runtime_checked → marker_written_or_rotation_enabled → adoption_complete`，每步可幂等重入：

- 从现有 `.env` 和获准主机变量得到权威 `DATA_ROOT`、`MODEL_ROOT`；
- 如果当前数据根存在 `PG_VERSION`，或 Compose 中存在对应运行中/创建中的容器，创建 `operation=legacy_adoption` 的启动标记，禁止 secret 轮换；
- 只有 `PG_VERSION` 不存在、PostgreSQL 数据目录未初始化且没有对应运行中/创建中的容器时，创建部署身份后才允许首次 `up`、`exec` 或 `cp` 前轮换；
- 状态身份一旦建立，普通命令不得改变 data/model/repository/secret roots；
- 更换 checkout 只允许在无未完成事务、无运行容器、identity 中绑定的 secret root 与新 checkout 经人工审核一致时使用；本设计不提供自动 secret-root 迁移；
- 更换 `DATA_ROOT` 或接管另一套 PostgreSQL 数据必须作为新部署处理，不能复用旧状态目录。

全新部署使用 `prepare_offline_env.sh --initialize-state`。如果 `.env` 不存在，该模式只读加载 `.env.example` 和 HOST allowlist 来推导 roots；判定条件全部满足后，先用 control WAL 发布 identity，再继续执行普通环境准备事务，并在最终 `.env` 中写入绝对 `DEPLOYMENT_STATE_ROOT`。该模式可幂等重入：identity 已存在且匹配时继续准备，不匹配时 fail closed。

初始化判定要求：canonical `DATA_ROOT` 必须由运维预先创建为空的 `0700` 目录并由部署账号所有；不存在 `PG_VERSION`；PostgreSQL 子目录不存在或为空；没有匹配 `dcagent-offline` project 的运行中、已创建或停止容器；不存在旧 marker 或无法恢复的 control transaction。`DATA_ROOT` 的父目录创建不属于工具事务，runbook 使用 `install -d -m 0700 <data-root>` 明确完成。任一条件不满足时必须改用 `adopt-existing`，不能通过 initialize 绕过 marker。

新目标机的固定顺序为：创建空 data/model roots → `prepare_offline_env.sh --initialize-state` → Compose `config/build/up`。旧目标机先执行 `recover_offline_deployment.sh adopt-existing --state-root ...`，再执行普通 `prepare_offline_env.sh`。initialize/adopt 完成后把绝对 `DEPLOYMENT_STATE_ROOT` 原子写入 `.env`；如果后续同时修改 `.env` 的 data root 和 state root，工具会把它视为显式选择另一套部署，而不是同一部署内变更。安全保证限定为：一个 state root 内绑定不可改变，且普通命令绝不隐式创建新 identity。

PowerShell 入口明确仅用于 Windows 开发机，不得指向或管理 Ubuntu 生产部署，因此不属于共享锁并发保证的范围。

## 安全属性

- 部署状态目录和事务目录为 `0700`；其中包含 secret 的文件为 `0600`。
- 日志、阶段记录和异常不得包含 secret、数据库 URL、模型响应或原始 evidence。
- 所有现有文件的类型和 owner 在 mutation 前检查；工具不对已有对象执行 `chown`；所有最终 mode 和 owner 在 mutation 后复验。
- backup 生命周期覆盖 secret 发布、权限复验和 `.env` 提交，不在整体事务完成前删除。
- 部署锁、启动标记和事务目录只能位于绑定的 `${DATA_ROOT}/.dcagent-deployment-state`；secret 只能位于绑定的仓库 secret 根，二者都不能通过 `.env` 重定向到其它位置。
- Windows PowerShell 入口保持兼容，但 Ubuntu 生产文档只使用 Bash/Python 路线。

## 测试设计

### 共享状态协议

- 锁竞争、30 秒超时和释放；
- 标记原子创建、重复创建幂等和异常对象 fail closed；
- 未完成事务检测；
- 阶段记录不包含敏感字段；
- Windows 测试环境使用可注入锁后端，Linux 目标测试使用真实 `fcntl.flock`。
- 两个独立 checkout 指向同一 `DATA_ROOT` 时必须竞争同一锁并读取同一启动标记；指向不同部署身份时不能复用状态。
- 普通命令缺少 identity 时拒绝；prepare `--initialize-state`/恢复 `adopt-existing` exclusive-create 并发时只有一个成功建立身份，另一个幂等验证或明确失败；initialize 对已有容器或非空 PostgreSQL 目录拒绝。

### 环境准备故障矩阵

依次在以下边界注入失败，并验证旧状态完整恢复：

- 创建任一父目录；
- chmod 和 chmod 后 metadata 复验；
- secret staging、backup、publish；
- staging 和 active secret 验证；
- `.env` 临时文件写入和原子替换；
- commit 前阶段记录；
- rollback 中的文件恢复；
- commit 后 cleanup。

故障注入必须在每个 operation intent 和 done 之间模拟强制终止，并覆盖 mkdir/chmod/active-to-backup/staging-to-active/env-replace/unlink 的“未执行、已执行、冲突”三种可观察状态，根据恢复表验证重启后的确定性动作。旧 `.env` 必须从磁盘 backup 恢复，不能依赖进程内存。

目录测试必须覆盖多层父目录、部分创建、逆序删除和已存在目录不被删除。secret 测试必须覆盖 PostgreSQL/ClickHouse 两个 pair、旧部署升级、空键、partial pair、owner 错误和 backup 部分失败。

### 初始化与竞态

- 计划阶段已存在普通 `PG_VERSION`；
- `PG_VERSION` 为符号链接或损坏链接；
- 第一次检查后、发布前才出现 `PG_VERSION`；
- 部署启动标记预先存在；
- `up`、`exec`、`cp` 执行前创建标记；
- `up`、`exec`、`cp` 返回失败仍保留标记；
- `config/build/down` 不创建标记。
- 未完成事务存在时，六个 allowlist verb 全部拒绝；未列出 verb 在参数解析阶段拒绝。

### Compose 主机变量

- `HOST_DATA_ROOT`、`HOST_MODEL_ROOT` 从清理后的进程环境展开；
- 未定义变量、未批准变量、引号、相对路径和符号链接被拒绝；
- Python 渲染预检与实际 Compose 子进程接收相同环境；
- 进程环境不得覆盖 `.env` 中的非主机变量或 Compose project/file/profile。
- `.env` 定义 HOST 变量时拒绝；固定 cwd、`--env-file`、project name、compose file 和环境 allowlist 与 Docker 子进程实际参数完全一致。

### 恢复执行器

- `inspect` 只输出脱敏元数据；
- `resume-rollback` 可从每个 WAL 中断阶段幂等恢复；
- `finalize-cleanup` 可从 pending receipt、committed transaction 或部分 cleanup 重入；
- `clear-start-marker` 任一检查失败时不删除标记；
- 所有 mutation action 在持锁范围内写入恢复审计 receipt；
- prepare initialize、recovery adopt/clear-marker 在 control WAL 每个 intent/done 间被终止后都能幂等恢复；identity 只能作为 runtime/marker 检查后的最后业务发布对象；
- `.env` 缺失或损坏时，恢复命令可通过显式 state root 定位 identity；普通命令仍 fail closed；
- 损坏 journal 只有在 active 状态重新验证并写入 evidence receipt 后才能移入 quarantine；
- live recovery 演练结束后没有未完成事务、backup 或 staging 残留。

### 回归与目标机门禁

- 完整 `tools/tests` 和 `backend/tests`；
- 原 PowerShell prepare/Compose 专项契约；
- Ruff check、Ruff format check、`uv --no-config lock --check`；
- 两个 Bash 脚本的 `bash -n`、LF 和 `100755`；
- 全新 Ubuntu 20.04 上依次运行：创建空 roots、`prepare_offline_env.sh --initialize-state`、Compose `config`、五个内部镜像 `build`、`up -d`、API `readyz`、Physoc probe、Ollama embed/generate/tags probe、owner/mode 检查和隔离的强制终止恢复演练；旧部署以显式 `adopt-existing` 替代 initialize。

目标机验收统一写入脱敏 JSON 报告，包含命令类别、开始/结束 UTC、exit code、耗时、服务健康状态、owner/mode 检查和恢复演练结果，不记录 secret、提示词、模型正文或原始 SSE。每步使用文档固定超时；任一步失败立即停止后续上线证据生成，但保留诊断状态，不自动执行破坏性 `down -v`。

固定目标机超时和结果：

- 锁等待：30 秒；超时返回非零并报告持锁状态文件，不打印进程环境；
- Compose `config`：60 秒，必须产生可解析 JSON；
- 镜像 `build`：1800 秒，五个内部镜像必须全部成功；
- Compose `up -d`：300 秒；
- API `readyz`：最长等待 300 秒，最终必须 HTTP 200；
- Physoc 和三个 Ollama probe：每项 60 秒，输出只能进入既有脱敏报告；
- 强制终止恢复演练：在隔离的临时 deployment identity、临时 data/model/secret/state roots 下执行，不启动生产 Compose project；120 秒内必须检测到遗留事务并 fail closed，随后通过恢复执行器完成 rollback 或 cleanup。演练结束必须删除临时 marker、active test secrets、data/model roots、state/history/quarantine 和 companion 目录，并确认没有相关容器、未完成事务、backup、staging 或审计外敏感材料残留。

Windows 开发机没有目标 Docker 拓扑时，只能报告本地门禁结果，不得声称 Ubuntu live gate 通过。

## 文档更新

以下文档需要同步说明共享锁、部署启动标记、轮换限制和人工恢复：

- `docs/intranet-deployment-configuration.md`；
- `docs/offline-platform-runbook.md`；
- `deploy/offline/README.md`；
- `README.md` 的 Ubuntu 生产部署章节。

根 README 继续只保留一段 Windows 开发机 `.ps1` 兼容说明。生产文档不得提供删除启动标记的快捷命令，必须列出人工清理前的四项检查。

## 验收标准

- Ubuntu 生产 Bash/Python 环境准备和 Compose 操作不能并发修改部署状态；Windows PowerShell 开发机路线不在该生产并发保证内；
- 首次 `up`、`exec` 或 `cp` 请求后无法再通过普通支持入口轮换 secret；
- 任一提交前故障不会留下新 `.env`、新 secret、错误权限或本次创建的父目录；
- rollback 不完整或强制终止后，恢复材料被保留且后续操作 fail closed；
- `${HOST_DATA_ROOT}`、`${HOST_MODEL_ROOT}` 在 Python 预检和 Compose 中解析一致；
- PowerShell 开发机兼容测试不回归；
- 所有本地自动化门禁通过；
- Ubuntu 20.04 目标机 live gate 有单独、可审计的执行结果。
