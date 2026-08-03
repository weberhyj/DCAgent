# Ubuntu 内网部署事务与启动门禁实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 Ubuntu 20.04 内网部署入口升级为共享排他锁、可恢复完整事务、部署启动标记和受审计人工恢复组成的单一安全协议，确保环境准备与 Compose 管理操作串行、可追溯且故障后 fail closed。

**Architecture:** `tools/offline_deployment_state.py` 提供部署身份、状态目录、原子 JSON、文件锁和 WAL 基础协议；`tools/offline_env.py` 负责生成不可变准备计划并在同一锁内提交 `.env`、目录、权限和 secret；`tools/offline_recovery.py` 负责确定性回滚、提交后清理及人工恢复 CLI。`tools/offline_compose.py` 只允许六个 verb，使用同一状态根和锁，在 `up`、`exec`、`cp` 调用 Docker 前写入持久启动标记。

**Tech Stack:** Python 3.12 标准库（`dataclasses`、`fcntl`、`hashlib`、`json`、`os`、`pathlib`、`uuid`）、Bash、Docker Compose v2、`unittest`、Ruff、uv、Ubuntu 20.04 POSIX owner/mode 契约。

---

## 文件职责与边界

- `tools/offline_deployment_state.py`：部署状态根规范化、identity、SHA-256、原子 JSON、`fsync`、Linux `flock`、marker、事务/receipt 路径和静态 schema 校验；不解析 `.env`，不执行 Docker，不决定业务回滚动作。
- `tools/offline_recovery.py`：读取 undo manifest 与 operation intent/done，根据磁盘状态判定并执行回滚或 cleanup；提供六个显式恢复动作及 control transaction 审计。
- `tools/offline_env.py`：保留 `.env` 和 secret 业务规则；新增只读 `PreparationPlan` 与完整提交事务；只有 `--initialize-state` 能为全新部署初始化 identity。
- `tools/offline_compose.py`：固定 Compose 参数、cwd、环境、项目名与文件；锁内执行状态检查、全 profile 渲染、marker 写入和 Docker 命令。
- `tools/intranet_deployment_gate.py`：在 Ubuntu 目标机编排新部署/旧部署验收步骤，只写脱敏 JSON 结果，不写 secret、提示词、模型正文或原始 SSE。
- `tools/recover_offline_deployment.sh`：最小 Bash 恢复入口，只转发到 `offline_recovery.py`。
- `tools/tests/test_offline_deployment_state.py`：状态协议、identity、锁、marker、事务 schema 与跨 checkout 行为。
- `tools/tests/test_offline_recovery.py`：每种 operation 的未执行/已执行/冲突判定、阶段恢复表、control WAL 和恢复 CLI。
- `tools/tests/test_offline_env.py`：规划器无副作用、初始化、完整提交/回滚、故障矩阵、竞态与 PostgreSQL 轮换门禁。
- `tools/tests/test_offline_compose.py`：verb allowlist、清理环境、锁范围、未完成事务、启动标记与 Docker 调用顺序。
- `tools/tests/test_intranet_deployment_gate.py`：目标机步骤、超时、短路、脱敏报告和隔离恢复演练契约。
- `tools/tests/test_ubuntu_deployment_entrypoints.py`：三个 Bash 入口的 LF、shebang、`set -Eeuo pipefail`、helper 映射与 `100755`。
- `tools/tests/test_compose_contract.py`、`tools/tests/test_structured_deployment_contract.py`：文档和部署命令契约。

### Task 1：共享状态根、部署身份、原子文件与排他锁

**Files:**
- Create: `tools/offline_deployment_state.py`
- Create: `tools/tests/test_offline_deployment_state.py`

- [ ] **Step 1：写状态协议 RED 测试**

创建 `tools/tests/test_offline_deployment_state.py`，先覆盖绝对路径规范化、identity canonical hash、固定权限、锁竞争、marker 幂等与异常对象 fail closed：

```python
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from tools.offline_deployment_state import (
    DeploymentIdentity,
    DeploymentStateError,
    StatePaths,
    acquire_deployment_lock,
    atomic_write_json,
    create_start_marker,
    derive_state_root,
    identity_digest,
    load_identity,
    normalize_absolute_root,
    write_identity_exclusive,
)


def current_ids() -> tuple[int, int]:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    gid = os.getgid() if hasattr(os, "getgid") else 0
    return uid, gid


class BusyLock:
    def acquire(self, _descriptor: int, _timeout_seconds: float) -> None:
        raise TimeoutError("busy")

    def release(self, _descriptor: int) -> None:
        raise AssertionError("release must not run after failed acquire")


class OfflineDeploymentStateTests(unittest.TestCase):
    def make_identity(self, root: Path, *, model_suffix: str = "models") -> DeploymentIdentity:
        return DeploymentIdentity.new(
            state_root=root,
            data_root=root.parent,
            model_root=root.parent / model_suffix,
            secret_root=root.parent / "secrets",
            deployment_uuid="0123456789ab4def8123456789abcdef",
        )

    def test_normalize_rejects_relative_dotdot_double_slash_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.mkdir()
            link = root / "link"
            link.symlink_to(target, target_is_directory=True)
            for raw in ("relative", f"{root}/../escape", "//server/share", str(link)):
                with self.subTest(raw=raw), self.assertRaises(DeploymentStateError):
                    normalize_absolute_root(raw, "DATA_ROOT")

    def test_identity_uses_canonical_json_and_stable_sha256(self) -> None:
        identity = DeploymentIdentity.new(
            state_root=Path("/srv/dcagent/data/.dcagent-deployment-state"),
            data_root=Path("/srv/dcagent/data"),
            model_root=Path("/srv/dcagent/models"),
            secret_root=Path("/srv/dcagent/repo/artifacts/secrets"),
            deployment_uuid="0123456789ab4def8123456789abcdef",
        )
        self.assertEqual(identity_digest(identity), identity_digest(identity))
        self.assertNotIn("repo_root", identity.to_mapping())

    def test_identity_is_exclusive_and_mode_0600(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = StatePaths(Path(directory))
            uid, gid = current_ids()
            paths.ensure_layout(uid=uid, gid=gid)
            identity = self.make_identity(paths.root)
            write_identity_exclusive(paths, identity)
            self.assertEqual(0o600, stat.S_IMODE(paths.identity.stat().st_mode))
            self.assertEqual(identity, load_identity(paths))
            write_identity_exclusive(paths, identity)
            with self.assertRaises(DeploymentStateError):
                write_identity_exclusive(
                    paths,
                    self.make_identity(paths.root, model_suffix="other"),
                )

    def test_busy_lock_fails_closed_without_leaking_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = StatePaths(Path(directory))
            uid, gid = current_ids()
            paths.ensure_layout(uid=uid, gid=gid)
            with self.assertRaisesRegex(DeploymentStateError, "30 seconds") as raised:
                with acquire_deployment_lock(paths, backend=BusyLock()):
                    self.fail("lock must not be entered")
            self.assertNotIn("PATH=", str(raised.exception))

    def test_start_marker_is_idempotent_and_has_fixed_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = StatePaths(Path(directory))
            uid, gid = current_ids()
            paths.ensure_layout(uid=uid, gid=gid)
            create_start_marker(paths, operation="up", deployment_identity_hash="a" * 64)
            create_start_marker(paths, operation="up", deployment_identity_hash="a" * 64)
            marker = json.loads(paths.start_marker.read_text(encoding="utf-8"))
            self.assertEqual(
                {"schema_version", "created_at", "operation", "deployment_identity_hash"},
                set(marker),
            )
            self.assertEqual(0o600, stat.S_IMODE(paths.start_marker.stat().st_mode))

    def test_marker_symlink_is_treated_as_started(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = StatePaths(Path(directory))
            uid, gid = current_ids()
            paths.ensure_layout(uid=uid, gid=gid)
            paths.start_marker.symlink_to(paths.root / "missing")
            with self.assertRaisesRegex(DeploymentStateError, "already started"):
                create_start_marker(
                    paths,
                    operation="exec",
                    deployment_identity_hash="a" * 64,
                )

    def test_state_root_is_derived_only_below_data_root(self) -> None:
        self.assertEqual(
            Path("/srv/dcagent/data/.dcagent-deployment-state"),
            derive_state_root(Path("/srv/dcagent/data")),
        )
```

- [ ] **Step 2：运行测试并确认模块缺失**

Run: `python -m unittest tools.tests.test_offline_deployment_state -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.offline_deployment_state'`.

- [ ] **Step 3：实现共享状态协议的固定接口**

在 `tools/offline_deployment_state.py` 实现以下公开接口和常量；所有 JSON 写入必须执行“同目录临时文件 → 文件 `fsync` → `os.replace` → 父目录 `fsync`”，所有读取必须验证普通非链接文件、`0600`、owner 和 schema：

```python
SCHEMA_VERSION = 1
LOCK_TIMEOUT_SECONDS = 30.0
TRANSACTION_ID = re.compile(r"^[0-9a-f]{32}$")
START_OPERATIONS = frozenset({"up", "exec", "cp", "legacy_adoption"})
TERMINAL_PHASES = frozenset({"committed"})


class DeploymentStateError(RuntimeError):
    """部署状态协议被破坏时使用的 fail-closed 异常。"""


@dataclass(frozen=True)
class DeploymentIdentity:
    schema_version: int
    deployment_uuid: str
    state_root: str
    data_root: str
    model_root: str
    secret_root: str

    @classmethod
    def new(
        cls,
        *,
        state_root: Path,
        data_root: Path,
        model_root: Path,
        secret_root: Path,
        deployment_uuid: str | None = None,
    ) -> "DeploymentIdentity":
        selected_uuid = deployment_uuid or uuid.uuid4().hex
        parsed_uuid = uuid.UUID(hex=selected_uuid)
        if (
            TRANSACTION_ID.fullmatch(selected_uuid) is None
            or parsed_uuid.version != 4
            or parsed_uuid.hex != selected_uuid
        ):
            raise DeploymentStateError("deployment_uuid must be lowercase UUIDv4 hex")
        return cls(
            schema_version=SCHEMA_VERSION,
            deployment_uuid=selected_uuid,
            state_root=str(state_root),
            data_root=str(data_root),
            model_root=str(model_root),
            secret_root=str(secret_root),
        )

    def to_mapping(self) -> dict[str, object]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class StatePaths:
    root: Path

    @property
    def lock(self) -> Path:
        return self.root / "deployment.lock"
    @property
    def start_marker(self) -> Path:
        return self.root / "deployment-started.json"
    @property
    def identity(self) -> Path:
        return self.root / "deployment-identity.json"
    @property
    def transactions(self) -> Path:
        return self.root / "transactions"
    @property
    def control_transactions(self) -> Path:
        return self.root / "control-transactions"
    @property
    def history(self) -> Path:
        return self.root / "history"
    @property
    def quarantine(self) -> Path:
        return self.root / "quarantine"

    def ensure_layout(self, *, uid: int, gid: int) -> None:
        for path in (
            self.root,
            self.transactions,
            self.control_transactions,
            self.history,
            self.quarantine,
        ):
            path.mkdir(mode=0o700, exist_ok=True)
            metadata = path.stat()
            if os.name == "posix" and (metadata.st_uid, metadata.st_gid) != (uid, gid):
                raise DeploymentStateError(f"unsafe state owner: {path}")
            path.chmod(0o700)
        descriptor = os.open(self.lock, os.O_CREAT | os.O_RDWR, 0o600)
        os.close(descriptor)
        self.lock.chmod(0o600)


class LockBackend(Protocol):
    def acquire(self, descriptor: int, timeout_seconds: float) -> None:
        """在给定超时内获取 descriptor 的排他锁。"""

    def release(self, descriptor: int) -> None:
        """释放已经成功获取的排他锁。"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def derive_state_root(data_root: Path) -> Path:
    return data_root / ".dcagent-deployment-state"


def identity_digest(identity: DeploymentIdentity) -> str:
    encoded = json.dumps(
        identity.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
```

同一步中继续实现这些精确签名：`atomic_write_json(path: Path, payload: Mapping[str, object], *, mode: int = 0o600) -> None`、`normalize_absolute_root(raw: str, name: str) -> Path`、`load_identity(paths: StatePaths) -> DeploymentIdentity`、`write_identity_exclusive(paths: StatePaths, identity: DeploymentIdentity) -> None`、`assert_identity_matches(paths: StatePaths, expected: DeploymentIdentity) -> DeploymentIdentity`、`acquire_deployment_lock(paths: StatePaths, *, timeout_seconds: float = LOCK_TIMEOUT_SECONDS, backend: LockBackend | None = None) -> Iterator[None]`、`create_start_marker(paths: StatePaths, *, operation: str, deployment_identity_hash: str) -> None`、`assert_start_marker_absent(paths: StatePaths) -> None`、`assert_no_incomplete_transactions(paths: StatePaths) -> None`。每个函数的校验和持久化顺序按本任务前述测试与设计约束固定，不增加隐式 state 初始化分支。

`normalize_absolute_root()` 必须逐段 `lstat()`，拒绝引号、NUL、前后空白、相对路径、原始 `..`、`//`、任意符号链接，并使用 `posixpath.normpath()` 规范化而不是 `Path.resolve()`。`FcntlLockBackend` 仅在 POSIX 实际调用时导入 `fcntl`，以便 Windows 单测通过注入 backend 运行。

- [ ] **Step 4：补充同数据根跨 checkout 与真实 Linux flock 测试**

追加测试：两个不同 `repo_root` 构造相同 `StatePaths` 后读取同一 identity/marker；Linux 上用第二进程持锁并断言 30 秒参数被传入，Windows 跳过真实 flock 测试但不跳过注入 backend 测试。

- [ ] **Step 5：运行 GREEN 测试和 Ruff**

Run: `python -m unittest tools.tests.test_offline_deployment_state -v`

Expected: PASS.

Run: `uv run --project backend ruff check tools/offline_deployment_state.py tools/tests/test_offline_deployment_state.py`

Expected: `All checks passed!`

- [ ] **Step 6：提交**

```bash
git add tools/offline_deployment_state.py tools/tests/test_offline_deployment_state.py
git commit -m "feat: add shared intranet deployment state protocol"
```

### Task 2：WAL、undo manifest、operation intent/done 与确定性恢复引擎

**Files:**
- Modify: `tools/offline_deployment_state.py`
- Create: `tools/offline_recovery.py`
- Create: `tools/tests/test_offline_recovery.py`

- [ ] **Step 1：写 operation 判定与恢复表 RED 测试**

创建 `tools/tests/test_offline_recovery.py`，用临时目录逐一覆盖 `mkdir`、`chmod`、`active_to_backup`、`staging_to_active`、`env_replace`、`unlink` 的未执行、已执行和冲突状态：

```python
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.offline_deployment_state import StatePaths, TransactionJournal
from tools.offline_recovery import (
    RecoveryConflict,
    classify_operation,
    finalize_committed_cleanup,
    resume_transaction_rollback,
)


def current_ids() -> tuple[int, int]:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    gid = os.getgid() if hasattr(os, "getgid") else 0
    return uid, gid


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def make_journal(root: Path, *, phase: str) -> tuple[StatePaths, TransactionJournal]:
    paths = StatePaths(root / "state")
    uid, gid = current_ids()
    paths.ensure_layout(uid=uid, gid=gid)
    journal = TransactionJournal.create(
        paths,
        deployment_identity_hash="a" * 64,
        object_categories={"environment", "secret", "directory"},
        secret_companion_root=root / "secrets" / ".dcagent-transactions",
    )
    journal.write_phase(phase)
    return paths, journal


class OfflineRecoveryTests(unittest.TestCase):
    def test_chmod_intent_distinguishes_not_run_done_and_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "managed"
            path.mkdir(mode=0o755)
            operation = {
                "sequence": 1,
                "kind": "chmod",
                "status": "intent",
                "path": str(path),
                "before_mode": 0o755,
                "after_mode": 0o700,
                "object_category": "directory",
            }
            self.assertEqual("not_executed", classify_operation(operation))
            path.chmod(0o700)
            self.assertEqual("executed", classify_operation(operation))
            path.chmod(0o711)
            with self.assertRaises(RecoveryConflict):
                classify_operation(operation)

    def test_env_replace_uses_persisted_backup_not_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env_path = root / ".env"
            old_env_path = root / "old.env"
            old_env_path.write_text("VALUE=old\n", encoding="utf-8")
            env_path.write_text("VALUE=new\n", encoding="utf-8")
            _, journal = make_journal(root, phase="env_committed")
            journal.persist_env_backup(old_env_path)
            journal.record_intent(
                1,
                {
                    "kind": "env_replace",
                    "path": str(env_path),
                    "before_digest": sha256_text("VALUE=old\n"),
                    "after_digest": sha256_text("VALUE=new\n"),
                    "before_absent": False,
                    "object_category": "environment",
                },
            )
            journal.record_done(1)
            resume_transaction_rollback(journal)
            self.assertEqual("VALUE=old\n", env_path.read_text(encoding="utf-8"))

    def test_conflict_sets_rollback_failed_and_retains_material(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, journal = make_journal(root, phase="publishing")
            managed = root / "managed"
            managed.mkdir(mode=0o711)
            journal.record_intent(
                1,
                {
                    "kind": "chmod",
                    "path": str(managed),
                    "before_mode": 0o755,
                    "after_mode": 0o700,
                    "object_category": "directory",
                },
            )
            with self.assertRaises(RecoveryConflict):
                resume_transaction_rollback(journal)
            self.assertEqual("rollback_failed", journal.read_phase().phase)
            self.assertTrue(journal.root.exists())

    def test_committed_cleanup_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths, journal = make_journal(Path(directory), phase="committed")
            receipt = paths.history / f"{journal.transaction_id}.json"
            finalize_committed_cleanup(journal)
            finalize_committed_cleanup(journal)
            self.assertFalse(journal.root.exists())
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            self.assertEqual("complete", payload["cleanup_status"])
```

- [ ] **Step 2：运行测试并确认接口缺失**

Run: `python -m unittest tools.tests.test_offline_recovery -v`

Expected: FAIL because `TransactionJournal` and `tools.offline_recovery` are not implemented.

- [ ] **Step 3：在状态模块实现事务 journal 数据结构**

增加固定阶段、对象类别和 operation schema；每个写操作先记录 intent 并 `fsync`，mutation 完成后原子改为 done：

```python
TRANSACTION_PHASES = frozenset(
    {
        "planned", "staging", "staged", "backing_up", "backup_complete",
        "publishing", "published", "verifying", "verified",
        "env_committing", "env_committed", "committed",
        "committed_cleanup_required", "rollback_in_progress", "rollback_failed",
    }
)
OPERATION_KINDS = frozenset(
    {"mkdir", "chmod", "active_to_backup", "staging_to_active", "env_replace", "unlink"}
)


@dataclass(frozen=True)
class PhaseRecord:
    transaction_id: str
    phase: str
    updated_at: str
    deployment_identity_hash: str
    object_categories: tuple[str, ...]


@dataclass(frozen=True)
class UndoEntry:
    path: str
    object_type: str
    existed: bool
    original_mode: int | None
    owner_uid: int
    owner_gid: int
    backup_name: str | None
    expected_action: str
    before: Mapping[str, object]
    after: Mapping[str, object]


class TransactionJournal:
    def __init__(
        self,
        *,
        root: Path,
        transaction_id: str,
        deployment_identity_hash: str,
        secret_companion_root: Path,
        object_categories: Collection[str],
    ) -> None:
        self.root = root
        self.transaction_id = transaction_id
        self.deployment_identity_hash = deployment_identity_hash
        self.secret_companion_root = secret_companion_root
        self.object_categories = tuple(sorted(set(object_categories)))
        self.operations = root / "operations"
        self.phase_path = root / "phase.json"
        self.undo_manifest_path = root / "undo-manifest.json"
```

同一步中为 `TransactionJournal` 完整实现以下精确方法：`create(paths: StatePaths, *, deployment_identity_hash: str, object_categories: Collection[str], secret_companion_root: Path, control: bool = False) -> TransactionJournal`、`open(root: Path, *, expected_identity_hash: str) -> TransactionJournal`、`write_phase(phase: str) -> None`、`read_phase() -> PhaseRecord`、`write_undo_manifest(entries: Sequence[UndoEntry]) -> None`、`record_intent(sequence: int, payload: Mapping[str, object]) -> Path`、`record_done(sequence: int) -> None`、`persist_env_backup(env_path: Path) -> None`、`write_history_receipt(*, cleanup_status: str) -> Path`。`create()` 使用 `uuid.uuid4().hex`，先以 `0700` 创建 journal/operations 与 companion/staging/backup，再写 `planned`；所有记录使用 `atomic_write_json()` 和 `0600`。

事务目录与 companion 路径固定为 `0700`，记录固定为 `0600`。journal 扫描必须双向校验 `${STATE_ROOT}/transactions/<id>` 与 `${SECRET_ROOT}/.dcagent-transactions/<id>`，孤立目录、类型异常和 identity hash 不匹配均抛出 `DeploymentStateError`。

- [ ] **Step 4：实现确定性分类、逆序回滚与提交后 cleanup**

在 `tools/offline_recovery.py` 实现以下固定入口：

```python
class RecoveryConflict(DeploymentStateError):
    """磁盘现状既不匹配 operation before 也不匹配 after。"""
```

实现四个精确入口：`classify_operation(operation: Mapping[str, object]) -> Literal["not_executed", "executed"]`、`reverse_operation(journal: TransactionJournal, operation: Mapping[str, object]) -> None`、`resume_transaction_rollback(journal: TransactionJournal) -> None`、`finalize_committed_cleanup(journal: TransactionJournal) -> None`。

具体规则必须逐字落实设计表：`mkdir` 仅删除本事务创建且仍为空的目录；`chmod` 只接受 before/after 两种 mode；两个 rename 操作用 active/staging/backup 位置组合和 secret pair 校验判断；`env_replace` 用持久化 backup 与 before/after SHA-256；`unlink` 用存在性和对象类型判断。回滚顺序固定为 `.env` → 新 active secret → 旧 secret → mode → 新建目录，任一冲突先持久化 `rollback_failed` 再抛错。

提交后 cleanup 顺序固定为：写 `history/<id>.json` 且 `cleanup_status=committed_cleanup_pending` → 删除 staging/backup 并 `fsync` → receipt 改为 `complete` → 删除 transaction journal 并 `fsync`。重复执行不得报冲突；identity hash 不一致必须 fail closed。

- [ ] **Step 5：增加每个 intent/done 中断点的表驱动测试**

用 `subTest(kind=kind, state=state)` 覆盖六类 operation 的三种磁盘状态，并覆盖 `planned` 至 `env_committed` 的回滚、`committed`/`committed_cleanup_required` 的 cleanup、pending receipt 重入、backup 部分失败和孤立 companion。

- [ ] **Step 6：运行 GREEN 测试和 Ruff**

Run: `python -m unittest tools.tests.test_offline_deployment_state tools.tests.test_offline_recovery -v`

Expected: PASS.

Run: `uv run --project backend ruff check tools/offline_deployment_state.py tools/offline_recovery.py tools/tests/test_offline_deployment_state.py tools/tests/test_offline_recovery.py`

Expected: `All checks passed!`

- [ ] **Step 7：提交**

```bash
git add tools/offline_deployment_state.py tools/offline_recovery.py tools/tests/test_offline_deployment_state.py tools/tests/test_offline_recovery.py
git commit -m "feat: add durable deployment transaction recovery"
```

### Task 3：环境准备规划器、显式初始化与完整业务事务

**Files:**
- Modify: `tools/offline_env.py:1-684`
- Modify: `tools/tests/test_offline_env.py:1-786`

- [ ] **Step 1：写规划器无副作用和初始化 RED 测试**

在 `tools/tests/test_offline_env.py` 增加：

```python
from tools.offline_env import build_preparation_plan, prepare_environment


def test_plan_is_read_only_and_records_all_mutations(self) -> None:
    with prepared_repository_fixture() as fixture:
        before = snapshot_tree(fixture.root)
        plan = build_preparation_plan(
            fixture.repo_root,
            environ=fixture.environ,
            initialize_state=True,
            verify_posix_metadata=False,
        )
        self.assertEqual(before, snapshot_tree(fixture.root))
        self.assertEqual(fixture.data_root, plan.data_root)
        self.assertEqual(fixture.model_root, plan.model_root)
        self.assertIn("DEPLOYMENT_STATE_ROOT", plan.env_updates)
        self.assertEqual(
            {"postgres-password", "database-url", "clickhouse-query-password", "clickhouse-ingest-password"},
            set(plan.publish_secret_names),
        )


def test_normal_prepare_requires_existing_identity(self) -> None:
    with prepared_repository_fixture() as fixture:
        with self.assertRaisesRegex(DeploymentError, "--initialize-state|adopt-existing"):
            prepare_environment(
                fixture.repo_root,
                environ=fixture.environ,
                verify_posix_metadata=False,
            )


def test_initialize_rejects_nonempty_postgres_or_existing_container(self) -> None:
    for condition in ("pg_version", "nonempty_postgres", "compose_container"):
        with self.subTest(condition=condition), prepared_repository_fixture(condition) as fixture:
            with self.assertRaises(DeploymentError):
                prepare_environment(
                    fixture.repo_root,
                    environ=fixture.environ,
                    initialize_state=True,
                    verify_posix_metadata=False,
                )


def test_rotate_rechecks_marker_and_pg_version_before_publish(self) -> None:
    with initialized_repository_fixture() as fixture:
        def inject_race(_plan: PreparationPlan) -> None:
            fixture.pg_version.write_text("16\n", encoding="ascii")

        with self.assertRaisesRegex(DeploymentError, "initialized PostgreSQL"):
            prepare_environment(
                fixture.repo_root,
                rotate_secrets=True,
                environ=fixture.environ,
                verify_posix_metadata=False,
                before_mutation=inject_race,
            )
        assert_original_environment_restored(fixture)
```

测试 helper 必须在测试文件中完整实现：fixture 预先创建空、当前账号所有、模式 `0700` 的 `DATA_ROOT`/`MODEL_ROOT`，模拟 Docker container inspect 返回；`snapshot_tree()` 记录路径、类型、mode 和内容摘要，不读取 secret 正文到断言消息。

- [ ] **Step 2：运行 RED 测试**

Run: `python -m unittest tools.tests.test_offline_env -v`

Expected: FAIL because `build_preparation_plan()`, `PreparationPlan`, `initialize_state` and `before_mutation` do not exist.

- [ ] **Step 3：把现有单体准备函数拆成不可变计划与执行器**

在 `tools/offline_env.py` 定义并使用：

```python
@dataclass(frozen=True)
class DirectoryMutation:
    path: Path
    existed: bool
    original_mode: int | None


@dataclass(frozen=True)
class PreparationPlan:
    repo_root: Path
    env_path: Path
    env_before: str | None
    env_after: str
    env_mode_before: int | None
    env_updates: Mapping[str, str]
    uid: int
    gid: int
    data_root: Path
    model_root: Path
    secret_root: Path
    state_paths: StatePaths
    identity: DeploymentIdentity
    directory_mutations: tuple[DirectoryMutation, ...]
    managed_secret_paths: Mapping[str, Path]
    publish_secret_names: tuple[str, ...]
    rotate_secrets: bool
```

实现精确入口：`build_preparation_plan(repo_root: Path, *, rotate_secrets: bool = False, initialize_state: bool = False, environ: Mapping[str, str] | None = None, verify_posix_metadata: bool = True) -> PreparationPlan`、`execute_preparation_plan(plan: PreparationPlan, *, verify_posix_metadata: bool = True, before_mutation: Callable[[PreparationPlan], None] | None = None) -> None`、`prepare_environment(repo_root: Path, *, rotate_secrets: bool = False, initialize_state: bool = False, environ: Mapping[str, str] | None = None, verify_posix_metadata: bool = True, before_mutation: Callable[[PreparationPlan], None] | None = None) -> None`。`prepare_environment()` 只负责引导 state、持锁、构建计划和执行计划；锁直到提交后 cleanup 或回滚结束才释放。

规划器必须完成全部 owner/mode/type/symlink/secret pair/ClickHouse 旧键升级检查，逐级计算所有待创建祖先，并且不得调用 `mkdir`、`chmod`、secret 生成、`os.replace` 或 `.env` 写入。路径展开只给 `resolve_env_path()` 传 `HOST_DATA_ROOT`/`HOST_MODEL_ROOT` allowlist；`.env` 出现这两个 HOST key 时拒绝。

- [ ] **Step 4：实现 `--initialize-state` control transaction**

CLI 增加互斥显式开关并保持普通调用不初始化：

```python
parser.add_argument("--rotate-secrets", action="store_true")
parser.add_argument("--initialize-state", action="store_true")
if args.rotate_secrets and not args.initialize_state and not env_has_state_root:
    raise DeploymentError("Deployment identity is missing; initialize or adopt first")
```

初始化顺序必须是：只读 `.env`/`.env.example` → 规范化 roots → 只创建 state root/lock → 持锁 → 检查 control transaction → 验证 data root 为空且无容器/marker/`PG_VERSION` → control WAL 记录 intent → exclusive-create identity 并 `fsync` → 记录 done → 执行普通准备事务。identity 已存在且字段完全匹配时幂等继续，不匹配时 fail closed；`.env` 最终写入绝对 `DEPLOYMENT_STATE_ROOT`。

- [ ] **Step 5：用 TransactionJournal 替换局部 secret 事务**

删除 `_publish_secret_set()` 内的临时局部事务和 `prepare_environment()` 的内存 rollback；按以下持久顺序实现：

```text
planned -> staging -> staged -> backing_up -> backup_complete ->
publishing -> published -> verifying -> verified ->
env_committing -> env_committed -> committed
```

任何 mutation 前持久化 undo manifest；每个 `mkdir`、`chmod`、active-to-backup、staging-to-active、env-replace、cleanup 先写 intent、再执行、再写 done。secret companion 固定在 `${SECRET_ROOT}/.dcagent-transactions/<id>`；旧 `.env` 复制到 state transaction backup。提交前异常调用 `resume_transaction_rollback()`，回滚成功后删除事务并仍返回原非零错误；回滚失败只报告事务目录和阶段。提交后异常转为 `committed_cleanup_required`，不回滚新状态。

- [ ] **Step 6：增加完整故障矩阵与竞态测试**

将现有 staging/publish 回滚测试扩展为表驱动 fault injector，边界固定为：`mkdir`、`chmod`、chmod 后复验、secret staging、backup、publish、staging 验证、active 验证、`.env` 临时写、`.env` replace、commit phase、rollback restore、post-commit cleanup。每个边界断言旧 `.env`、旧 secret、原 mode、原有目录完整恢复，本事务创建的多层目录逆序删除；rollback 故障断言材料保留且下一次 prepare fail closed。

- [ ] **Step 7：运行 GREEN 测试和原环境契约**

Run: `python -m unittest tools.tests.test_offline_env tools.tests.test_offline_deployment_state tools.tests.test_offline_recovery -v`

Expected: PASS.

Run: `uv run --project backend ruff check tools/offline_env.py tools/offline_deployment_state.py tools/offline_recovery.py tools/tests/test_offline_env.py`

Expected: `All checks passed!`

- [ ] **Step 8：提交**

```bash
git add tools/offline_env.py tools/tests/test_offline_env.py
git commit -m "feat: make intranet environment preparation transactional"
```

### Task 4：人工恢复 CLI、control transaction 与审计 receipt

**Files:**
- Modify: `tools/offline_recovery.py`
- Modify: `tools/tests/test_offline_recovery.py`

- [ ] **Step 1：写六个动作与脱敏输出 RED 测试**

增加 CLI 测试：

```python
from contextlib import redirect_stdout
from io import StringIO

from tools.offline_recovery import main


def test_inspect_outputs_only_sanitized_metadata(self) -> None:
    with recovery_fixture(phase="publishing") as fixture:
        output = StringIO()
        with redirect_stdout(output):
            exit_code = main([
                "inspect",
                "--state-root", str(fixture.state_root),
                "--transaction", fixture.transaction_id,
            ])
        self.assertEqual(0, exit_code)
        payload = json.loads(output.getvalue())
        self.assertEqual("publishing", payload["phase"])
        self.assertNotIn(fixture.secret_value, output.getvalue())
        self.assertNotIn("database_url", output.getvalue().casefold())


def test_clear_marker_keeps_marker_if_any_gate_fails(self) -> None:
    for gate in ("container", "pg_version", "postgres_nonempty", "transaction"):
        with self.subTest(gate=gate), recovery_fixture(gate=gate) as fixture:
            with self.assertRaises(DeploymentStateError):
                main(["clear-start-marker", "--state-root", str(fixture.state_root)])
            self.assertTrue(os.path.lexists(fixture.marker))


def test_adopt_existing_writes_marker_before_identity_for_initialized_data(self) -> None:
    with legacy_deployment_fixture(pg_initialized=True) as fixture:
        fixture.injector.stop_after("marker_done")
        with self.assertRaises(InjectedStop):
            main(["adopt-existing", "--state-root", str(fixture.state_root)])
        self.assertTrue(fixture.marker.exists())
        self.assertFalse(fixture.identity.exists())
        main(["adopt-existing", "--state-root", str(fixture.state_root)])
        self.assertTrue(fixture.identity.exists())
```

- [ ] **Step 2：运行 RED 测试**

Run: `python -m unittest tools.tests.test_offline_recovery -v`

Expected: FAIL because recovery CLI actions are not implemented.

- [ ] **Step 3：实现严格 argparse 和显式 state root 定位**

`main(argv=None)` 只注册下列子命令，所有 state root 都调用与生产路径相同的规范化和 owner/mode/symlink 检查：

```text
inspect --state-root ABS --transaction ID
resume-rollback --state-root ABS --transaction ID
finalize-cleanup --state-root ABS --transaction ID
clear-start-marker --state-root ABS
adopt-existing --state-root ABS
acknowledge-repaired --state-root ABS --transaction ID --evidence ABS
```

`inspect` 不获取 mutation 权限、不写 receipt，只输出 `transaction_id`、`phase`、`object_categories`、`recommended_action`、`deployment_identity_hash`。其余动作从状态检查到 `history/recovery-<uuid>.json` 落盘都持有同一排他锁。

- [ ] **Step 4：实现 mutation action 的 control WAL**

为 `adopt-existing`、`clear-start-marker`、`acknowledge-repaired` 使用 `${STATE_ROOT}/control-transactions/<id>`，阶段和顺序固定：

```python
ADOPTION_PHASES = (
    "adoption_planned",
    "identity_created",
    "runtime_checked",
    "marker_written_or_rotation_enabled",
    "adoption_complete",
)
CLEAR_MARKER_PHASES = (
    "clear_planned",
    "runtime_checked",
    "marker_backed_up",
    "receipt_written",
    "clear_complete",
)
```

旧部署接管先从现有 `.env` 和 HOST allowlist 得出 data/model/secret roots；`identity_created` 阶段只把 identity candidate 写入 control journal，不发布权威 `deployment-identity.json`。有容器、`PG_VERSION` 或非空 PostgreSQL 数据目录时先 durable marker，随后才 exclusive-create 权威 identity，最后进入 `adoption_complete`，因此 identity 仍是最后一个业务发布对象。`clear-start-marker` 必须检查无 DC-Agent 容器、无 `PG_VERSION`、PostgreSQL 目录不存在或为空、无未完成普通/control 事务；先把 marker 移到 control backup，审计 receipt durable 后才清理 backup。任意 intent/done 中断后重复相同命令幂等继续。

- [ ] **Step 5：实现 repair acknowledgment 与 quarantine**

`acknowledge-repaired` 先读取 evidence 普通文件并仅记录其 SHA-256、大小和绝对路径 basename；重新验证 active `.env`、identity、两个 secret pair、owner/mode 后，原子移动损坏 transaction 到 `quarantine/<id>`，写恢复 receipt。不得记录 evidence 正文、secret digest、数据库 URL或原始异常对象。

- [ ] **Step 6：运行 GREEN 测试和 Ruff**

Run: `python -m unittest tools.tests.test_offline_recovery -v`

Expected: PASS.

Run: `uv run --project backend ruff check tools/offline_recovery.py tools/tests/test_offline_recovery.py`

Expected: `All checks passed!`

- [ ] **Step 7：提交**

```bash
git add tools/offline_recovery.py tools/tests/test_offline_recovery.py
git commit -m "feat: add audited intranet deployment recovery commands"
```

### Task 5：Compose verb allowlist、共享锁、主机环境与启动标记

**Files:**
- Modify: `tools/offline_compose.py:1-454`
- Modify: `tools/tests/test_offline_compose.py:1-318`

- [ ] **Step 1：写参数、环境、锁和 marker RED 测试**

在 `tools/tests/test_offline_compose.py` 增加：

```python
def test_only_six_compose_verbs_are_allowed(self) -> None:
    for verb in ("config", "build", "up", "down", "exec", "cp"):
        with self.subTest(verb=verb):
            self.assertEqual(verb, validate_compose_arguments([verb]).verb)
    for verb in ("run", "create", "start", "restart", "scale", "pull", "push", "logs", "ps", "version"):
        with self.subTest(verb=verb), self.assertRaises(DeploymentError):
            validate_compose_arguments([verb])


def test_child_environment_is_allowlisted_and_host_roots_survive(self) -> None:
    env_file = {
        "DATA_ROOT": "${HOST_DATA_ROOT}",
        "MODEL_ROOT": "${HOST_MODEL_ROOT}",
        "API_SECRET": "must-not-override",
    }
    process = {
        "PATH": "/usr/bin",
        "HOME": "/home/dcagent",
        "HOST_DATA_ROOT": "/srv/dcagent/data",
        "HOST_MODEL_ROOT": "/srv/dcagent/models",
        "API_SECRET": "attacker",
    }
    child = build_compose_environment(env_file, process)
    self.assertEqual("/srv/dcagent/data", child["HOST_DATA_ROOT"])
    self.assertNotIn("API_SECRET", child)
    self.assertNotIn("DOCKER_CONTEXT", child)


def test_mutating_verbs_write_marker_before_docker_and_keep_it_on_failure(self) -> None:
    for verb in ("up", "exec", "cp"):
        with self.subTest(verb=verb), compose_fixture() as fixture:
            fixture.runner.return_code = 9
            self.assertEqual(9, run_compose([verb], fixture.repo_root, environ=fixture.environ, runner=fixture.runner))
            self.assertLess(fixture.events.index("marker"), fixture.events.index("docker-compose"))
            self.assertTrue(fixture.state_paths.start_marker.exists())


def test_nonmutating_verbs_do_not_write_marker(self) -> None:
    for verb in ("config", "build", "down"):
        with self.subTest(verb=verb), compose_fixture() as fixture:
            run_compose([verb], fixture.repo_root, environ=fixture.environ, runner=fixture.runner)
            self.assertFalse(fixture.state_paths.start_marker.exists())


def test_lock_is_held_until_compose_process_exits(self) -> None:
    with compose_fixture() as fixture:
        fixture.runner.assert_lock_held = fixture.lock_probe
        run_compose(["build"], fixture.repo_root, environ=fixture.environ, runner=fixture.runner)
        self.assertTrue(fixture.runner.lock_was_held)
```

- [ ] **Step 2：运行 RED 测试**

Run: `python -m unittest tools.tests.test_offline_compose -v`

Expected: FAIL because argument parsing does not return a verb, unsupported verbs pass, and lock/marker/environment helpers do not exist.

- [ ] **Step 3：实现结构化参数解析与固定命令前缀**

增加：

```python
ALLOWED_VERBS = frozenset({"config", "build", "up", "down", "exec", "cp"})
MUTATING_VERBS = frozenset({"up", "exec", "cp"})
ALLOWED_PROCESS_ENV = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "XDG_RUNTIME_DIR",
    "DOCKER_CONFIG", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR",
})
HOST_PATH_ENV = frozenset({"HOST_DATA_ROOT", "HOST_MODEL_ROOT"})


@dataclass(frozen=True)
class ComposeInvocation:
    arguments: tuple[str, ...]
    verb: str
```

实现精确入口 `validate_compose_arguments(arguments: Sequence[str]) -> ComposeInvocation` 和 `build_compose_environment(env_values: Mapping[str, str], process_environ: Mapping[str, str]) -> dict[str, str]`。前者返回原参数的不可变 tuple 与唯一 verb；后者从空字典开始复制 `ALLOWED_PROCESS_ENV`，验证并加入两个 HOST 值，拒绝危险 Docker/Compose 键，并删除全部 `.env` key。

解析器只允许现有批准的 wrapper 全局参数，首个非全局参数必须在 allowlist；`exec`/`cp` 的 `--` 以后不再识别 verb。命令前缀固定加入 `--project-name dcagent-offline`，所有 Docker 子进程 `cwd=repo_root`。

- [ ] **Step 4：统一 HOST 路径解析与 identity 绑定**

`.env` 不得定义 `HOST_DATA_ROOT`/`HOST_MODEL_ROOT`；只从调用进程复制这两个值。`DATA_ROOT`/`MODEL_ROOT` 只能是绝对值或完整单 token，传给 `normalize_absolute_root()` 后必须等于 identity 中的批准位置。config 渲染和最终 Compose 子进程必须接收同一个 `child_environ` 对象；检测到 `DOCKER_HOST`、`DOCKER_CONTEXT`、`DOCKER_TLS_VERIFY` 或任意 `COMPOSE_*` 时在运行任何 Docker 命令前拒绝。

- [ ] **Step 5：把状态检查、marker 和 Docker 执行放入同一锁范围**

`run_compose(arguments: Sequence[str], repo_root: Path, *, environ: Mapping[str, str] | None = None, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> int` 的固定执行顺序是：结构化解析参数；只读加载 `.env` 并解析固定 state root/identity；获取共享排他锁；检查普通/control transaction、`rollback_failed` 和 `committed_cleanup_required`；检查 Docker context；渲染全 profile JSON 并校验拓扑；对 `up/exec/cp` durable marker；运行最终 Docker 命令；子进程退出后释放锁。不得在锁外调用 context inspect、config 或最终 Compose。marker 已存在且 schema/hash 一致时幂等；普通文件内容损坏、目录、符号链接、损坏链接或不可读对象都按“部署已启动”处理，不能被普通入口清除。

- [ ] **Step 6：增加未完成事务对六个 verb 的拒绝测试**

对普通 transaction、control transaction、`rollback_failed` 和 `committed_cleanup_required` 四种状态，逐一测试六个 allowlist verb 均在 Docker 调用前失败；对非法 verb 断言在读取 `.env` 和创建 state root 前失败。

- [ ] **Step 7：运行 GREEN 测试和 PowerShell 回归契约**

Run: `python -m unittest tools.tests.test_offline_compose tools.tests.test_offline_deployment_state -v`

Expected: PASS.

Run: `python -m unittest tools.tests.test_compose_contract -v`

Expected: PASS, including unchanged PowerShell development compatibility assertions.

- [ ] **Step 8：提交**

```bash
git add tools/offline_compose.py tools/tests/test_offline_compose.py
git commit -m "feat: serialize compose operations and persist start marker"
```

### Task 6：恢复 Bash 入口、LF 与 executable mode 契约

**Files:**
- Create: `tools/recover_offline_deployment.sh`
- Modify: `tools/tests/test_ubuntu_deployment_entrypoints.py:1-56`
- Verify: `.gitattributes`

- [ ] **Step 1：先扩展入口契约测试**

把 `expected_helpers` 扩展为：

```python
expected_helpers = {
    "prepare_offline_env.sh": "offline_env.py",
    "invoke_offline_compose.sh": "offline_compose.py",
    "recover_offline_deployment.sh": "offline_recovery.py",
}
```

Git mode 检查也加入第三个路径，并把期望数量从 `2` 改为 `3`。

- [ ] **Step 2：运行 RED 测试**

Run: `python -m unittest tools.tests.test_ubuntu_deployment_entrypoints -v`

Expected: FAIL because `tools/recover_offline_deployment.sh` does not exist.

- [ ] **Step 3：创建最小 Bash 入口并设置 executable bit**

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
exec python3 "$SCRIPT_DIR/offline_recovery.py" "$@"
```

Run: `git update-index --add --chmod=+x tools/recover_offline_deployment.sh`

- [ ] **Step 4：验证语法、LF、mode 与测试**

Run: `bash -n tools/prepare_offline_env.sh tools/invoke_offline_compose.sh tools/recover_offline_deployment.sh`

Expected: no output, exit 0.

Run: `python -m unittest tools.tests.test_ubuntu_deployment_entrypoints -v`

Expected: PASS.

- [ ] **Step 5：提交**

```bash
git add tools/recover_offline_deployment.sh tools/tests/test_ubuntu_deployment_entrypoints.py
git commit -m "feat: add Ubuntu deployment recovery entrypoint"
```

### Task 7：Ubuntu 目标机验收与隔离强制终止恢复演练

**Files:**
- Create: `tools/intranet_deployment_gate.py`
- Create: `tools/tests/test_intranet_deployment_gate.py`

- [ ] **Step 1：写步骤顺序、超时、短路与脱敏 RED 测试**

创建测试：

```python
from tools.intranet_deployment_gate import GateError, run_gate


def test_fresh_gate_uses_fixed_steps_and_timeouts(self) -> None:
    with gate_fixture() as fixture:
        report = run_gate(fixture.config, runner=fixture.runner)
        self.assertEqual(
            ["prepare", "compose_config", "compose_build", "compose_up", "readyz", "physoc", "ollama_embed", "ollama_generate", "ollama_tags", "metadata", "recovery_drill"],
            [step["category"] for step in report["steps"]],
        )
        self.assertEqual(60, fixture.runner.timeout_for("compose_config"))
        self.assertEqual(1800, fixture.runner.timeout_for("compose_build"))
        self.assertEqual(300, fixture.runner.timeout_for("compose_up"))
        self.assertEqual(120, fixture.runner.timeout_for("recovery_drill"))


def test_failure_stops_online_evidence_but_keeps_sanitized_diagnostics(self) -> None:
    with gate_fixture(fail_at="readyz") as fixture:
        with self.assertRaises(GateError):
            run_gate(fixture.config, runner=fixture.runner)
        payload = json.loads(fixture.report_path.read_text(encoding="utf-8"))
        self.assertEqual("failed", payload["status"])
        serialized = json.dumps(payload, ensure_ascii=False)
        self.assertNotIn(fixture.secret, serialized)
        self.assertNotIn(fixture.prompt, serialized)
        self.assertNotIn("data:", serialized)


def test_recovery_drill_uses_isolated_roots_and_leaves_no_residue(self) -> None:
    with gate_fixture() as fixture:
        run_gate(fixture.config, runner=fixture.runner)
        fixture.assert_no_test_containers()
        fixture.assert_no_transaction_companions()
        fixture.assert_isolated_roots_removed()
```

- [ ] **Step 2：运行 RED 测试**

Run: `python -m unittest tools.tests.test_intranet_deployment_gate -v`

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3：实现固定步骤和原子脱敏报告**

公开接口固定为：

```python
@dataclass(frozen=True)
class GateConfig:
    repo_root: Path
    report_path: Path
    deployment_mode: Literal["fresh", "adopt"]
    state_root: Path | None


class GateError(RuntimeError):
    """任一目标机验收步骤失败或留下残留状态。"""
```

实现精确入口 `run_gate(config: GateConfig, *, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> dict[str, object]` 和 `main(argv: list[str] | None = None) -> int`。CLI 固定接受 `--mode fresh|adopt`、可选的 `--state-root` 和必需的 `--report`；`adopt` 缺 state root 或 `fresh` 携带 state root 都直接返回非零。

新部署调用 `prepare_offline_env.sh --initialize-state`；旧部署必须要求 `--state-root` 并调用 `recover_offline_deployment.sh adopt-existing` 后再普通 prepare。随后固定执行 Compose config、五个内部镜像 build、up、API readyz、Physoc、Ollama embeddings/generate/tags、owner/mode 检查和隔离恢复演练。报告每步只保存 category、started_at、finished_at、exit_code、duration_ms、sanitized_status；输出用临时文件、`fsync`、`os.replace` 原子写入。

- [ ] **Step 4：实现隔离恢复演练**

演练使用 `tempfile.mkdtemp()` 创建独立 data/model/secret/state roots 和 deployment identity，不启动 `dcagent-offline` Compose project。子进程在一个已持久化 intent 与 done 之间被 SIGKILL；随后普通 prepare 必须检测遗留事务并 fail closed，再由 `resume-rollback` 或 `finalize-cleanup` 完成。`finally` 必须确认并删除临时 marker、active test secrets、roots、state/history/quarantine、companion，并检查无相关容器；任一残留使 gate 失败但不执行 `down -v`。

- [ ] **Step 5：运行 GREEN 测试和 Ruff**

Run: `python -m unittest tools.tests.test_intranet_deployment_gate -v`

Expected: PASS with fake runner; this is not evidence that Ubuntu live gate passed.

Run: `uv run --project backend ruff check tools/intranet_deployment_gate.py tools/tests/test_intranet_deployment_gate.py`

Expected: `All checks passed!`

- [ ] **Step 6：提交**

```bash
git add tools/intranet_deployment_gate.py tools/tests/test_intranet_deployment_gate.py
git commit -m "feat: add auditable Ubuntu intranet deployment gate"
```

### Task 8：Ubuntu 文档、轮换限制、恢复 runbook 与契约测试

**Files:**
- Modify: `docs/intranet-deployment-configuration.md:1-437`
- Modify: `docs/offline-platform-runbook.md:1-240`
- Modify: `deploy/offline/README.md:1-500`
- Modify: `README.md:90-320`
- Modify: `tools/tests/test_compose_contract.py`
- Modify: `tools/tests/test_structured_deployment_contract.py`

- [ ] **Step 1：先写文档契约 RED 测试**

增加断言，四份生产文档必须包含：

```python
required_fragments = (
    "./tools/prepare_offline_env.sh --initialize-state",
    "./tools/recover_offline_deployment.sh adopt-existing",
    "./tools/recover_offline_deployment.sh inspect",
    "./tools/recover_offline_deployment.sh resume-rollback",
    "./tools/recover_offline_deployment.sh finalize-cleanup",
    "DEPLOYMENT_STATE_ROOT",
    "deployment-started.json",
    "30 秒",
)
for path in production_documents:
    text = path.read_text(encoding="utf-8")
    for fragment in required_fragments:
        self.assertIn(fragment, text, path)
    self.assertNotRegex(text, r"rm\s+.*deployment-started\.json")
```

根 README 只允许一段 Windows `.ps1` 开发兼容说明；Ubuntu 生产命令不得出现 `pwsh`、`Copy-Item`、`New-Item`、`$LASTEXITCODE` 或 PowerShell 续行符。

- [ ] **Step 2：运行 RED 契约测试**

Run: `python -m unittest tools.tests.test_compose_contract tools.tests.test_structured_deployment_contract -v`

Expected: FAIL because current docs use ordinary prepare and do not document state initialization/recovery.

- [ ] **Step 3：更新全新部署和旧部署固定顺序**

四份文档统一使用：

```bash
install -d -m 0700 /srv/dcagent/data /srv/dcagent/models
./tools/prepare_offline_env.sh --initialize-state
./tools/invoke_offline_compose.sh config
./tools/invoke_offline_compose.sh build \
  schema-migration embedding-service reranker-service api ingestion-worker
./tools/invoke_offline_compose.sh up -d
```

旧部署统一写为：

```bash
./tools/recover_offline_deployment.sh adopt-existing \
  --state-root /absolute/data/root/.dcagent-deployment-state
./tools/prepare_offline_env.sh
```

明确普通 prepare/Compose 不会隐式创建 identity，更换 `DATA_ROOT` 必须作为新部署处理，PowerShell 只用于 Windows 开发机。

- [ ] **Step 4：写明锁、marker、secret 轮换和人工恢复**

说明六个 Compose verb、30 秒锁超时、`up/exec/cp` 前 durable marker、失败后 marker 保留、`config/build/down` 不写 marker。说明 marker 或任意形态 `PG_VERSION` 存在后普通 `--rotate-secrets` 永久拒绝；不提供在线 PostgreSQL role 密码修改或单行删除 marker 命令。

人工清 marker 前必须逐项列出：无 DC-Agent 容器、无 `PG_VERSION`、PostgreSQL 目录不存在或未初始化、无未完成事务；只允许调用 `clear-start-marker`，不展示 `rm`。

- [ ] **Step 5：写明故障分类和恢复命令**

文档分别说明自动回滚、`rollback_failed`、`committed_cleanup_required`、损坏 journal/quarantine，并给出 inspect、resume-rollback、finalize-cleanup、acknowledge-repaired 的完整命令。强调日志和 evidence receipt 不含 secret、数据库 URL、模型正文或原始 SSE。

- [ ] **Step 6：写明目标机 gate 和证据边界**

加入：

```bash
python3 tools/intranet_deployment_gate.py \
  --mode fresh \
  --report artifacts/benchmarks/intranet-deployment-gate.json
```

列出 config 60 秒、build 1800 秒、up/readyz 300 秒、各 probe 60 秒、恢复演练 120 秒；开发机只运行本地测试不得声称 Ubuntu live gate 通过。

- [ ] **Step 7：运行 GREEN 文档契约**

Run: `python -m unittest tools.tests.test_compose_contract tools.tests.test_structured_deployment_contract -v`

Expected: PASS.

- [ ] **Step 8：提交**

```bash
git add README.md docs/intranet-deployment-configuration.md docs/offline-platform-runbook.md deploy/offline/README.md tools/tests/test_compose_contract.py tools/tests/test_structured_deployment_contract.py
git commit -m "docs: document transactional intranet deployment recovery"
```

### Task 9：全量回归、本地门禁与 Ubuntu 20.04 live gate

**Files:**
- Verify: all files changed by Tasks 1-8

- [ ] **Step 1：运行 tools 全量测试**

Run: `python -m unittest discover -s tools/tests -p "test_*.py" -v`

Expected: all tests PASS; platform-specific real `fcntl` tests may SKIP only on non-POSIX hosts.

- [ ] **Step 2：运行 backend 全量测试**

Run: `uv run --project backend pytest backend/tests -q`

Expected: all tests PASS; only pre-existing explicitly documented skips are allowed.

- [ ] **Step 3：运行 Ruff、格式和锁文件检查**

Run: `uv run --project backend ruff check backend/app backend/tests tools`

Expected: `All checks passed!`

Run: `uv run --project backend ruff format --check backend/app backend/tests tools`

Expected: all files already formatted.

Run: `uv --no-config lock --check --project backend`

Expected: exit 0 with no lock drift.

- [ ] **Step 4：运行 Bash、LF、executable mode 和 Git whitespace 检查**

Run: `bash -n tools/prepare_offline_env.sh tools/invoke_offline_compose.sh tools/recover_offline_deployment.sh`

Expected: no output, exit 0.

Run: `git ls-files --eol tools/prepare_offline_env.sh tools/invoke_offline_compose.sh tools/recover_offline_deployment.sh`

Expected: all report `i/lf` and `w/lf`.

Run: `git ls-files --stage tools/prepare_offline_env.sh tools/invoke_offline_compose.sh tools/recover_offline_deployment.sh`

Expected: all three entries start with `100755`.

Run: `git diff --check`

Expected: no output, exit 0.

- [ ] **Step 5：在全新 Ubuntu 20.04 目标机运行 live gate**

```bash
install -d -m 0700 /srv/dcagent/data /srv/dcagent/models
python3 tools/intranet_deployment_gate.py \
  --mode fresh \
  --report artifacts/benchmarks/intranet-deployment-gate.json
```

Expected: report status is `passed`; Compose config、五个内部镜像 build、up、API readyz、Physoc、Ollama embeddings/generate/tags、owner/mode 和隔离恢复演练全部通过，且无测试 transaction/backup/staging/container 残留。若当前机器没有 Ubuntu Docker/Physoc/Ollama 拓扑，只记录“live gate 未运行”，不得写成通过。

- [ ] **Step 6：在旧部署演练机运行接管 gate**

```bash
python3 tools/intranet_deployment_gate.py \
  --mode adopt \
  --state-root /absolute/data/root/.dcagent-deployment-state \
  --report artifacts/benchmarks/intranet-deployment-adoption-gate.json
```

Expected: identity 与现有 data/model/secret roots 绑定；若存在容器、`PG_VERSION` 或已初始化 PostgreSQL 目录，`operation=legacy_adoption` marker 在 identity 前 durable；后续普通 prepare 通过且 secret 轮换被拒绝。

- [ ] **Step 7：提交验证过程中产生的必要测试/格式修正**

只有在门禁暴露实现缺陷时修改对应任务文件并重跑该任务与全量门禁；不要提交 live gate 报告中可能包含目标机标识的本地 artifact。

```bash
git add tools/offline_deployment_state.py tools/offline_recovery.py tools/offline_env.py tools/offline_compose.py tools/intranet_deployment_gate.py tools/recover_offline_deployment.sh tools/tests/test_offline_deployment_state.py tools/tests/test_offline_recovery.py tools/tests/test_offline_env.py tools/tests/test_offline_compose.py tools/tests/test_intranet_deployment_gate.py tools/tests/test_ubuntu_deployment_entrypoints.py tools/tests/test_compose_contract.py tools/tests/test_structured_deployment_contract.py README.md docs/intranet-deployment-configuration.md docs/offline-platform-runbook.md deploy/offline/README.md
git commit -m "test: complete intranet deployment transaction gates"
```

如果没有产生代码或文档修正，则跳过该提交，保持工作树干净。
