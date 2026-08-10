# Supervisor Structured Worker Manager Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an idempotent Ubuntu Bash script that installs and manages the `app.structured_worker` process under Supervisor.

**Architecture:** One script, `tools/manage_structured_worker_supervisor.sh`, owns four concerns: argument validation, read-only Supervisor configuration rendering, the internal worker `run` entry point, and Supervisor lifecycle commands. The generated Supervisor program invokes the same script with `run`, so environment loading and Python execution stay in one tested path. A standard-library Python contract test exercises help, rendering, invalid input, and shell syntax without requiring root or a live Supervisor daemon.

**Tech Stack:** Bash on Ubuntu, Supervisor, Python `unittest`, existing backend virtual environment, Markdown runbook.

## Global Constraints

- Do not install the Supervisor OS package from the repository script.
- Manage only `app.structured_worker`; do not manage API, retrieval worker, embedding, reranker, or databases.
- Keep application secrets outside the repository and never print or copy environment-file contents.
- Run the worker as a configured non-root user.
- Do not change structured worker code, queue semantics, database schema, or public API models.
- Do not add an uninstall command or delete an existing Supervisor configuration.
- Reject paths containing whitespace or newlines because the generated Supervisor command is intentionally unambiguous and shell-safe.
- Default Python executable: `<project-root>/backend/.venv/bin/python`.
- Fixed Supervisor program name: `dcagent-structured-worker`.

---

### Task 1: Add the failing Supervisor script contract tests

**Files:**
- Create: `tools/tests/test_structured_worker_supervisor_contract.py`
- Read: `docs/superpowers/specs/2026-08-10-supervisor-structured-worker-manager-design.md`

**Interfaces:**
- Consumes: the future script path `tools/manage_structured_worker_supervisor.sh` and its `help`, `render-config`, and rejected `uninstall` commands.
- Produces: executable contract tests that later tasks must satisfy without root or a running Supervisor daemon.

- [ ] **Step 1: Write the failing tests**

Use `unittest.TestCase` so the tests run with the backend virtual environment without adding pytest. Skip shell-specific tests only when `bash` is unavailable. The test module should include these concrete behaviors:

```python
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
SCRIPT = ROOT / "tools" / "manage_structured_worker_supervisor.sh"


class StructuredWorkerSupervisorContractTest(unittest.TestCase):
    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is required for the Ubuntu deployment script")
        return subprocess.run(
            [bash, str(SCRIPT), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_help_lists_lifecycle_commands(self) -> None:
        result = self.run_script("help")
        self.assertEqual(result.returncode, 0, result.stderr)
        for command in ("install", "start", "stop", "restart", "status", "logs", "render-config"):
            self.assertIn(command, result.stdout)

    def test_script_has_valid_bash_syntax(self) -> None:
        bash = shutil.which("bash")
        if bash is None:
            self.skipTest("bash is required for the Ubuntu deployment script")
        result = subprocess.run(
            [bash, "-n", str(SCRIPT)], text=True, capture_output=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_render_config_contains_worker_contract_without_secret_value(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw) / "DCAgent"
            (project / "backend" / "app").mkdir(parents=True)
            (project / "backend" / "app" / "structured_worker.py").write_text("# test\n", encoding="utf-8")
            env_file = project / "dcagent.env"
            env_file.write_text("DATABASE_URL=postgresql://test\nSECRET_VALUE=do-not-print\n", encoding="utf-8")
            env_file.chmod(0o600)
            python_path = Path(sys.executable)
            user = subprocess.check_output(["id", "-un"], text=True).strip()
            result = self.run_script(
                "render-config", "--project-root", str(project), "--user", user,
                "--env-file", str(env_file), "--python", str(python_path),
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("[program:dcagent-structured-worker]", result.stdout)
        self.assertIn("-m app.structured_worker", result.stdout)
        self.assertIn("autostart=true", result.stdout)
        self.assertIn("autorestart=true", result.stdout)
        self.assertIn(str(env_file), result.stdout)
        self.assertNotIn("do-not-print", result.stdout)

    def test_invalid_project_fails_without_installing_system_files(self) -> None:
        result = self.run_script(
            "render-config", "--project-root", "/does/not/exist", "--user", "root",
            "--env-file", "/does/not/exist.env",
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("/etc/supervisor", result.stdout + result.stderr)

    def test_uninstall_is_not_supported(self) -> None:
        result = self.run_script("uninstall")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("uninstall", result.stderr.lower())

    def test_render_config_rejects_group_writable_environment_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            project = Path(raw) / "DCAgent"
            (project / "backend" / "app").mkdir(parents=True)
            (project / "backend" / "app" / "structured_worker.py").write_text("# test\n", encoding="utf-8")
            env_file = project / "dcagent.env"
            env_file.write_text("DATABASE_URL=postgresql://test\n", encoding="utf-8")
            env_file.chmod(0o620)
            user = subprocess.check_output(["id", "-un"], text=True).strip()
            result = self.run_script(
                "render-config", "--project-root", str(project), "--user", user,
                "--env-file", str(env_file), "--python", sys.executable,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("writable", result.stderr.lower())
```

- [ ] **Step 2: Run the new tests and verify the expected red state**

Run on Ubuntu or WSL from the repository root:

```bash
./backend/.venv/bin/python -m unittest tools.tests.test_structured_worker_supervisor_contract -v
```

Expected result before implementation: the tests fail because `tools/manage_structured_worker_supervisor.sh` does not exist. On Windows without Bash, the shell tests are skipped; do not treat skipped shell tests as deployment verification.

- [ ] **Step 3: Preserve the red evidence without committing a broken tree**

Record the expected failure output in the task notes, leave the test uncommitted, and proceed directly
to Task 2. The contract test and implementation will be committed together only after the suite is
green, keeping `main` deployable.

### Task 2: Implement the idempotent Supervisor manager script

**Files:**
- Create: `tools/manage_structured_worker_supervisor.sh`
- Test: `tools/tests/test_structured_worker_supervisor_contract.py`

**Interfaces:**
- Consumes: `install`, `start`, `stop`, `restart`, `status`, `logs`, `render-config`, and internal `run` commands.
- Produces: `/etc/supervisor/conf.d/dcagent-structured-worker.conf`, `/var/log/dcagent/structured-worker.log`, and `/var/log/dcagent/structured-worker-error.log` on install.

- [ ] **Step 1: Add strict shell bootstrap and argument helpers**

Start the script with:

```bash
#!/usr/bin/env bash
set -euo pipefail

PROGRAM_NAME="dcagent-structured-worker"
CONF_PATH="/etc/supervisor/conf.d/${PROGRAM_NAME}.conf"
LOG_DIR="/var/log/dcagent"
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
```

Implement `die`, `usage`, `require_command`, `reject_unsafe_value`, `absolute_path`, and a parser that accepts only the documented long options. `reject_unsafe_value` must reject empty values, newlines, and whitespace; unknown options and missing values must exit non-zero.

- [ ] **Step 2: Implement input validation**

Implement `validate_install_inputs(project_root, service_user, env_file, python_path)` to check:

```bash
[[ "$(uname -s)" == "Linux" ]] || die "Linux is required"
[[ "${EUID}" -eq 0 ]] || die "install requires root"
command -v supervisorctl >/dev/null || die "supervisorctl is not installed"
[[ -d "${project_root}/backend/app" ]] || die "backend/app is missing"
[[ -f "${project_root}/backend/app/structured_worker.py" ]] || die "structured worker is missing"
[[ -x "${python_path}" ]] || die "Python executable is not executable"
[[ -f "${env_file}" && -r "${env_file}" ]] || die "environment file is unreadable"
```

Also require a real service user via `id`, reject group/world-writable environment files using `stat -c '%A'`, verify the service user can read the project, manager script, Python executable, and environment file with `sudo -u`, and require `/etc/supervisor/conf.d` to exist. `render-config` performs the non-root checks that do not require `supervisorctl`; `install` performs all checks.

- [ ] **Step 3: Implement configuration rendering**

Implement `render_config` so the exact output includes:

```ini
[program:dcagent-structured-worker]
directory=<project-root>/backend
command=<absolute-script> run --project-root <project-root> --env-file <env-file> --python <python>
user=<service-user>
autostart=true
autorestart=true
startsecs=5
startretries=10
stopsignal=TERM
stopasgroup=true
killasgroup=true
environment=PYTHONUNBUFFERED="1"
stdout_logfile=/var/log/dcagent/structured-worker.log
stderr_logfile=/var/log/dcagent/structured-worker-error.log
stdout_logfile_maxbytes=50MB
stderr_logfile_maxbytes=50MB
stdout_logfile_backups=10
stderr_logfile_backups=10
```

`render-config` writes only this text to standard output. It must include the environment-file path but never read or print the environment-file contents.

- [ ] **Step 4: Implement the internal `run` path**

Implement `run_worker` with no root requirement:

```bash
set -a
# shellcheck disable=SC1090
source "${env_file}"
set +a
cd "${project_root}/backend"
exec "${python_path}" -m app.structured_worker
```

Validate the same file and executable before sourcing. Do not enable `set -x`, print variables, or wrap Python in a second long-lived shell process.

- [ ] **Step 5: Implement install and lifecycle commands**

The `install` path must create the log directory with the service user as owner, render to a temporary file in the Supervisor configuration directory, compare with the existing file using `cmp -s`, and replace it with `/usr/bin/install -o root -g root -m 0644` only when changed. Then run:

```bash
supervisorctl reread
supervisorctl update
```

Start a stopped program; restart only when the generated configuration changed. `start`, `stop`, `restart`, and `status` call `supervisorctl` with the fixed program name. `logs` runs `tail -F` on both worker log files. Unsupported `uninstall` must fail through the usage path.

- [ ] **Step 6: Make the script executable and run the contract tests**

```bash
chmod 0755 tools/manage_structured_worker_supervisor.sh
./backend/.venv/bin/python -m unittest tools.tests.test_structured_worker_supervisor_contract -v
```

Expected result: all contract tests pass.

- [ ] **Step 7: Commit the implementation slice**

```bash
git add tools/manage_structured_worker_supervisor.sh tools/tests/test_structured_worker_supervisor_contract.py
git commit -m "feat: manage structured worker with Supervisor"
```

### Task 3: Document Ubuntu installation and recovery

**Files:**
- Modify: `README.md` near the backend startup instructions
- Modify: `docs/offline-platform-runbook.md` in the non-Docker/Ubuntu operations section

**Interfaces:**
- Consumes: the script commands and fixed Supervisor program name from Task 2.
- Produces: operator instructions that use the same environment file as the API and explain how to diagnose queued jobs.

- [ ] **Step 1: Add the documented installation command**

Document this command with paths explicitly marked for replacement:

```bash
sudo ./tools/manage_structured_worker_supervisor.sh install \
  --project-root /opt/DCAgent \
  --user dcagent \
  --env-file /etc/dcagent/dcagent.env
```

State that the environment file must contain the same database and ClickHouse settings used by the API and must be readable by `dcagent` but not writable by group/other users.

- [ ] **Step 2: Add lifecycle and queued-job checks**

Document:

```bash
sudo ./tools/manage_structured_worker_supervisor.sh status
sudo ./tools/manage_structured_worker_supervisor.sh restart
sudo ./tools/manage_structured_worker_supervisor.sh logs
```

Explain that `RUNNING` is required, and that a job remaining at `queued`/`checkpointRow: 0` should be investigated first by checking worker logs and matching API/worker environment files.

- [ ] **Step 3: Run documentation and whitespace checks**

```bash
git diff --check
```

- [ ] **Step 4: Commit the documentation slice**

```bash
git add README.md docs/offline-platform-runbook.md
git commit -m "docs: document Supervisor structured worker operations"
```

### Task 4: Run the full verification and Ubuntu acceptance checks

**Files:**
- Test: `tools/tests/test_structured_worker_supervisor_contract.py`
- Verify: `tools/manage_structured_worker_supervisor.sh`

**Interfaces:**
- Consumes: all scripts, docs, and tests from Tasks 1-3.
- Produces: evidence that local checks pass and a command sequence for the actual Ubuntu host.

- [ ] **Step 1: Run shell syntax and contract verification**

```bash
bash -n tools/manage_structured_worker_supervisor.sh
./backend/.venv/bin/python -m unittest tools.tests.test_structured_worker_supervisor_contract -v
```

- [ ] **Step 2: Run affected backend regression tests**

```bash
PYTHONPATH=backend ./backend/.venv/bin/python -m unittest \
  tests.test_structured_api tests.test_structured_worker -v
```

If the environment lacks optional `pyarrow`, report that limitation instead of claiming the worker suite passed.

- [ ] **Step 3: Perform Ubuntu host acceptance**

```bash
sudo ./tools/manage_structured_worker_supervisor.sh install \
  --project-root /opt/DCAgent \
  --user dcagent \
  --env-file /etc/dcagent/dcagent.env
sudo ./tools/manage_structured_worker_supervisor.sh status
sudo ./tools/manage_structured_worker_supervisor.sh logs
```

Confirm Supervisor reports `dcagent-structured-worker RUNNING`, the logs show polling without configuration errors, and a real structured job advances beyond `queued` with `checkpointRow: 0`.

- [ ] **Step 4: Review the final diff and working tree**

```bash
git diff --check
git status --short
git log -4 --oneline
```

Expected result: no whitespace errors, no untracked runtime secrets/logs, and only the intended script, contract test, and documentation changes.
