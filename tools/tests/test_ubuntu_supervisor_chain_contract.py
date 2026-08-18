from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools" / "start_ubuntu_supervisor_chain.sh"


class UbuntuSupervisorChainContractTest(unittest.TestCase):
    def setUp(self) -> None:
        if shutil.which("bash") is None:
            self.skipTest("bash is required for the Ubuntu deployment contract")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.work = Path(self.temp_dir.name)
        self.calls = self.work / "calls.log"
        supervisor = self.work / "supervisorctl"
        supervisor.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'supervisor %s\\n' \"$*\" >> \"$CALLS_LOG\"\n",
            encoding="utf-8",
        )
        supervisor.chmod(0o755)
        bootstrap = self.work / "bootstrap.sh"
        bootstrap.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'bootstrap\\n' >> \"$CALLS_LOG\"\n",
            encoding="utf-8",
        )
        bootstrap.chmod(0o755)
        self.supervisor = supervisor
        self.bootstrap = bootstrap

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def run_script(self, *, bootstrap: Path | None = None) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment.update(
            {
                "DCAGENT_SUPERVISORCTL": str(self.supervisor),
                "DCAGENT_RETRIEVAL_BOOTSTRAP": str(bootstrap or self.bootstrap),
                "CALLS_LOG": str(self.calls),
            }
        )
        return subprocess.run(
            ["bash", str(SCRIPT)],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_starts_models_bootstraps_then_starts_application(self) -> None:
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            self.calls.read_text(encoding="utf-8").splitlines(),
            [
                "supervisor start dcagent-llama-embedding dcagent-llama-reranker dcagent-ollama-llm",
                "bootstrap",
                "supervisor start dcagent-api dcagent-structured-worker",
                "supervisor status dcagent-llama-embedding dcagent-llama-reranker dcagent-ollama-llm dcagent-api dcagent-structured-worker",
            ],
        )

    def test_does_not_start_application_when_bootstrap_fails(self) -> None:
        failing = self.work / "failing-bootstrap.sh"
        failing.write_text(
            "#!/usr/bin/env bash\n"
            "printf 'bootstrap-failed\\n' >> \"$CALLS_LOG\"\n"
            "exit 1\n",
            encoding="utf-8",
        )
        failing.chmod(0o755)

        result = self.run_script(bootstrap=failing)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            self.calls.read_text(encoding="utf-8").splitlines(),
            [
                "supervisor start dcagent-llama-embedding dcagent-llama-reranker dcagent-ollama-llm",
                "bootstrap-failed",
            ],
        )

    def test_shell_syntax_is_valid(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
