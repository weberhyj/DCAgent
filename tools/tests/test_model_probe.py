from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[tuple[list[str], bool]] = []
        self.override_snapshots: list[dict[str, object]] = []
        self.override_modes: list[int] = []

    def __call__(self, command, *, shell, **_kwargs):
        from tools.benchmarks.model_probe import CommandResult

        self.calls.append((list(command), shell))
        override_paths = [
            Path(command[index + 1])
            for index, value in enumerate(command[:-1])
            if value == "-f"
        ]
        if len(override_paths) > 1 and override_paths[-1].exists():
            self.override_snapshots.append(
                json.loads(override_paths[-1].read_text(encoding="utf-8"))
            )
            self.override_modes.append(stat.S_IMODE(override_paths[-1].stat().st_mode))
        outcome = self.outcomes.pop(0)
        if callable(outcome):
            outcome = outcome(command)
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, CommandResult):
            return outcome
        exit_code, payload = outcome
        if (
            isinstance(payload, dict)
            and "--project-name" in command
            and "config" in command
        ):
            payload = dict(payload)
            payload["name"] = command[command.index("--project-name") + 1]
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        return CommandResult(exit_code=exit_code, stdout=stdout)


def rendered_config(
    kind: str,
    *,
    checksum: str | None = None,
    candidate_path: Path | None = None,
    candidate_name: str = "local-candidate",
) -> dict[str, object]:
    services = {
        "embedding-service": {"networks": {"offline": {}}, "ports": []},
    }
    if kind == "generation-model":
        services["llama"] = {"networks": {"offline": {}}, "ports": []}
    if checksum is not None and candidate_path is not None:
        environment = {
            "MODEL_PROBE_CANDIDATE_NAME": candidate_name,
            "MODEL_PROBE_CANDIDATE_VERSION": "1",
            "MODEL_PROBE_CANDIDATE_SHA256": checksum,
        }
        services["embedding-service"].update(
            {
                "command": ["python", "-c", "fixed-helper"],
                "environment": environment,
            }
        )
        if kind == "generation-model":
            services["llama"].update(
                {
                    "command": ["--model", "/probe-candidate/model.gguf"],
                    "volumes": [
                        {
                            "source": str(candidate_path),
                            "target": "/probe-candidate/model.gguf",
                        }
                    ],
                }
            )
        else:
            services["embedding-service"]["volumes"] = [
                {
                    "source": str(candidate_path),
                    "target": "/probe-candidate",
                }
            ]
    return {
        "name": "dc-agent-offline-probe",
        "networks": {"offline": {"internal": True}},
        "services": services,
    }


def probe_outcomes(
    kind: str,
    checksum: str,
    candidate_path: Path,
    metrics: dict[str, object],
    *,
    metadata: dict[str, object] | None = None,
    benchmark_exit_code: int = 0,
):
    candidate_name = "bge-small" if kind == "embedding-model" else "local-candidate"
    return [
        (
            0,
            rendered_config(
                kind,
                checksum=checksum,
                candidate_path=candidate_path,
                candidate_name=candidate_name,
            ),
        ),
        (0, ""),
        (
            0,
            metadata
            or {
                "name": candidate_name,
                "version": "1",
                "modelChecksum": checksum,
                "publicNetworkAttempted": False,
            },
        ),
        (benchmark_exit_code, metrics),
        (0, ""),
    ]


def embedding_metrics() -> dict[str, object]:
    return {
        "family": "BGE",
        "variant": "small",
        "actualTokenRange": [300, 800],
        "modelMetadata": {
            "modelName": "bge-small",
            "modelVersion": "1",
            "dimensions": 8,
        },
        "batchDocumentsPerSecond": 37.5,
        "residentMemoryBytes": 1_500_000_000,
        "queryEmbedding": {
            "1": {"p50Ms": 200, "p95Ms": 350},
            "5": {"p50Ms": 400, "p95Ms": 700},
            "15": {"p50Ms": 700, "p95Ms": 1200},
        },
    }


def generation_metrics() -> dict[str, object]:
    return {
        "family": "qwen2",
        "parameterSize": "1.5B",
        "quantization": "Q4_K_M",
        "contexts": {
            "512": {
                "actualContextTokens": 512,
                "queueWaitP50Ms": 100,
                "queueWaitP95Ms": 200,
                "firstTokenP50Ms": 1000,
                "firstTokenP95Ms": 2000,
                "availableFirstTokenP50Ms": 900,
                "availableFirstTokenP95Ms": 1800,
                "outputTokensPerSecond": 20,
                "failureRate": 0,
            },
            "1024": {
                "actualContextTokens": 1024,
                "queueWaitP50Ms": 150,
                "queueWaitP95Ms": 350,
                "firstTokenP50Ms": 1300,
                "firstTokenP95Ms": 2500,
                "availableFirstTokenP50Ms": 1150,
                "availableFirstTokenP95Ms": 2150,
                "outputTokensPerSecond": 18,
                "failureRate": 0.01,
            },
            "2048": {
                "actualContextTokens": 2048,
                "queueWaitP50Ms": 300,
                "queueWaitP95Ms": 700,
                "firstTokenP50Ms": 1800,
                "firstTokenP95Ms": 4000,
                "availableFirstTokenP50Ms": 1500,
                "availableFirstTokenP95Ms": 3300,
                "outputTokensPerSecond": 15,
                "failureRate": 0.02,
            },
        },
        "maxOutputTokens": 256,
        "modelMetadata": {
            "architecture": "qwen2",
            "parameterSize": "1.5B",
            "quantization": "Q4_K_M",
        },
        "publicNetworkAttempted": "not_available",
    }


def write_minimal_gguf(
    path: Path,
    *,
    name: str = "local-candidate",
    irrelevant_value: str = "irrelevant-large-metadata-placeholder",
) -> None:
    metadata = (
        ("general.architecture", 8, "qwen2"),
        ("general.name", 8, name),
        ("general.version", 8, "1"),
        ("general.size_label", 8, "1.5B"),
        ("general.file_type", 4, 15),
        ("tokenizer.ggml.model", 8, irrelevant_value),
    )
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<IQQ", 3, 0, len(metadata)))
        for key, value_type, value in metadata:
            encoded_key = key.encode("utf-8")
            handle.write(struct.pack("<Q", len(encoded_key)))
            handle.write(encoded_key)
            handle.write(struct.pack("<I", value_type))
            if value_type == 8:
                encoded_value = str(value).encode("utf-8")
                handle.write(struct.pack("<Q", len(encoded_value)))
                handle.write(encoded_value)
            else:
                handle.write(struct.pack("<I", int(value)))


class ModelProbeTest(unittest.TestCase):
    def test_direct_script_help_is_import_safe(self) -> None:
        script = Path(__file__).resolve().parents[1] / "benchmarks" / "model_probe.py"
        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=script.parents[2],
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--candidate-lock", completed.stdout)

    def test_probe_source_does_not_reference_an_unshipped_executable(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "benchmarks" / "model_probe.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("dc-agent-model-probe", source)

    def test_fixed_python_helpers_compile_and_generation_measures_throughput(self) -> None:
        from tools.benchmarks import model_probe

        scripts = (
            model_probe.EMBEDDING_METADATA_SCRIPT,
            model_probe.EMBEDDING_BENCHMARK_SCRIPT,
            model_probe.RERANKER_METADATA_SCRIPT,
            model_probe.RERANKER_BENCHMARK_SCRIPT,
            model_probe.GENERATION_METADATA_SCRIPT,
            model_probe.GENERATION_BENCHMARK_SCRIPT,
        )
        for index, script in enumerate(scripts):
            with self.subTest(index=index):
                compile(script, f"<model-probe-helper-{index}>", "exec")
        self.assertNotIn(
            '"outputTokensPerSecond": 0',
            model_probe.GENERATION_BENCHMARK_SCRIPT,
        )
        self.assertIn("failureRate", model_probe.GENERATION_BENCHMARK_SCRIPT)
        self.assertIn("model path is missing", model_probe.GENERATION_METADATA_SCRIPT)
        self.assertIn("math.ceil", model_probe.GENERATION_BENCHMARK_SCRIPT)
        self.assertIn("math.ceil", model_probe.EMBEDDING_BENCHMARK_SCRIPT)
        self.assertIn("math.ceil", model_probe.RERANKER_BENCHMARK_SCRIPT)
        self.assertNotIn('"busyTimeoutRate": 0', model_probe.RERANKER_BENCHMARK_SCRIPT)

    def test_embedding_metrics_use_service_metadata_and_actual_tokenizer(self) -> None:
        from tools.benchmarks import model_probe

        self.assertIn("/v1/metadata", model_probe.EMBEDDING_BENCHMARK_SCRIPT)
        self.assertIn("AutoTokenizer", model_probe.EMBEDDING_BENCHMARK_SCRIPT)
        self.assertIn("timed_post", model_probe.EMBEDDING_BENCHMARK_SCRIPT)
        self.assertNotIn('"family": "BGE"', model_probe.EMBEDDING_BENCHMARK_SCRIPT)
        self.assertNotIn('"variant": os.environ', model_probe.EMBEDDING_BENCHMARK_SCRIPT)
        self.assertIn("actualTokenRange", model_probe.EMBEDDING_BENCHMARK_SCRIPT)

    def test_generation_helper_uses_real_tokenizer_and_fifteen_way_concurrency(self) -> None:
        from tools.benchmarks import model_probe

        script = model_probe.GENERATION_BENCHMARK_SCRIPT
        self.assertIn("/tokenize", script)
        self.assertIn("/detokenize", script)
        self.assertIn("max_workers=15", script)
        self.assertIn("actualContextTokens", script)
        self.assertIn("predicted_per_second", script)
        self.assertIn("content", script)
        self.assertIn("if text and first is None", script)
        self.assertIn("prompt_ms", script)
        self.assertIn("predicted_ms", script)
        self.assertIn("total_ms - prompt_ms - predicted_ms", script)
        self.assertIn("available_first = max(0, first - queue_wait_ms)", script)
        self.assertIn("availableFirstTokenP95Ms", script)
        self.assertNotIn("queue = (time.perf_counter() - started)", script)
        self.assertNotIn('"publicNetworkAttempted": False', script)

    def test_available_slot_first_token_removes_queue_time_once(self) -> None:
        from tools.benchmarks.model_probe import available_first_token_ms

        self.assertEqual(available_first_token_ms(900, 400), 500)
        self.assertEqual(available_first_token_ms(400, 900), 0)

    def test_generation_loader_audit_unavailable_fails_closed_in_report(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("generation-model") as fixture:
            runner = FakeRunner(
                probe_outcomes("generation-model", fixture["checksum"], fixture["artifact"], generation_metrics())
            )
            result = run_model_probe(
                compose_file=fixture["compose"],
                embedding_service="embedding-service",
                llama_service="llama",
                discovery_label="discovery",
                candidate_entry=fixture["entry"],
                artifact_root=fixture["root"],
                report_path=fixture["report"],
                runner=runner,
            )
            self.assertFalse(result.passed)
            self.assertIn("public_network_audit_unavailable", result.failures)
            payload = json.loads(fixture["report"].read_text(encoding="utf-8"))
            self.assertEqual(payload["networkPolicy"]["loaderAudit"], "unavailable")

    def test_generation_with_llama_loader_audit_can_pass(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("generation-model") as fixture:
            metadata = {
                "name": "local-candidate",
                "version": "1",
                "modelChecksum": fixture["checksum"],
                "loaderAuditAvailable": True,
                "publicNetworkAttempted": False,
                "serviceMetadata": {
                    "model_path": "/probe-candidate/model.gguf",
                    "model_loader_network_audit": "complete",
                    "public_network_attempted": False,
                },
            }
            runner = FakeRunner(
                probe_outcomes(
                    "generation-model",
                    fixture["checksum"],
                    fixture["artifact"],
                    generation_metrics(),
                    metadata=metadata,
                )
            )
            result = run_model_probe(
                compose_file=fixture["compose"],
                embedding_service="embedding-service",
                llama_service="llama",
                discovery_label="discovery",
                candidate_entry=fixture["entry"],
                artifact_root=fixture["root"],
                report_path=fixture["report"],
                runner=runner,
            )
            self.assertTrue(result.passed)
            payload = json.loads(fixture["report"].read_text(encoding="utf-8"))
            self.assertEqual(payload["networkPolicy"]["loaderAudit"], "llama_props")

    def test_candidate_paths_are_kind_specific_and_gguf_metadata_is_read(self) -> None:
        from tools.benchmarks.model_probe import load_candidate_artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "bge"
            model_dir.mkdir()
            (model_dir / "embedding-metadata.json").write_text("{}", encoding="utf-8")
            gguf = root / "qwen.gguf"
            write_minimal_gguf(gguf)
            directory_entry = {
                "name": "bge-small",
                "kind": "embedding-model",
                "version": "1",
                "sha256": "a" * 64,
                "license": "approved",
                "localPath": "bge",
            }
            with self.assertRaisesRegex(ValueError, "directory"):
                load_candidate_artifact(
                    {**directory_entry, "localPath": "qwen.gguf"},
                    artifact_root=root,
                    artifact_hasher=lambda _path: "a" * 64,
                )
            checksum = hashlib.sha256(gguf.read_bytes()).hexdigest()
            candidate = load_candidate_artifact(
                {
                    "name": "local-candidate",
                    "kind": "generation-model",
                    "version": "1",
                    "sha256": checksum,
                    "license": "approved",
                    "localPath": "qwen.gguf",
                },
                artifact_root=root,
            )
            self.assertEqual(candidate.metadata["general.architecture"], "qwen2")
            self.assertEqual(candidate.metadata["general.file_type"], 15)
            self.assertNotIn("tokenizer.ggml.model", candidate.metadata)

    def test_cli_selects_one_probeable_candidate_from_a_complete_lock(self) -> None:
        from tools.benchmarks import model_probe

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "artifacts.lock.json"
            entries = [
                {
                    "name": "one",
                    "kind": "generation-model",
                    "version": "1",
                    "sha256": "a" * 64,
                    "license": "approved",
                    "localPath": "one.gguf",
                },
                {
                    "name": "two",
                    "kind": "embedding-model",
                    "version": "1",
                    "sha256": "b" * 64,
                    "license": "approved",
                    "localPath": "two",
                },
            ]
            lock.write_text(json.dumps({"artifacts": entries}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate-name"):
                model_probe._load_candidate_lock(lock)
            selected, _ = model_probe._load_candidate_lock(lock, candidate_name="two")
            self.assertEqual(selected["name"], "two")

    def test_probe_project_name_is_stable_legal_and_path_specific(self) -> None:
        from tools.benchmarks.model_probe import probe_project_name

        first = probe_project_name(Path("C:/worktree-a/deploy/offline/compose.yaml"))
        repeated = probe_project_name(Path("C:/worktree-a/deploy/offline/compose.yaml"))
        second = probe_project_name(Path("C:/worktree-b/deploy/offline/compose.yaml"))
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^[a-z0-9][a-z0-9_-]+$")

    def test_rejects_candidate_when_first_token_gate_is_missed(self) -> None:
        from tools.benchmarks.model_probe import ModelGate, evaluate_model_probe

        result = evaluate_model_probe(
            ModelGate(
                max_query_embedding_p95_ms=1500,
                max_queue_feedback_p95_ms=2000,
                max_first_token_p95_ms=10000,
            ),
            {
                "query_embedding_p95_ms": 700,
                "queue_feedback_p95_ms": 900,
                "first_token_p95_ms": 13000,
            },
        )
        self.assertFalse(result.passed)
        self.assertEqual(result.failures, ["first_token_p95_ms"])

    def test_gate_fails_closed_for_missing_non_numeric_and_non_finite_metrics(self) -> None:
        from tools.benchmarks.model_probe import ModelGate, evaluate_model_probe

        gate = ModelGate(1500, 2000, 10000)
        for value in (None, "not_available", math.nan, math.inf, True):
            with self.subTest(value=value):
                result = evaluate_model_probe(
                    gate,
                    {
                        "query_embedding_p95_ms": value,
                        "queue_feedback_p95_ms": 2000,
                        "first_token_p95_ms": 10000,
                    },
                )
                self.assertEqual(result.failures, ["query_embedding_p95_ms"])
        self.assertTrue(
            evaluate_model_probe(
                gate,
                {
                    "query_embedding_p95_ms": 1500,
                    "queue_feedback_p95_ms": 2000,
                    "first_token_p95_ms": 10000,
                },
            ).passed
        )

    def test_candidate_entry_requires_strict_checksum_and_bounded_local_path(self) -> None:
        from tools.benchmarks.model_probe import load_candidate_artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "models" / "candidate.gguf"
            artifact.parent.mkdir()
            write_minimal_gguf(artifact, name="qwen-local")
            checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
            entry = {
                "name": "qwen-local",
                "kind": "generation-model",
                "version": "1",
                "sha256": checksum,
                "license": "approved",
                "localPath": "models/candidate.gguf",
            }
            candidate = load_candidate_artifact(entry, artifact_root=root)
            self.assertEqual(candidate.path, artifact.resolve())
            self.assertEqual(candidate.sha256, checksum)

            invalid_entries = (
                {**entry, "sha256": checksum.upper()},
                {**entry, "sha256": "a" * 63},
                {**entry, "localPath": "https://example.com/model"},
                {**entry, "localPath": "../outside.gguf"},
                {**entry, "extra": "not-allowed"},
            )
            for invalid in invalid_entries:
                with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                    load_candidate_artifact(invalid, artifact_root=root)

            artifact.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                load_candidate_artifact(entry, artifact_root=root)

    def test_candidate_entry_allows_an_absolute_local_path_outside_lock_root(self) -> None:
        from tools.benchmarks.model_probe import load_candidate_artifact

        with tempfile.TemporaryDirectory() as lock_directory, tempfile.TemporaryDirectory() as artifact_directory:
            artifact = Path(artifact_directory) / "candidate.gguf"
            write_minimal_gguf(artifact, name="qwen-local")
            checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()
            candidate = load_candidate_artifact(
                {
                    "name": "qwen-local",
                    "kind": "generation-model",
                    "version": "1",
                    "sha256": checksum,
                    "license": "approved",
                    "localPath": str(artifact.resolve()),
                },
                artifact_root=Path(lock_directory),
            )
            self.assertEqual(candidate.path, artifact.resolve())

    def test_candidate_load_rejects_replacement_after_hasher_returns_old_digest(self) -> None:
        from tools.benchmarks.model_probe import load_candidate_artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate.gguf"
            write_minimal_gguf(artifact, name="local-candidate", irrelevant_value="original")
            original_digest = hashlib.sha256(artifact.read_bytes()).hexdigest()

            def replacing_hasher(path: Path) -> str:
                replacement = path.with_suffix(".replacement.gguf")
                write_minimal_gguf(
                    replacement,
                    name="local-candidate",
                    irrelevant_value="replacement-with-same-identity",
                )
                os.replace(replacement, path)
                return original_digest

            entry = {
                "name": "local-candidate",
                "kind": "generation-model",
                "version": "1",
                "sha256": original_digest,
                "license": "approved",
                "localPath": artifact.name,
            }
            with self.assertRaisesRegex(ValueError, "changed|checksum"):
                load_candidate_artifact(
                    entry,
                    artifact_root=root,
                    artifact_hasher=replacing_hasher,
                )

    def test_directory_candidate_rejects_same_stat_nested_file_replacement(self) -> None:
        from tools.benchmarks.model_probe import load_candidate_artifact, sha256_artifact

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate-model"
            nested = artifact / "nested"
            nested.mkdir(parents=True)
            weights = nested / "weights.bin"
            weights.write_bytes(b"GOOD")
            (artifact / "embedding-metadata.json").write_text(
                json.dumps(
                    {
                        "modelName": "bge-small",
                        "modelVersion": "1",
                        "dimensions": 8,
                    }
                ),
                encoding="utf-8",
            )
            original_digest = sha256_artifact(artifact)
            root_stat = artifact.stat()
            nested_stat = nested.stat()
            weights_stat = weights.stat()
            calls = 0

            def replacing_hasher(_path: Path) -> str:
                nonlocal calls
                calls += 1
                if calls == 1:
                    replacement = root / "replacement.bin"
                    replacement.write_bytes(b"EVIL")
                    os.utime(
                        replacement,
                        ns=(weights_stat.st_atime_ns, weights_stat.st_mtime_ns),
                    )
                    os.replace(replacement, weights)
                    os.utime(
                        nested,
                        ns=(nested_stat.st_atime_ns, nested_stat.st_mtime_ns),
                    )
                    os.utime(
                        artifact,
                        ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns),
                    )
                return original_digest

            entry = {
                "name": "bge-small",
                "kind": "embedding-model",
                "version": "1",
                "sha256": original_digest,
                "license": "approved",
                "localPath": artifact.name,
            }
            with self.assertRaisesRegex(ValueError, "changed|checksum"):
                load_candidate_artifact(
                    entry,
                    artifact_root=root,
                    artifact_hasher=replacing_hasher,
                )
            self.assertEqual(weights.read_bytes(), b"EVIL")

    def test_directory_fingerprint_budget_counts_directories_and_files(self) -> None:
        from tools.benchmarks import model_probe

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "candidate-model"
            artifact.mkdir()
            (artifact / "empty-a").mkdir()
            (artifact / "empty-b").mkdir()
            (artifact / "embedding-metadata.json").write_text(
                json.dumps(
                    {
                        "modelName": "bge-small",
                        "modelVersion": "1",
                        "dimensions": 8,
                    }
                ),
                encoding="utf-8",
            )
            checksum = model_probe.sha256_artifact(artifact)
            entry = {
                "name": "bge-small",
                "kind": "embedding-model",
                "version": "1",
                "sha256": checksum,
                "license": "approved",
                "localPath": artifact.name,
            }
            with patch.object(model_probe, "MAX_ARTIFACT_TREE_ENTRIES", 2):
                with self.assertRaisesRegex(ValueError, "budget"):
                    model_probe.load_candidate_artifact(entry, artifact_root=root)

    def test_gguf_metadata_rejects_excessive_nesting_with_bounded_error(self) -> None:
        from tools.benchmarks.model_probe import read_gguf_metadata

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep.gguf"
            key = b"general.name"
            with path.open("wb") as handle:
                handle.write(b"GGUF")
                handle.write(struct.pack("<IQQ", 3, 0, 1))
                handle.write(struct.pack("<Q", len(key)))
                handle.write(key)
                handle.write(struct.pack("<I", 9))
                for _ in range(1500):
                    handle.write(struct.pack("<IQ", 9, 1))
                value = b"local-candidate"
                handle.write(struct.pack("<IQ", 8, len(value)))
                handle.write(value)
            with self.assertRaisesRegex(ValueError, "depth|budget"):
                read_gguf_metadata(path)

    def test_gguf_metadata_rejects_excessive_cumulative_value_work(self) -> None:
        from tools.benchmarks.model_probe import read_gguf_metadata

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wide.gguf"
            key = b"tokenizer.ggml.tokens"
            with path.open("wb") as handle:
                handle.write(b"GGUF")
                handle.write(struct.pack("<IQQ", 3, 0, 1))
                handle.write(struct.pack("<Q", len(key)))
                handle.write(key)
                handle.write(struct.pack("<I", 9))
                handle.write(struct.pack("<IQ", 0, 1_000_000))
                handle.write(b"\0" * 1_000_000)
            with self.assertRaisesRegex(ValueError, "budget"):
                read_gguf_metadata(path)

    def test_compose_commands_are_fixed_argv_and_never_use_a_shell(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("embedding-model") as fixture:
            metadata = self._metadata(fixture["checksum"])
            runner = FakeRunner(
                probe_outcomes(
                    "embedding-model",
                    fixture["checksum"],
                    fixture["artifact"],
                    embedding_metrics(),
                    metadata=metadata,
                )
            )
            report = run_model_probe(
                compose_file=fixture["compose"],
                embedding_service="embedding-service",
                llama_service="llama",
                discovery_label="discovery",
                candidate_entry=fixture["entry"],
                artifact_root=fixture["root"],
                report_path=fixture["report"],
                runner=runner,
            )
            prefix = ["docker", "--context", "default", "compose"]
            self.assertTrue(report.passed)
            self.assertIn("config", runner.calls[0][0])
            up = runner.calls[1][0]
            self.assertEqual(
                up[up.index("up") : up.index("up") + 7],
                ["up", "-d", "--wait", "--no-build", "--pull", "never", "embedding-service"],
            )
            self.assertEqual(runner.calls[0][0][:4], prefix)
            self.assertIn("python", runner.calls[2][0])
            self.assertIn("-c", runner.calls[2][0])
            self.assertNotIn("dc-agent-model-probe", runner.calls[2][0])
            self.assertEqual(runner.calls[-1][0][-2:], ["--remove-orphans", "--volumes"])
            self.assertTrue(all(shell is False for _, shell in runner.calls))

            with self.assertRaises(ValueError):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service; shutdown",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=FakeRunner([]),
                )

    def test_runtime_candidate_replacement_fails_closed_and_still_cleans_up(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("generation-model") as fixture:
            metadata = {
                "name": "local-candidate",
                "version": "1",
                "modelChecksum": fixture["checksum"],
                "loaderAuditAvailable": True,
                "publicNetworkAttempted": False,
            }

            def replace_after_up(_command):
                replacement = fixture["artifact"].with_suffix(".runtime.gguf")
                write_minimal_gguf(
                    replacement,
                    name="local-candidate",
                    irrelevant_value="runtime-replacement-with-same-identity",
                )
                os.replace(replacement, fixture["artifact"])
                return (0, "")

            runner = FakeRunner(
                [
                    (
                        0,
                        rendered_config(
                            "generation-model",
                            checksum=fixture["checksum"],
                            candidate_path=fixture["artifact"],
                        ),
                    ),
                    replace_after_up,
                    (0, metadata),
                    (0, generation_metrics()),
                    (0, ""),
                ]
            )
            with self.assertRaisesRegex(ValueError, "changed|checksum"):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )
            self.assertIn("down", runner.calls[-1][0])

    def test_candidate_replacement_after_config_is_rejected_before_up(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("generation-model") as fixture:
            def replace_during_config(_command):
                replacement = fixture["artifact"].with_suffix(".config.gguf")
                write_minimal_gguf(
                    replacement,
                    name="local-candidate",
                    irrelevant_value="config-replacement-with-same-identity",
                )
                os.replace(replacement, fixture["artifact"])
                return (
                    0,
                    rendered_config(
                        "generation-model",
                        checksum=fixture["checksum"],
                        candidate_path=fixture["artifact"],
                    ),
                )

            runner = FakeRunner([replace_during_config])
            with self.assertRaisesRegex(ValueError, "changed|checksum"):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )
            self.assertFalse(any("up" in command for command, _shell in runner.calls))

    def test_candidate_replacement_during_benchmark_is_rejected_before_pass(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("generation-model") as fixture:
            metadata = {
                "name": "local-candidate",
                "version": "1",
                "modelChecksum": fixture["checksum"],
                "loaderAuditAvailable": True,
                "publicNetworkAttempted": False,
            }

            def replace_during_benchmark(_command):
                replacement = fixture["artifact"].with_suffix(".benchmark.gguf")
                write_minimal_gguf(
                    replacement,
                    name="local-candidate",
                    irrelevant_value="benchmark-replacement-with-same-identity",
                )
                os.replace(replacement, fixture["artifact"])
                return (0, generation_metrics())

            runner = FakeRunner(
                [
                    (
                        0,
                        rendered_config(
                            "generation-model",
                            checksum=fixture["checksum"],
                            candidate_path=fixture["artifact"],
                        ),
                    ),
                    (0, ""),
                    (0, metadata),
                    replace_during_benchmark,
                    (0, ""),
                ]
            )
            with self.assertRaisesRegex(ValueError, "changed|checksum"):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )
            self.assertIn("down", runner.calls[-1][0])

    def test_override_injects_each_locked_candidate_into_real_image_commands(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        cases = (
            ("embedding-model", embedding_metrics()),
            (
                "reranker-model",
                {
                    "family": "BGE",
                    "top20To10P50Ms": 500,
                    "top20To10P95Ms": 900,
                    "residentMemoryBytes": 500_000_000,
                    "modelChecksum": None,
                    "busyTimeoutRate": 0,
                },
            ),
            ("generation-model", generation_metrics()),
        )
        for kind, metrics in cases:
            with self.subTest(kind=kind), self._probe_files(kind) as fixture:
                if kind == "reranker-model":
                    metrics = {**metrics, "modelChecksum": fixture["checksum"]}
                runner = FakeRunner(
                    probe_outcomes(
                        kind, fixture["checksum"], fixture["artifact"], metrics
                    )
                )
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )
                self.assertTrue(runner.override_snapshots)
                if os.name != "nt":
                    self.assertTrue(all(mode & 0o077 == 0 for mode in runner.override_modes))
                override = runner.override_snapshots[0]
                encoded = json.dumps(override, sort_keys=True)
                sources = {
                    volume["source"]
                    for service in override["services"].values()
                    for volume in service.get("volumes", [])
                }
                self.assertIn(str(fixture["artifact"].resolve()), sources)
                self.assertIn(fixture["checksum"], encoded)
                self.assertIn(fixture["entry"]["name"], encoded)
                self.assertIn('"1"', encoded)
                if kind == "generation-model":
                    self.assertIn("/probe-candidate/model.gguf", encoded)
                    self.assertIn("llama", override["services"])
                else:
                    self.assertIn("/probe-candidate", encoded)
                    self.assertIn("embedding-service", override["services"])
                if kind == "embedding-model":
                    command = override["services"]["embedding-service"]["command"]
                    self.assertEqual(command[:2], ["python", "-c"])
                    self.assertIn("sys.addaudithook", command[2])
                    self.assertIn("public-network-attempted", command[2])

    def test_config_preflight_rejects_non_internal_or_published_probe_services_before_up(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        invalid_configs = (
            {
                "name": "dc-agent-offline-probe",
                "networks": {"offline": {"internal": False}},
                "services": {
                    "embedding-service": {"networks": {"offline": {}}, "ports": []}
                },
            },
            {
                "name": "dc-agent-offline-probe",
                "networks": {"offline": {"internal": True}},
                "services": {
                    "embedding-service": {
                        "networks": {"offline": {}},
                        "ports": [{"published": "8081"}],
                    }
                },
            },
        )
        for rendered in invalid_configs:
            with self.subTest(rendered=rendered), self._probe_files(
                "embedding-model"
            ) as fixture:
                runner = FakeRunner([(0, rendered)])
                result = run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )
                self.assertFalse(result.passed)
                self.assertIn("compose_private_network", result.failures)
                self.assertEqual(len(runner.calls), 1)
                self.assertNotIn("up", runner.calls[0][0])

    def test_config_preflight_requires_candidate_injection_in_final_rendered_services(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("embedding-model") as fixture:
            runner = FakeRunner([(0, rendered_config("embedding-model"))])
            result = run_model_probe(
                compose_file=fixture["compose"],
                embedding_service="embedding-service",
                llama_service="llama",
                discovery_label="discovery",
                candidate_entry=fixture["entry"],
                artifact_root=fixture["root"],
                report_path=fixture["report"],
                runner=runner,
            )
            self.assertFalse(result.passed)
            self.assertIn("candidate_injection", result.failures)
            self.assertEqual(len(runner.calls), 1)

    def test_probe_rejects_remote_docker_context_before_compose(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("embedding-model") as fixture, patch.dict(
            os.environ, {"DOCKER_HOST": "tcp://public.example:2375"}, clear=False
        ):
            with self.assertRaisesRegex(ValueError, "local Docker"):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=FakeRunner([]),
                )

    def test_cross_process_mutex_rejects_a_second_candidate_and_releases_afterward(self) -> None:
        from tools.benchmarks.model_probe import ProbeMutex, run_model_probe

        with self._probe_files("embedding-model") as fixture:
            lock_path = fixture["compose"].parent / ".model-probe.lock"
            with ProbeMutex(lock_path):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    run_model_probe(
                        compose_file=fixture["compose"],
                        embedding_service="embedding-service",
                        llama_service="llama",
                        discovery_label="discovery",
                        candidate_entry=fixture["entry"],
                        artifact_root=fixture["root"],
                        report_path=fixture["report"],
                        runner=FakeRunner([]),
                    )
            self.assertFalse(lock_path.exists())

    def test_probe_mutex_rejects_symlink_without_modifying_victim(self) -> None:
        from tools.benchmarks.model_probe import ProbeMutex

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            victim = root / "victim.lock"
            lock = root / "probe.lock"
            victim.write_bytes(b"victim")
            try:
                lock.symlink_to(victim)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(ValueError, "symbolic link|symlink"):
                with ProbeMutex(lock):
                    pass
            self.assertEqual(victim.read_bytes(), b"victim")

    def test_kind_specific_gate_reports_non_applicable_metrics_without_zeroes(self) -> None:
        from tools.benchmarks.model_probe import ModelGate, evaluate_candidate_gate

        gate = ModelGate(1500, 2000, 10000)
        embedding = evaluate_candidate_gate(
            "embedding-model", gate, {"query_embedding_p95_ms": 1000}
        )
        generation = evaluate_candidate_gate(
            "generation-model",
            gate,
            {"queue_feedback_p95_ms": 1000, "first_token_p95_ms": 9000},
        )
        reranker = evaluate_candidate_gate("reranker-model", gate, {})
        self.assertTrue(embedding.result.passed)
        self.assertEqual(embedding.metrics["queue_feedback_p95_ms"], "not_applicable")
        self.assertEqual(embedding.metrics["first_token_p95_ms"], "not_applicable")
        self.assertTrue(generation.result.passed)
        self.assertEqual(generation.metrics["query_embedding_p95_ms"], "not_applicable")
        self.assertTrue(reranker.result.passed)
        self.assertTrue(
            all(value == "not_applicable" for value in reranker.metrics.values())
        )

    def test_candidate_service_is_stopped_in_finally_after_probe_exception(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("embedding-model") as fixture:
            runner = FakeRunner(
                [
                    (
                        0,
                        rendered_config(
                            "embedding-model",
                            checksum=fixture["checksum"],
                            candidate_path=fixture["artifact"],
                            candidate_name=fixture["entry"]["name"],
                        ),
                    ),
                    (0, ""),
                    RuntimeError("probe crashed"),
                    (0, ""),
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "probe crashed"):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )
            commands = [call[0] for call in runner.calls]
            self.assertIn("config", commands[0])
            self.assertIn("down", commands[-1])
            self.assertNotIn("llama", [item for command in commands for item in command])
            override_paths = [
                Path(command[index + 1])
                for command in commands
                for index, value in enumerate(command[:-1])
                if value == "-f" and command[index + 1] != str(fixture["compose"].resolve())
            ]
            self.assertTrue(override_paths)
            self.assertTrue(all(not path.exists() for path in override_paths))
            self.assertFalse((fixture["compose"].parent / ".model-probe.lock").exists())

    def test_cleanup_failure_does_not_replace_original_probe_exception(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("embedding-model") as fixture:
            runner = FakeRunner(
                [
                    (
                        0,
                        rendered_config(
                            "embedding-model",
                            checksum=fixture["checksum"],
                            candidate_path=fixture["artifact"],
                            candidate_name=fixture["entry"]["name"],
                        ),
                    ),
                    (0, ""),
                    RuntimeError("ORIGINAL probe failure"),
                    RuntimeError("cleanup failure"),
                ]
            )
            with self.assertRaisesRegex(RuntimeError, "ORIGINAL probe failure"):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )

    def test_override_cleanup_failure_does_not_replace_original_probe_exception(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        original_unlink = Path.unlink

        def selective_unlink(path: Path, missing_ok: bool = False) -> None:
            if path.name.endswith(".override.json"):
                raise OSError("override cleanup failure")
            original_unlink(path, missing_ok=missing_ok)

        with self._probe_files("embedding-model") as fixture:
            runner = FakeRunner(
                [
                    (
                        0,
                        rendered_config(
                            "embedding-model",
                            checksum=fixture["checksum"],
                            candidate_path=fixture["artifact"],
                            candidate_name=fixture["entry"]["name"],
                        ),
                    ),
                    (0, ""),
                    RuntimeError("ORIGINAL probe failure"),
                    (0, ""),
                ]
            )
            with patch.object(Path, "unlink", selective_unlink):
                with self.assertRaisesRegex(
                    RuntimeError, "ORIGINAL probe failure"
                ) as caught:
                    run_model_probe(
                        compose_file=fixture["compose"],
                        embedding_service="embedding-service",
                        llama_service="llama",
                        discovery_label="discovery",
                        candidate_entry=fixture["entry"],
                        artifact_root=fixture["root"],
                        report_path=fixture["report"],
                        runner=runner,
                    )
            notes = getattr(caught.exception, "__notes__", [])
            self.assertTrue(any("override cleanup" in note for note in notes))

    def test_metadata_checksum_mismatch_and_public_network_attempt_fail_closed(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        for metadata, expected_failure in (
            (self._metadata("b" * 64), "metadata_checksum"),
            (
                {**self._metadata("a" * 64), "publicNetworkAttempted": True},
                "public_network_attempt",
            ),
        ):
            with self.subTest(expected_failure=expected_failure), self._probe_files(
                "embedding-model", forced_checksum="a" * 64
            ) as fixture:
                runner = FakeRunner(
                    probe_outcomes(
                        "embedding-model",
                        fixture["checksum"],
                        fixture["artifact"],
                        embedding_metrics(),
                        metadata=metadata,
                    )
                )
                result = run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                    artifact_hasher=lambda _path: "a" * 64,
                )
                self.assertFalse(result.passed)
                self.assertIn(expected_failure, result.failures)

    def test_metadata_identity_must_match_the_locked_candidate(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("embedding-model") as fixture:
            metadata = self._metadata(fixture["checksum"])
            metadata["name"] = "different-candidate"
            runner = FakeRunner(
                probe_outcomes(
                    "embedding-model",
                    fixture["checksum"],
                    fixture["artifact"],
                    embedding_metrics(),
                    metadata=metadata,
                )
            )
            result = run_model_probe(
                compose_file=fixture["compose"],
                embedding_service="embedding-service",
                llama_service="llama",
                discovery_label="discovery",
                candidate_entry=fixture["entry"],
                artifact_root=fixture["root"],
                report_path=fixture["report"],
                runner=runner,
            )
            self.assertFalse(result.passed)
            self.assertIn("metadata_identity", result.failures)

    def test_reranker_that_misses_1500ms_is_recorded_disabled(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        metrics = {
            "family": "BGE",
            "top20To10P50Ms": 900,
            "top20To10P95Ms": 1600,
            "residentMemoryBytes": 900_000_000,
            "modelChecksum": None,
            "busyTimeoutRate": 0.03,
        }
        with self._probe_files("reranker-model") as fixture:
            metrics["modelChecksum"] = fixture["checksum"]
            runner = FakeRunner(
                probe_outcomes(
                    "reranker-model", fixture["checksum"], fixture["artifact"], metrics
                )
            )
            result = run_model_probe(
                compose_file=fixture["compose"],
                embedding_service="embedding-service",
                llama_service="llama",
                discovery_label="discovery",
                candidate_entry=fixture["entry"],
                artifact_root=fixture["root"],
                report_path=fixture["report"],
                runner=runner,
            )
            payload = json.loads(fixture["report"].read_text(encoding="utf-8"))
            self.assertTrue(result.passed)
            self.assertEqual(payload["reranker"]["status"], "disabled")
            self.assertEqual(
                payload["reranker"]["disabledReason"],
                "top20_to10_p95_ms_exceeds_1500",
            )

    def test_generation_probe_uses_fixed_contexts_output_cap_and_preserves_metrics(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("generation-model") as fixture:
            raw_metrics = generation_metrics()
            runner = FakeRunner(
                probe_outcomes(
                    "generation-model",
                    fixture["checksum"],
                    fixture["artifact"],
                    raw_metrics,
                )
            )
            result = run_model_probe(
                compose_file=fixture["compose"],
                embedding_service="embedding-service",
                llama_service="llama",
                discovery_label="discovery",
                candidate_entry=fixture["entry"],
                artifact_root=fixture["root"],
                report_path=fixture["report"],
                runner=runner,
            )
            benchmark = runner.calls[3][0]
            payload = json.loads(fixture["report"].read_text(encoding="utf-8"))
            self.assertFalse(result.passed)
            self.assertIn("public_network_audit_unavailable", result.failures)
            self.assertIn("1,5,15", benchmark)
            self.assertIn("512,1024,2048", benchmark)
            self.assertIn("256", benchmark)
            self.assertEqual(payload["rawMetrics"], raw_metrics)
            self.assertEqual(payload["gateMetrics"]["queue_feedback_p95_ms"], 700)
            self.assertEqual(payload["gateMetrics"]["first_token_p95_ms"], 3300)

    def test_missing_required_metric_and_nonzero_command_prevent_pass(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        cases = []
        missing = embedding_metrics()
        del missing["residentMemoryBytes"]
        cases.append((missing, 0, "resident_memory_bytes"))
        cases.append((embedding_metrics(), 9, "command:benchmark"))
        for metrics, exit_code, expected in cases:
            with self.subTest(expected=expected), self._probe_files("embedding-model") as fixture:
                runner = FakeRunner(
                    probe_outcomes(
                        "embedding-model",
                        fixture["checksum"],
                        fixture["artifact"],
                        metrics,
                        benchmark_exit_code=exit_code,
                    )
                )
                result = run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=runner,
                )
                self.assertFalse(result.passed)
                self.assertIn(expected, result.failures)

    def test_report_is_deterministic_atomic_and_contains_audit_fields(self) -> None:
        from tools.benchmarks.model_probe import run_model_probe

        with self._probe_files("embedding-model") as fixture:
            outcomes = probe_outcomes(
                "embedding-model",
                fixture["checksum"],
                fixture["artifact"],
                embedding_metrics(),
            )
            for _ in range(2):
                run_model_probe(
                    compose_file=fixture["compose"],
                    embedding_service="embedding-service",
                    llama_service="llama",
                    discovery_label="discovery",
                    candidate_entry=fixture["entry"],
                    artifact_root=fixture["root"],
                    report_path=fixture["report"],
                    runner=FakeRunner(list(outcomes)),
                )
                current = fixture["report"].read_bytes()
                if "first" not in fixture:
                    fixture["first"] = current
                else:
                    self.assertEqual(current, fixture["first"])
            payload = json.loads(fixture["report"].read_text(encoding="utf-8"))
            self.assertEqual(
                set(payload),
                {
                    "candidate",
                    "commandExitCodes",
                    "discoveryLabel",
                    "gate",
                    "gateMetrics",
                    "gateResult",
                    "metadata",
                    "offlineOnly",
                    "networkPolicy",
                    "probeMatrix",
                    "rawMetrics",
                    "reranker",
                },
            )
            self.assertEqual(payload["candidate"]["sha256"], fixture["checksum"])
            self.assertEqual(payload["commandExitCodes"], {
                "benchmark": 0,
                "config": 0,
                "down": 0,
                "metadata": 0,
                "up": 0,
            })
            self.assertTrue(payload["gateResult"]["passed"])
            self.assertEqual(list(fixture["report"].parent.glob("*.tmp")), [])

    def test_cli_uses_a_deterministic_default_report_path(self) -> None:
        from tools.benchmarks import model_probe
        from tools.benchmarks.report import CapacityResult

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            compose = root / "compose.yaml"
            compose.write_text("services: {}\n", encoding="utf-8")
            lock = root / "candidate-lock.json"
            lock.write_text(
                json.dumps(
                    {
                        "artifacts": [
                            {
                                "name": "candidate",
                                "kind": "embedding-model",
                                "version": "1",
                                "sha256": "a" * 64,
                                "license": "approved",
                                "localPath": "candidate.bin",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                model_probe,
                "run_model_probe",
                return_value=CapacityResult(True, []),
            ) as run:
                exit_code = model_probe.main(
                    [
                        "--compose",
                        str(compose),
                        "--embedding-service",
                        "embedding-service",
                        "--llama-service",
                        "llama",
                        "--candidate-lock",
                        str(lock),
                        "--label",
                        "discovery",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(
                run.call_args.kwargs["report_path"],
                Path("artifacts/benchmarks/model-probe-report.json"),
            )

    @staticmethod
    def _metadata(checksum: str) -> dict[str, object]:
        return {
            "name": "bge-small",
            "version": "1",
            "modelChecksum": checksum,
            "publicNetworkAttempted": False,
        }

    class _ProbeFiles:
        def __init__(self, kind: str, forced_checksum: str | None):
            self.temporary = tempfile.TemporaryDirectory()
            self.root = Path(self.temporary.name)
            self.compose = self.root / "compose.yaml"
            self.compose.write_text("services: {}\n", encoding="utf-8")
            (self.root / ".env").write_text("MODEL_ROOT=/models\n", encoding="utf-8")
            if kind == "generation-model":
                self.artifact = self.root / "candidate.gguf"
                write_minimal_gguf(self.artifact)
                actual = hashlib.sha256(self.artifact.read_bytes()).hexdigest()
            else:
                from tools.benchmarks.model_probe import sha256_artifact

                self.artifact = self.root / "candidate-model"
                self.artifact.mkdir()
                (self.artifact / "model.bin").write_bytes(b"offline-candidate")
                (self.artifact / "embedding-metadata.json").write_text(
                    json.dumps(
                        {
                            "modelName": "bge-small" if kind == "embedding-model" else "local-candidate",
                            "modelVersion": "1",
                            "dimensions": 8,
                        }
                    ),
                    encoding="utf-8",
                )
                actual = sha256_artifact(self.artifact)
            self.checksum = forced_checksum or actual
            candidate_name = "bge-small" if kind == "embedding-model" else "local-candidate"
            self.values = {
                "root": self.root,
                "artifact": self.artifact,
                "compose": self.compose,
                "report": self.root / "reports" / "probe.json",
                "checksum": self.checksum,
                "entry": {
                    "name": candidate_name,
                    "kind": kind,
                    "version": "1",
                    "sha256": self.checksum,
                    "license": "approved",
                    "localPath": self.artifact.name,
                },
            }

        def __enter__(self):
            return self.values

        def __exit__(self, exc_type, exc, traceback):
            self.temporary.cleanup()

    def _probe_files(self, kind: str, forced_checksum: str | None = None):
        return self._ProbeFiles(kind, forced_checksum)


if __name__ == "__main__":
    unittest.main()
