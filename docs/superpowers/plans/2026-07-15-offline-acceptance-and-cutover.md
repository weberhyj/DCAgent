# Offline Acceptance and Cutover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce machine-verifiable capacity, recovery, security, quality, rollout, and rollback evidence before switching DC-Agent away from legacy full scans.

**Architecture:** Tooling tasks create deterministic runners and JSON reports; separate target-host execution tasks run long benchmarks and restore drills. The 32GB and 64GB profiles never share a pass result. Shadow and cohort promotion use explicit machine-readable gates, and legacy cleanup is blocked until a successful rollback drill, a 14-day retention window, and a verified pre-drop backup.

**Tech Stack:** Docker Compose, Locust, PostgreSQL base backup/WAL archive, ClickHouse native backup, Qdrant snapshots, PowerShell, Python report gates, FastAPI integration tests, Vitest.

---

Small constructors such as `report()`, `comparison()`, `config_file()`, and `fixture_with_overlapping_allowed_and_forbidden_text()` are defined in the test module that uses them or imported from the Phase 2/3 offline test-support module; they are not hidden implementation dependencies.

### Task 1: Implement complete backup, WAL, and restore contracts

**Files:**
- Create: `deploy/offline/postgresql.conf`
- Modify: `deploy/offline/compose.yaml`
- Modify: `deploy/offline/.env.example`
- Create: `tools/backup_offline.ps1`
- Create: `tools/restore_offline.ps1`
- Create: `tools/backup_report.schema.json`
- Create: `tools/tests/test_backup_contract.py`
- Create: `docs/offline-backup-runbook.md`

- [ ] **Step 1: Write failing static and report-schema tests**

```python
class BackupContractTest(unittest.TestCase):
    def test_backup_covers_every_plane_and_wal(self) -> None:
        backup = Path("tools/backup_offline.ps1").read_text(encoding="utf-8")
        restore = Path("tools/restore_offline.ps1").read_text(encoding="utf-8")
        for token in ("pg_basebackup", "WAL_ARCHIVE_ROOT", "recoveryPointLsn", "clickhouse", "qdrant", "RAW_DATA_ROOT", "SHA256"):
            self.assertIn(token, backup)
        for token in ("isolated", "rowCount", "pointCount", "permissionChecks", "representativeQueries"):
            self.assertIn(token, restore)

    def test_postgres_archives_wal_to_separate_mount(self) -> None:
        text = Path("deploy/offline/postgresql.conf").read_text(encoding="utf-8")
        self.assertIn("archive_mode = on", text)
        self.assertIn("archive_command", text)
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_backup_contract -v
```

Expected: FAIL because scripts, WAL configuration, and report schema do not exist.

- [ ] **Step 3: Implement safe backup and restore scripts**

`postgresql.conf` enables WAL archiving to `${WAL_ARCHIVE_ROOT}`, mounted from a separate disk/NAS. `backup_offline.ps1 -Destination -ManifestPath -KeepDays -DryRun` resolves every path, rejects the live-data volume, runs `pg_basebackup --wal-method=stream`, captures the WAL range plus the latest application recovery marker/LSN, backs up raw/Parquet, ClickHouse published tables/dictionaries, Qdrant published collections, configuration, Alembic revisions, models, tokenizer/vocabulary/DF/token-count artifacts, and atomically writes the requested SHA-256 manifest path. It never deletes before validating destination and retention scope.

`restore_offline.ps1 -Manifest -Target` restores only to an isolated target, validates checksums, starts isolated services, restores PostgreSQL/WAL, ClickHouse, Qdrant, raw/Parquet/config/model artifacts in dependency order, and writes a JSON report. It never overwrites the live root. The runbook defines daily backups, 24-hour RPO, four-hour RTO, encryption, quarterly drills, and missing-artifact pointer recovery.

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tools.tests.test_backup_contract -v
git add deploy/offline/postgresql.conf deploy/offline/compose.yaml deploy/offline/.env.example tools/backup_offline.ps1 tools/restore_offline.ps1 tools/backup_report.schema.json tools/tests/test_backup_contract.py docs/offline-backup-runbook.md
git commit -m "ops: add complete offline backup contracts"
```

### Task 2: Implement and execute a timed isolated restore drill

**Files:**
- Create: `tools/run_restore_drill.py`
- Create: `tools/tests/test_restore_drill.py`
- Modify: `docs/offline-backup-runbook.md`

- [ ] **Step 1: Write failing RPO/RTO gate tests**

```python
class RestoreDrillTest(unittest.TestCase):
    def test_fails_when_backup_is_old_or_restore_exceeds_four_hours(self) -> None:
        result = evaluate_restore_report({
            "backupAgeSeconds": 90000,
            "recoveryPointAgeSeconds": 90000,
            "restoreElapsedSeconds": 15000,
            "checksumsValid": True,
            "rowCountsMatch": True,
            "pointCountsMatch": True,
            "permissionChecksPassed": True,
            "representativeQueriesPassed": True,
        })
        self.assertFalse(result.passed)
        self.assertEqual(set(result.failures), {"rpo", "rto"})
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_restore_drill -v
```

Expected: FAIL because the drill runner and evaluator do not exist.

- [ ] **Step 3: Implement the runner and execute on the target host**

The runner accepts an existing scheduled backup manifest and its WAL archive; it does not create a fresh backup immediately before measuring RPO. It verifies WAL continuity to the recorded recovery LSN/application marker, deliberately verifies that a damaged checksum fails, restores to an isolated Compose project/root, measures elapsed time, checks recovery-point age, row/point counts, active manifest, permission matrix, and representative queries, then destroys only the isolated project. It writes `artifacts/acceptance/restore-drill-<timestamp>.json` and exits nonzero if any required field is missing, recovery-point age exceeds 86,400 seconds, or RTO exceeds 14,400 seconds.

```powershell
$profile = $env:OFFLINE_ACCEPTED_PROFILE
if ($profile -notin @("32gb", "64gb")) { throw "Set OFFLINE_ACCEPTED_PROFILE to 32gb or 64gb" }
$backup = $env:OFFLINE_BACKUP_MANIFEST
if (-not (Test-Path -LiteralPath $backup)) { throw "Set OFFLINE_BACKUP_MANIFEST to a scheduled backup manifest" }
py tools/run_restore_drill.py --compose deploy/offline/compose.yaml --profile $profile --backup-manifest $backup --report-root artifacts/acceptance
```

Expected: a schema-valid report with `passed=true`; otherwise rollout remains blocked.

- [ ] **Step 4: Commit runner code, not generated reports**

```powershell
py -m unittest tools.tests.test_restore_drill -v
git add tools/run_restore_drill.py tools/tests/test_restore_drill.py docs/offline-backup-runbook.md
git commit -m "test: add timed isolated restore drill"
```

### Task 3: Add reproducible profile and batch-window benchmark modes

**Files:**
- Modify: `tools/benchmarks/manifests/acceptance-30m-5m.json`
- Create: `tools/benchmarks/manifests/acceptance-30m-5m-32gb.json`
- Create: `tools/benchmarks/manifests/acceptance-30m-5m-64gb.json`
- Create: `tools/benchmarks/profiles/32gb.json`
- Create: `tools/benchmarks/profiles/64gb.json`
- Create: `tools/benchmarks/build_acceptance_fixtures.py`
- Create: `tools/benchmarks/fixtures/acceptance-initial.json`
- Create: `tools/benchmarks/fixtures/acceptance-daily-5-percent.json`
- Create: `tools/benchmarks/fixtures/acceptance-weekly-full.json`
- Create: `deploy/offline/compose.32gb.yaml`
- Create: `deploy/offline/compose.64gb.yaml`
- Modify: `tools/benchmarks/run_capacity_benchmark.py`
- Modify: `tools/benchmarks/locustfile.py`
- Modify: `tools/benchmarks/model_probe.py`
- Create: `tools/check_artifacts.py`
- Create: `tools/benchmarks/compare_profiles.py`
- Create: `tools/tests/test_capacity_profiles.py`

- [ ] **Step 1: Write failing profile/batch gate tests**

```python
class CapacityProfilesTest(unittest.TestCase):
    def test_profiles_are_exact_and_cannot_inherit_pass(self) -> None:
        profile32 = load_profile("tools/benchmarks/profiles/32gb.json")
        profile64 = load_profile("tools/benchmarks/profiles/64gb.json")
        self.assertEqual(profile32.model_slots, 1)
        self.assertEqual(profile64.model_slots, 2)
        self.assertEqual(load_locked_manifest(profile32.manifest_path).vector_dimensions, profile32.vector_dimensions)
        self.assertEqual(load_locked_manifest(profile64.manifest_path).model_slots, profile64.model_slots)
        self.assertEqual(profile32.gates["online-warm"]["queue_feedback_p95_ms"], ("lte", 2000))
        self.assertEqual(profile64.gates["batch-daily"]["online_structured_p95_ms"], ("lte", 5000))
        result = compare_profiles(
            report("32gb", passed=False, manifest_sha="a"),
            report("64gb", passed=True, manifest_sha="a"),
        )
        self.assertFalse(result["32gb"]["passed"])
        self.assertTrue(result["64gb"]["passed"])

    def test_batch_mode_requires_online_load_metrics(self) -> None:
        with self.assertRaises(ValueError):
            validate_batch_report({"batchMode": "daily-5-percent", "onlineMetrics": None})

    def test_artifact_check_fails_on_missing_local_file(self) -> None:
        with self.assertRaises(ArtifactCheckError):
            check_artifact_lock({"artifacts": [{"localPath": "missing.bin", "sha256": "a" * 64}]})

    def test_batch_fixture_manifests_are_complete_and_deterministic(self) -> None:
        initial = load_fixture_manifest("tools/benchmarks/fixtures/acceptance-initial.json")
        daily = load_fixture_manifest("tools/benchmarks/fixtures/acceptance-daily-5-percent.json")
        weekly = load_fixture_manifest("tools/benchmarks/fixtures/acceptance-weekly-full.json")
        self.assertEqual((initial.structured_rows, initial.semantic_points), (30_000_000, 5_000_000))
        self.assertEqual(initial.changes["structured"]["new"], 30_000_000)
        self.assertLessEqual(daily.changed_fraction, 0.05)
        self.assertTrue(weekly.force_full_rebuild)
        self.assertTrue(all(len(item.source_sha256) == 64 for item in (initial, daily, weekly)))
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_capacity_profiles -v
```

Expected: FAIL because fixed profiles, `--batch`, and concurrent online-load validation do not exist.

- [ ] **Step 3: Implement exact profiles and runner flags**

`check_artifacts.py` validates every local path, SHA-256, license, and model/container digest in `deploy/offline/artifacts.lock.json` and exits nonzero before a benchmark starts. `compose.32gb.yaml` and `compose.64gb.yaml` are explicit Docker Compose overrides that apply the profile's `mem_limit`, `cpus`, and `blkio_config` to PostgreSQL, ClickHouse, Qdrant, Redis, ClamAV, Embedding, API, worker, and llama.cpp; the runner always includes the matching override, so declared budgets are enforced rather than merely documented. `32gb.json` fixes `vector_dimensions`, `model_slots=1`, `manifest_path=tools/benchmarks/manifests/acceptance-30m-5m-32gb.json`, Qdrant on-disk vectors, validated quantization, and these initial memory ceilings in GiB: ClickHouse 8, Qdrant 6, llama.cpp 4, Embedding 3, PostgreSQL 1.5, Redis 0.5, API 1, worker 2, ClamAV 1, leaving at least 5 for the OS/safety margin. `64gb.json` fixes its own selected `vector_dimensions`, `model_slots=2`, `manifest_path=tools/benchmarks/manifests/acceptance-30m-5m-64gb.json`, with ClickHouse 18, Qdrant 10, llama.cpp 10, Embedding 4, PostgreSQL 2, Redis 1, API 2, worker 4, ClamAV 1, leaving at least 12 for OS/safety. CPU and I/O quotas prevent batch work from starving online services. Both profile manifests retain the same 30M/5M scale but have their own model checksum, exact gate profiles, and report directory. Extend the runner with:

Each profile file must contain explicit values rather than prose:

```json
{
  "profile": "32gb",
  "vector_dimensions": 512,
  "model_slots": 1,
  "manifest_path": "tools/benchmarks/manifests/acceptance-30m-5m-32gb.json",
  "memory_gib": {"clickhouse": 8, "qdrant": 6, "llama": 4, "embedding": 3, "postgres": 1.5, "redis": 0.5, "api": 1, "worker": 2, "clamav": 1},
  "gates": {
    "online-cold": {"structured_p95_ms": ["lte", 5000], "document_p95_ms": ["lte", 5000], "mixed_p95_ms": ["lte", 5000], "queue_feedback_p95_ms": ["lte", 2000], "error_rate": ["lte", 0.01]},
    "online-warm": {"structured_p95_ms": ["lte", 5000], "document_p95_ms": ["lte", 5000], "mixed_p95_ms": ["lte", 5000], "queue_feedback_p95_ms": ["lte", 2000], "warm_cache_hit_rate": ["gte", 0.20], "first_token_p95_ms": ["lte", 10000], "error_rate": ["lte", 0.01]},
    "batch-initial": {"elapsed_seconds": ["lte", 172800], "online_structured_p95_ms": ["lte", 5000], "online_document_p95_ms": ["lte", 5000], "online_mixed_p95_ms": ["lte", 5000], "online_queue_feedback_p95_ms": ["lte", 2000], "online_error_rate": ["lte", 0.01]},
    "batch-daily": {"elapsed_seconds": ["lte", 28800], "online_structured_p95_ms": ["lte", 5000], "online_document_p95_ms": ["lte", 5000], "online_mixed_p95_ms": ["lte", 5000], "online_queue_feedback_p95_ms": ["lte", 2000], "online_error_rate": ["lte", 0.01], "changed_fraction": ["lte", 0.05]},
    "batch-weekly": {"elapsed_seconds": ["lte", 129600], "online_structured_p95_ms": ["lte", 5000], "online_document_p95_ms": ["lte", 5000], "online_mixed_p95_ms": ["lte", 5000], "online_queue_feedback_p95_ms": ["lte", 2000], "online_error_rate": ["lte", 0.01]}
  }
}
```

The 64GB file is equally explicit; no report may borrow the other profile's gates:

```json
{
  "profile": "64gb",
  "vector_dimensions": 768,
  "model_slots": 2,
  "manifest_path": "tools/benchmarks/manifests/acceptance-30m-5m-64gb.json",
  "memory_gib": {"clickhouse": 18, "qdrant": 10, "llama": 10, "embedding": 4, "postgres": 2, "redis": 1, "api": 2, "worker": 4, "clamav": 1},
  "gates": {
    "online-cold": {"structured_p95_ms": ["lte", 5000], "document_p95_ms": ["lte", 5000], "mixed_p95_ms": ["lte", 5000], "queue_feedback_p95_ms": ["lte", 2000], "error_rate": ["lte", 0.01]},
    "online-warm": {"structured_p95_ms": ["lte", 5000], "document_p95_ms": ["lte", 5000], "mixed_p95_ms": ["lte", 5000], "queue_feedback_p95_ms": ["lte", 2000], "warm_cache_hit_rate": ["gte", 0.20], "first_token_p95_ms": ["lte", 10000], "error_rate": ["lte", 0.01]},
    "batch-initial": {"elapsed_seconds": ["lte", 172800], "online_structured_p95_ms": ["lte", 5000], "online_document_p95_ms": ["lte", 5000], "online_mixed_p95_ms": ["lte", 5000], "online_queue_feedback_p95_ms": ["lte", 2000], "online_error_rate": ["lte", 0.01]},
    "batch-daily": {"elapsed_seconds": ["lte", 28800], "online_structured_p95_ms": ["lte", 5000], "online_document_p95_ms": ["lte", 5000], "online_mixed_p95_ms": ["lte", 5000], "online_queue_feedback_p95_ms": ["lte", 2000], "online_error_rate": ["lte", 0.01], "changed_fraction": ["lte", 0.05]},
    "batch-weekly": {"elapsed_seconds": ["lte", 129600], "online_structured_p95_ms": ["lte", 5000], "online_document_p95_ms": ["lte", 5000], "online_mixed_p95_ms": ["lte", 5000], "online_queue_feedback_p95_ms": ["lte", 2000], "online_error_rate": ["lte", 0.01]}
  }
}
```

`compare_profiles.py` defines `LockedBenchmarkManifest` by combining the Phase 1 template with the selected model name/version/SHA, exact vector dimension, exact model slots, Compose override checksum, and mode-specific `MetricGate` tuples. `load_locked_manifest()` rejects a candidate dimension not listed by the base template.

```text
--cache cold|warm
--batch none|initial|daily-5-percent|weekly-full
--with-online-load
--profile 32gb|64gb
```

Batch modes require an input fixture manifest containing generator version/seed, canonical source SHA-256, expected structured rows, semantic points, and expected changed/new/deleted counts. `build_acceptance_fixtures.py` deterministically writes the three checked-in manifests and `--check` fails if regeneration differs. `acceptance-initial.json` builds the complete 30M/5M snapshot; `acceptance-daily-5-percent.json` changes exactly 1.5M structured rows and 250k semantic records (5%) with stable IDs and verifies dense-cache reuse; `acceptance-weekly-full.json` keeps the same logical counts but sets `force_full_rebuild=true`. The runner verifies all DAG jobs completed, no quarantined publication dependency exists, row/point counts match, and the PostgreSQL pointer advances exactly once to the tested batch. Initial <=48h, daily <=8h, and weekly <=36h are numeric gates. `--with-online-load` keeps the 15-user 40/40/20 workload running during ingestion and fails if any online gate misses. `compare_profiles` validates profile name, hardware, manifest checksum, 30-minute duration, every mode-specific metric gate, command exits, and never copies a pass between profiles.

The fixture builder is fully deterministic and stores no 30M-row data in Git:

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

FIXTURES = {
    "acceptance-initial.json": {
        "fixture_id": "acceptance-initial-v1",
        "generator_version": "2026-07-15",
        "seed": 20260715,
        "structured_rows": 30_000_000,
        "semantic_points": 5_000_000,
        "changes": {
            "structured": {"new": 30_000_000, "changed": 0, "deleted": 0},
            "semantic": {"new": 5_000_000, "changed": 0, "deleted": 0},
        },
        "changed_fraction": 0.0,
        "force_full_rebuild": False,
    },
    "acceptance-daily-5-percent.json": {
        "fixture_id": "acceptance-daily-5-percent-v1",
        "generator_version": "2026-07-15",
        "seed": 20260716,
        "structured_rows": 30_000_000,
        "semantic_points": 5_000_000,
        "changes": {
            "structured": {"new": 0, "changed": 1_500_000, "deleted": 0},
            "semantic": {"new": 0, "changed": 250_000, "deleted": 0},
        },
        "changed_fraction": 0.05,
        "force_full_rebuild": False,
    },
    "acceptance-weekly-full.json": {
        "fixture_id": "acceptance-weekly-full-v1",
        "generator_version": "2026-07-15",
        "seed": 20260722,
        "structured_rows": 30_000_000,
        "semantic_points": 5_000_000,
        "changes": {
            "structured": {"new": 0, "changed": 0, "deleted": 0},
            "semantic": {"new": 0, "changed": 0, "deleted": 0},
        },
        "changed_fraction": 0.0,
        "force_full_rebuild": True,
    },
}

def render(payload: dict[str, object]) -> dict[str, object]:
    source = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {"schema_version": 1, **payload, "source_sha256": hashlib.sha256(source).hexdigest()}

def main(check: bool = False) -> None:
    root = Path("tools/benchmarks/fixtures")
    for filename, payload in FIXTURES.items():
        target = root / filename
        rendered = json.dumps(render(payload), sort_keys=True, indent=2) + "\n"
        if check and target.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"fixture differs: {target}")
        if not check:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")

if __name__ == "__main__":
    main(check="--check" in sys.argv[1:])
```

- [ ] **Step 4: Run tests and commit**

```powershell
py -m unittest tools.tests.test_capacity_profiles -v
py tools/benchmarks/build_acceptance_fixtures.py --check
git add tools/benchmarks/manifests/acceptance-30m-5m.json tools/benchmarks/manifests/acceptance-30m-5m-32gb.json tools/benchmarks/manifests/acceptance-30m-5m-64gb.json tools/benchmarks/profiles/32gb.json tools/benchmarks/profiles/64gb.json tools/benchmarks/build_acceptance_fixtures.py tools/benchmarks/fixtures/acceptance-initial.json tools/benchmarks/fixtures/acceptance-daily-5-percent.json tools/benchmarks/fixtures/acceptance-weekly-full.json deploy/offline/compose.32gb.yaml deploy/offline/compose.64gb.yaml tools/benchmarks/run_capacity_benchmark.py tools/benchmarks/locustfile.py tools/benchmarks/model_probe.py tools/check_artifacts.py tools/benchmarks/compare_profiles.py tools/tests/test_capacity_profiles.py
git commit -m "test: add reproducible capacity and batch profiles"
```

### Task 4: Execute separate 32GB and 64GB acceptance runs

**Files:**
- Modify: `docs/offline-platform-runbook.md`

- [ ] **Step 1: Verify target-host preconditions**

```powershell
py tools/compose_smoke.py
py tools/check_artifacts.py --manifest deploy/offline/artifacts.lock.json
py tools/benchmarks/model_probe.py --compose deploy/offline/compose.yaml --compose-override deploy/offline/compose.32gb.yaml --candidate-lock deploy/offline/artifacts.lock.json --locked-manifest tools/benchmarks/manifests/acceptance-30m-5m-32gb.json --profile 32gb
py tools/benchmarks/model_probe.py --compose deploy/offline/compose.yaml --compose-override deploy/offline/compose.64gb.yaml --candidate-lock deploy/offline/artifacts.lock.json --locked-manifest tools/benchmarks/manifests/acceptance-30m-5m-64gb.json --profile 64gb
```

Expected: every command exits 0. The two model probes run under their actual Compose CPU/memory/I/O overrides, record CPU/RAM/NVMe/software/model checksums separately, and cannot reuse the Phase 1 discovery label as a profile pass.

- [ ] **Step 2: Run cold and warm online workloads**

```powershell
py tools/benchmarks/run_capacity_benchmark.py --compose-override deploy/offline/compose.32gb.yaml --manifest tools/benchmarks/manifests/acceptance-30m-5m-32gb.json --profile 32gb --cache cold --batch none
py tools/benchmarks/run_capacity_benchmark.py --compose-override deploy/offline/compose.32gb.yaml --manifest tools/benchmarks/manifests/acceptance-30m-5m-32gb.json --profile 32gb --cache warm --batch none
py tools/benchmarks/run_capacity_benchmark.py --compose-override deploy/offline/compose.64gb.yaml --manifest tools/benchmarks/manifests/acceptance-30m-5m-64gb.json --profile 64gb --cache cold --batch none
py tools/benchmarks/run_capacity_benchmark.py --compose-override deploy/offline/compose.64gb.yaml --manifest tools/benchmarks/manifests/acceptance-30m-5m-64gb.json --profile 64gb --cache warm --batch none
```

Expected: separate report directories; 15 users, 1,800 seconds, required p50/p95/error/cache/model metrics, and no inherited result.

- [ ] **Step 3: Run batch windows with concurrent online load**

```powershell
$profile = $env:OFFLINE_ACCEPTED_PROFILE
if ($profile -notin @("32gb", "64gb")) { throw "Set OFFLINE_ACCEPTED_PROFILE to 32gb or 64gb" }
$manifest = if ($profile -eq "32gb") { "tools/benchmarks/manifests/acceptance-30m-5m-32gb.json" } else { "tools/benchmarks/manifests/acceptance-30m-5m-64gb.json" }
$override = if ($profile -eq "32gb") { "deploy/offline/compose.32gb.yaml" } else { "deploy/offline/compose.64gb.yaml" }
py tools/benchmarks/run_capacity_benchmark.py --compose-override $override --manifest $manifest --profile $profile --batch initial --with-online-load --input tools/benchmarks/fixtures/acceptance-initial.json
py tools/benchmarks/run_capacity_benchmark.py --compose-override $override --manifest $manifest --profile $profile --batch daily-5-percent --with-online-load --input tools/benchmarks/fixtures/acceptance-daily-5-percent.json
py tools/benchmarks/run_capacity_benchmark.py --compose-override $override --manifest $manifest --profile $profile --batch weekly-full --with-online-load --input tools/benchmarks/fixtures/acceptance-weekly-full.json
```

Two consecutive window misses block rollout and force one recorded architecture decision: semantic aggregation, base-plus-delta Qdrant, 64GB, or a second indexing node.

- [ ] **Step 4: Record report locations**

Add the report paths and pass/fail summary to `docs/offline-platform-runbook.md`; generated benchmark JSON remains outside Git.

```powershell
git add docs/offline-platform-runbook.md
git commit -m "docs: record offline capacity evidence"
```

### Task 5: Exercise permission, provenance, and component failures

**Files:**
- Create: `backend/tests/integration/test_offline_security.py`
- Create: `backend/tests/integration/test_offline_failures.py`
- Create: `tools/failure_matrix.json`
- Create: `docs/offline-failure-runbook.md`

- [ ] **Step 1: Write failing permission and prompt-isolation tests**

```python
class OfflineSecurityTest(unittest.TestCase):
    def test_forbidden_record_never_reaches_hit_aggregation_or_prompt(self) -> None:
        result = run_query_as("tenant-a-finance", fixture_with_overlapping_allowed_and_forbidden_text())
        self.assertNotIn("forbidden-point", result.evidence_ids)
        self.assertNotIn("forbidden-row", result.structured_row_ids)
        self.assertNotIn("forbidden secret", result.model_prompt)
```

- [ ] **Step 2: Verify RED**

```powershell
$env:RUN_OFFLINE_INTEGRATION="1"
Set-Location backend
py -m unittest tests.integration.test_offline_security tests.integration.test_offline_failures -v
Set-Location ..
Remove-Item Env:RUN_OFFLINE_INTEGRATION
```

Expected: FAIL because the new security/failure tests and matrix fixtures do not exist.

- [ ] **Step 3: Add the machine-readable failure matrix**

Cover ClickHouse outage/timeout, Qdrant outage/no evidence, Redis clear during a job, Embedding outage/metadata mismatch, reranker busy/timeout, llama.cpp stop while streaming, disk threshold, corrupt PDF/PPT/XLSB, missing formula cache, mixed batch IDs, stale lease token, and failed publication. Every case declares expected HTTP/event type, retained facts/evidence, pointer state, and recovery command.

- [ ] **Step 4: Run gated integration tests**

```powershell
$env:RUN_OFFLINE_INTEGRATION="1"
Set-Location backend
py -m unittest tests.integration.test_offline_security tests.integration.test_offline_failures -v
Set-Location ..
Remove-Item Env:RUN_OFFLINE_INTEGRATION
```

Expected: no permission leak, no mixed publication, typed degradation, old pointer retained on failed batch, and Redis-loss recovery.

- [ ] **Step 5: Commit**

```powershell
git add backend/tests/integration/test_offline_security.py backend/tests/integration/test_offline_failures.py tools/failure_matrix.json docs/offline-failure-runbook.md
git commit -m "test: verify offline security and failures"
```

### Task 6: Build machine-verifiable shadow comparison gates

**Files:**
- Create: `tools/shadow_compare.py`
- Create: `tools/provision_shadow_audit_reader.py`
- Create: `tools/shadow_report.schema.json`
- Create: `tools/tests/test_shadow_compare.py`
- Create: `tools/evaluation/approved-questions.json`
- Modify: `docs/offline-query-runbook.md`

- [ ] **Step 1: Write failing gate/exit tests**

```python
class ShadowCompareTest(unittest.TestCase):
    def test_numeric_disagreement_or_permission_leak_blocks_promotion(self) -> None:
        report = evaluate_shadow([
            comparison("structured-1", numeric_disagreement=True),
            comparison("document-1", permission_leak=False, recall_at_5=1.0),
        ])
        self.assertFalse(report.passed)
        self.assertIn("numeric_disagreement", report.failures)

    def test_minimum_sample_and_category_coverage_are_required(self) -> None:
        report = evaluate_shadow([comparison("structured-1")] * 10)
        self.assertFalse(report.passed)
        self.assertIn("minimum_sample_count", report.failures)

    def test_reader_provisioning_grants_only_shadow_audit_capability(self) -> None:
        repository = FakeAuthorizationAdminRepository()
        provision_shadow_audit_reader(repository, principal="acceptance-runner")
        self.assertEqual(repository.permissions_for("acceptance-runner"), {"shadow_audit:read"})
        self.assertEqual(repository.query_scopes_for("acceptance-runner"), set())
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_shadow_compare -v
```

Expected: FAIL because schema, evaluator, and nonzero gate exit do not exist.

- [ ] **Step 3: Implement synthetic and protected production-set inputs**

The checked-in synthetic set includes structured, document, mixed, and no-answer questions. `provision_shadow_audit_reader.py` idempotently creates/enables the named principal, the exact `shadow_audit:read` permission, one dedicated role, and its role assignment through the authorization repository; it grants no tenant/department/classification scope and writes a restrictive JSON auth file with `{"headers":{"X-Identity":"acceptance-runner"}}`, matching the deployed `IDENTITY_HEADER`. That file is used only against the loopback/internal admin listener. A separate production set is supplied through `--production-set <path>` and must contain schema version, dataset version, checksum, access owner, and question records. `shadow_compare.py --api-base ... --audit-auth-file ...` sends requests through the real `ShadowAnswerEngine`, then polls the Phase 3 protected `/api/admin/shadow-comparisons/{request_id}` endpoint with bounded backoff until the offline branch's configured terminal timeout plus five seconds. It fails if the terminal row is still missing or contains a raw question/answer field and never simulates the dual run locally. Reports store hashes/metrics by default, not raw sensitive prompts/evidence.

The schema and evaluator fix these gates: at least 100 questions total and at least 20 each for structured, document, mixed, and no-answer; zero numeric disagreement, permission leaks, and `publication_mismatch`; Recall@5 >=0.85; citation correctness >=0.95; no-answer accuracy >=0.90; grounded-claim rate >=0.95; queue-feedback p95 <=2,000 ms; available-slot first-token p95 <=10,000 ms; and `unexpected_error` rate <=1%. `terminal_status`, `terminal_reason`, `legacy_status`, `offline_status`, and `offline_timeout` are required for every comparison; cancelled rows are reported separately and failed/offline-timeout rows count toward the error gate. `--fail-on-gate` exits 2 on any missing field, insufficient sample/category coverage, or failed threshold. Shadow always returns the legacy answer to users.

- [ ] **Step 4: Run and commit**

```powershell
py -m unittest tools.tests.test_shadow_compare -v
$auditAuth = "artifacts/secrets/shadow-audit-auth.json"
py tools/provision_shadow_audit_reader.py --database-url-file $env:DATABASE_URL_SECRET_FILE --principal acceptance-runner --auth-file $auditAuth
py tools/shadow_compare.py --api-base http://127.0.0.1:8000/api --audit-auth-file $auditAuth --synthetic-set tools/evaluation/approved-questions.json --production-set $env:PRODUCTION_EVALUATION_SET_PATH --engine shadow --fail-on-gate
git add tools/shadow_compare.py tools/provision_shadow_audit_reader.py tools/shadow_report.schema.json tools/tests/test_shadow_compare.py tools/evaluation/approved-questions.json docs/offline-query-runbook.md
git commit -m "test: add machine-verifiable shadow gates"
```

### Task 7: Implement cohort rollout and configuration attestation

**Files:**
- Modify: `.env.example`
- Modify: `backend/.env.example`
- Modify: `backend/app/main.py`
- Modify: `backend/app/routes.py`
- Modify: `backend/tests/test_query_engine_flags.py`
- Create: `tools/check_config.py`
- Create: `tools/tests/test_check_config.py`

- [ ] **Step 1: Write failing principal/department and target-config tests**

```python
class QueryEnginePolicyTest(unittest.TestCase):
    def test_rolls_out_to_principal_then_department(self) -> None:
        policy = QueryEnginePolicy("offline", principals={"u1"}, departments={"d2"}, all_users=False)
        self.assertEqual(policy.select("u1", "d9"), "offline")
        self.assertEqual(policy.select("u9", "d2"), "offline")
        self.assertEqual(policy.select("u9", "d9"), "legacy")
        self.assertEqual(QueryEnginePolicy("offline", set(), set(), all_users=True).select("u9", "d9"), "offline")

class CheckConfigTest(unittest.TestCase):
    def test_checks_explicit_target_files_not_only_current_process(self) -> None:
        self.assertEqual(check_targets([config_file({"QUERY_ENGINE": "offline"})], {"QUERY_ENGINE": "offline"}), [])
```

- [ ] **Step 2: Verify RED**

```powershell
Set-Location backend
py -m unittest tests.test_query_engine_flags -v
Set-Location ..
py -m unittest tools.tests.test_check_config -v
```

Expected: FAIL because cohort policy and target attestation do not exist.

- [ ] **Step 3: Implement staged selection and attestation**

Support `QUERY_ENGINE=legacy|shadow|offline`, `OFFLINE_QUERY_PRINCIPALS`, `OFFLINE_QUERY_DEPARTMENTS`, and explicit `OFFLINE_QUERY_ALL_USERS=true|false`. When all-users is false and both cohort sets are empty, selection remains legacy; all-users true is the only full-promotion meaning. `tools/check_config.py --target <env-file>` accepts repeated target files, parses only requested non-secret keys, writes `artifacts/acceptance/config-attestation-<stage>.json` with target/config hashes, and exits nonzero on missing/mismatch. Roll out in order: legacy → shadow → one principal → one department → all users.

- [ ] **Step 4: Run tests and commit**

```powershell
Set-Location backend
py -m unittest tests.test_query_engine_flags tests.test_api_contract -v
Set-Location ..
py -m unittest tools.tests.test_check_config -v
py tools/check_config.py --target deploy/offline/.env --require QUERY_ENGINE=legacy --require OFFLINE_QUERY_ALL_USERS=false --report artifacts/acceptance/config-attestation-baseline.json
git add .env.example backend/.env.example backend/app/main.py backend/app/routes.py backend/tests/test_query_engine_flags.py tools/check_config.py tools/tests/test_check_config.py
git commit -m "feat: add cohort-aware offline rollout"
```

### Task 8: Execute rollback drill and guard irreversible cleanup

**Files:**
- Create: `tools/run_cutover_drill.py`
- Create: `tools/pre_drop_legacy_check.py`
- Create: `tools/build_release_candidate.py`
- Create: `tools/export_legacy_vectors.py`
- Create: `tools/release_candidate.schema.json`
- Create: `tools/tests/test_cutover_guards.py`
- Create: `tools/tests/test_legacy_export.py`
- Modify: `docs/offline-query-runbook.md`

- [ ] **Step 1: Write failing cleanup-guard tests**

```python
class CutoverGuardTest(unittest.TestCase):
    def test_cleanup_requires_retention_backup_and_rollback_evidence(self) -> None:
        with self.assertRaises(CleanupBlocked):
            verify_cleanup_prerequisites({
                "allOfflineSince": "2026-07-14T00:00:00Z",
                "retentionDays": 14,
                "backupId": "",
                "rollbackDrillPassed": False,
            }, now="2026-07-15T00:00:00Z")

    def test_retention_gate_requires_fourteen_daily_slo_reports_and_same_manifest(self) -> None:
        result = verify_retention_window(
            retention_start="2026-07-01T00:00:00Z",
            now="2026-07-15T00:00:00Z",
            active_manifest_id="manifest-32",
            daily_reports=daily_slo_reports(days=13, manifest_id="manifest-32"),
        )
        self.assertFalse(result.passed)
        self.assertIn("daily_report_coverage", result.failures)

    def test_retention_day_report_is_signed_and_bound_to_profile_manifest_and_config(self) -> None:
        report = record_retention_day(
            clock=fixed_utc_clock("2026-07-02T12:00:00Z"),
            profile="32gb",
            active_manifest_id="manifest-32",
            config_attestation_sha256="d" * 64,
            metrics=passing_online_metrics(),
        )
        self.assertEqual(report["date"], "2026-07-02")
        self.assertEqual(report["observedAt"], "2026-07-02T12:00:00Z")
        self.assertEqual(report["profile"], "32gb")
        self.assertEqual(report["activeManifestId"], "manifest-32")
        self.assertEqual(len(report["signature"]), 64)
        with self.assertRaises(SystemExit):
            parse_cutover_args(["--record-retention-day", "--date", "2026-07-01"])

class LegacyExportTest(unittest.TestCase):
    def test_export_is_streamed_and_records_count_and_sha256(self) -> None:
        report = export_fixture_vectors([{"id": "a", "vector": [1.0]}, {"id": "b", "vector": [2.0]}])
        self.assertEqual(report["rowCount"], 2)
        self.assertEqual(len(report["exportSha256"]), 64)

class ReleaseCandidateTest(unittest.TestCase):
    def test_candidate_selection_requires_explicit_profile_manifest_and_retention(self) -> None:
        backup = fixture_backup_manifest(backup_id="backup-1", manifest_sha256="a" * 64)
        legacy = fixture_legacy_export_report(
            backup_id="backup-1",
            row_count=12,
            export_sha256="b" * 64,
            report_sha256="c" * 64,
        )
        with self.assertRaises(ValueError):
            build_release_candidate(
                report_root=fixture_report_root("32gb"),
                profile=None,
                active_manifest_id=None,
                retention_start=None,
                pre_drop_backup_manifest=backup,
                legacy_export_report=legacy,
            )
        candidate = build_release_candidate(
            report_root=fixture_report_root("32gb"),
            profile="32gb",
            active_manifest_id="manifest-32",
            retention_start="2026-07-01T00:00:00Z",
            pre_drop_backup_manifest=backup,
            legacy_export_report=legacy,
        )
        self.assertEqual(candidate.profile, "32gb")
        self.assertEqual(candidate.active_manifest_id, "manifest-32")
        self.assertEqual(candidate.retention_start, "2026-07-01T00:00:00Z")
        self.assertEqual(candidate.pre_drop_backup.id, "backup-1")
        self.assertEqual(candidate.legacy_export.row_count, 12)
```

- [ ] **Step 2: Verify RED**

```powershell
py -m unittest tools.tests.test_cutover_guards tools.tests.test_legacy_export -v
```

Expected: FAIL because drill, cleanup guard, release builder, and streamed legacy export do not exist.

- [ ] **Step 3: Execute promotion and rollback**

`run_cutover_drill.py` changes only cohort flags, recreates API, runs health/security/representative queries, observes the requested soak window, switches back to legacy unless `--keep-promoted` is given, verifies the same active publication remains usable, and writes a signed JSON report. Every stage requires zero permission/version leaks, <=1% unexpected errors, structured/document p95 <=5 seconds, queue feedback p95 <=2 seconds, available-slot first-token p95 <=10 seconds, and rollback completion <=5 minutes. Minimum traffic is 50 queries for principal, 500 for department, and 2,000 for all-users. After final all-user promotion, retain legacy for 14 successful days. `--record-retention-day` runs the same 15-user 40/40/20 online SLO probe, validates the active profile/manifest and all-users config attestation, derives `observedAt` and `date` only from the system UTC clock, and atomically writes `day-YYYY-MM-DD.json` under the explicit output directory. The CLI does not accept a caller-supplied date, and the recorder rejects a duplicate current-day file, a non-monotonic previous report, failed metrics, a profile/manifest change, or a rollback. Schedule that command once per UTC day for 14 days. `--verify-retention` then requires 14 consecutive signed daily reports from the explicit directory, verifies each signature/observed timestamp against its filename and the preceding day, requires the same active manifest/config attestation throughout, no rollback, and elapsed wall time of at least 14 days; it writes a retention report and exits nonzero before any backup/export if a day or gate is missing. Only after that retention gate passes, run the scheduled off-host backup command and `export_legacy_vectors.py`; the exporter streams legacy rows to an immutable archive and writes a schema-valid report containing row count, export SHA-256, source schema revision, and backup ID. `build_release_candidate.py` requires explicit `--profile 32gb|64gb`, `--active-manifest-id`, `--retention-start`, `--pre-drop-backup-manifest`, and `--legacy-export-report`. It loads only reports whose embedded profile equals the requested profile, rejects mixed profile or mixed active-manifest IDs, requires exactly one latest report for each required cohort/rollback stage plus the retention report, hashes every report/config attestation, and records the supplied active manifest, selected profile, backup/export IDs, and retention start. It never chooses a profile or timestamp by filename ordering. It writes schema-valid `artifacts/acceptance/release-candidate.json`. `pre_drop_legacy_check.py` requires that file plus target-config attestation, shadow/capacity/security/restore reports, rollback-drill pass, a pre-drop backup ID, the legacy JSON-vector row count/export hash, and retention elapsed; otherwise it exits nonzero.

`export_legacy_vectors.py` opens the database URL only from the named secret file, uses a server-side cursor ordered by stable row ID with `fetchmany(1000)`, writes canonical JSONL containing only row ID and the legacy vector, updates SHA-256 while writing, fsyncs and atomically renames the archive, and then writes its report. It refuses an existing output unless `--replace` is explicitly supplied and verifies that the referenced pre-drop backup manifest passed checksum validation before attaching its backup ID.

`release_candidate.schema.json` requires `schemaVersion`, `profile`, `activeManifestId`, `retentionStart`, `retentionReport`, `reportChecksums`, `configAttestationChecksums`, `preDropBackup` (`id`, `manifestSha256`), `legacyExport` (`rowCount`, `exportSha256`, `reportSha256`), `rollbackDrillReport`, and `createdAt`; additional properties are rejected.

```powershell
py tools/run_cutover_drill.py --from legacy --to offline --cohort principal:u1 --soak-minutes 60 --minimum-queries 50 --rollback legacy --report-root artifacts/acceptance
py tools/check_config.py --target deploy/offline/.env --require QUERY_ENGINE=legacy --report artifacts/acceptance/config-attestation-after-principal-rollback.json
py tools/run_cutover_drill.py --from legacy --to offline --cohort department:d2 --soak-minutes 1440 --minimum-queries 500 --rollback legacy --report-root artifacts/acceptance
py tools/check_config.py --target deploy/offline/.env --require QUERY_ENGINE=legacy --report artifacts/acceptance/config-attestation-after-department-rollback.json
py tools/run_cutover_drill.py --from legacy --to offline --cohort all-users --soak-minutes 4320 --minimum-queries 2000 --rollback legacy --report-root artifacts/acceptance
py tools/run_cutover_drill.py --from legacy --to offline --cohort all-users --soak-minutes 4320 --minimum-queries 2000 --keep-promoted --report-root artifacts/acceptance
py tools/check_config.py --target deploy/offline/.env --require QUERY_ENGINE=offline --require OFFLINE_QUERY_ALL_USERS=true --report artifacts/acceptance/config-attestation-all-users.json
if ($env:OFFLINE_ACCEPTED_PROFILE -notin @("32gb", "64gb")) { throw "Set OFFLINE_ACCEPTED_PROFILE before verifying retention" }
if (-not $env:ACTIVE_PUBLICATION_MANIFEST_ID) { throw "Set ACTIVE_PUBLICATION_MANIFEST_ID before verifying retention" }
if (-not $env:OFFLINE_ALL_USERS_PROMOTED_AT) { throw "Set OFFLINE_ALL_USERS_PROMOTED_AT before verifying retention" }
py tools/run_cutover_drill.py --record-retention-day --profile $env:OFFLINE_ACCEPTED_PROFILE --active-manifest-id $env:ACTIVE_PUBLICATION_MANIFEST_ID --config-attestation artifacts/acceptance/config-attestation-all-users.json --minimum-queries 2000 --output-dir artifacts/acceptance/retention
# Schedule the preceding --record-retention-day command once per UTC day for 14 consecutive days.
py tools/run_cutover_drill.py --verify-retention --retention-start $env:OFFLINE_ALL_USERS_PROMOTED_AT --minimum-days 14 --active-manifest-id $env:ACTIVE_PUBLICATION_MANIFEST_ID --daily-report-dir artifacts/acceptance/retention --report-root artifacts/acceptance
$preDropManifest = Join-Path $env:OFFLINE_PRE_DROP_BACKUP_ROOT "backup-manifest.json"
$powershell = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Path
if (-not $powershell) { throw "Windows PowerShell is required to run backup_offline.ps1" }
& $powershell -NoProfile -ExecutionPolicy Bypass -File tools/backup_offline.ps1 -Destination $env:OFFLINE_PRE_DROP_BACKUP_ROOT -ManifestPath $preDropManifest -KeepDays 30
py tools/export_legacy_vectors.py --database-url-file $env:DATABASE_URL_SECRET_FILE --backup-manifest $preDropManifest --output artifacts/acceptance/legacy-vectors.jsonl --report artifacts/acceptance/legacy-vector-export.json
py tools/build_release_candidate.py --report-root artifacts/acceptance --profile $env:OFFLINE_ACCEPTED_PROFILE --active-manifest-id $env:ACTIVE_PUBLICATION_MANIFEST_ID --retention-start $env:OFFLINE_ALL_USERS_PROMOTED_AT --pre-drop-backup-manifest $preDropManifest --legacy-export-report artifacts/acceptance/legacy-vector-export.json --output artifacts/acceptance/release-candidate.json
py tools/pre_drop_legacy_check.py --release-record artifacts/acceptance/release-candidate.json
```

- [ ] **Step 4: Commit guards**

```powershell
py -m unittest tools.tests.test_cutover_guards tools.tests.test_legacy_export -v
git add tools/run_cutover_drill.py tools/pre_drop_legacy_check.py tools/build_release_candidate.py tools/export_legacy_vectors.py tools/release_candidate.schema.json tools/tests/test_cutover_guards.py tools/tests/test_legacy_export.py docs/offline-query-runbook.md
git commit -m "test: guard offline cutover and rollback"
```

### Task 9: Remove legacy retrieval only after every guard passes

**Files:**
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/embeddings.py`
- Modify: `backend/app/ingestion.py`
- Modify: `backend/app/database.py`
- Create: `backend/alembic/versions/20260715_05_remove_legacy_vectors.py`
- Modify: `backend/tests/test_sql_repository.py`
- Modify: `backend/tests/test_knowledge_ingestion_pipeline.py`

- [ ] **Step 1: Run the irreversible-operation guard**

```powershell
py tools/pre_drop_legacy_check.py --release-record artifacts/acceptance/release-candidate.json
```

Expected: exit 0 only after all-offline attestation, 14 days, rollback evidence, and backup/export identifiers.

- [ ] **Step 2: Remove only unreachable production code**

`20260715_05_remove_legacy_vectors.py` has `down_revision = "20260715_04"`. Delete Python full scans and JSON-vector production writes only after the guard verifies the already-created legacy export count/hash and pre-drop backup manifest from Task 8. Do not generate either artifact in this task. Keep `InMemoryChatRepository` test doubles until API tests use `AnswerEngine` fakes. Migration downgrade recreates a nullable compatibility column but documents that vector contents require restoration from the named backup.

- [ ] **Step 3: Run migration and full regression**

```powershell
Set-Location backend
alembic upgrade head
py -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
npm.cmd --prefix frontend run test:run
npm.cmd --prefix frontend run build
npm.cmd --prefix admin-frontend run test:run
npm.cmd --prefix admin-frontend run build
```

- [ ] **Step 4: Commit cleanup**

```powershell
git add backend/app/repository.py backend/app/sql_repository.py backend/app/embeddings.py backend/app/ingestion.py backend/app/database.py backend/alembic/versions/20260715_05_remove_legacy_vectors.py backend/tests/test_sql_repository.py backend/tests/test_knowledge_ingestion_pipeline.py
git commit -m "refactor: remove accepted legacy retrieval"
```

### Task 10: Produce the final release record

**Files:**
- Create: `docs/offline-release-checklist.md`
- Modify: `README.md`
- Modify: `docs/offline-platform-runbook.md`
- Modify: `docs/offline-query-runbook.md`
- Modify: `docs/offline-backup-runbook.md`

- [ ] **Step 1: Link every required artifact**

The checklist records artifact/model licenses/checksums, Alembic head, active manifest, separate 32GB/64GB reports, selected profile, batch-window reports, restore drill, failure matrix, shadow report, cohort/rollback drill, all 14 signed daily retention reports plus the final `retentionReport`, pre-drop backup/export IDs, current config hashes, and test/build logs.

- [ ] **Step 2: Run final verification**

```powershell
Set-Location backend
py -m unittest discover -s tests -p "test_*.py" -v
Set-Location ..
py -m unittest discover -s tools/tests -p "test_*.py" -v
npm.cmd --prefix frontend run test:run
npm.cmd --prefix frontend run build
npm.cmd --prefix admin-frontend run test:run
npm.cmd --prefix admin-frontend run build
git diff --check
git status --short --branch
```

Environment-gated tests are named as blocked unless their report files exist and pass; they are never labeled passed from documentation alone.

- [ ] **Step 3: Commit release documentation**

```powershell
git add README.md docs/offline-release-checklist.md docs/offline-platform-runbook.md docs/offline-query-runbook.md docs/offline-backup-runbook.md
git commit -m "docs: record offline production acceptance"
```

## Phase 4 completion gate

- A selected hardware profile has its own reproducible cold/warm and batch-under-load pass.
- A real isolated restore drill proves backup age <=24 hours and restore time <=4 hours.
- Permission, provenance, version pinning, component outage, Redis loss, and degradation tests pass.
- Shadow comparison passes machine-readable numeric, retrieval, citation, no-answer, grounding, permission, version, and latency gates.
- Principal and department cohorts are exercised before all-user promotion.
- Rollback to legacy is executed and recorded before the 14-day retention window starts.
- Legacy full-scan/vector data is removed only after the pre-drop guard validates every report and backup/export identifier.
