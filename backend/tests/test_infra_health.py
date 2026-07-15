from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from app.infra.health import DependencyCheck, DependencyHealthRegistry
from app.main import create_app
from app.repository import InMemoryChatRepository
from app.seed import build_seed_state


class InfraHealthTest(unittest.TestCase):
    def test_liveness_does_not_require_external_services(self) -> None:
        calls = 0

        def qdrant_check() -> tuple[bool, str]:
            nonlocal calls
            calls += 1
            return False, "unavailable"

        registry = DependencyHealthRegistry(
            [DependencyCheck("qdrant", qdrant_check)]
        )
        client = TestClient(
            create_app(
                InMemoryChatRepository(build_seed_state()),
                health_registry=registry,
            )
        )

        liveness = client.get("/api/healthz")

        self.assertEqual(liveness.status_code, 200)
        self.assertEqual(liveness.json(), {"status": "ok"})
        self.assertEqual(calls, 0)

        readiness = client.get("/api/readyz")

        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(
            readiness.json(),
            {
                "status": "not_ready",
                "dependencies": {
                    "qdrant": {"ok": False, "detail": "unavailable"}
                },
            },
        )
        self.assertEqual(calls, 1)

    def test_registry_reports_success_and_empty_registry_is_ready(self) -> None:
        registry = DependencyHealthRegistry(
            [DependencyCheck("redis", lambda: (True, "ready"))]
        )

        self.assertEqual(
            registry.report(),
            {"redis": {"ok": True, "detail": "ready"}},
        )
        self.assertTrue(registry.ready())
        self.assertEqual(DependencyHealthRegistry().report(), {})
        self.assertTrue(DependencyHealthRegistry().ready())

    def test_registry_sanitizes_check_exceptions(self) -> None:
        def failing_check() -> tuple[bool, str]:
            raise RuntimeError(
                "postgresql://admin:secret@example.invalid/private-database"
            )

        registry = DependencyHealthRegistry(
            [DependencyCheck("postgresql", failing_check)]
        )

        report = registry.report()

        self.assertEqual(report["postgresql"]["ok"], False)
        self.assertEqual(report["postgresql"]["detail"], "check failed")
        self.assertNotIn("secret", str(report))
        self.assertFalse(registry.ready())

    def test_registry_rejects_malformed_check_results(self) -> None:
        cases = (
            lambda: True,
            lambda: (True,),
            lambda: ("yes", "ready"),
            lambda: (True, object()),
        )

        for index, check in enumerate(cases):
            with self.subTest(index=index):
                registry = DependencyHealthRegistry(
                    [DependencyCheck("invalid", check)]  # type: ignore[arg-type]
                )
                self.assertEqual(
                    registry.report(),
                    {
                        "invalid": {
                            "ok": False,
                            "detail": "invalid check result",
                        }
                    },
                )
                self.assertFalse(registry.ready())

    def test_registry_rejects_duplicate_dependency_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate dependency check"):
            DependencyHealthRegistry(
                [
                    DependencyCheck("redis", lambda: (False, "unavailable")),
                    DependencyCheck("redis", lambda: (True, "ready")),
                ]
            )

    def test_dependency_check_is_frozen_and_slotted(self) -> None:
        check = DependencyCheck("redis", lambda: (True, "ready"))

        with self.assertRaises((AttributeError, TypeError)):
            check.name = "changed"  # type: ignore[misc]
        self.assertFalse(hasattr(check, "__dict__"))

    def test_health_compatibility_endpoint_is_unchanged(self) -> None:
        client = TestClient(
            create_app(InMemoryChatRepository(build_seed_state()))
        )

        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
