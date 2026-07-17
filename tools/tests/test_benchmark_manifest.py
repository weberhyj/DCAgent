import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from types import MappingProxyType
from pathlib import Path

from tools.benchmarks.manifest import BenchmarkManifest


ROOT = Path(__file__).parents[1]


class BenchmarkManifestTest(unittest.TestCase):
    @staticmethod
    def _direct_payload() -> dict[str, object]:
        return {
            "clickhouse_rows": 1,
            "qdrant_points": 1,
            "vector_dimension_candidates": (8,),
            "virtual_users": 1,
            "think_time_seconds": 1,
            "duration_seconds": 1,
            "request_mix": {"structured": 100},
            "filter_selectivity": (0.5,),
            "dense_candidates": 1,
            "sparse_candidates": 1,
            "fused_evidence_limit": 1,
            "context_tokens": 1,
            "output_tokens": 1,
            "include_sparse_vectors": True,
            "gate_profiles": {"smoke": ({"metric": "x", "lte": 1},)},
        }

    def test_acceptance_manifest_matches_approved_capacity_profile(self) -> None:
        manifest = BenchmarkManifest.load(ROOT / "benchmarks/manifests/acceptance-30m-5m.json")

        self.assertEqual(manifest.clickhouse_rows, 30_000_000)
        self.assertEqual(manifest.qdrant_points, 5_000_000)
        self.assertEqual(manifest.vector_dimension_candidates, (512, 768))
        self.assertEqual(manifest.virtual_users, 15)
        self.assertEqual(manifest.think_time_seconds, 5)
        self.assertEqual(manifest.duration_seconds, 1_800)
        self.assertEqual(manifest.request_mix, {"structured": 40, "document": 40, "mixed": 20})
        self.assertEqual(manifest.filter_selectivity, (0.01, 0.1, 0.5))
        self.assertEqual(manifest.dense_candidates, 50)
        self.assertEqual(manifest.sparse_candidates, 50)
        self.assertEqual(manifest.fused_evidence_limit, 10)
        self.assertEqual(manifest.context_tokens, 2_048)
        self.assertEqual(manifest.output_tokens, 256)
        self.assertTrue(manifest.include_sparse_vectors)
        self.assertEqual(
            set(manifest.gate_profiles),
            {"online-cold", "online-warm", "batch-initial", "batch-daily", "batch-weekly"},
        )

        for profile_name in ("online-cold", "online-warm"):
            gates = manifest.gate_profiles[profile_name]
            self.assertTrue(any(g["metric"] == "queue_feedback_p95_ms" and g["lte"] <= 2_000 for g in gates))
            self.assertTrue(any(g["metric"] == "first_token_p95_ms" and g["lte"] <= 10_000 for g in gates))
        for profile_name in ("batch-initial", "batch-daily", "batch-weekly"):
            gates = manifest.gate_profiles[profile_name]
            self.assertTrue(
                any(g["metric"] == "online_queue_feedback_p95_ms" and g["lte"] <= 2_000 for g in gates)
            )

    def test_smoke_manifest_is_phase_one_service_only(self) -> None:
        manifest = BenchmarkManifest.load(ROOT / "benchmarks/manifests/smoke.json")
        self.assertEqual(manifest.clickhouse_rows, 10_000)
        self.assertEqual(manifest.qdrant_points, 2_000)
        self.assertEqual(manifest.vector_dimension_candidates, (32,))
        self.assertEqual(manifest.virtual_users, 3)
        self.assertEqual(manifest.duration_seconds, 120)
        self.assertTrue(manifest.gate_profiles)
        self.assertTrue(all(g["metric"].endswith("_round_trip_ms") for gs in manifest.gate_profiles.values() for g in gs))

    def test_manifest_is_frozen_and_slots(self) -> None:
        manifest = BenchmarkManifest.load(ROOT / "benchmarks/manifests/smoke.json")
        self.assertFalse(hasattr(manifest, "__dict__"))
        self.assertIsInstance(manifest.request_mix, MappingProxyType)
        self.assertIsInstance(manifest.gate_profiles, MappingProxyType)
        self.assertIsInstance(manifest.gate_profiles["service-round-trip"][0], MappingProxyType)
        with self.assertRaises(FrozenInstanceError):
            manifest.virtual_users = 4  # type: ignore[misc]
        with self.assertRaises(TypeError):
            manifest.request_mix["other"] = 1  # type: ignore[index]
        with self.assertRaises(TypeError):
            manifest.gate_profiles["service-round-trip"][0]["lte"] = 2_000  # type: ignore[index]

    def test_scalar_shapes_fail_with_value_error(self) -> None:
        valid = {
            "clickhouse_rows": 1,
            "qdrant_points": 1,
            "vector_dimension_candidates": [8],
            "virtual_users": 1,
            "think_time_seconds": 1,
            "duration_seconds": 1,
            "request_mix": {"structured": 100},
            "filter_selectivity": [0.5],
            "dense_candidates": 1,
            "sparse_candidates": 1,
            "fused_evidence_limit": 1,
            "context_tokens": 1,
            "output_tokens": 1,
            "include_sparse_vectors": True,
            "gate_profiles": {"smoke": [{"metric": "x", "lte": 1}]},
        }
        for key, value in {
            "vector_dimension_candidates": 8,
            "filter_selectivity": 0.5,
            "gate_profiles": [],
        }.items():
            with self.subTest(key=key):
                payload = dict(valid, **{key: value})
                with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as fh:
                    json.dump(payload, fh)
                    path = Path(fh.name)
                try:
                    with self.assertRaises(ValueError):
                        BenchmarkManifest.load(path)
                finally:
                    path.unlink(missing_ok=True)

    def test_deeply_frozen_manifest_has_json_serializable_copy(self) -> None:
        manifest = BenchmarkManifest.load(ROOT / "benchmarks/manifests/smoke.json")
        payload = manifest.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["request_mix"], {"structured": 40, "document": 40, "mixed": 20})
        self.assertIsInstance(payload["gate_profiles"]["service-round-trip"][0], dict)

    def test_direct_constructor_also_validates_and_deep_freezes(self) -> None:
        payload = self._direct_payload()
        manifest = BenchmarkManifest(**payload)
        self.assertIsInstance(manifest.request_mix, MappingProxyType)
        self.assertIsInstance(manifest.gate_profiles, MappingProxyType)
        with self.assertRaises(TypeError):
            manifest.request_mix["other"] = 1  # type: ignore[index]
        with self.assertRaises(TypeError):
            manifest.gate_profiles["smoke"][0]["lte"] = 2  # type: ignore[index]

        payload["request_mix"] = {"structured": 99}
        with self.assertRaises(ValueError):
            BenchmarkManifest(**payload)

    def test_huge_integer_gate_limit_is_rejected_as_value_error(self) -> None:
        payload = self._direct_payload()
        payload["gate_profiles"] = {"smoke": ({"metric": "x", "lte": 10**10000},)}
        with self.assertRaises(ValueError):
            BenchmarkManifest(**payload)

    def test_rejects_invalid_values_and_gate_shape(self) -> None:
        valid = {
            "clickhouse_rows": 1,
            "qdrant_points": 1,
            "vector_dimension_candidates": [8],
            "virtual_users": 1,
            "think_time_seconds": 1,
            "duration_seconds": 1,
            "request_mix": {"structured": 100},
            "filter_selectivity": [0.5],
            "dense_candidates": 1,
            "sparse_candidates": 1,
            "fused_evidence_limit": 1,
            "context_tokens": 1,
            "output_tokens": 1,
            "include_sparse_vectors": True,
            "gate_profiles": {"smoke": [{"metric": "service_round_trip_ms", "lte": 100}]},
        }
        cases = [
            ("request_mix", {"structured": 99}),
            ("clickhouse_rows", 0),
            ("vector_dimension_candidates", []),
            ("filter_selectivity", [0, 1.1]),
            ("gate_profiles", {"smoke": [{"metric": "x", "lte": "100"}]}),
            ("gate_profiles", {"smoke": [{"metric": "x", "lte": 1, "gte": 0}]}),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                payload = dict(valid)
                if key == "request_mix":
                    payload[key] = value
                else:
                    payload[key] = value
                with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as fh:
                    json.dump(payload, fh)
                    path = Path(fh.name)
                try:
                    with self.assertRaises(ValueError):
                        BenchmarkManifest.load(path)
                finally:
                    path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
