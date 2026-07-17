import unittest

from tools.benchmarks.fixtures import (
    clickhouse_numbers_sql,
    iter_qdrant_points,
)


class BenchmarkFixtureTest(unittest.TestCase):
    def test_qdrant_points_are_deterministic_and_bounded(self) -> None:
        first = list(iter_qdrant_points(total=7, dimensions=4, batch_size=3, seed=42))
        second = list(iter_qdrant_points(total=7, dimensions=4, batch_size=3, seed=42))
        self.assertEqual([len(batch) for batch in first], [3, 3, 1])
        self.assertEqual(first, second)
        self.assertEqual(first[0][0]["id"], 0)
        self.assertEqual(len(first[0][0]["vector"]["dense"]), 4)

    def test_qdrant_fixture_validates_inputs(self) -> None:
        for kwargs in (
            {"total": -1, "dimensions": 4, "batch_size": 2, "seed": 1},
            {"total": 1, "dimensions": 0, "batch_size": 2, "seed": 1},
            {"total": 1, "dimensions": 4, "batch_size": 0, "seed": 1},
            {"total": True, "dimensions": 4, "batch_size": 2, "seed": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    list(iter_qdrant_points(**kwargs))

    def test_clickhouse_fixture_uses_server_side_numbers(self) -> None:
        sql = clickhouse_numbers_sql(30_000_000)
        self.assertIn("numbers(30000000)", sql)
        self.assertIn("INSERT INTO", sql.upper())
        self.assertNotIn("range(30000000)", sql)
        with self.assertRaises(ValueError):
            clickhouse_numbers_sql(-1)


if __name__ == "__main__":
    unittest.main()
