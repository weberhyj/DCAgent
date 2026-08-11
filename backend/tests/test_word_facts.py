from __future__ import annotations

import unittest

from sqlalchemy import inspect

from app.database import Database
from app.models import KnowledgeFactModel as ExportedKnowledgeFactModel
from app.word_facts import KnowledgeFactModel, canonical_fact_field, normalize_fact_key


class WordFactContractTests(unittest.TestCase):
    def test_fact_model_remains_available_from_models(self) -> None:
        self.assertIs(ExportedKnowledgeFactModel, KnowledgeFactModel)

    def test_fact_normalization_keeps_display_values(self) -> None:
        fact = KnowledgeFactModel.create(
            id="fact-1",
            source_id="kb-people",
            chunk_id="chunk-1",
            entity=" \u5f20\u4e09 ",
            field="\u5e74\u9f84",
            value="28\u5c81",
            confidence=0.98,
            locator={"paragraph": 3},
        )

        self.assertEqual(fact.entity, "\u5f20\u4e09")
        self.assertEqual(fact.entity_normalized, "\u5f20\u4e09")
        self.assertEqual(fact.field, "\u5e74\u9f84")
        self.assertEqual(fact.field_normalized, "\u5e74\u9f84")
        self.assertEqual(fact.value, "28\u5c81")
        self.assertEqual(fact.locator, {"paragraph": 3})

    def test_fact_key_normalization_folds_unicode_case_and_separators(self) -> None:
        self.assertEqual(normalize_fact_key(" \uff21-ge! "), "age")

    def test_canonical_field_resolves_only_configured_aliases(self) -> None:
        self.assertEqual(canonical_fact_field("\u5e74\u7eaa"), "\u5e74\u9f84")
        with self.assertRaisesRegex(ValueError, "unknown fact field"):
            canonical_fact_field("\u672a\u914d\u7f6e\u5b57\u6bb5")

    def test_fact_rejects_invalid_identifier_and_confidence(self) -> None:
        with self.assertRaises(ValueError):
            KnowledgeFactModel.create(
                id="fact-1",
                source_id=" ",
                chunk_id="chunk-1",
                entity="\u5f20\u4e09",
                field="\u5e74\u9f84",
                value="28\u5c81",
                confidence=0.98,
                locator={},
            )
        with self.assertRaisesRegex(ValueError, "confidence"):
            KnowledgeFactModel.create(
                id="fact-1",
                source_id="kb-people",
                chunk_id="chunk-1",
                entity="\u5f20\u4e09",
                field="\u5e74\u9f84",
                value="28\u5c81",
                confidence=1.01,
                locator={},
            )


class WordFactSchemaTests(unittest.TestCase):
    def test_head_migration_creates_fact_lookup_indexes(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        try:
            database.create_schema()
            inspector = inspect(database.engine)
            self.assertIn("knowledge_facts", inspector.get_table_names())
            indexes = {item["name"] for item in inspector.get_indexes("knowledge_facts")}
            self.assertIn("ix_knowledge_facts_entity_field", indexes)
            self.assertIn("ix_knowledge_facts_source_id", indexes)
            self.assertIn("ix_knowledge_facts_chunk_id", indexes)
        finally:
            database.engine.dispose()

    def test_fact_foreign_keys_cascade_to_source_and_chunk(self) -> None:
        database = Database("sqlite+pysqlite:///:memory:")
        try:
            database.create_schema()
            foreign_keys = {
                (
                    tuple(item["constrained_columns"]),
                    item["referred_table"],
                    tuple(item["referred_columns"]),
                    item["options"].get("ondelete"),
                )
                for item in inspect(database.engine).get_foreign_keys("knowledge_facts")
            }
            self.assertEqual(
                {
                    (("source_id",), "knowledge_sources", ("id",), "CASCADE"),
                    (("chunk_id",), "knowledge_chunks", ("id",), "CASCADE"),
                },
                foreign_keys,
            )
        finally:
            database.engine.dispose()


if __name__ == "__main__":
    unittest.main()
