from __future__ import annotations

import unittest

from sqlalchemy import inspect

from app.database import Database
from app.models import KnowledgeFactModel as ExportedKnowledgeFactModel
from app.word_facts import (
    FACT_FIELD_ALIASES,
    KnowledgeFactModel,
    WordFactClarification,
    WordFactMatch,
    WordFactualIntent,
    canonical_fact_field,
    fact_value_has_embedded_key_value,
    normalize_fact_key,
    resolve_word_factual_intent,
    validate_word_fact_answer,
)


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

    def test_shared_field_registry_contains_only_docx_extractable_fields(self) -> None:
        self.assertEqual(set(FACT_FIELD_ALIASES), {"年龄", "性别", "职务"})
        with self.assertRaisesRegex(ValueError, "unknown fact field"):
            canonical_fact_field("部门")

    def test_ascii_field_alias_requires_identifier_boundaries(self) -> None:
        for value in ("https://example.com/page:1", "message: queued"):
            with self.subTest(value=value):
                self.assertFalse(
                    fact_value_has_embedded_key_value(value, field="职务")
                )
        self.assertTrue(fact_value_has_embedded_key_value("age: 28", field="职务"))
        self.assertTrue(
            fact_value_has_embedded_key_value("28岁 性别：女", field="年龄")
        )

    def test_answer_validation_preserves_ascii_token_boundaries(self) -> None:
        intent = WordFactualIntent("张三", "张三", "职务", "职务")

        def word_fact_match(value: str) -> WordFactMatch:
            return WordFactMatch(
                fact=KnowledgeFactModel.create(
                    id=f"fact-{value}",
                    source_id="kb-people",
                    chunk_id="chunk-1",
                    entity="张三",
                    field="职务",
                    value=value,
                    confidence=0.98,
                    locator={},
                ),
                source_name="people.docx",
                classification="内部",
            )

        unsafe_value = "engineer age 28"
        self.assertFalse(
            validate_word_fact_answer(
                intent,
                [word_fact_match(unsafe_value)],
                f"张三的职务是{unsafe_value}。",
            )
        )
        for value in (
            "高级工程师（方向：AI）",
            "值班经理 08:30",
            "https://example.com/page:1",
            "message: queued",
        ):
            with self.subTest(value=value):
                self.assertTrue(
                    validate_word_fact_answer(
                        intent,
                        [word_fact_match(value)],
                        f"张三的职务是{value}。",
                    )
                )

    def test_age_gender_and_job_aliases_share_plan_canonical_fields(self) -> None:
        cases = (
            ("\u5c81\u6570", "\u5e74\u9f84"),
            ("\u6027\u522b", "\u6027\u522b"),
            ("\u804c\u4f4d", "\u804c\u52a1"),
        )

        for alias, canonical in cases:
            with self.subTest(alias=alias):
                fact = KnowledgeFactModel.create(
                    id=f"fact-{canonical}",
                    source_id="kb-people",
                    chunk_id="chunk-1",
                    entity="\u5f20\u4e09",
                    field=alias,
                    value="\u793a\u4f8b\u503c",
                    confidence=0.97,
                    locator={"paragraph": 0},
                )
                self.assertEqual(fact.field, canonical)
                self.assertEqual(fact.field_normalized, normalize_fact_key(canonical))

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


class WordFactualIntentTests(unittest.TestCase):
    def test_natural_gender_question_resolves_to_only_gender_field(self) -> None:
        resolution = resolve_word_factual_intent("张三是男是女")

        self.assertEqual(
            resolution,
            WordFactualIntent("张三", "张三", "性别", "性别"),
        )

    def test_age_and_job_aliases_resolve_canonical_fields(self) -> None:
        self.assertEqual(
            resolve_word_factual_intent("张三几岁"),
            WordFactualIntent("张三", "张三", "年龄", "年龄"),
        )
        job = resolve_word_factual_intent("请问张三担任什么职位？")
        self.assertIsInstance(job, WordFactualIntent)
        assert isinstance(job, WordFactualIntent)
        self.assertEqual(job.field, "职务")

    def test_open_introduction_is_not_factual(self) -> None:
        self.assertIsNone(resolve_word_factual_intent("介绍张三"))

    def test_open_questions_with_field_vocabulary_are_not_factual(self) -> None:
        for question in (
            "介绍财务部门",
            "公司的岗位职责是什么",
            "张三的年龄变化趋势是什么",
        ):
            with self.subTest(question=question):
                self.assertIsNone(resolve_word_factual_intent(question))

    def test_he_inside_person_or_organization_name_is_not_a_list_separator(self) -> None:
        for question, entity in (
            ("李和平几岁", "李和平"),
            ("仁和公司的职务是什么", "仁和公司"),
        ):
            with self.subTest(question=question):
                resolution = resolve_word_factual_intent(question)
                self.assertIsInstance(resolution, WordFactualIntent)
                assert isinstance(resolution, WordFactualIntent)
                self.assertEqual(resolution.entity, entity)

    def test_multiple_fields_and_entities_request_clarification(self) -> None:
        field_resolution = resolve_word_factual_intent("张三的年龄和性别")
        self.assertIsInstance(field_resolution, WordFactClarification)
        assert isinstance(field_resolution, WordFactClarification)
        self.assertEqual(field_resolution.candidates, ("年龄", "性别"))

        entity_resolution = resolve_word_factual_intent("张三和李四几岁")
        self.assertIsInstance(entity_resolution, WordFactClarification)

    def test_multiple_fields_allow_a_bounded_final_question_tail(self) -> None:
        for question in (
            "张三的年龄和性别是什么",
            "张三年龄、性别是什么",
        ):
            with self.subTest(question=question):
                resolution = resolve_word_factual_intent(question)
                self.assertIsInstance(resolution, WordFactClarification)
                assert isinstance(resolution, WordFactClarification)
                self.assertEqual(resolution.candidates, ("年龄", "性别"))

    def test_list_separator_between_entities_requests_clarification(self) -> None:
        resolution = resolve_word_factual_intent("张三、李四几岁")

        self.assertIsInstance(resolution, WordFactClarification)
        assert isinstance(resolution, WordFactClarification)
        self.assertEqual(resolution.candidates, ("张三", "李四"))

    def test_entity_above_route_scalar_limit_is_bounded_factual_clarification(self) -> None:
        accepted = resolve_word_factual_intent("张" * 256 + "几岁")
        resolution = resolve_word_factual_intent("张" * 257 + "几岁")

        self.assertIsInstance(accepted, WordFactualIntent)
        self.assertIsInstance(resolution, WordFactClarification)
        assert isinstance(resolution, WordFactClarification)
        self.assertIsNone(resolution.entity)
        self.assertEqual(resolution.target_fields, ("年龄",))
        self.assertEqual(resolution.candidates, ())


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
