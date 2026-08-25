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
    find_query_field_aliases,
    normalize_fact_key,
    query_field_matches,
    query_field_terms,
    query_file_reference_terms,
    query_overlap_terms,
    query_subject_terms,
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

    def test_query_field_aliases_are_broader_without_changing_fact_contract(self) -> None:
        matches = find_query_field_aliases("蜘蛛侠的位置是什么？")

        self.assertEqual([(item.field, item.alias) for item in matches], [("位置", "位置")])
        self.assertIn("主要活动区域", query_field_terms("蜘蛛侠的位置是什么？"))
        self.assertIn("地理位置", query_field_terms("蜘蛛侠的位置是什么？"))
        self.assertNotIn("位置", FACT_FIELD_ALIASES)

    def test_longest_query_field_alias_wins_and_expands_synonyms(self) -> None:
        matches = find_query_field_aliases("蜘蛛侠的主要活动区域是什么")

        self.assertEqual([(item.field, item.alias) for item in matches], [("位置", "主要活动区域")])
        terms = query_overlap_terms("蜘蛛侠的位置")
        self.assertIn("主要活动区域", terms)
        self.assertIn("地理位置", terms)

    def test_query_subject_terms_exclude_field_and_question_scaffolding(self) -> None:
        self.assertEqual(query_subject_terms("蜘蛛侠的位置是什么？"), ("蜘蛛侠",))
        self.assertEqual(query_subject_terms("位置是什么？"), ())

    def test_query_subject_terms_preserve_cjk_names_and_organizations(self) -> None:
        self.assertEqual(query_subject_terms("李和平的年龄"), ("李和平",))
        self.assertEqual(query_subject_terms("中山大学的所在地"), ("中山大学",))

    def test_query_subject_terms_keep_explicit_region_and_year(self) -> None:
        self.assertEqual(
            query_subject_terms("华东在2025年的销售额"),
            ("2025", "华东"),
        )
        self.assertEqual(query_subject_terms("华东2025销售额"), ("2025", "华东"))

    def test_query_subject_terms_split_qualified_topics_without_breaking_names(self) -> None:
        self.assertEqual(
            query_subject_terms("中山大学在北京的地址"),
            ("中山大学", "北京"),
        )
        self.assertEqual(
            query_subject_terms("A公司北京地址"),
            ("a公司", "北京"),
        )
        self.assertEqual(
            query_subject_terms("A公司的北京地址"),
            ("a公司", "北京"),
        )
        self.assertEqual(query_subject_terms("张三和李四的联系方式"), ("张三", "李四"))
        self.assertEqual(query_subject_terms("李和平的年龄"), ("李和平",))

    def test_query_subject_terms_omit_time_window_scaffolding(self) -> None:
        self.assertEqual(
            query_subject_terms("多伦多在2026年8月16日的平均温度"),
            ("2026", "多伦多"),
        )

    def test_query_subject_terms_omit_relative_age_time_qualifiers(self) -> None:
        for question in (
            "张三今年几岁",
            "张三现在多少岁",
            "张三目前多大",
            "张三当前年龄是多少",
            "张三如今多少岁",
            "张三现年多少岁",
        ):
            with self.subTest(question=question):
                self.assertEqual(query_subject_terms(question), ("张三",))

    def test_query_subject_terms_drop_standalone_schema_nouns_but_keep_compound_names(
        self,
    ) -> None:
        self.assertEqual(query_subject_terms("公司的岗位职责是什么"), ())
        self.assertEqual(query_subject_terms("华为公司的地址"), ("华为公司",))
        self.assertEqual(query_subject_terms("财务部门的电话"), ("财务部门",))

    def test_query_subject_terms_split_location_and_ignore_clock_fragments(self) -> None:
        self.assertEqual(
            query_subject_terms("中山大学在北京的地址"),
            ("中山大学", "北京"),
        )
        self.assertEqual(
            query_subject_terms(
                "多伦多在2026年8月16日00:00到1:00这段时间的平均温度"
            ),
            ("2026", "多伦多"),
        )

    def test_query_field_aliases_cover_natural_location_and_contact_questions(self) -> None:
        self.assertEqual(find_query_field_aliases("蜘蛛侠在哪里？")[0].field, "位置")
        self.assertEqual(find_query_field_aliases("怎么联系张三？")[0].field, "联系方式")

    def test_natural_gender_question_forms_are_specific(self) -> None:
        for question in ("蜘蛛侠是男是女", "蜘蛛侠男还是女", "蜘蛛侠是男或女"):
            with self.subTest(question=question):
                self.assertEqual(
                    [item.field for item in find_query_field_aliases(question)],
                    ["性别"],
                )
        self.assertEqual(find_query_field_aliases("男女混合活动")[0:1], ())

    def test_natural_age_question_forms_do_not_become_quantity_queries(self) -> None:
        for question in (
            "张三有多少岁",
            "张三多少岁",
            "张三今年多少岁",
            "张三现在多少岁",
            "张三现年多少岁",
        ):
            with self.subTest(question=question):
                matches = find_query_field_aliases(question)
                self.assertEqual([item.field for item in matches], ["年龄"])
                self.assertNotIn("数量", [item.field for item in matches])

    def test_exact_word_fact_route_accepts_natural_age_question_forms(self) -> None:
        for question in (
            "张三有多少岁",
            "张三多少岁",
            "张三今年多少岁",
            "张三现在多少岁",
            "张三现年多少岁",
        ):
            with self.subTest(question=question):
                intent = resolve_word_factual_intent(question)
                self.assertIsInstance(intent, WordFactualIntent)
                assert isinstance(intent, WordFactualIntent)
                self.assertEqual(intent.entity, "张三")
                self.assertEqual(intent.field, "年龄")

    def test_explicit_file_reference_terms_ignore_question_scaffolding(self) -> None:
        self.assertEqual(
            query_file_reference_terms("请查询蜘蛛侠资料.docx中的位置"),
            ("蜘蛛侠资料docx",),
        )
        self.assertEqual(
            query_file_reference_terms("上传的A公司.xlsx里的地址"),
            ("a公司xlsx",),
        )
        self.assertEqual(
            query_file_reference_terms("文件名是abc.xlsx中的金额"),
            ("abcxlsx",),
        )
        self.assertEqual(
            query_file_reference_terms(r"请查询 E:/data/abc.xlsx 中的金额"),
            ("abcxlsx",),
        )
        self.assertEqual(query_file_reference_terms("请查询文件类型为.docx的资料"), ())

    def test_file_reference_cleaning_preserves_basename_that_is_a_prefix_word(self) -> None:
        self.assertEqual(query_file_reference_terms("介绍.docx中的位置"), ("介绍docx",))
        self.assertEqual(query_file_reference_terms("查询.xlsx中的金额"), ("查询xlsx",))

    def test_file_reference_terms_keep_prefix_named_basenames(self) -> None:
        self.assertEqual(
            query_file_reference_terms("关于项目.docx中的内容"),
            ("项目docx", "关于项目docx"),
        )
        self.assertEqual(
            query_file_reference_terms("在北京.xlsx中的销售额"),
            ("北京xlsx", "在北京xlsx"),
        )
        self.assertEqual(
            query_file_reference_terms("请查询关于项目.docx中的内容"),
            ("项目docx", "关于项目docx"),
        )

    def test_file_reference_terms_preserve_action_word_basename_candidates(self) -> None:
        self.assertEqual(
            query_file_reference_terms("介绍报告.docx中的内容"),
            ("报告docx", "介绍报告docx"),
        )
        # Marker-aware wording still keeps the exact basename as a fallback;
        # source scoping can therefore match a real upload named
        # ``查询报告.xlsx`` instead of only the cleaned ``报告.xlsx``.
        self.assertEqual(
            query_file_reference_terms("文件名是查询报告.xlsx中的内容"),
            ("报告xlsx", "查询报告xlsx"),
        )
        # A conversational prefix followed by ``一下`` must not create a
        # bogus raw candidate such as ``一下报告xlsx``.
        self.assertEqual(
            query_file_reference_terms("介绍一下报告.xlsx中的内容"),
            ("报告xlsx",),
        )

    def test_file_reference_terms_reject_placeholder_and_unbounded_extensions(self) -> None:
        self.assertEqual(query_file_reference_terms("文件是 .xlsx"), ())
        self.assertEqual(query_file_reference_terms("文件名为.docx"), ())
        self.assertEqual(query_file_reference_terms("报告.xlsxabc中的内容"), ())
        self.assertEqual(query_file_reference_terms("报告.xlsx_foo中的内容"), ())

    def test_file_reference_terms_strip_leading_natural_language_scaffolding(self) -> None:
        # Conversational wording must not become part of the hard filename
        # constraint.  In particular, ``文件`` is a marker in these forms,
        # not part of the uploaded basename.
        cases = {
            "请读取文件abc.xlsx": ("abcxlsx",),
            "请介绍报告.xlsx": ("报告xlsx",),
            "查一下abc.xlsb": ("abcxlsb",),
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(query_file_reference_terms(question), expected)

    def test_file_reference_terms_strip_generic_question_prefixes(self) -> None:
        expected = {
            "请问蜘蛛侠资料.docx的位置": ("蜘蛛侠资料docx",),
            "能告诉我蜘蛛侠资料.docx的位置吗": ("蜘蛛侠资料docx",),
            # ``关于`` may be a genuine basename prefix, so retain both the
            # cleaned conversational candidate and the exact full basename.
            "关于蜘蛛侠资料.docx的位置": ("蜘蛛侠资料docx", "关于蜘蛛侠资料docx"),
            "帮我查询蜘蛛侠资料.docx中的位置": ("蜘蛛侠资料docx",),
            "介绍一下蜘蛛侠资料.docx的位置": ("蜘蛛侠资料docx",),
            "查询一下蜘蛛侠资料.docx的位置": ("蜘蛛侠资料docx",),
            "打开一下蜘蛛侠资料.docx的位置": ("蜘蛛侠资料docx",),
            "我想查看蜘蛛侠资料.docx的位置": ("蜘蛛侠资料docx",),
        }
        for question, expected_terms in expected.items():
            with self.subTest(question=question):
                self.assertEqual(query_file_reference_terms(question), expected_terms)

    def test_filename_field_words_do_not_become_subject_terms(self) -> None:
        self.assertEqual(query_subject_terms("文件地点.xlsx中的内容"), ())
        self.assertEqual(query_subject_terms("文件年龄.docx中的内容"), ())

    def test_query_field_aliases_ignore_field_words_inside_file_names(self) -> None:
        self.assertEqual(find_query_field_aliases("文件年龄.docx中的内容"), ())
        matches = find_query_field_aliases("报告.xlsx中的销售额")
        self.assertEqual([(item.field, item.alias) for item in matches], [("销售额", "销售额")])

    def test_generic_work_word_does_not_create_a_second_field_for_workplace(self) -> None:
        matches = find_query_field_aliases("工作地点在哪里")
        self.assertEqual([(item.field, item.alias) for item in matches], [("位置", "在哪里")])

    def test_metric_aliases_do_not_cross_expand_unrelated_fields(self) -> None:
        temperature_terms = query_field_terms("平均温度")
        self.assertIn("气温", temperature_terms)
        self.assertNotIn("湿度", temperature_terms)
        self.assertNotIn("评分", temperature_terms)
        self.assertNotIn("风速", temperature_terms)
        self.assertNotIn("最高温度", temperature_terms)
        self.assertNotIn("最低温度", temperature_terms)

    def test_average_temperature_compatibility_rejects_extrema(self) -> None:
        self.assertTrue(query_field_matches("平均温度", "温度 | 19.7"))
        self.assertFalse(query_field_matches("平均温度", "最低温度 | 10"))

    def test_semantic_field_fallback_rejects_broad_narrative_terms(self) -> None:
        negative_cases = (
            ("年龄", "张三的出生地是北京。"),
            ("性别", "男女混合活动正在报名。"),
            ("职务", "张三负责活动策划。"),
            ("联系方式", "请联系客户确认订单。"),
        )
        for field, text in negative_cases:
            with self.subTest(field=field, text=text):
                self.assertFalse(query_field_matches(field, text))

    def test_semantic_field_fallback_accepts_explicit_narrative_forms(self) -> None:
        positive_cases = (
            ("年龄", "张三出生于1990年。"),
            ("性别", "张三是一名女性。"),
            ("职务", "张三担任工程师。"),
            ("联系方式", "联系电话是13800000000。"),
        )
        for field, text in positive_cases:
            with self.subTest(field=field, text=text):
                self.assertTrue(query_field_matches(field, text))

    def test_birth_narrative_is_scoped_to_age_or_location(self) -> None:
        self.assertTrue(query_field_matches("年龄", "张三出生于1990年。"))
        self.assertTrue(query_field_matches("年龄", "张三生于2020-01-02。"))
        self.assertFalse(query_field_matches("年龄", "张三出生于北京。"))
        self.assertTrue(query_field_matches("位置", "张三出生于北京。"))
        self.assertTrue(query_field_matches("位置", "张三生于New York。"))
        self.assertFalse(query_field_matches("位置", "张三出生于1990年。"))

    def test_ascii_field_alias_requires_identifier_boundaries(self) -> None:
        for value in ("https://example.com/page:1", "message: queued"):
            with self.subTest(value=value):
                self.assertFalse(fact_value_has_embedded_key_value(value, field="职务"))
        self.assertTrue(fact_value_has_embedded_key_value("age: 28", field="职务"))
        self.assertTrue(fact_value_has_embedded_key_value("28岁 性别：女", field="年龄"))

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
