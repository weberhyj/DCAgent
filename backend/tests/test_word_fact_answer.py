from __future__ import annotations

import unittest
from collections.abc import Sequence

from app.word_fact_answer import WordFactAnswerService
from app.word_facts import KnowledgeFactModel, WordFactMatch, WordFactualIntent


def match(
    entity: str,
    field: str,
    value: str,
    *,
    source_id: str = "kb-people",
    source_name: str | None = None,
    chunk_id: str = "chunk-1",
) -> WordFactMatch:
    fact = KnowledgeFactModel.create(
        id=f"fact-{source_id}-{chunk_id}-{value}",
        source_id=source_id,
        chunk_id=chunk_id,
        entity=entity,
        field=field,
        value=value,
        confidence=0.98,
        locator={"paragraph": 1},
    )
    return WordFactMatch(
        fact=fact,
        source_name=source_name or source_id,
        classification="内部",
    )


class FakeFacts:
    def __init__(self, matches: Sequence[WordFactMatch]) -> None:
        self.matches = list(matches)
        self.calls: list[tuple[WordFactualIntent, tuple[str, ...]]] = []

    def find_knowledge_facts(
        self,
        intent: WordFactualIntent,
        *,
        permission_tags: Sequence[str] = (),
    ) -> list[WordFactMatch]:
        self.calls.append((intent, tuple(permission_tags)))
        return list(self.matches)


class WordFactAnswerServiceTests(unittest.TestCase):
    def test_age_answer_contains_no_gender_or_job(self) -> None:
        service = WordFactAnswerService(FakeFacts([match("张三", "年龄", "28岁")]))

        result = service.try_answer("conv-1", "张三几岁", "quick", [])

        self.assertIsNotNone(result)
        assert result is not None
        text = result.reply.paragraphs[0].text
        self.assertEqual(text, "张三的年龄是28岁。")
        self.assertNotIn("性别", text)
        self.assertNotIn("职务", text)
        self.assertEqual([step.tool_name for step in result.steps], ["query_word_fact"])
        self.assertEqual(result.reply.paragraphs[0].citations[0].excerpt, "年龄：28岁")

    def test_conflicting_values_return_source_clarification(self) -> None:
        service = WordFactAnswerService(
            FakeFacts(
                [
                    match("张三", "年龄", "28岁", source_id="kb-a"),
                    match("张三", "年龄", "29岁", source_id="kb-b"),
                ]
            )
        )

        result = service.try_answer("conv-1", "张三几岁", "quick", [])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("存在多个年龄值", result.reply.paragraphs[0].text)
        self.assertNotIn("张三的年龄是28岁", result.reply.paragraphs[0].text)

    def test_same_value_from_multiple_sources_still_clarifies_same_name_entity(self) -> None:
        service = WordFactAnswerService(
            FakeFacts(
                [
                    match("张三", "年龄", "28岁", source_id="kb-a"),
                    match("张三", "年龄", "28岁", source_id="kb-b"),
                ]
            )
        )

        result = service.try_answer("conv-1", "张三几岁", "quick", [])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("请确认来源", result.reply.paragraphs[0].text)

    def test_missing_target_field_returns_not_found_not_rag(self) -> None:
        result = WordFactAnswerService(FakeFacts([])).try_answer(
            "conv-1", "张三几岁", "deep", []
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.reply.paragraphs[0].text, "未找到张三的年龄。")
        self.assertEqual(result.evidence_count, 0)

    def test_duplicate_records_in_one_source_use_one_value_and_citation(self) -> None:
        service = WordFactAnswerService(
            FakeFacts(
                [
                    match("张三", "年龄", "28岁", chunk_id="chunk-1"),
                    match("张三", "年龄", "28岁", chunk_id="chunk-2"),
                ]
            )
        )

        result = service.try_answer("conv-1", "张三几岁", "quick", [])

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.reply.paragraphs[0].text, "张三的年龄是28岁。")
        self.assertEqual(len(result.reply.paragraphs[0].citations), 1)
        self.assertEqual(result.evidence_count, 1)

    def test_permission_tags_are_forwarded_to_the_fact_repository(self) -> None:
        facts = FakeFacts([match("张三", "年龄", "28岁")])
        service = WordFactAnswerService(facts, permission_tags=("内部", "机密"))

        service.try_answer("conv-1", "张三几岁", "quick", [])

        self.assertEqual(facts.calls[0][1], ("内部", "机密"))


if __name__ == "__main__":
    unittest.main()
