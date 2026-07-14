import unittest

from app.answer_text import normalize_plain_text_answer


class NormalizePlainTextAnswerTests(unittest.TestCase):
    def test_normalizes_formatted_list_and_inline_citation(self):
        text = "- **数联**：数据要素联通\n- **智联**：智能与算力连接\n- **光联**：城市光网支撑。[1]"

        self.assertEqual(
            normalize_plain_text_answer(text),
            "数联：数据要素联通\n智联：智能与算力连接\n光联：城市光网支撑。",
        )

    def test_preserves_plain_text_values(self):
        values = ["城市一张网 2.0", "2 * 3", "user_name", "__init__", "- 5°C"]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(normalize_plain_text_answer(value), value)

    def test_normalizes_multiple_bold_spans_on_one_line(self):
        self.assertEqual(
            normalize_plain_text_answer("a **b** c **d** e"),
            "a b c d e",
        )

    def test_normalizes_adjacent_bold_spans(self):
        self.assertEqual(normalize_plain_text_answer("**a****b**"), "ab")

    def test_normalizes_formatted_list_with_multiple_bold_spans(self):
        self.assertEqual(
            normalize_plain_text_answer("- **a** and **b**"),
            "a and b",
        )

    def test_normalizes_chinese_inline_bold_spans(self):
        cases = {
            "这是**重点**内容": "这是重点内容",
            "现金流风险与**回款周期**直接相关。": "现金流风险与回款周期直接相关。",
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_plain_text_answer(value), expected)

    def test_preserves_code_like_and_escaped_stars(self):
        values = [
            "`**literal**`",
            "```\n**literal**\n```",
            r"\**literal**",
            "2**3**4",
            "name**literal**value",
        ]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(normalize_plain_text_answer(value), value)

    def test_preserves_malformed_bold_delimiters(self):
        values = [
            "**foo **bar**",
            "**foo",
            r"**literal\**",
            r"- **literal\**",
        ]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(normalize_plain_text_answer(value), value)

    def test_preserves_indented_formatted_lists(self):
        values = ["    - **literal**", "\t- **literal**", "  - **literal**"]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(normalize_plain_text_answer(value), value)

    def test_preserves_indented_code_lines(self):
        values = ["示例：\n    **literal**", "示例：\n\t**literal**"]

        for value in values:
            with self.subTest(value=value):
                self.assertEqual(normalize_plain_text_answer(value), value)


if __name__ == "__main__":
    unittest.main()
