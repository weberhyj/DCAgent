from __future__ import annotations

import unittest

from app.llm_runtime import normalize_llm_provider, validate_production_llm_provider


class LLMRuntimeTest(unittest.TestCase):
    def test_normalizes_hyphenated_provider_without_mutating_environ(self) -> None:
        environ = {"LLM_PROVIDER": "  physoc-deepseek  "}

        provider = normalize_llm_provider(environ)

        self.assertEqual(provider, "physoc_deepseek")
        self.assertEqual(environ, {"LLM_PROVIDER": "  physoc-deepseek  "})

    def test_production_rejects_non_generating_providers(self) -> None:
        for environ in (
            {},
            {"LLM_PROVIDER": ""},
            {"LLM_PROVIDER": "template"},
            {"LLM_PROVIDER": "mock"},
            {"LLM_PROVIDER": "  TeMpLaTe  "},
            {"LLM_PROVIDER": "  MoCk  "},
        ):
            with self.subTest(environ=environ):
                with self.assertRaisesRegex(
                    ValueError, "Production runtime requires a real LLM provider"
                ):
                    validate_production_llm_provider(environ)

    def test_production_accepts_real_providers(self) -> None:
        cases = (
            ({"LLM_PROVIDER": "physoc_deepseek"}, "physoc_deepseek"),
            ({"LLM_PROVIDER": "  Physoc-DeepSeek  "}, "physoc_deepseek"),
            ({"LLM_PROVIDER": "openai_compatible"}, "openai_compatible"),
        )

        for environ, expected_provider in cases:
            with self.subTest(environ=environ):
                self.assertEqual(validate_production_llm_provider(environ), expected_provider)


if __name__ == "__main__":
    unittest.main()
