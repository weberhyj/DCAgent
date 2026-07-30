from __future__ import annotations

import hashlib
import json
import unittest

from app.ollama_client import OllamaBusy, OllamaResponseError, OllamaServiceError
from app.ollama_reranker_backend import (
    RERANK_PROMPT_PROFILE,
    RERANK_PROMPT_PROFILE_SHA256,
    OllamaGenerativeRerankerBackend,
)


class RecordingClient:
    def __init__(
        self,
        response: object | None = None,
        *,
        responses: list[object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = (
            {"response": '{"scores":[{"index":0,"score":0.5}]}'} if response is None else response
        )
        self.responses = list(responses) if responses is not None else None
        self.error = error
        self.calls: list[tuple[str, object]] = []
        self.close_calls = 0

    def post_json(self, path: str, payload: object) -> object:
        self.calls.append((path, payload))
        if self.error is not None:
            raise self.error
        if self.responses is not None:
            return self.responses.pop(0)
        return self.response

    def close(self) -> None:
        self.close_calls += 1


def make_backend(
    client: RecordingClient | None = None,
    **overrides: object,
) -> OllamaGenerativeRerankerBackend:
    arguments: dict[str, object] = {
        "model": "qwen2.5:7b",
        "path": "/api/generate",
        "keep_alive": "5m",
    }
    arguments.update(overrides)
    return OllamaGenerativeRerankerBackend(client or RecordingClient(), **arguments)  # type: ignore[arg-type]


def response_for(scores: list[dict[str, object]]) -> dict[str, str]:
    return {"response": json.dumps({"scores": scores}, separators=(",", ":"))}


class OllamaGenerativeRerankerBackendTest(unittest.TestCase):
    def test_prompt_profile_has_stable_sha256(self) -> None:
        self.assertIsInstance(RERANK_PROMPT_PROFILE, str)
        self.assertTrue(RERANK_PROMPT_PROFILE)
        self.assertEqual(
            RERANK_PROMPT_PROFILE_SHA256,
            hashlib.sha256(RERANK_PROMPT_PROFILE.encode("utf-8")).hexdigest(),
        )

    def test_constructor_rejects_invalid_configuration(self) -> None:
        cases = (
            {"model": ""},
            {"model": "  "},
            {"model": 42},
            {"keep_alive": ""},
            {"keep_alive": "\t"},
            {"keep_alive": 42},
            {"path": "/api/chat"},
            {"path": "api/generate"},
            {"path": 42},
            {"format_json": "true"},
            {"format_json": 1},
            {"num_predict": 0},
            {"num_predict": -1},
            {"num_predict": True},
            {"num_predict": 2.5},
            {"num_predict": "256"},
        )
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                make_backend(**overrides)

    def test_rerank_rejects_invalid_query_without_calling_client(self) -> None:
        client = RecordingClient()
        backend = make_backend(client)
        for query in ("", " \t\n", 42, None):
            with self.subTest(query=query), self.assertRaises(ValueError):
                backend.rerank(query, ["document"])  # type: ignore[arg-type]
        self.assertEqual(client.calls, [])

    def test_rerank_rejects_invalid_passage_collection_without_calling_client(self) -> None:
        client = RecordingClient()
        backend = make_backend(client)
        for passages in ("document", b"document", [], (), 42, None):
            with self.subTest(passages=passages), self.assertRaises(ValueError):
                backend.rerank("query", passages)  # type: ignore[arg-type]
        self.assertEqual(client.calls, [])

    def test_rerank_rejects_blank_or_nonstring_passage_without_calling_client(self) -> None:
        client = RecordingClient()
        backend = make_backend(client)
        for passages in ([""], ["ok", "  "], ["ok", 42], [None]):
            with self.subTest(passages=passages), self.assertRaises(ValueError):
                backend.rerank("query", passages)  # type: ignore[arg-type]
        self.assertEqual(client.calls, [])

    def test_rerank_sends_one_deterministic_generation_request_for_query_batch(self) -> None:
        client = RecordingClient(
            response_for(
                [
                    {"index": 0, "score": 0.75},
                    {"index": 1, "score": 0.25},
                ]
            )
        )
        backend = make_backend(client, num_predict=321)

        result = backend.rerank("secret query", ["first secret passage", "second passage"])

        self.assertEqual(result, [0.75, 0.25])
        self.assertEqual(len(client.calls), 1)
        path, payload = client.calls[0]
        self.assertEqual(path, "/api/generate")
        self.assertEqual(
            payload,
            {
                "model": "qwen2.5:7b",
                "prompt": payload["prompt"],  # type: ignore[index]
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0, "num_predict": 321},
                "format": "json",
            },
        )
        prompt = payload["prompt"]  # type: ignore[index]
        self.assertIsInstance(prompt, str)
        self.assertIn("Return exactly one JSON object and no prose", prompt)
        self.assertIn('object must have only the key "scores"', prompt)
        self.assertIn("one item per document", prompt)
        self.assertIn('only the keys "index" and "score"', prompt)
        self.assertIn("zero-based", prompt)
        self.assertIn("numeric number from 0 through 1", prompt)
        self.assertIn("retrieval relevance", prompt)
        self.assertIn("Query:\nsecret query", prompt)
        self.assertIn("Document 0:\nfirst secret passage", prompt)
        self.assertIn("Document 1:\nsecond passage", prompt)
        self.assertLess(prompt.index("Document 0:"), prompt.index("Document 1:"))

    def test_rerank_builds_the_same_prompt_for_the_same_input(self) -> None:
        response = response_for([{"index": 0, "score": 0.5}])
        first_client = RecordingClient(response)
        second_client = RecordingClient(response)

        make_backend(first_client).rerank("query", ["passage"])
        make_backend(second_client).rerank("query", ["passage"])

        self.assertEqual(first_client.calls[0][1], second_client.calls[0][1])

    def test_format_json_false_omits_format_field(self) -> None:
        client = RecordingClient(response_for([{"index": 0, "score": 1}]))
        backend = make_backend(client, format_json=False)

        self.assertEqual(backend.rerank("query", ["passage"]), [1.0])

        payload = client.calls[0][1]
        self.assertIsInstance(payload, dict)
        self.assertNotIn("format", payload)

    def test_rerank_restores_shuffled_indices_and_converts_integer_scores(self) -> None:
        client = RecordingClient(
            response_for(
                [
                    {"index": 2, "score": 1},
                    {"index": 0, "score": 0},
                    {"index": 1, "score": 0.5},
                ]
            )
        )

        result = make_backend(client).rerank("query", ["a", "b", "c"])

        self.assertEqual(result, [0.0, 0.5, 1.0])
        self.assertTrue(all(isinstance(score, float) for score in result))

    def test_response_accepts_only_json_object_with_exact_top_level_key(self) -> None:
        invalid = (
            '{"scores":[{"index":0,"score":0.5}],"extra":1}',
            "{}",
            "[]",
            '"text"',
            "null",
        )
        for model_response in invalid:
            with self.subTest(model_response=model_response), self.assertRaises(ValueError):
                make_backend(RecordingClient({"response": model_response})).rerank(
                    "query", ["passage"]
                )

    def test_response_rejects_prose_code_fences_and_multiple_json_values(self) -> None:
        valid_json = '{"scores":[{"index":0,"score":0.5}]}'
        invalid = (
            f"Here is the result: {valid_json}",
            f"```json\n{valid_json}\n```",
            f"{valid_json} trailing prose",
            f"{valid_json}{valid_json}",
        )
        for model_response in invalid:
            with self.subTest(model_response=model_response), self.assertRaises(ValueError):
                make_backend(RecordingClient({"response": model_response})).rerank(
                    "query", ["passage"]
                )

    def test_response_rejects_duplicate_object_keys_at_every_level_without_leaking(self) -> None:
        secret = "RAW_DUPLICATE_SECRET_123"
        invalid = (
            f'{{"scores":"{secret}","scores":[{{"index":0,"score":0.5}}]}}',
            f'{{"scores":[{{"index":"{secret}","index":0,"score":0.5}}]}}',
            f'{{"scores":[{{"index":0,"score":"{secret}","score":0.5}}]}}',
        )
        for model_response in invalid:
            with self.subTest(model_response=model_response):
                with self.assertRaises(ValueError) as caught:
                    make_backend(RecordingClient({"response": model_response})).rerank(
                        "query", ["passage"]
                    )
                self.assertNotIn(secret, str(caught.exception))

    def test_response_accepts_leading_and_trailing_json_whitespace(self) -> None:
        client = RecordingClient({"response": ' \n\t{"scores":[{"index":0,"score":0.5}]}\r\n '})
        self.assertEqual(make_backend(client).rerank("query", ["passage"]), [0.5])

    def test_response_rejects_missing_or_nonstring_response_field(self) -> None:
        for response in ({}, {"response": None}, {"response": 42}, {"response": {}}):
            with self.subTest(response=response), self.assertRaises(ValueError):
                make_backend(RecordingClient(response)).rerank("query", ["passage"])

    def test_response_rejects_non_mapping_outer_response(self) -> None:
        for response in ([], "text", 42):
            with self.subTest(response=response), self.assertRaises(ValueError):
                make_backend(RecordingClient(response)).rerank("query", ["passage"])

    def test_score_items_require_exact_keys_and_mapping_type(self) -> None:
        invalid_scores = (
            [{"index": 0}],
            [{"score": 0.5}],
            [{"index": 0, "score": 0.5, "extra": True}],
            [[0, 0.5]],
            ["score"],
        )
        for scores in invalid_scores:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                make_backend(RecordingClient(response_for(scores))).rerank(  # type: ignore[arg-type]
                    "query", ["passage"]
                )

    def test_response_rejects_missing_duplicate_out_of_range_and_bool_indices(self) -> None:
        invalid_scores = (
            [{"index": 0, "score": 0.1}],
            [{"index": 0, "score": 0.1}, {"index": 0, "score": 0.2}],
            [{"index": 0, "score": 0.1}, {"index": 2, "score": 0.2}],
            [{"index": 0, "score": 0.1}, {"index": -1, "score": 0.2}],
            [{"index": 0, "score": 0.1}, {"index": True, "score": 0.2}],
            [{"index": 0, "score": 0.1}, {"index": 1.0, "score": 0.2}],
        )
        for scores in invalid_scores:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                make_backend(RecordingClient(response_for(scores))).rerank("query", ["one", "two"])

    def test_response_rejects_non_list_and_wrong_score_count(self) -> None:
        invalid_responses = (
            {"response": '{"scores":null}'},
            {"response": '{"scores":{}}'},
            response_for([]),
            response_for(
                [
                    {"index": 0, "score": 0.1},
                    {"index": 1, "score": 0.2},
                ]
            ),
        )
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                make_backend(RecordingClient(response)).rerank("query", ["passage"])

    def test_response_rejects_bool_nonnumeric_nonfinite_and_out_of_range_scores(self) -> None:
        invalid_scores = (
            True,
            False,
            "0.5",
            None,
            float("nan"),
            float("inf"),
            float("-inf"),
            -0.01,
            1.01,
        )
        for score in invalid_scores:
            with self.subTest(score=score), self.assertRaises(ValueError):
                make_backend(RecordingClient(response_for([{"index": 0, "score": score}]))).rerank(
                    "query", ["passage"]
                )

    def test_response_rejects_huge_integer_score_with_sanitized_value_error(self) -> None:
        huge_integer = 10**400
        raw_response = f'{{"scores":[{{"index":0,"score":{huge_integer}}}]}}'

        with self.assertRaises(ValueError) as caught:
            make_backend(RecordingClient({"response": raw_response})).rerank(
                "query-secret", ["passage-secret"]
            )

        message = str(caught.exception)
        self.assertNotIn(str(huge_integer), message)
        self.assertNotIn("query-secret", message)
        self.assertNotIn("passage-secret", message)

    def test_adapter_errors_do_not_include_query_passage_or_raw_response(self) -> None:
        secrets = ("QUERY_SECRET_123", "PASSAGE_SECRET_456", "RAW_SECRET_789")
        client = RecordingClient({"response": f"not-json-{secrets[2]}"})

        with self.assertRaises(ValueError) as caught:
            make_backend(client).rerank(secrets[0], [secrets[1]])

        message = str(caught.exception)
        for secret in secrets:
            self.assertNotIn(secret, message)

    def test_client_errors_propagate_unchanged(self) -> None:
        for error in (
            OllamaServiceError("sanitized service error"),
            OllamaBusy("sanitized busy error"),
            OllamaResponseError("sanitized response error"),
        ):
            with self.subTest(error=type(error).__name__):
                client = RecordingClient(error=error)
                with self.assertRaises(type(error)) as caught:
                    make_backend(client).rerank("query", ["passage"])
                self.assertIs(caught.exception, error)

    def test_score_pairs_rejects_invalid_collection_without_calls(self) -> None:
        client = RecordingClient()
        backend = make_backend(client)
        for pairs in ("query passage", b"query passage", [], (), 42, None):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                backend.score_pairs(pairs)  # type: ignore[arg-type]
        self.assertEqual(client.calls, [])

    def test_score_pairs_rejects_malformed_pairs_without_calls(self) -> None:
        invalid = (
            [("query",)],
            [("query", "passage", "extra")],
            [["query", "passage"]],
            ["query"],
            [("", "passage")],
            [("query", " ")],
            [(42, "passage")],
            [("query", None)],
            [("valid", "valid"), ("", "bad")],
        )
        for pairs in invalid:
            client = RecordingClient()
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                make_backend(client).score_pairs(pairs)  # type: ignore[arg-type]
            self.assertEqual(client.calls, [])

    def test_score_pairs_groups_distinct_queries_and_restores_interleaved_order(self) -> None:
        client = RecordingClient(
            responses=[
                response_for(
                    [
                        {"index": 1, "score": 0.2},
                        {"index": 0, "score": 0.9},
                    ]
                ),
                response_for(
                    [
                        {"index": 0, "score": 0.4},
                        {"index": 1, "score": 0.7},
                    ]
                ),
            ]
        )
        backend = make_backend(client)

        result = backend.score_pairs(
            [
                ("query-a", "a-first"),
                ("query-b", "b-first"),
                ("query-a", "a-second"),
                ("query-b", "b-second"),
            ]
        )

        self.assertEqual(result, [0.9, 0.4, 0.2, 0.7])
        self.assertEqual(len(client.calls), 2)
        first_prompt = client.calls[0][1]["prompt"]  # type: ignore[index]
        second_prompt = client.calls[1][1]["prompt"]  # type: ignore[index]
        self.assertIn("Query:\nquery-a", first_prompt)
        self.assertLess(first_prompt.index("a-first"), first_prompt.index("a-second"))
        self.assertIn("Query:\nquery-b", second_prompt)
        self.assertLess(second_prompt.index("b-first"), second_prompt.index("b-second"))

    def test_close_delegates_to_client(self) -> None:
        client = RecordingClient()

        make_backend(client).close()

        self.assertEqual(client.close_calls, 1)


if __name__ == "__main__":
    unittest.main()
