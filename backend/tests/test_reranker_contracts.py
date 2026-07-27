from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from app.reranker_contracts import (
    MAX_RERANK_PASSAGES,
    MAX_RERANK_REQUEST_BYTES,
    MAX_RERANK_TEXT_BYTES,
    RerankerMetadataResponse,
    RerankerModelMetadata,
    RerankerRequest,
    RerankerResponse,
    reranker_request_json_size,
)


def metadata() -> RerankerModelMetadata:
    return RerankerModelMetadata(
        "Qwen/Qwen3-Reranker-0.6B",
        "1",
        "a" * 64,
        "b" * 64,
        "1",
    )


class RerankerContractTest(unittest.TestCase):
    def test_metadata_round_trips_exact_wire_fields(self) -> None:
        payload = RerankerMetadataResponse.from_metadata(metadata())

        self.assertEqual(
            payload.model_dump(by_alias=True),
            {
                "modelName": "Qwen/Qwen3-Reranker-0.6B",
                "modelVersion": "1",
                "modelChecksum": "a" * 64,
                "promptProfileSha256": "b" * 64,
                "protocolVersion": "1",
            },
        )
        self.assertEqual(payload.to_metadata(), metadata())

    def test_metadata_rejects_extras_uppercase_checksums_and_coercion(self) -> None:
        valid = RerankerMetadataResponse.from_metadata(metadata()).model_dump(by_alias=True)
        invalid_payloads = (
            {**valid, "dimensions": 1024},
            {**valid, "modelChecksum": "A" * 64},
            {**valid, "promptProfileSha256": "B" * 64},
            {**valid, "protocolVersion": 1},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises((ValidationError, ValueError)):
                    RerankerMetadataResponse.model_validate(payload)

    def test_request_enforces_count_text_and_total_utf8_bounds(self) -> None:
        request = RerankerRequest(query="policy", passages=["required"])
        self.assertEqual(request.query, "policy")

        invalid_payloads = (
            {"query": "", "passages": ["valid"]},
            {"query": "valid", "passages": []},
            {"query": "valid", "passages": ["valid"] * (MAX_RERANK_PASSAGES + 1)},
            {"query": "x" * (MAX_RERANK_TEXT_BYTES + 1), "passages": ["valid"]},
            {"query": "valid", "passages": ["x" * (MAX_RERANK_TEXT_BYTES + 1)]},
            {"query": 1, "passages": ["valid"]},
            {"query": "valid", "passages": [1]},
            {"query": "valid", "passages": ["x" * MAX_RERANK_TEXT_BYTES] * 24},
        )

        for payload in invalid_payloads:
            with self.subTest(query_type=type(payload["query"]).__name__):
                with self.assertRaises(ValidationError):
                    RerankerRequest.model_validate(payload)

        self.assertGreater(
            reranker_request_json_size("valid", ["x" * MAX_RERANK_TEXT_BYTES] * 24),
            MAX_RERANK_REQUEST_BYTES,
        )

    def test_response_requires_matching_finite_unit_interval_scores(self) -> None:
        valid = {
            **RerankerMetadataResponse.from_metadata(metadata()).model_dump(by_alias=True),
            "passageCount": 2,
            "scores": [0.8, 0.2],
        }
        response = RerankerResponse.model_validate(valid)
        self.assertEqual(response.scores, [0.8, 0.2])

        invalid_payloads = (
            {**valid, "passageCount": 1},
            {**valid, "scores": [float("nan"), 0.2]},
            {**valid, "scores": [1.1, 0.2]},
            {**valid, "scores": [-0.1, 0.2]},
            {**valid, "passageCount": "2"},
            {**valid, "extra": True},
        )
        for payload in invalid_payloads:
            with self.subTest(payload=json.dumps(payload, default=str)):
                with self.assertRaises(ValidationError):
                    RerankerResponse.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
