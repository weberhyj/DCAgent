from __future__ import annotations

import unittest

from app.retrieval_index_worker import main


class RecordingPublisher:
    def __init__(self) -> None:
        self.calls = []

    def build(self, collection_name, *, activate, batch_size, validation_sample_size):
        self.calls.append((collection_name, activate, batch_size, validation_sample_size))


class RetrievalIndexWorkerTest(unittest.TestCase):
    def test_cli_builds_validated_collection_without_activation_by_default(self) -> None:
        publisher = RecordingPublisher()

        result = main(
            ["--collection", "knowledge_chunks_qwen3_v12"],
            environ={},
            publisher_factory=lambda _environ: publisher,
        )

        self.assertEqual(result, 0)
        self.assertEqual(
            publisher.calls,
            [("knowledge_chunks_qwen3_v12", False, 64, 50)],
        )

    def test_cli_accepts_only_locked_flags_and_valid_collection_names(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                ["--collection", "knowledge_chunks_current", "--activate"],
                environ={},
                publisher_factory=lambda _environ: RecordingPublisher(),
            )
        with self.assertRaises(SystemExit):
            main(
                ["--collection", "knowledge_chunks_qwen3_v1", "--unknown"],
                environ={},
                publisher_factory=lambda _environ: RecordingPublisher(),
            )

    def test_cli_rejects_embedding_batches_above_protocol_limit(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                ["--collection", "knowledge_chunks_qwen3_v1", "--batch-size", "65"],
                environ={},
                publisher_factory=lambda _environ: RecordingPublisher(),
            )


if __name__ == "__main__":
    unittest.main()
