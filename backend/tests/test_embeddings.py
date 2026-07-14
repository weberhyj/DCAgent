from __future__ import annotations

import unittest

from app.embeddings import HashingEmbeddingProvider, cosine_similarity


class EmbeddingProviderTest(unittest.TestCase):
    def test_hashing_embedding_ranks_related_finance_text_higher_than_unrelated_text(self) -> None:
        provider = HashingEmbeddingProvider()

        query = provider.embed("回款风险")
        related = provider.embed("应收账款增加，回款周期拉长，造成现金流压力。")
        unrelated = provider.embed("GPU rack temperature and network port utilization")

        self.assertEqual(len(query), provider.dimensions)
        self.assertAlmostEqual(sum(value * value for value in query), 1.0, places=6)
        self.assertGreater(cosine_similarity(query, related), cosine_similarity(query, unrelated))


if __name__ == "__main__":
    unittest.main()
