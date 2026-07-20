from __future__ import annotations

import unittest

from app.physoc_sse import PhysocStreamError, collect_physoc_response, iter_message_data


class PhysocSseTests(unittest.TestCase):
    def test_iterates_default_and_explicit_message_events(self) -> None:
        lines = [
            ": heartbeat\n",
            'data: {"response":"hello",\n',
            'data: "done":false}\n',
            "\n",
            "event: status\n",
            'data: {"response":"ignored","done":true}\n',
            "\n",
            "event: message\n",
            'data: {"response":" world","done":true}\n',
            "\n",
        ]

        self.assertEqual(
            list(iter_message_data(lines)),
            [
                '{"response":"hello",\n"done":false}',
                '{"response":" world","done":true}',
            ],
        )

    def test_collects_response_until_done_with_optional_matching_model(self) -> None:
        lines = [
            'data: {"response":"hello ","done":false}\n',
            "\n",
            "event: message\n",
            'data: {"response":"world","done":true,"model":"physoc-v1"}\n',
        ]

        self.assertEqual(
            collect_physoc_response(lines, expected_model="physoc-v1"),
            "hello world",
        )

    def test_rejects_incomplete_or_invalid_message_payloads(self) -> None:
        invalid_streams = {
            "invalid JSON": ["data: not-json\n", "\n"],
            "non-object JSON": ["data: []\n", "\n"],
            "non-string response": ['data: {"response":1,"done":true}\n', "\n"],
            "non-boolean done": ['data: {"response":"x","done":1}\n', "\n"],
            "empty model": [
                'data: {"response":"x","done":true,"model":""}\n',
                "\n",
            ],
            "model mismatch": [
                'data: {"response":"x","done":true,"model":"other"}\n',
                "\n",
            ],
            "EOF before done": ['data: {"response":"x","done":false}\n', "\n"],
            "empty final answer": ['data: {"response":"","done":true}\n', "\n"],
            "no message completion": [
                "event: status\n",
                'data: {"response":"ignored","done":true}\n',
                "\n",
            ],
        }

        for label, lines in invalid_streams.items():
            with self.subTest(label=label):
                with self.assertRaises(PhysocStreamError):
                    collect_physoc_response(lines, expected_model="physoc-v1")

    def test_rejects_responses_over_the_character_limit(self) -> None:
        lines = [
            'data: {"response":"abc","done":false}\n',
            "\n",
            'data: {"response":"de","done":true}\n',
            "\n",
        ]

        with self.assertRaisesRegex(PhysocStreamError, "response size"):
            collect_physoc_response(
                lines,
                expected_model="physoc-v1",
                max_response_chars=4,
            )


if __name__ == "__main__":
    unittest.main()
