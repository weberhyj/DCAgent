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

    def test_empty_event_fields_fall_back_to_message(self) -> None:
        for event_line in ("event:\r\n", "event\r\n"):
            with self.subTest(event_line=event_line):
                self.assertEqual(
                    collect_physoc_response(
                        [
                            event_line,
                            'data: {"response":"ok","done":true}\r\n',
                            "\r\n",
                        ],
                        expected_model="physoc-v1",
                    ),
                    "ok",
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

    def test_rejects_nonstandard_json_constants(self) -> None:
        for constant in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(constant=constant):
                with self.assertRaises(PhysocStreamError):
                    collect_physoc_response(
                        [
                            f'data: {{"response":"ok","done":true,"extra":{constant}}}\n',
                            "\n",
                        ],
                        expected_model="physoc-v1",
                    )

    def test_rejects_oversized_event_metadata(self) -> None:
        lines = [
            'data: {"response":"ok","done":true,"extra":"' + ("x" * 128) + '"}\n',
            "\n",
        ]

        with self.assertRaisesRegex(PhysocStreamError, "Physoc event size exceeded"):
            collect_physoc_response(
                lines,
                expected_model="physoc-v1",
                max_event_chars=64,
            )

    def test_rejects_event_limit_exceeded_across_data_lines(self) -> None:
        first_data = '{"response":"ok",'
        second_data = '"done":true}'
        lines = [f"data: {first_data}\n", f"data: {second_data}\n", "\n"]

        with self.assertRaisesRegex(PhysocStreamError, "Physoc event size exceeded"):
            collect_physoc_response(
                lines,
                expected_model="physoc-v1",
                max_event_chars=len(first_data) + len(second_data),
            )

    def test_event_limit_resets_after_each_record(self) -> None:
        lines = [
            'data: {"response":"a","done":false}\n',
            "\n",
            'data: {"response":"b","done":true}\n',
            "\n",
        ]

        self.assertEqual(
            collect_physoc_response(
                lines,
                expected_model="physoc-v1",
                max_event_chars=34,
            ),
            "ab",
        )

    def test_rejects_oversized_single_response_before_accumulation(self) -> None:
        response = "x" * 128

        with self.assertRaisesRegex(PhysocStreamError, "response size"):
            collect_physoc_response(
                [f'data: {{"response":"{response}","done":true}}\n', "\n"],
                expected_model="physoc-v1",
                max_event_chars=256,
                max_response_chars=64,
            )

    def test_response_limit_accepts_exact_size_and_rejects_one_more(self) -> None:
        exact_lines = [
            'data: {"response":"ab","done":false}\n',
            "\n",
            'data: {"response":"cd","done":true}\n',
            "\n",
        ]
        oversized_lines = [
            'data: {"response":"ab","done":false}\n',
            "\n",
            'data: {"response":"cde","done":true}\n',
            "\n",
        ]

        self.assertEqual(
            collect_physoc_response(
                exact_lines,
                expected_model="physoc-v1",
                max_response_chars=4,
            ),
            "abcd",
        )
        with self.assertRaisesRegex(PhysocStreamError, "response size"):
            collect_physoc_response(
                oversized_lines,
                expected_model="physoc-v1",
                max_response_chars=4,
            )

    def test_done_response_does_not_consume_more_input(self) -> None:
        def lines():
            yield 'data: {"response":"ok","done":true}\n'
            yield "\n"
            raise AssertionError("input consumed after done")

        self.assertEqual(
            collect_physoc_response(
                lines(),
                expected_model="physoc-v1",
                max_event_chars=64,
            ),
            "ok",
        )

    def test_strips_one_bom_only_from_the_first_line(self) -> None:
        self.assertEqual(
            collect_physoc_response(
                [
                    '\ufeffdata: {"response":"ok","done":true}\n',
                    "\n",
                ],
                expected_model="physoc-v1",
            ),
            "ok",
        )

        invalid_bom_streams = [
            [
                '\ufeff\ufeffdata: {"response":"bad","done":true}\n',
                "\n",
            ],
            [
                ": first line\n",
                '\ufeffdata: {"response":"bad","done":true}\n',
                "\n",
            ],
        ]
        for lines in invalid_bom_streams:
            with self.subTest(lines=lines):
                with self.assertRaises(PhysocStreamError):
                    collect_physoc_response(lines, expected_model="physoc-v1")

    def test_rejects_invalid_event_and_response_limits(self) -> None:
        invalid_limits = (0, -1, True, 1.5, "10")

        def unreadable_lines():
            raise AssertionError("invalid limits must be rejected before input is consumed")
            yield ""

        for limit in invalid_limits:
            with self.subTest(kind="event", limit=limit):
                with self.assertRaises(PhysocStreamError):
                    list(iter_message_data(unreadable_lines(), max_event_chars=limit))
            with self.subTest(kind="response", limit=limit):
                with self.assertRaises(PhysocStreamError):
                    collect_physoc_response(
                        unreadable_lines(),
                        expected_model="physoc-v1",
                        max_response_chars=limit,
                    )


if __name__ == "__main__":
    unittest.main()
