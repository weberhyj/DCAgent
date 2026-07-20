from __future__ import annotations

import unittest

from app.physoc_sse import (
    PhysocStreamError,
    collect_physoc_response,
    iter_message_data,
    iter_sse_lines,
)


class PhysocSseTests(unittest.TestCase):
    def test_iter_sse_lines_splits_raw_chunks_and_preserves_split_utf8(self) -> None:
        chunks = [b"data: " + bytes([0xE4]), bytes([0xBD, 0xA0]) + b"\n", b"\n"]

        self.assertEqual(list(iter_sse_lines(chunks)), ["data: 你\n", "\n"])

    def test_iter_sse_lines_accepts_cr_lf_and_split_crlf_terminators(self) -> None:
        chunks = [b"one\rtwo\r", b"\nthree\n", b"four\r"]

        self.assertEqual(
            list(iter_sse_lines(chunks)),
            ["one\n", "two\n", "three\n", "four\n"],
        )

    def test_collect_accepts_bare_cr_sse_records(self) -> None:
        chunks = [
            b'data: {"response":"hello ","done":false}\r\r',
            b'data: {"response":"world","done":true}\r\r',
        ]

        self.assertEqual(
            collect_physoc_response(
                iter_sse_lines(chunks),
                expected_model="physoc-v1",
            ),
            "hello world",
        )

    def test_iter_sse_lines_rejects_invalid_utf8(self) -> None:
        with self.assertRaisesRegex(PhysocStreamError, "UTF-8"):
            list(iter_sse_lines([b"data: \xff\n"]))

    def test_iter_sse_lines_rejects_unterminated_and_complete_oversized_lines(self) -> None:
        for chunks in ([b"x" * 5], [b"x" * 5 + b"\n"]):
            with self.subTest(chunks=chunks):
                with self.assertRaisesRegex(PhysocStreamError, "line"):
                    list(iter_sse_lines(chunks, max_line_bytes=4))

    def test_iter_sse_lines_bounds_all_sse_field_types(self) -> None:
        for prefix in (b":", b"event: ", b"unknown: "):
            with self.subTest(prefix=prefix):
                with self.assertRaisesRegex(PhysocStreamError, "line"):
                    list(iter_sse_lines([prefix + b"x" * 8 + b"\n"], max_line_bytes=8))

    def test_iter_sse_lines_stops_consuming_after_unterminated_line_limit(self) -> None:
        def chunks():
            yield b"abc"
            yield b"de"
            raise AssertionError("raw iterator consumed after the line limit was exceeded")

        with self.assertRaisesRegex(PhysocStreamError, "line"):
            list(iter_sse_lines(chunks(), max_line_bytes=4))

    def test_iter_sse_lines_rejects_stream_over_total_byte_limit_before_append(self) -> None:
        with self.assertRaisesRegex(PhysocStreamError, "stream"):
            list(iter_sse_lines([b"abc", b"de"], max_stream_bytes=4))

    def test_iter_sse_lines_validates_positive_integer_limits_before_consuming(self) -> None:
        def unreadable_chunks():
            raise AssertionError("invalid limits must be rejected before input is consumed")
            yield b""

        for limit_name in ("max_line_bytes", "max_stream_bytes"):
            for limit in (0, -1, True, 1.5, "10"):
                with self.subTest(limit_name=limit_name, limit=limit):
                    with self.assertRaises(PhysocStreamError):
                        list(iter_sse_lines(unreadable_chunks(), **{limit_name: limit}))

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

    def test_collect_rejects_duplicate_json_keys(self) -> None:
        for duplicate in ("response", "done", "model"):
            with self.subTest(duplicate=duplicate):
                payload = (
                    '{"response":"ok","done":true,"model":"physoc-v1",'
                    f'"{duplicate}":' + ('"other"' if duplicate != "done" else "false") + "}"
                )
                with self.assertRaises(PhysocStreamError):
                    collect_physoc_response(
                        [f"data: {payload}\n", "\n"], expected_model="physoc-v1"
                    )

    def test_collect_rejects_more_than_max_events_and_does_not_accumulate_empty_response(
        self,
    ) -> None:
        lines = []
        for _ in range(3):
            lines.extend(['data: {"response":"","done":false}\n', "\n"])
        lines.extend(['data: {"response":"ok","done":true}\n', "\n"])

        with self.assertRaisesRegex(PhysocStreamError, "events"):
            collect_physoc_response(lines, expected_model="physoc-v1", max_events=3)

    def test_collect_bounds_an_unending_stream_of_empty_response_events(self) -> None:
        consumed = 0

        def lines():
            nonlocal consumed
            while True:
                consumed += 1
                yield 'data: {"response":"","done":false}\n'
                yield "\n"

        with self.assertRaisesRegex(PhysocStreamError, "events"):
            collect_physoc_response(lines(), expected_model="physoc-v1", max_events=3)
        self.assertEqual(consumed, 4)

    def test_collect_accepts_exact_max_events_and_done_empty_fragments(self) -> None:
        lines = []
        for _ in range(3):
            lines.extend(['data: {"response":"","done":false}\n', "\n"])
        lines.extend(['data: {"response":"ok","done":true}\n', "\n"])

        self.assertEqual(
            collect_physoc_response(lines, expected_model="physoc-v1", max_events=4),
            "ok",
        )

    def test_collect_rejects_invalid_max_events_before_consuming(self) -> None:
        def unreadable_lines():
            raise AssertionError("invalid max_events must be rejected before input is consumed")
            yield ""

        for limit in (0, -1, True, 1.5, "10"):
            with self.subTest(limit=limit):
                with self.assertRaises(PhysocStreamError):
                    collect_physoc_response(
                        unreadable_lines(),
                        expected_model="physoc-v1",
                        max_events=limit,
                    )


if __name__ == "__main__":
    unittest.main()
