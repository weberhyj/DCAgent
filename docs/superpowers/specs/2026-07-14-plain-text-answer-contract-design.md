# Plain-Text Answer Contract Design

## Goal

Prevent model formatting markers such as `- **数联**` from appearing literally in the user chat while keeping assistant answers safe, readable, and compatible with the current API and frontend.

## Root Cause

The model may return Markdown, but the answer pipeline currently removes only inline citation markers. The user frontend deliberately renders `paragraph.text` through Vue text interpolation, so Markdown is displayed as literal characters instead of being interpreted.

This is an output-contract mismatch between model generation and frontend rendering. It is not a character-encoding or CSS problem.

## Decision

Assistant answers use a plain-text contract end to end:

- The system prompt and request prompt explicitly require plain text and forbid Markdown and HTML formatting.
- The backend normalizes common residual Markdown before constructing the response message.
- The frontend continues to render text with Vue interpolation.
- The frontend does not use `v-html` and does not add a Markdown renderer.

This keeps the browser surface safe by default and avoids introducing HTML sanitization or Markdown-rendering dependencies for a formatting issue that can be handled at the answer boundary.

## Components and Data Flow

`backend/app/llm.py` owns the generation policy. Both the system prompt and the per-request rules tell the model to return readable plain text without headings, list markers, emphasis markers, code fences, links, or HTML tags.

`backend/app/answer_text.py` owns deterministic answer cleanup. A focused normalization function processes external model text before it is stored in `ResponseParagraphModel`. It also retains the existing removal of `[1]`-style inline citation markers.

`frontend/src/components/chat/ChatTranscript.vue` remains unchanged. It receives normalized text and renders it through `{{ paragraph.text }}`. Newlines remain available for readable paragraph separation, but no content is interpreted as executable markup.

The resulting flow is:

`prompt rules -> model output -> backend plain-text normalization -> API response -> safe Vue text interpolation`

## Normalization Rules

The first implementation is deliberately limited to the formatting pattern observed in the defect:

| Input pattern | Output rule |
| --- | --- |
| `- **数联**：数据要素联通` | `数联：数据要素联通` |
| Three consecutive lines starting with `- **...**` | Keep three lines in the same order; remove each list prefix and paired `**` markers |
| `光联：城市光网支撑。[1]` | `光联：城市光网支撑。` |
| `城市一张网 2.0` | Preserve exactly |
| `2 * 3` | Preserve exactly |
| `user_name` or `__init__` | Preserve exactly |
| `- 5°C` | Preserve exactly because it is not the confirmed formatted-list pattern |

The normalizer therefore:

- Removes a leading unordered-list prefix only when that line immediately contains paired `**...**` Markdown emphasis.
- Removes paired `**` markers from non-empty emphasized content.
- Removes `[1]`, `[2]`, and similar inline citation markers using the existing behavior.
- Preserves line breaks, answer wording, ordinary symbols, and outer trimming behavior.

It does not strip underscore emphasis, headings, blockquotes, links, inline code, code fences, HTML tags, or arbitrary punctuation in this phase. The prompt contract discourages those formats, and Vue still displays any unexpected pattern as inert text. Broader normalization should be added only with a concrete failing example and a protection test for legitimate text.

## Error Handling and Safety

Normalization is local and deterministic; it does not make another model or network call. Existing provider timeout, HTTP, and malformed-response handling remains unchanged.

If an unfamiliar formatting pattern is returned, the frontend still treats it as inert text. This may leave a visible marker, but it cannot become executable HTML through the answer-rendering path.

## Testing

Backend unit tests cover:

- A response such as `- **数联**：数据要素联通` keeps the Chinese wording but removes the bullet and emphasis markers.
- Multiple list lines retain their order and line separation.
- Inline citation markers are removed together with residual formatting.
- Plain text and meaningful symbols, including `城市一张网 2.0`, `2 * 3`, `user_name`, `__init__`, and `- 5°C`, are unchanged.
- Prompt construction explicitly requires plain text and forbids Markdown and HTML.
- The OpenAI-compatible provider returns normalized paragraph text.

Relevant backend regression tests and the frontend build run after implementation. No frontend rendering behavior is changed, so the existing safe-interpolation contract remains the acceptance condition.

## Scope

In scope:

- Plain-text prompt rules.
- Focused backend normalization of model answers.
- Unit and provider-level regression coverage.

Out of scope:

- Rendering rich Markdown in the user frontend.
- Adding `v-html`, a Markdown package, or an HTML sanitizer.
- Changing citation metadata attached to response paragraphs.
- Reformatting historical answers already persisted outside the current response path.
- Unrelated chat UI or model-provider changes.

## Success Criteria

New assistant answers no longer show common Markdown formatting symbols such as `**` or leading list markers. The wording remains intact, normal plain-text content is not damaged, inline citations remain hidden from the answer body, and the frontend continues to render all answer content as safe text.
