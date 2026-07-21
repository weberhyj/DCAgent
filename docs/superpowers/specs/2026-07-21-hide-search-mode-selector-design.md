# Hide Search Mode Selector Design

## Goal

Temporarily hide the user-facing “快速检索 / 深度分析 / 全库检索” selector while keeping the existing conversation workflow operational and preserving backend compatibility for later mode work.

## Scope

Only the user frontend composer changes. Both the centered first-search composer and the docked/embedded composer use `ComposerBar.vue`, so one component change covers every user-facing entry point.

The backend mode schema, API payload, Agent behavior, admin audit labels, existing records, and `ComposerMode` TypeScript union remain unchanged.

## Behavior

- The mode dropdown is not rendered in the user interface.
- Every new user message continues to send `mode: "deep"`, matching the component’s current default and avoiding a backend behavior change.
- Search submission, disabled states, attachment button, loading indicator, answer display, and error handling remain unchanged.
- Historical conversations and admin audit records may continue to display any existing `quick`, `deep`, or `source` value.

## Component Changes

`frontend/src/components/chat/ComposerBar.vue` will:

- remove the `BaseSelect` import and template node;
- remove `mode`, `modeOptions`, `modeModel`, and the selector accessibility label;
- retain the `ComposerMode` type and define a fixed internal default mode of `deep`;
- emit `{ content, mode: "deep" }` from `submit()`;
- change the desktop composer grid from four columns to three so the text input expands into the removed selector space;
- keep the loading indicator visible without restoring a selector-sized permanent column;
- remove selector-only CSS rules.

## Testing

`frontend/src/components/chat/__tests__/ComposerBar.spec.ts` will be updated test-first to assert:

1. the selector trigger and all three mode labels are absent;
2. submitting trimmed text emits one `send` event with `mode: "deep"`;
3. the existing loading-state behavior remains visible and disables input/submission.

The complete frontend test suite and production build must pass.

## Reversibility

This is intentionally a presentation-layer change. Restoring selectable search modes later requires reintroducing the selector and mode state in `ComposerBar.vue`; no backend or persisted-data migration is required.

## Non-Goals

- deleting mode values from frontend or backend types;
- changing retrieval depth or Agent execution;
- changing admin audit mode labels;
- introducing a new “standard search” API mode;
- removing historical mode data;
- implementing the future quick/deep/full search behavior.
