# Single Search Knowledge QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the frontend from a history/sidebar chat workspace into a no-sidebar company knowledge-base Q&A page that keeps the current-session chat bubbles.

**Architecture:** Keep the existing backend conversation APIs for now, but hide all history-oriented UI. Add a frontend `loadFreshSession()` flow that loads knowledge sources and ensures the active session starts empty. `ChatShell` becomes the single page shell with a top-right knowledge management entry, centered search intro, current-session bubbles, and the existing composer/Three.js transition.

**Tech Stack:** Vue 3 Composition API, TypeScript, Vitest, Vue Test Utils, GSAP, Three.js, existing FastAPI API contract.

---

### Task 1: Fresh Session Composable

**Files:**
- Modify: `frontend/src/composables/useChat.ts`
- Test: `frontend/src/composables/useChat.spec.ts`

- [ ] Add a failing test: when `fetchConversations()` returns messages, `loadFreshSession()` calls `createConversation()` and uses the empty returned bundle.
- [ ] Run `npm.cmd run test:run -- useChat` and confirm the new test fails because `loadFreshSession` does not exist.
- [ ] Implement `loadFreshSession()` in `useChat.ts`.
- [ ] Re-run `npm.cmd run test:run -- useChat` and confirm it passes.

### Task 2: No-Sidebar Search Shell

**Files:**
- Modify: `frontend/src/components/chat/ChatShell.vue`
- Test: `frontend/src/components/chat/__tests__/ChatShell.spec.ts`

- [ ] Add a failing test that mounts `ChatShell` with mocked `useChat()` and asserts it shows the DCAgent knowledge-search entry, shows `资料库管理`, and does not show `新建搜查` or `检索搜查档案`.
- [ ] Run `npm.cmd run test:run -- ChatShell` and confirm the test fails against the current sidebar layout.
- [ ] Remove `ConversationSidebar` from `ChatShell`, call `loadFreshSession()` on mount, add top-right knowledge management, and add the centered company-search intro.
- [ ] Re-run `npm.cmd run test:run -- ChatShell` and confirm it passes.

### Task 3: Verification

**Files:**
- No production edits expected.

- [ ] Run `npm.cmd run test:run`.
- [ ] Run `npm.cmd run build`.
- [ ] Use browser automation to verify desktop/mobile load without a left sidebar, show the centered search entry, keep the current-session bubble flow after sending, and keep the Three.js loading pulse transition.
