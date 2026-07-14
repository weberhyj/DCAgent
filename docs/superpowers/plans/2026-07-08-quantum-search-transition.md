# Quantum Search Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a loading transition when the centered small composer sends a search request and before the large chat panel expands.

**Architecture:** `ChatShell` derives a transition state from `launchFromEmpty`, `sending`, and `isEmptyConversation`. `ComposerBar` receives a `searching` prop to render a focused loading state. `QuantumNetworkBackground` receives an `autoPulse` prop and owns the repeated pulse loop internally.

**Tech Stack:** Vue 3 Composition API, TypeScript, GSAP, Three.js, Vitest, Vue Test Utils.

---

### Task 1: Composer Loading State

**Files:**
- Test: `frontend/src/components/chat/__tests__/ComposerBar.spec.ts`
- Modify: `frontend/src/components/chat/ComposerBar.vue`

- [ ] Write a failing component test that mounts `ComposerBar` with `variant="center"`, `sending=true`, and `searching=true`, then asserts it shows `资料库搜查中`, disables the input, disables the submit button, and renders a loading marker.
- [ ] Run `npm.cmd run test:run -- ComposerBar` and confirm the new test fails before implementation.
- [ ] Add the `searching` prop, loading label, spinner node, and disabled input handling.
- [ ] Re-run `npm.cmd run test:run -- ComposerBar` and confirm the test passes.

### Task 2: Transition State And Auto Pulse

**Files:**
- Modify: `frontend/src/components/chat/ChatShell.vue`
- Modify: `frontend/src/components/chat/QuantumNetworkBackground.vue`

- [ ] Add `isSearchTransitioning = computed(() => launchFromEmpty.value && sending.value && isEmptyConversation.value)` in `ChatShell`.
- [ ] Pass `:auto-pulse="isSearchTransitioning"` to `QuantumNetworkBackground` and `:searching="isSearchTransitioning"` to the centered `ComposerBar`.
- [ ] Add an `autoPulse` prop to `QuantumNetworkBackground`.
- [ ] Implement an interval-owned auto-pulse loop that fires while `autoPulse` is true, stops on false/unmount, and triggers pulses near the current visible network center.

### Task 3: Verification

**Files:**
- No production edits expected.

- [ ] Run `npm.cmd run test:run`.
- [ ] Run `npm.cmd run build`.
- [ ] Use browser automation to send from the centered composer and verify the screenshot sequence shows loading before expansion and repeated background motion.
