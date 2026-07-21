# Hide Attachment Entry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide the user-facing attachment entry while retaining its code for later reactivation and keeping the composer layout compact.

**Architecture:** `ComposerBar.vue` will keep the existing attachment button implementation behind a local `isAttachmentEntryVisible = false` switch. The form will use a two-column grid while hidden and a three-column modifier when the switch is enabled, so input/loading and send controls remain aligned without an empty attachment column.

**Tech Stack:** Vue 3 Composition API, TypeScript, Vue Test Utils, Vitest, Vite.

---

### Task 1: Add regression coverage for the hidden attachment entry

**Files:**
- Modify: `frontend/src/components/chat/__tests__/ComposerBar.spec.ts`

- [ ] **Step 1: Add a failing visibility test**

Mount `ComposerBar` with `sending: false` and assert that the retained attachment button is not rendered by default:

```ts
it('hides the attachment entry by default while retaining the composer', () => {
  const wrapper = mount(ComposerBar, {
    props: {
      sending: false,
    },
  })

  expect(wrapper.find('.tool-button').exists()).toBe(false)
  expect(wrapper.find('button[aria-label="添加附件"]').exists()).toBe(false)
  expect(wrapper.find('input').exists()).toBe(true)
  expect(wrapper.find('button[type="submit"]').exists()).toBe(true)
  expect(composerBarSource).toContain('const isAttachmentEntryVisible = false')
  expect(composerBarSource).toContain('v-if="isAttachmentEntryVisible"')
  expect(composerBarSource).toContain('<Paperclip :size="21" />')
})
```

- [ ] **Step 2: Add a failing two-column layout contract**

Replace the existing `keeps composer controls in their three-column layout while searching` test. Read `ComposerBar.vue?raw` using the existing `getStyleRule` helper and assert that the default composer, input/loading, and send button use the hidden-entry columns:

```ts
it('uses a compact two-column layout while the attachment entry is hidden', () => {
  const composerRule = getStyleRule(composerBarSource, '.composer')
  const inputRule = getStyleRule(composerBarSource, '.composer-input')
  const loadingRule = getStyleRule(composerBarSource, '.composer-loading')
  const sendRule = getStyleRule(composerBarSource, '.send-button')

  expect(composerRule).toContain('grid-template-columns: minmax(0, 1fr) 42px')
  expect(inputRule).toContain('grid-column: 1')
  expect(inputRule).toContain('grid-row: 1')
  expect(loadingRule).toContain('grid-column: 1')
  expect(loadingRule).toContain('grid-row: 1')
  expect(sendRule).toContain('grid-column: 2')
  expect(sendRule).toContain('grid-row: 1')
})
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run from `frontend/`:

```bash
npm.cmd exec vitest run src/components/chat/__tests__/ComposerBar.spec.ts
```

Expected: the new visibility assertion fails because the attachment button is currently rendered, and the new layout assertion fails because the component still uses the three-column grid.

### Task 2: Implement the visibility switch and compact layout

**Files:**
- Modify: `frontend/src/components/chat/ComposerBar.vue`

- [ ] **Step 1: Add the local visibility switch without deleting attachment code**

Keep the `Paperclip` import, `attachmentLabel`, and existing `BaseButton` markup. Add this constant in `<script setup>`:

```ts
const isAttachmentEntryVisible = false
```

Render the retained button conditionally and add a form class for the future visible layout:

```vue
<form class="composer" :class="{ 'attachments-visible': isAttachmentEntryVisible }" @submit.prevent="submit">
  <BaseButton
    v-if="isAttachmentEntryVisible"
    type="button"
    class="tool-button"
    variant="ghost"
    size="icon"
    :disabled="isInputDisabled"
    :aria-label="attachmentLabel"
  >
    <Paperclip :size="21" />
  </BaseButton>
```

- [ ] **Step 2: Make the hidden state the compact base layout**

Change the base grid and control columns to:

```css
.composer {
  grid-template-columns: minmax(0, 1fr) 42px;
}

.composer-input {
  grid-column: 1;
  grid-row: 1;
}

.composer-loading {
  grid-column: 1;
  grid-row: 1;
}

.send-button {
  grid-column: 2;
  grid-row: 1;
}
```

Add a modifier that restores the retained attachment layout when the switch is later enabled:

```css
.composer.attachments-visible {
  grid-template-columns: auto minmax(0, 1fr) 42px;
}

.composer.attachments-visible .tool-button {
  grid-column: 1;
  grid-row: 1;
}

.composer.attachments-visible .composer-input,
.composer.attachments-visible .composer-loading {
  grid-column: 2;
}

.composer.attachments-visible .send-button {
  grid-column: 3;
}
```

At the mobile breakpoint, use `45px` for the compact send column and add the matching `.attachments-visible` override with `auto minmax(0, 1fr) 45px`, preserving the current mobile behavior when re-enabled.

- [ ] **Step 3: Run the focused tests and verify GREEN**

Run:

```bash
npm.cmd exec vitest run src/components/chat/__tests__/ComposerBar.spec.ts
```

Expected: all `ComposerBar` tests pass, including the existing fixed-`deep` payload, loading accessibility, and layout tests.

### Task 3: Run the complete validation suite

**Files:**
- No additional files.

- [ ] **Step 1: Run all frontend tests**

```bash
npm.cmd run test:run
```

Expected: all test files and tests pass with zero failures.

- [ ] **Step 2: Run type checking and production build**

```bash
npm.cmd run build
```

Expected: `vue-tsc --noEmit` and `vite build` exit successfully. Existing large-chunk warnings may remain.

- [ ] **Step 3: Check formatting and scope**

```bash
git diff --check
git status --short --branch
```

Expected: no whitespace errors; only the intended spec, plan, component, and test changes are present.

- [ ] **Step 4: Commit the implementation**

```bash
git add frontend/src/components/chat/ComposerBar.vue frontend/src/components/chat/__tests__/ComposerBar.spec.ts docs/superpowers/plans/2026-07-21-hide-attachment-entry.md
git commit -m "feat: hide attachment entry"
```
