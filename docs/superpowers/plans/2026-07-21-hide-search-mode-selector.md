# Hide Search Mode Selector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hide the unfinished quick/deep/full search selector from every user composer while continuing to submit the existing `deep` mode internally.

**Architecture:** Keep the backend and public conversation payload unchanged. Simplify the shared Vue `ComposerBar` component to a fixed internal mode, expand its input into the removed selector space, and preserve the existing loading indicator by overlaying it within the input grid column.

**Tech Stack:** Vue 3 `<script setup>`, TypeScript, Vite 7, Vitest 4, Vue Test Utils, npm.

---

### Task 1: Specify the hidden selector and fixed payload behavior

**Files:**
- Modify: `frontend/src/components/chat/__tests__/ComposerBar.spec.ts`
- Test: `frontend/src/components/chat/__tests__/ComposerBar.spec.ts`

- [ ] **Step 1: Replace the selector-copy test and add the fixed-mode submission test**

Replace the existing `uses user-safe search mode copy instead of source tracing copy` test with these tests, leaving the loading-state test intact:

```ts
it('hides the unfinished search mode choices', () => {
  const wrapper = mount(ComposerBar, {
    props: {
      sending: false,
    },
  })

  expect(wrapper.find('[data-testid="base-select-trigger"]').exists()).toBe(false)
  expect(wrapper.text()).not.toContain('快速检索')
  expect(wrapper.text()).not.toContain('深度分析')
  expect(wrapper.text()).not.toContain('全库检索')
})

it('submits trimmed text with the fixed deep mode', async () => {
  const wrapper = mount(ComposerBar, {
    props: {
      sending: false,
    },
  })

  await wrapper.get('input').setValue('  分析现金流风险  ')
  await wrapper.get('form').trigger('submit')

  expect(wrapper.emitted('send')).toEqual([
    [{ content: '分析现金流风险', mode: 'deep' }],
  ])
  expect((wrapper.get('input').element as HTMLInputElement).value).toBe('')
})
```

- [ ] **Step 2: Run the focused component tests and verify RED**

Run from `frontend`:

```powershell
npm.cmd run test:run -- src/components/chat/__tests__/ComposerBar.spec.ts
```

Expected: the new hidden-selector test fails because `ComposerBar` still renders the `BaseSelect` trigger and the current “深度分析” label. The fixed `deep` payload test may already pass because it records the behavior that must remain unchanged.

- [ ] **Step 3: Commit only the failing behavior specification**

Do not commit a deliberately failing test. Keep the failing test uncommitted and continue directly to Task 2 so the RED and GREEN changes land together.

### Task 2: Remove the selector and preserve the composer contract

**Files:**
- Modify: `frontend/src/components/chat/ComposerBar.vue`
- Test: `frontend/src/components/chat/__tests__/ComposerBar.spec.ts`

- [ ] **Step 1: Remove selector-only imports and state**

Change the top of `ComposerBar.vue` from selector state to a fixed internal mode:

```ts
<script setup lang="ts">
import { Paperclip, SendHorizontal } from 'lucide-vue-next'
import { computed, shallowRef } from 'vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import type { ComposerMode } from '@/types/chat'
```

Keep `computed` for `isInputDisabled` and `isSubmitDisabled`, but remove:

```ts
import BaseSelect from '@/components/ui/BaseSelect.vue'
const mode = shallowRef<ComposerMode>('deep')
const modeOptions = [
  { label: '快速检索', value: 'quick' },
  { label: '深度分析', value: 'deep' },
  { label: '全库检索', value: 'source' },
]
const modeSelectLabel = '搜查模式'
const modeModel = computed({ ... })
```

Add the route-neutral fixed payload constant next to `content`:

```ts
const content = shallowRef('')
const DEFAULT_COMPOSER_MODE: ComposerMode = 'deep'
```

- [ ] **Step 2: Emit the fixed internal mode**

Update `submit()` to use the constant:

```ts
function submit() {
  const trimmed = content.value.trim()
  if (!trimmed) return
  emit('send', { content: trimmed, mode: DEFAULT_COMPOSER_MODE })
  content.value = ''
}
```

- [ ] **Step 3: Remove the selector template node**

Keep the loading block and remove the `BaseSelect` block entirely:

```vue
<div v-if="props.searching" class="composer-loading" data-testid="composer-loading" aria-live="polite">
  <span class="loading-ring" aria-hidden="true" />
  <span>{{ searchingLabel }}</span>
</div>
```

The next normal-flow element after this block remains the submit `BaseButton`. Do not add a replacement mode label, hidden select, or empty selector placeholder.

- [ ] **Step 4: Expand the input and anchor loading inside its grid column**

Change the normal composer grid to three columns:

```css
.composer {
  position: relative;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) 42px;
  /* retain the remaining existing declarations */
}
```

Remove `.mode-select` from the shared height rule:

```css
.composer-input {
  height: 36px;
}
```

Place the loading state in the same input grid cell so it does not create a fourth column:

```css
.composer-loading {
  z-index: 1;
  grid-column: 2;
  grid-row: 1;
  justify-self: end;
  pointer-events: none;
  /* retain the existing flex, spacing, color, and typography declarations */
}

.composer-wrap.searching .composer-input {
  padding-right: 150px;
}
```

Delete the selector-only rules:

```css
.mode-select :deep(.base-select-trigger) { ... }
.mode-select :deep(.base-select-trigger:hover),
.mode-select :deep(.base-select-trigger[data-state="open"]) { ... }
```

In the existing mobile media query, replace the obsolete `.mode-select { display: none; }` rule with:

```css
.composer-wrap.searching .composer-input {
  padding-right: 42px;
}

.composer-loading span:last-child {
  display: none;
}
```

This keeps the loading ring visible on narrow screens without covering most of the input.

- [ ] **Step 5: Run the focused test and verify GREEN**

Run from `frontend`:

```powershell
npm.cmd run test:run -- src/components/chat/__tests__/ComposerBar.spec.ts
```

Expected: all `ComposerBar` tests pass. The component has no selector trigger or mode labels, submission still emits `mode: "deep"`, and the loading state remains visible.

- [ ] **Step 6: Commit the component behavior**

```powershell
git add frontend/src/components/chat/ComposerBar.vue frontend/src/components/chat/__tests__/ComposerBar.spec.ts
git commit -m "feat: hide unfinished search modes"
```

### Task 3: Verify the complete user frontend

**Files:**
- Verify: `frontend/src/components/chat/ComposerBar.vue`
- Verify: `frontend/src/components/chat/__tests__/ComposerBar.spec.ts`

- [ ] **Step 1: Run the full frontend unit suite**

Run from `frontend`:

```powershell
npm.cmd run test:run
```

Expected: all frontend test files and tests pass with no failures.

- [ ] **Step 2: Run the production build**

Run from `frontend`:

```powershell
npm.cmd run build
```

Expected: `vue-tsc` and Vite finish successfully and produce the frontend build output.

- [ ] **Step 3: Check for stale user-facing mode references in the composer**

Run from the repository root:

```powershell
rg -n "快速检索|深度分析|全库检索|mode-select|BaseSelect" frontend/src/components/chat/ComposerBar.vue frontend/src/components/chat/__tests__/ComposerBar.spec.ts
```

Expected: matches exist only in the negative assertions in `ComposerBar.spec.ts`; `ComposerBar.vue` contains none of the selector labels, selector class, or `BaseSelect` import.

- [ ] **Step 4: Run repository hygiene checks**

Run from the repository root:

```powershell
git diff --check
git status --short --branch
```

Expected: `git diff --check` exits 0 and the implementation worktree is clean after the Task 2 commit.

- [ ] **Step 5: Record the deferred behavior boundary**

Do not delete `ComposerMode`, backend mode validation, admin mode labels, or historical mode values. The future quick/deep/full implementation must be a separate feature and may restore the selector without a data migration.
