# Unified Page Logo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the supplied SVG as the visible user/admin brand mark and browser favicon in both Vite applications while preserving existing brand text, layout, accessibility, and the user's local page-title edit.

**Architecture:** Each Vite app receives an independent byte-for-byte public copy of the SVG at `/favicon-logo.svg`. Existing text marks become decorative `<img>` elements, and each HTML entry declares the same public favicon path. Focused Vitest contracts protect assets, markup, and accessibility; production builds and a temporary Playwright check verify delivery and dimensions in real browsers.

**Tech Stack:** Vue 3 SFCs, TypeScript, Vitest, Vue Test Utils, Vite public assets, SVG, Python Playwright.

---

## Execution Context

Before Task 1, use `using-git-worktrees` to create:

- Worktree: `D:\project\DC-Agent\.worktrees\unified-page-logo`
- Branch: `codex/unified-page-logo`
- Base: latest committed `main`

The primary workspace contains an unrelated, uncommitted `frontend/index.html` title change to `DC智识中枢`. It is not present in the worktree and must never be copied into feature commits. In the worktree, add only the favicon link to the committed HTML title. During eventual integration, preserve the primary workspace title with a path-scoped stash before merge and restore it afterward.

## File Structure

- Create: `frontend/public/favicon-logo.svg` — public user-app logo and favicon asset.
- Create: `admin-frontend/public/favicon-logo.svg` — independent admin-app copy of the same asset.
- Create: `frontend/src/favicon.spec.ts` — user entry/asset contract.
- Create: `admin-frontend/src/favicon.spec.ts` — admin entry/asset contract.
- Modify: `frontend/index.html` — user favicon declaration only.
- Modify: `admin-frontend/index.html` — admin favicon declaration.
- Create: `frontend/src/components/chat/__tests__/CompanyBrand.spec.ts` — visible user-brand contract.
- Modify: `frontend/src/components/chat/CompanyBrand.vue` — replace the `DC` text tile with the supplied image.
- Create: `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts` — visible admin-brand contract.
- Modify: `admin-frontend/src/components/layout/AdminLayout.vue` — replace the admin `DC` tile with the supplied image.

No shared package, Vite-root change, SVG transformation, navigation change, or unrelated icon replacement is included.

### Task 1: Add Public Assets and Favicons

**Files:**

- Create: `frontend/src/favicon.spec.ts`
- Create: `admin-frontend/src/favicon.spec.ts`
- Create: `frontend/public/favicon-logo.svg`
- Create: `admin-frontend/public/favicon-logo.svg`
- Modify: `frontend/index.html`
- Modify: `admin-frontend/index.html`

- [ ] **Step 1: Write the failing user favicon contract**

Create `frontend/src/favicon.spec.ts`:

```ts
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('user application favicon', () => {
  it('declares the supplied public SVG favicon', () => {
    const html = readFileSync(join(process.cwd(), 'index.html'), 'utf8')
    const assetPath = join(process.cwd(), 'public', 'favicon-logo.svg')

    expect(html).toContain('<link rel="icon" type="image/svg+xml" href="/favicon-logo.svg" />')
    expect(existsSync(assetPath)).toBe(true)
    expect(readFileSync(assetPath, 'utf8')).toContain('viewBox="0 0 66.77 66.77"')
  })
})
```

- [ ] **Step 2: Write the failing admin favicon contract**

Create `admin-frontend/src/favicon.spec.ts`:

```ts
import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('administration application favicon', () => {
  it('declares the supplied public SVG favicon', () => {
    const html = readFileSync(join(process.cwd(), 'index.html'), 'utf8')
    const assetPath = join(process.cwd(), 'public', 'favicon-logo.svg')

    expect(html).toContain('<link rel="icon" type="image/svg+xml" href="/favicon-logo.svg" />')
    expect(existsSync(assetPath)).toBe(true)
    expect(readFileSync(assetPath, 'utf8')).toContain('viewBox="0 0 66.77 66.77"')
  })
})
```

- [ ] **Step 3: Run both contracts and verify the red state**

Run:

```powershell
cd D:\project\DC-Agent\.worktrees\unified-page-logo\frontend
npm.cmd run test:run -- src/favicon.spec.ts

cd ..\admin-frontend
npm.cmd run test:run -- src/favicon.spec.ts
```

Expected: both tests FAIL because the favicon declaration and public SVG do not exist.

- [ ] **Step 4: Add byte-for-byte SVG copies with `apply_patch`**

Read `C:\Users\56252\Desktop\favicon-logo.svg` and use `apply_patch` to add its complete, unchanged UTF-8 contents to both:

```text
frontend/public/favicon-logo.svg
admin-frontend/public/favicon-logo.svg
```

Do not edit XML attributes, gradients, colors, paths, dimensions, or whitespace inside the supplied SVG.

Verify all three files are identical:

```powershell
Get-FileHash -Algorithm SHA256 `
  'C:\Users\56252\Desktop\favicon-logo.svg', `
  'D:\project\DC-Agent\.worktrees\unified-page-logo\frontend\public\favicon-logo.svg', `
  'D:\project\DC-Agent\.worktrees\unified-page-logo\admin-frontend\public\favicon-logo.svg'
```

Expected: all three SHA-256 hashes are identical.

The expected source hash is:

```text
18C6F80D423C1B1E01E4A6E02F66E1375567FD909DE1AF55C8013430A504126D
```

- [ ] **Step 5: Add favicon declarations without changing titles**

In both HTML files, add this line immediately after the viewport meta tag:

```html
    <link rel="icon" type="image/svg+xml" href="/favicon-logo.svg" />
```

Do not edit either `<title>` line. In particular, the feature branch must not introduce the primary workspace's uncommitted `DC智识中枢` title.

- [ ] **Step 6: Run contracts and builds**

Run:

```powershell
cd D:\project\DC-Agent\.worktrees\unified-page-logo\frontend
npm.cmd run test:run -- src/favicon.spec.ts
npm.cmd run build

cd ..\admin-frontend
npm.cmd run test:run -- src/favicon.spec.ts
npm.cmd run build
```

Expected: both focused tests PASS and both production builds exit 0.

- [ ] **Step 7: Commit the public assets and favicon declarations**

Run from the worktree root:

```powershell
git add frontend/public/favicon-logo.svg frontend/src/favicon.spec.ts frontend/index.html admin-frontend/public/favicon-logo.svg admin-frontend/src/favicon.spec.ts admin-frontend/index.html
git commit -m "feat: add unified page favicons"
```

Expected: one commit containing only the two public assets, two favicon contracts, and two favicon declarations.

### Task 2: Replace the User Brand Mark

**Files:**

- Create: `frontend/src/components/chat/__tests__/CompanyBrand.spec.ts`
- Modify: `frontend/src/components/chat/CompanyBrand.vue`

- [ ] **Step 1: Write the failing user-brand test**

Create `frontend/src/components/chat/__tests__/CompanyBrand.spec.ts`:

```ts
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import CompanyBrand from '../CompanyBrand.vue'

describe('CompanyBrand', () => {
  it('renders the supplied decorative logo and keeps the product name', () => {
    const wrapper = mount(CompanyBrand)
    const logo = wrapper.get('img.company-brand__mark')

    expect(logo.attributes('src')).toBe('/favicon-logo.svg')
    expect(logo.attributes('alt')).toBe('')
    expect(logo.attributes('aria-hidden')).toBe('true')
    expect(wrapper.get('.company-brand__name').text()).toBe('DC-Agent')
  })
})
```

- [ ] **Step 2: Run the user-brand test and verify the red state**

Run from `frontend`:

```powershell
npm.cmd run test:run -- src/components/chat/__tests__/CompanyBrand.spec.ts
```

Expected: FAIL because `company-brand__mark` is currently a text `<span>`, not an image.

- [ ] **Step 3: Replace the user text mark with the image**

Replace the current mark element with:

```vue
    <img class="company-brand__mark" src="/favicon-logo.svg" alt="" aria-hidden="true">
```

Replace the base mark CSS with:

```css
.company-brand__mark {
  display: block;
  width: 38px;
  height: 38px;
  flex: 0 0 auto;
  object-fit: contain;
}
```

Keep the existing mobile width and height, but remove the mobile `border-radius` rule:

```css
  .company-brand__mark {
    width: 34px;
    height: 34px;
  }
```

Do not change the header, `DC-Agent` text, position, gap, or responsive name rules.

- [ ] **Step 4: Run focused and full user tests**

Run:

```powershell
npm.cmd run test:run -- src/components/chat/__tests__/CompanyBrand.spec.ts
npm.cmd run test:run
npm.cmd run build
```

Expected: focused test, all user tests, and production build exit 0.

- [ ] **Step 5: Commit the user brand replacement**

Run from the worktree root:

```powershell
git add frontend/src/components/chat/CompanyBrand.vue frontend/src/components/chat/__tests__/CompanyBrand.spec.ts
git commit -m "feat: replace user page brand mark"
```

### Task 3: Replace the Administration Brand Mark

**Files:**

- Create: `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts`
- Modify: `admin-frontend/src/components/layout/AdminLayout.vue`

- [ ] **Step 1: Write the failing admin-brand test**

Create `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts`:

```ts
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import AdminLayout from '../AdminLayout.vue'

describe('AdminLayout', () => {
  it('renders the supplied decorative logo and keeps the administration brand contract', () => {
    const wrapper = mount(AdminLayout, {
      global: {
        mocks: {
          $route: {
            path: '/',
            meta: { title: '管理概览' },
          },
        },
        stubs: {
          RouterLink: {
            template: '<a><slot /></a>',
          },
          RouterView: true,
        },
      },
    })
    const logo = wrapper.get('img.admin-brand__mark')

    expect(logo.attributes('src')).toBe('/favicon-logo.svg')
    expect(logo.attributes('alt')).toBe('')
    expect(logo.attributes('aria-hidden')).toBe('true')
    expect(wrapper.get('.admin-brand__copy strong').text()).toBe('DC-Agent')
    expect(wrapper.get('.admin-brand').attributes('aria-label')).toBe('返回管理概览')
  })
})
```

- [ ] **Step 2: Run the admin-brand test and verify the red state**

Run from `admin-frontend`:

```powershell
npm.cmd run test:run -- src/components/layout/__tests__/AdminLayout.spec.ts
```

Expected: FAIL because `admin-brand__mark` is currently a text `<span>`, not an image.

- [ ] **Step 3: Replace the admin text mark with the image**

Replace the current mark element with:

```vue
        <img class="admin-brand__mark" src="/favicon-logo.svg" alt="" aria-hidden="true">
```

Replace the mark CSS with:

```css
.admin-brand__mark {
  display: block;
  width: 38px;
  height: 38px;
  flex: 0 0 auto;
  object-fit: contain;
}
```

Do not change the `RouterLink`, accessible label, product text, sidebar spacing, navigation, or mobile behavior.

- [ ] **Step 4: Run focused and full admin tests**

Run:

```powershell
npm.cmd run test:run -- src/components/layout/__tests__/AdminLayout.spec.ts
npm.cmd run test:run
npm.cmd run build
```

Expected: focused test, all admin tests, and production build exit 0.

- [ ] **Step 5: Commit the admin brand replacement**

Run from the worktree root:

```powershell
git add admin-frontend/src/components/layout/AdminLayout.vue admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts
git commit -m "feat: replace admin page brand mark"
```

### Task 4: Run Final Regression and Browser Verification

**Files:**

- Verify: all files changed by Tasks 1-3
- Create temporarily, then delete: `tools/tmp_logo_browser_check.py`

- [ ] **Step 1: Run both complete test/build pipelines**

Run:

```powershell
cd D:\project\DC-Agent\.worktrees\unified-page-logo\frontend
npm.cmd run test:run
npm.cmd run build

cd ..\admin-frontend
npm.cmd run test:run
npm.cmd run build
```

Expected: both full suites and builds exit 0.

- [ ] **Step 2: Confirm the Playwright server helper usage**

Run from the worktree root:

```powershell
py 'C:\Users\56252\.codex\skills\webapp-testing\scripts\with_server.py' --help
```

Expected: help lists repeated `--server`/`--port` arguments and a trailing command.

- [ ] **Step 3: Create the temporary browser check**

Use `apply_patch` to create `tools/tmp_logo_browser_check.py`:

```python
from __future__ import annotations

from playwright.sync_api import expect, sync_playwright


def verify_logo(page, url: str, selector: str, expected_size: int) -> None:
    page.goto(url, wait_until="networkidle")
    logo = page.locator(selector).first
    expect(logo).to_be_visible(timeout=20_000)
    expect(logo).to_have_attribute("src", "/favicon-logo.svg")
    dimensions = logo.evaluate(
        "element => ({ width: element.getBoundingClientRect().width, height: element.getBoundingClientRect().height })"
    )
    if (
        abs(dimensions["width"] - expected_size) > 0.5
        or abs(dimensions["height"] - expected_size) > 0.5
    ):
        raise AssertionError(
            f"Expected {expected_size}px logo, got {dimensions}"
        )

    favicon = page.locator('link[rel="icon"][href="/favicon-logo.svg"]')
    expect(favicon).to_have_count(1)
    response = page.request.get(f"{url}/favicon-logo.svg")
    if not response.ok:
        raise AssertionError(f"Favicon request failed: {response.status}")


def main() -> None:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        verify_logo(page, "http://127.0.0.1:5177", "img.company-brand__mark", 38)
        verify_logo(page, "http://127.0.0.1:5178", "img.admin-brand__mark", 38)
        page.set_viewport_size({"width": 390, "height": 844})
        verify_logo(page, "http://127.0.0.1:5177", "img.company-brand__mark", 34)
        browser.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run real-browser verification**

Run:

```powershell
py 'C:\Users\56252\.codex\skills\webapp-testing\scripts\with_server.py' `
  --server 'tools\start_smoke_backend.cmd' --port 8015 `
  --server 'tools\start_smoke_frontend.cmd' --port 5177 `
  --server 'tools\start_smoke_admin.cmd' --port 5178 `
  --timeout 60 `
  -- py 'tools\tmp_logo_browser_check.py'
```

Expected: exit 0; both visible images have non-zero dimensions, and both favicon requests return success.

- [ ] **Step 5: Delete the temporary script and verify scope**

Use `apply_patch` to delete `tools/tmp_logo_browser_check.py`, then run:

```powershell
git diff --check main...HEAD
git status --short --branch
git log --oneline main..HEAD
git diff --name-only main...HEAD
```

Expected:

- Worktree clean.
- Exactly three implementation commits.
- Changed paths are only the ten planned asset, HTML, component, and test files.
- No `<title>` change appears in the feature branch.
- Both copied SVG hashes still match the supplied source.

## Integration Safety

If the user chooses local merge after verification, run from the primary workspace:

```powershell
git stash push -m "preserve local user page title" -- frontend/index.html
git merge --ff-only codex/unified-page-logo
git stash apply 'stash@{0}'
```

Then verify `frontend/index.html` contains both:

```html
    <link rel="icon" type="image/svg+xml" href="/favicon-logo.svg" />
    <title>DC智识中枢</title>
```

Do not run `git stash drop 'stash@{0}'` until the title is visibly restored and `git diff -- frontend/index.html` still shows only the user's title change.

After that verification succeeds, remove only this temporary stash:

```powershell
git stash drop 'stash@{0}'
```

## Completion Criteria

- Both public SVGs are byte-for-byte identical to the supplied file.
- User and administration brand marks render the SVG at their existing dimensions.
- Existing `DC-Agent` text and accessible navigation labels remain.
- Both browser tabs declare and serve `/favicon-logo.svg`.
- No `v-html`, SVG injection, shared package, or Vite-root configuration is added.
- Both full test suites, production builds, and Playwright checks pass.
- The primary workspace's uncommitted `DC智识中枢` title change remains local and intact.
