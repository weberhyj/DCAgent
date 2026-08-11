# Admin Frontend Configurable Subpath Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and serve the Vue administration frontend from a configurable URL subpath, defaulting to `/admin/`, while keeping backend requests rooted at `/api`.

**Architecture:** A pure configuration helper validates and normalizes `VITE_ADMIN_BASE_PATH` for Vite's build-time `base`. Vue Router consumes Vite's emitted `import.meta.env.BASE_URL`, and public assets use the same runtime value so routes and assets cannot drift. Nginx serves the built SPA below the matching subpath and preserves `/api/` as a separate root-level upstream.

**Tech Stack:** Vue 3, Vue Router 4, Vite 7, TypeScript 5, Vitest 4, pnpm 11, Nginx

## Global Constraints

- The administration frontend base is configured by `VITE_ADMIN_BASE_PATH` and defaults to `/admin/`.
- The normalized base must start and end with `/`; `/` remains a supported explicit root deployment.
- Reject absolute URLs, protocol-relative URLs, query strings, fragments, `.` segments, and `..` segments.
- Backend requests remain rooted at `/api`; never prepend the administration base.
- Changing the administration subpath requires rebuilding the frontend.
- Do not modify the user-facing `frontend` application or backend API routes.
- Use pnpm for all frontend dependency, test, and build commands.

---

## File map

- Create `admin-frontend/src/config/adminBasePath.ts`: pure base-path normalization and validation.
- Create `admin-frontend/src/config/adminBasePath.spec.ts`: unit coverage for accepted and rejected values.
- Modify `admin-frontend/vite.config.ts`: load the build environment and set Vite `base` from the helper.
- Modify `admin-frontend/src/router/index.ts`: expose a router factory and pass the Vite base into history.
- Create `admin-frontend/src/router/index.spec.ts`: prove named routes resolve beneath a supplied base.
- Modify `admin-frontend/index.html`: make the favicon follow Vite's base placeholder.
- Modify `admin-frontend/src/components/layout/AdminLayout.vue`: make the brand image follow `import.meta.env.BASE_URL`.
- Modify `admin-frontend/src/favicon.spec.ts`: protect the base-aware favicon declaration.
- Modify `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts`: protect the base-aware brand image source.
- Modify `admin-frontend/src/services/api.spec.ts`: explicitly protect the root-level `/api` Axios configuration.
- Create `deploy/ubuntu/admin-frontend-subpath.nginx.conf.example`: production Nginx location contract.
- Create `deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md`: build, publish, reload, smoke-test, and rollback instructions.

---

### Task 1: Validate the configurable Vite base

**Files:**
- Create: `admin-frontend/src/config/adminBasePath.ts`
- Create: `admin-frontend/src/config/adminBasePath.spec.ts`
- Modify: `admin-frontend/vite.config.ts`

**Interfaces:**
- Produces: `normalizeAdminBasePath(value?: string): string`
- Consumes: `VITE_ADMIN_BASE_PATH` from the active Vite mode environment.
- Default output: `/admin/`.

- [ ] **Step 1: Write failing normalization tests**

Create `admin-frontend/src/config/adminBasePath.spec.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { normalizeAdminBasePath } from './adminBasePath'

describe('normalizeAdminBasePath', () => {
  it('defaults to the administration subpath', () => {
    expect(normalizeAdminBasePath()).toBe('/admin/')
    expect(normalizeAdminBasePath('')).toBe('/admin/')
    expect(normalizeAdminBasePath('   ')).toBe('/admin/')
  })

  it('normalizes custom and root deployments', () => {
    expect(normalizeAdminBasePath('operations')).toBe('/operations/')
    expect(normalizeAdminBasePath('/operations')).toBe('/operations/')
    expect(normalizeAdminBasePath('/operations/')).toBe('/operations/')
    expect(normalizeAdminBasePath('/')).toBe('/')
  })

  it.each([
    'https://intranet.example/admin/',
    '//intranet.example/admin/',
    '/admin/?debug=1',
    '/admin/#overview',
    '/admin/../root/',
    '/admin/./overview/',
    '\\admin\\',
  ])('rejects unsafe base path %s', (value) => {
    expect(() => normalizeAdminBasePath(value)).toThrow('Invalid VITE_ADMIN_BASE_PATH')
  })
})
```

- [ ] **Step 2: Run the focused test and confirm it fails**

Run:

```bash
pnpm --filter dc-agent-admin-frontend test:run -- src/config/adminBasePath.spec.ts
```

Expected: FAIL because `adminBasePath.ts` does not exist.

- [ ] **Step 3: Implement the pure normalizer**

Create `admin-frontend/src/config/adminBasePath.ts`:

```ts
const DEFAULT_ADMIN_BASE_PATH = '/admin/'

function invalidBasePath(value: string): never {
  throw new Error(`Invalid VITE_ADMIN_BASE_PATH: ${JSON.stringify(value)}`)
}

export function normalizeAdminBasePath(value?: string): string {
  const candidate = value?.trim() || DEFAULT_ADMIN_BASE_PATH

  if (
    candidate.includes('\\')
    || candidate.includes('?')
    || candidate.includes('#')
    || candidate.startsWith('//')
    || /^[a-z][a-z\d+.-]*:/i.test(candidate)
  ) {
    return invalidBasePath(candidate)
  }

  if (candidate === '/') return '/'

  const segments = candidate.split('/').filter(Boolean)
  if (segments.length === 0 || segments.some(segment => segment === '.' || segment === '..')) {
    return invalidBasePath(candidate)
  }

  return `/${segments.join('/')}/`
}
```

- [ ] **Step 4: Run the normalizer tests**

Run:

```bash
pnpm --filter dc-agent-admin-frontend test:run -- src/config/adminBasePath.spec.ts
```

Expected: PASS.

- [ ] **Step 5: Wire the normalized value into Vite**

Modify `admin-frontend/vite.config.ts` to use the active mode environment:

```ts
/// <reference types="vitest/config" />
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import { normalizeAdminBasePath } from './src/config/adminBasePath'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const apiProxyTarget = env.VITE_API_PROXY_TARGET || 'http://127.0.0.1:9313'

  return {
    base: normalizeAdminBasePath(env.VITE_ADMIN_BASE_PATH),
    plugins: [vue()],
    resolve: {
      extensions: ['.ts', '.tsx', '.mjs', '.js', '.jsx', '.json'],
      alias: {
        '@': '/src',
      },
    },
    server: {
      port: 5177,
      host: true,
      allowedHosts: true,
      proxy: {
        '/api': {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
    },
    test: {
      globals: true,
      environment: 'jsdom',
      include: ['src/**/*.spec.ts'],
    },
  }
})
```

- [ ] **Step 6: Run type checking and the focused tests**

Run:

```bash
pnpm --filter dc-agent-admin-frontend test:run -- src/config/adminBasePath.spec.ts
pnpm --filter dc-agent-admin-frontend exec vue-tsc --noEmit -p tsconfig.json
```

Expected: both commands PASS.

- [ ] **Step 7: Commit the configuration slice**

```bash
git add admin-frontend/src/config/adminBasePath.ts \
  admin-frontend/src/config/adminBasePath.spec.ts \
  admin-frontend/vite.config.ts
git commit -m "feat: configure admin frontend public base"
```

---

### Task 2: Align Vue Router, public assets, and API isolation

**Files:**
- Modify: `admin-frontend/src/router/index.ts`
- Create: `admin-frontend/src/router/index.spec.ts`
- Modify: `admin-frontend/index.html`
- Modify: `admin-frontend/src/favicon.spec.ts`
- Modify: `admin-frontend/src/components/layout/AdminLayout.vue`
- Modify: `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts`
- Modify: `admin-frontend/src/services/api.spec.ts`

**Interfaces:**
- Produces: `createAdminRouter(baseUrl?: string): Router`.
- Consumes: `import.meta.env.BASE_URL` as the default router and public-asset base.
- Preserves: Axios `baseURL: '/api'`.

- [ ] **Step 1: Write a failing router-base test**

Create `admin-frontend/src/router/index.spec.ts`:

```ts
import { describe, expect, it } from 'vitest'
import { createAdminRouter } from './index'

describe('administration router base', () => {
  it('resolves named routes below the configured public base', () => {
    const router = createAdminRouter('/operations/')

    expect(router.resolve({ name: 'overview' }).href).toBe('/operations/overview')
    expect(router.resolve({ name: 'knowledge' }).href).toBe('/operations/knowledge')
    expect(router.resolve({
      name: 'knowledge-source-detail',
      params: { sourceId: 'source-1' },
    }).href).toBe('/operations/knowledge/source-1')
  })
})
```

- [ ] **Step 2: Update existing asset and API contract tests so they fail first**

In `admin-frontend/src/favicon.spec.ts`, change the favicon assertion to:

```ts
expect(html).toContain(
  '<link rel="icon" type="image/svg+xml" href="%BASE_URL%favicon-logo.svg" />',
)
```

In `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts`, replace the old root-path
expectation with a source contract that verifies the component uses Vite's base and keep the existing
rendered-asset assertion for the Vitest root base:

```ts
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

const layoutSource = readFileSync(
  fileURLToPath(new URL('../AdminLayout.vue', import.meta.url)),
  'utf8',
)

expect(layoutSource).toContain('import.meta.env.BASE_URL')
expect(logo.attributes('src')).toBe('/favicon-logo.svg')
```

In `admin-frontend/src/services/api.spec.ts`, hoist and expose the Axios create mock together with
the request mock so initialization order is explicit:

```ts
const { axiosCreateMock, httpMock } = vi.hoisted(() => {
  const requestMock = {
    delete: vi.fn(),
    get: vi.fn(),
    post: vi.fn(),
    put: vi.fn(),
  }

  return {
    axiosCreateMock: vi.fn(() => requestMock),
    httpMock: requestMock,
  }
})

vi.mock('axios', () => ({
  default: {
    create: axiosCreateMock,
  },
}))
```

Add this test:

```ts
it('keeps backend requests rooted outside the administration subpath', async () => {
  await loadApi()

  expect(axiosCreateMock).toHaveBeenCalledWith(expect.objectContaining({
    baseURL: '/api',
  }))
})
```

Use `axiosCreateMock.mockClear()` instead of `mockReset()` during cleanup so the factory implementation
remains installed for later dynamic imports.

- [ ] **Step 3: Run the focused tests and confirm they fail**

Run:

```bash
pnpm --filter dc-agent-admin-frontend test:run -- \
  src/router/index.spec.ts \
  src/favicon.spec.ts \
  src/components/layout/__tests__/AdminLayout.spec.ts \
  src/services/api.spec.ts
```

Expected: FAIL because the router factory, HTML placeholder, and base-aware brand source are missing.

- [ ] **Step 4: Add the router factory**

In `admin-frontend/src/router/index.ts`, retain the existing route records but extract them to a typed
constant and create the router through this factory:

```ts
import {
  createRouter,
  createWebHistory,
  type RouteRecordRaw,
  type Router,
} from 'vue-router'
import AdminLayout from '@/components/layout/AdminLayout.vue'

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    component: AdminLayout,
    children: [
      {
        path: '',
        redirect: { name: 'overview' },
      },
      {
        path: 'overview',
        name: 'overview',
        component: () => import('@/views/OverviewPage.vue'),
        meta: { title: '管理概览' },
      },
      {
        path: 'knowledge',
        name: 'knowledge',
        component: () => import('@/views/KnowledgeManagementPage.vue'),
        meta: { title: '知识库管理' },
      },
      {
        path: 'knowledge/:sourceId',
        name: 'knowledge-source-detail',
        component: () => import('@/views/KnowledgeSourceDetailPage.vue'),
        meta: { title: '资料解析详情' },
      },
      {
        path: 'agent-runs',
        name: 'agent-runs',
        component: () => import('@/views/AgentAuditPage.vue'),
        meta: { title: 'Agent 执行审计' },
      },
      {
        path: 'quality',
        component: () => import('@/views/QualityModuleLayout.vue'),
        children: [
          {
            path: '',
            redirect: { name: 'quality-cases' },
          },
          {
            path: 'cases',
            name: 'quality-cases',
            component: () => import('@/views/QualityCasesPage.vue'),
            meta: { title: '质量评测' },
          },
          {
            path: 'reports',
            name: 'quality-reports',
            component: () => import('@/views/QualityReportsPage.vue'),
            meta: { title: '评测报告' },
          },
          {
            path: 'reports/:batchId',
            name: 'quality-report-detail',
            component: () => import('@/views/QualityReportDetailPage.vue'),
            meta: { title: '评测报告详情' },
          },
        ],
      },
    ],
  },
]

export function createAdminRouter(baseUrl = import.meta.env.BASE_URL): Router {
  return createRouter({
    history: createWebHistory(baseUrl),
    routes,
  })
}

export default createAdminRouter()
```

Do not rename the listed route names, paths, components, or metadata while moving them into the
typed constant.

- [ ] **Step 5: Make public assets base-aware**

In `admin-frontend/index.html`, use Vite's HTML placeholder:

```html
<link rel="icon" type="image/svg+xml" href="%BASE_URL%favicon-logo.svg" />
```

In the `<script setup>` section of `admin-frontend/src/components/layout/AdminLayout.vue`, add:

```ts
const brandLogoUrl = `${import.meta.env.BASE_URL}favicon-logo.svg`
```

Use it in the template:

```vue
<img class="admin-brand__mark" :src="brandLogoUrl" alt="" aria-hidden="true">
```

- [ ] **Step 6: Run the focused tests**

Run:

```bash
pnpm --filter dc-agent-admin-frontend test:run -- \
  src/router/index.spec.ts \
  src/favicon.spec.ts \
  src/components/layout/__tests__/AdminLayout.spec.ts \
  src/services/api.spec.ts
```

Expected: PASS.

- [ ] **Step 7: Run the complete administration test suite and type check**

Run:

```bash
pnpm --filter dc-agent-admin-frontend test:run
pnpm --filter dc-agent-admin-frontend exec vue-tsc --noEmit -p tsconfig.json
```

Expected: PASS with no route regressions and no TypeScript errors.

- [ ] **Step 8: Commit the runtime alignment slice**

```bash
git add admin-frontend/src/router/index.ts \
  admin-frontend/src/router/index.spec.ts \
  admin-frontend/index.html \
  admin-frontend/src/favicon.spec.ts \
  admin-frontend/src/components/layout/AdminLayout.vue \
  admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts \
  admin-frontend/src/services/api.spec.ts
git commit -m "feat: align admin routes and assets with public base"
```

---

### Task 3: Document Nginx deployment and verify production builds

**Files:**
- Create: `deploy/ubuntu/admin-frontend-subpath.nginx.conf.example`
- Create: `deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md`

**Interfaces:**
- Consumes: a build generated with `VITE_ADMIN_BASE_PATH=/admin/`.
- Produces: an Nginx contract serving files from `/var/www/dcagent/admin/` and proxying `/api/` to `127.0.0.1:8000`.

- [ ] **Step 1: Add the Nginx example**

Create `deploy/ubuntu/admin-frontend-subpath.nginx.conf.example`:

```nginx
location = /admin {
    return 301 /admin/;
}

location /admin/ {
    root /var/www/dcagent;
    try_files $uri $uri/ /admin/index.html;
}

location /api/ {
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_pass http://127.0.0.1:8000;
}
```

- [ ] **Step 2: Add the Ubuntu deployment runbook**

Create `deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md` with these exact operational stages:

1. Verify Node `>=20.19.0`, pnpm `11.16.0`, and a clean dependency install using
   `pnpm install --frozen-lockfile`.
2. Build from the repository root with
   `VITE_ADMIN_BASE_PATH=/admin/ pnpm --filter dc-agent-admin-frontend build`.
3. Publish `admin-frontend/dist/` into a newly staged directory under `/var/www/dcagent/`, then
   atomically rename the stage to `/var/www/dcagent/admin/` during an approved maintenance window.
4. Merge the reviewed Nginx example into the active server block.
5. Run `sudo nginx -t` before `sudo systemctl reload nginx`.
6. Probe `/admin/`, `/admin/knowledge`, `/admin/favicon-logo.svg`, and `/api/version`.
7. Confirm HTML references `/admin/assets/` and API requests never use `/admin/api/`.
8. Roll back by restoring the previous administration static directory and reloading Nginx; do not
   change backend or ClickHouse services.

The runbook must state that the Nginx location and `VITE_ADMIN_BASE_PATH` must match exactly and that
changing the subpath requires rebuilding.

- [ ] **Step 3: Build with the default `/admin/` base**

Run:

```bash
pnpm --filter dc-agent-admin-frontend build
rg -n '/admin/assets/|/admin/favicon-logo.svg' admin-frontend/dist/index.html
```

Expected: build PASS; generated HTML contains `/admin/assets/` and `/admin/favicon-logo.svg`.

- [ ] **Step 4: Build with a custom base**

On PowerShell:

```powershell
$env:VITE_ADMIN_BASE_PATH='/operations/'
pnpm --filter dc-agent-admin-frontend build
Remove-Item Env:VITE_ADMIN_BASE_PATH
rg -n '/operations/assets/|/operations/favicon-logo.svg' admin-frontend/dist/index.html
```

Expected: build PASS; generated HTML contains `/operations/assets/` and
`/operations/favicon-logo.svg`.

- [ ] **Step 5: Re-run all administration checks after the custom build**

Run:

```bash
pnpm --filter dc-agent-admin-frontend test:run
pnpm --filter dc-agent-admin-frontend exec vue-tsc --noEmit -p tsconfig.json
git diff --check
```

Expected: all commands PASS. `admin-frontend/dist/` remains ignored and must not be committed.

- [ ] **Step 6: Commit deployment documentation**

```bash
git add deploy/ubuntu/admin-frontend-subpath.nginx.conf.example \
  deploy/ubuntu/ADMIN_FRONTEND_SUBPATH.md
git commit -m "docs: add admin subpath deployment runbook"
```

---

### Task 4: Final review and handoff

**Files:**
- Review all files changed by Tasks 1-3.

**Interfaces:**
- Consumes: the complete implementation and deployment documentation.
- Produces: a reviewed branch ready to integrate and deploy.

- [ ] **Step 1: Review the complete diff against the design**

Run:

```bash
git diff origin/main...HEAD -- admin-frontend deploy/ubuntu
```

Verify explicitly:

- no user-facing `frontend` files changed;
- Axios still uses root-level `/api`;
- the default build base is `/admin/`;
- custom paths require only `VITE_ADMIN_BASE_PATH` and an aligned Nginx location;
- no `.env`, secret, `dist/`, or dependency output is tracked.

- [ ] **Step 2: Run the final verification commands**

```bash
pnpm --filter dc-agent-admin-frontend test:run
pnpm --filter dc-agent-admin-frontend build
git diff --check origin/main...HEAD
git status --short
```

Expected: tests and build PASS; diff check reports no errors; status contains no generated output.

- [ ] **Step 3: Perform code-quality review**

Review configuration safety, Vue Router behavior, asset paths, API isolation, test quality, Nginx
fallback behavior, and rollback completeness. Fix only findings within the approved subpath scope,
then repeat Step 2.

- [ ] **Step 4: Prepare deployment handoff**

Report:

- the configured build command;
- the static publish directory;
- the Nginx configuration path;
- the four smoke-test URLs;
- the final commit IDs;
- any live Nginx verification that could not be run outside the intranet host.
