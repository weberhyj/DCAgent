# Frontend and Backend Version Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在用户端左下角显示用户端版本，并在管理端服务状态区域显示后端版本，同时保持三个应用的版本源彼此独立。

**Architecture:** 用户端由 Vite 在构建时从 `frontend/package.json` 注入只读常量，`ChatShell` 直接渲染该构建产物版本。后端通过无外部依赖的 `/api/version` 暴露 `app.__version__`，管理端通过既有 Axios 服务和一个小型 composable 获取并降级显示该值。

**Tech Stack:** Python 3.12、FastAPI、Pydantic、Vue 3、Composition API、`<script setup lang="ts">`、Vite、Axios、Vitest、Vue Test Utils

---

## 文件结构

- Modify: `backend/app/schemas.py` — 定义固定的后端版本响应模型。
- Modify: `backend/app/routes.py` — 提供 `GET /api/version`。
- Modify: `backend/tests/test_api_contract.py` — 验证版本接口与 `app.__version__` 一致。
- Modify: `frontend/vite.config.ts` — 从用户端 manifest 读取并注入 `__APP_VERSION__`。
- Create: `frontend/src/env.d.ts` — 声明构建时版本常量类型。
- Create: `frontend/src/version.ts` — 暴露用户端只读版本标签。
- Create: `frontend/src/version.spec.ts` — 验证注入版本与用户端 manifest 一致。
- Modify: `frontend/src/components/chat/ChatShell.vue` — 在当前用户页面左下角展示版本。
- Modify: `frontend/src/components/chat/__tests__/ChatShell.spec.ts` — 验证版本位置和内容。
- Modify: `admin-frontend/src/services/api.ts` — 增加后端版本读取函数。
- Modify: `admin-frontend/src/services/api.spec.ts` — 验证版本 API 请求契约。
- Create: `admin-frontend/src/composables/useBackendVersion.ts` — 管理成功、失败和卸载后的版本状态。
- Create: `admin-frontend/src/composables/useBackendVersion.spec.ts` — 验证成功、失败降级。
- Modify: `admin-frontend/src/components/layout/AdminLayout.vue` — 在服务状态区域展示后端版本。
- Modify: `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts` — 验证管理端只显示后端版本。
- Modify: `frontend/package.json`, `frontend/package-lock.json` — 独立提升用户端 patch 版本。
- Modify: `admin-frontend/package.json`, `admin-frontend/package-lock.json` — 独立提升管理端 patch 版本。
- Keep: `backend/app/__init__.py` — 当前工作区已由待发布的后端改动提升到 `0.1.9`，本计划不重复提升同一待发布版本。

### Task 1: 后端版本接口

**Files:**
- Modify: `backend/app/schemas.py`
- Modify: `backend/app/routes.py`
- Test: `backend/tests/test_api_contract.py`

- [ ] **Step 1: 写失败的 API 契约测试**

在 `ApiContractTest` 中增加：

```python
def test_reports_backend_application_version_without_dependency_checks(self) -> None:
    from app import __version__

    response = self.client.get("/api/version")

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json(), {"version": __version__})
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
uv run --project backend pytest backend/tests/test_api_contract.py::ApiContractTest::test_reports_backend_application_version_without_dependency_checks -q
```

Expected: FAIL，`/api/version` 返回 `404`。

- [ ] **Step 3: 增加固定响应模型和路由**

在 `backend/app/schemas.py` 增加：

```python
class ApplicationVersion(BaseModel):
    version: str
```

在 `backend/app/routes.py` 导入版本源和模型：

```python
from . import __version__
from .schemas import ApplicationVersion
```

在健康检查路由附近增加：

```python
@router.get("/version", response_model=ApplicationVersion)
async def application_version() -> ApplicationVersion:
    return ApplicationVersion(version=__version__)
```

- [ ] **Step 4: 运行后端定向测试**

Run:

```bash
uv run --project backend pytest backend/tests/test_api_contract.py::ApiContractTest::test_reports_backend_application_version_without_dependency_checks -q
```

Expected: PASS。

### Task 2: 用户端构建版本源

**Files:**
- Modify: `frontend/vite.config.ts`
- Create: `frontend/src/env.d.ts`
- Create: `frontend/src/version.ts`
- Create: `frontend/src/version.spec.ts`

- [ ] **Step 1: 写失败的用户端版本源测试**

创建 `frontend/src/version.spec.ts`：

```typescript
import manifest from '../package.json'
import { describe, expect, it } from 'vitest'
import { APP_VERSION, APP_VERSION_LABEL } from './version'

describe('user frontend version', () => {
  it('uses the independent frontend package version', () => {
    expect(APP_VERSION).toBe(manifest.version)
    expect(APP_VERSION_LABEL).toBe(`v${manifest.version}`)
  })
})
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
npm --prefix frontend run test:run -- src/version.spec.ts
```

Expected: FAIL，`./version` 不存在。

- [ ] **Step 3: 注入并封装用户端版本**

在 `frontend/vite.config.ts` 顶部增加 Node 文件读取：

```typescript
import { readFileSync } from 'node:fs'

const packageManifest = JSON.parse(
  readFileSync(new URL('./package.json', import.meta.url), 'utf8'),
) as { version: string }

if (!/^\d+\.\d+\.\d+$/.test(packageManifest.version)) {
  throw new Error('frontend/package.json must contain a semantic version')
}
```

在 `defineConfig` 返回对象中增加：

```typescript
define: {
  __APP_VERSION__: JSON.stringify(packageManifest.version),
},
```

创建 `frontend/src/env.d.ts`：

```typescript
/// <reference types="vite/client" />

declare const __APP_VERSION__: string
```

创建 `frontend/src/version.ts`：

```typescript
export const APP_VERSION = __APP_VERSION__
export const APP_VERSION_LABEL = `v${APP_VERSION}`
```

- [ ] **Step 4: 运行版本源测试**

Run:

```bash
npm --prefix frontend run test:run -- src/version.spec.ts
```

Expected: PASS。

### Task 3: 用户端左下角显示版本

**Files:**
- Modify: `frontend/src/components/chat/ChatShell.vue`
- Test: `frontend/src/components/chat/__tests__/ChatShell.spec.ts`

- [ ] **Step 1: 写失败的页面显示测试**

在 `ChatShell` 测试中导入 manifest，并增加：

```typescript
import manifest from '../../../../package.json'

it('shows only the user frontend version in the lower-left page corner', () => {
  const wrapper = mount(ChatShell, {
    global: {
      stubs: {
        QuantumNetworkBackground: true,
        ChatTranscript: true,
        ComposerBar: true,
      },
    },
  })

  const version = wrapper.get('[data-testid="user-app-version"]')
  expect(version.text()).toBe(`v${manifest.version}`)
  expect(version.classes()).toContain('app-version')
  expect(wrapper.text()).not.toContain('后端版本')
})
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
npm --prefix frontend run test:run -- src/components/chat/__tests__/ChatShell.spec.ts
```

Expected: FAIL，找不到 `user-app-version`。

- [ ] **Step 3: 增加左下角版本元素**

在 `ChatShell.vue` 脚本中导入：

```typescript
import { APP_VERSION_LABEL } from '@/version'
```

在 `main.app-shell` 内增加：

```vue
<span class="app-version" data-testid="user-app-version">
  {{ APP_VERSION_LABEL }}
</span>
```

增加样式：

```css
.app-version {
  position: absolute;
  bottom: 18px;
  left: 26px;
  z-index: 4;
  color: rgba(207, 230, 239, 0.56);
  font-family: var(--font-mono);
  font-size: 11px;
  letter-spacing: 0.04em;
  pointer-events: none;
}
```

在移动端 media query 中把位置调整为 `bottom: 12px; left: 14px;`。

- [ ] **Step 4: 运行用户端组件测试**

Run:

```bash
npm --prefix frontend run test:run -- src/components/chat/__tests__/ChatShell.spec.ts
```

Expected: PASS。

### Task 4: 管理端后端版本读取状态

**Files:**
- Modify: `admin-frontend/src/services/api.ts`
- Modify: `admin-frontend/src/services/api.spec.ts`
- Create: `admin-frontend/src/composables/useBackendVersion.ts`
- Create: `admin-frontend/src/composables/useBackendVersion.spec.ts`

- [ ] **Step 1: 写失败的 API 服务测试**

在 `admin-frontend/src/services/api.spec.ts` 增加：

```typescript
it('loads the backend application version', async () => {
  httpMock.get.mockResolvedValue({ data: { version: '0.1.9' } })
  const { fetchBackendVersion } = await loadApi()

  await expect(fetchBackendVersion()).resolves.toBe('0.1.9')
  expect(httpMock.get).toHaveBeenCalledWith('/version')
})
```

- [ ] **Step 2: 运行服务测试并确认失败**

Run:

```bash
npm --prefix admin-frontend run test:run -- src/services/api.spec.ts
```

Expected: FAIL，`fetchBackendVersion` 不存在。

- [ ] **Step 3: 实现严格的版本响应解析**

在 `admin-frontend/src/services/api.ts` 增加：

```typescript
interface ApplicationVersionResponse {
  version: string
}

export async function fetchBackendVersion() {
  const { data } = await http.get<ApplicationVersionResponse>('/version')
  if (!/^\d+\.\d+\.\d+$/.test(data.version)) {
    throw new Error('Invalid backend version response')
  }
  return data.version
}
```

- [ ] **Step 4: 写 composable 失败测试**

创建 `admin-frontend/src/composables/useBackendVersion.spec.ts`：

```typescript
import { effectScope } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useBackendVersion } from './useBackendVersion'

const fetchBackendVersionMock = vi.hoisted(() => vi.fn())

vi.mock('@/services/api', () => ({
  fetchBackendVersion: fetchBackendVersionMock,
}))

describe('useBackendVersion', () => {
  beforeEach(() => {
    fetchBackendVersionMock.mockReset()
  })

  it('formats a successfully loaded backend version', async () => {
    fetchBackendVersionMock.mockResolvedValue('0.1.9')
    const scope = effectScope()
    const state = scope.run(() => useBackendVersion())
    if (!state) throw new Error('failed to create backend version state')

    await state.load()

    expect(state.displayVersion.value).toBe('v0.1.9')
    scope.stop()
  })

  it('falls back without throwing when the request fails', async () => {
    fetchBackendVersionMock.mockRejectedValue(new Error('unavailable'))
    const scope = effectScope()
    const state = scope.run(() => useBackendVersion())
    if (!state) throw new Error('failed to create backend version state')

    await expect(state.load()).resolves.toBeUndefined()
    expect(state.displayVersion.value).toBe('版本未知')
    scope.stop()
  })
})
```

- [ ] **Step 5: 实现 composable**

创建 `admin-frontend/src/composables/useBackendVersion.ts`：

```typescript
import { computed, onScopeDispose, shallowRef } from 'vue'
import { fetchBackendVersion } from '@/services/api'

export function useBackendVersion() {
  const version = shallowRef<string | null>(null)
  let active = true

  onScopeDispose(() => {
    active = false
  })

  async function load() {
    try {
      const loadedVersion = await fetchBackendVersion()
      if (active) version.value = loadedVersion
    }
    catch {
      if (active) version.value = null
    }
  }

  const displayVersion = computed(() => (
    version.value ? `v${version.value}` : '版本未知'
  ))

  return { displayVersion, load }
}
```

- [ ] **Step 6: 运行服务和 composable 测试**

Run:

```bash
npm --prefix admin-frontend run test:run -- src/services/api.spec.ts src/composables/useBackendVersion.spec.ts
```

Expected: PASS。

### Task 5: 管理端显示后端版本

**Files:**
- Modify: `admin-frontend/src/components/layout/AdminLayout.vue`
- Test: `admin-frontend/src/components/layout/__tests__/AdminLayout.spec.ts`

- [ ] **Step 1: 写失败的管理布局测试**

把 Vue/Vitest 导入调整为以下内容，并增加可复用 mock：

```typescript
import { shallowRef } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const loadBackendVersion = vi.hoisted(() => vi.fn())

vi.mock('@/composables/useBackendVersion', () => ({
  useBackendVersion: () => ({
    displayVersion: shallowRef('v0.1.9'),
    load: loadBackendVersion,
  }),
}))
```

在 `describe` 中增加 `beforeEach(() => loadBackendVersion.mockReset())`，并增加测试：

```typescript
it('shows the backend service version in the service status area', () => {
  const wrapper = mount(AdminLayout, {
    global: {
      mocks: {
        $route: { path: '/', meta: { title: '管理概览' } },
      },
      stubs: {
        RouterLink: { template: '<a><slot /></a>' },
        RouterView: true,
      },
    },
  })

expect(wrapper.get('[data-testid="backend-version"]').text()).toBe('v0.1.9')
expect(loadBackendVersion).toHaveBeenCalledTimes(1)
expect(wrapper.text()).not.toContain('前端版本')
})
```

- [ ] **Step 2: 运行布局测试并确认失败**

Run:

```bash
npm --prefix admin-frontend run test:run -- src/components/layout/__tests__/AdminLayout.spec.ts
```

Expected: FAIL，找不到 `backend-version`。

- [ ] **Step 3: 接入 composable 并显示版本**

在 `AdminLayout.vue` 中增加：

```typescript
import { onMounted } from 'vue'
import { useBackendVersion } from '@/composables/useBackendVersion'

const { displayVersion, load: loadBackendVersion } = useBackendVersion()

onMounted(() => {
  void loadBackendVersion()
})
```

把服务状态区域调整为：

```vue
<span class="admin-topbar__service">
  <i aria-hidden="true" />
  服务运行中
  <small data-testid="backend-version">{{ displayVersion }}</small>
</span>
```

增加弱化分隔样式：

```css
.admin-topbar__service small {
  padding-left: 8px;
  border-left: 1px solid #c5d0dc;
  color: #7a8999;
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 11px;
}
```

- [ ] **Step 4: 运行管理布局测试**

Run:

```bash
npm --prefix admin-frontend run test:run -- src/components/layout/__tests__/AdminLayout.spec.ts
```

Expected: PASS。

### Task 6: 独立升级前端版本并完成全量验证

**Files:**
- Modify: `frontend/package.json`
- Modify: `frontend/package-lock.json`
- Modify: `admin-frontend/package.json`
- Modify: `admin-frontend/package-lock.json`
- Verify: `backend/app/__init__.py`

- [ ] **Step 1: 分别提升两个发生修改的前端应用版本**

Run:

```bash
python tools/bump_version.py frontend patch
python tools/bump_version.py admin-frontend patch
```

Expected:

```text
frontend version: 0.1.0 -> 0.1.1
admin-frontend version: 0.1.0 -> 0.1.1
```

后端当前工作区的待发布版本已经是 `0.1.9`，不再次执行 patch；管理端将通过接口显示该版本。

- [ ] **Step 2: 运行版本契约测试**

Run:

```bash
uv run --project backend pytest tools/tests/test_version_contract.py -q
```

Expected: PASS。

- [ ] **Step 3: 运行后端完整测试**

Run:

```bash
uv run --project backend pytest backend/tests -q
```

Expected: 所有测试通过，允许仓库既有 skip。

- [ ] **Step 4: 运行两个前端完整测试和构建**

Run:

```bash
npm --prefix frontend run test:run
npm --prefix frontend run build
npm --prefix admin-frontend run test:run
npm --prefix admin-frontend run build
```

Expected: 两个应用测试与构建全部通过。

- [ ] **Step 5: 运行仓库静态检查**

Run:

```bash
fast lint --ty
git diff --check
```

Expected: 两条命令均无错误。

- [ ] **Step 6: 检查提交范围**

Run:

```bash
git status --short
git diff -- backend/app/schemas.py backend/app/routes.py backend/tests/test_api_contract.py frontend admin-frontend
```

Expected: 版本显示改动与既有未提交 Reranker 改动均清晰可辨，不覆盖用户或此前任务的修改。
