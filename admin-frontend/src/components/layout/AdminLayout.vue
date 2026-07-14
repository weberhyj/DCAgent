<script setup lang="ts">
import { Bot, Database, LayoutDashboard, ListChecks, ShieldCheck } from 'lucide-vue-next'

const navigation = [
  { name: 'overview', label: '管理概览', icon: LayoutDashboard },
  { name: 'knowledge', label: '知识库管理', icon: Database },
  { name: 'quality-cases', label: '质量评测', icon: ListChecks, modulePath: '/quality' },
  { name: 'agent-runs', label: 'Agent 审计', icon: Bot },
]
</script>

<template>
  <div class="admin-shell">
    <aside class="admin-sidebar">
      <RouterLink class="admin-brand" :to="{ name: 'overview' }" aria-label="返回管理概览">
        <span class="admin-brand__mark">DC</span>
        <span class="admin-brand__copy">
          <strong>DC-Agent</strong>
        </span>
      </RouterLink>

      <nav class="admin-nav" aria-label="管理后台导航">
        <RouterLink
          v-for="item in navigation"
          :key="item.name"
          class="admin-nav__item"
          :class="{ 'is-module-active': item.modulePath && ($route.path || '').startsWith(item.modulePath) }"
          :to="{ name: item.name }"
          :data-testid="item.modulePath ? 'nav-quality' : undefined"
        >
          <component :is="item.icon" :size="18" />
          <span>{{ item.label }}</span>
        </RouterLink>
      </nav>

      <div class="admin-sidebar__status">
        <ShieldCheck :size="16" />
        <span>
          <strong>只读治理模式</strong>
          <small>Agent 不执行写入操作</small>
        </span>
      </div>
    </aside>

    <div class="admin-workspace">
      <header class="admin-topbar">
        <div>
          <strong>{{ $route.meta.title }}</strong>
        </div>
        <span class="admin-topbar__service">
          <i aria-hidden="true" />
          服务运行中
        </span>
      </header>

      <main class="admin-content">
        <RouterView />
      </main>
    </div>
  </div>
</template>

<style scoped>
.admin-shell {
  display: grid;
  grid-template-columns: 236px minmax(0, 1fr);
  min-height: 100vh;
}

.admin-sidebar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: grid;
  grid-template-rows: auto 1fr auto;
  align-self: start;
  height: 100vh;
  padding: 22px 16px 18px;
  border-right: 1px solid rgba(172, 188, 204, 0.72);
  background: rgba(246, 249, 252, 0.9);
  backdrop-filter: blur(18px);
}

.admin-brand {
  display: flex;
  align-items: center;
  gap: 11px;
  min-width: 0;
  padding: 0 6px 22px;
  color: inherit;
  text-decoration: none;
}

.admin-brand__mark {
  display: grid;
  place-items: center;
  width: 38px;
  height: 38px;
  flex: 0 0 auto;
  border-radius: 8px;
  color: #ffffff;
  background: #113e91;
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 700;
  box-shadow: 0 8px 20px rgba(17, 62, 145, 0.2);
}

.admin-brand__copy {
  display: grid;
  gap: 2px;
  min-width: 0;
}

.admin-brand__copy strong {
  color: #14202c;
  font-size: 14px;
}

.admin-nav {
  display: grid;
  align-content: start;
  gap: 5px;
}

.admin-nav__item {
  display: grid;
  grid-template-columns: 24px minmax(0, 1fr);
  gap: 9px;
  align-items: center;
  min-height: 42px;
  padding: 0 11px;
  border: 1px solid transparent;
  border-radius: 7px;
  color: #536579;
  font-size: 13px;
  font-weight: 600;
  text-decoration: none;
}

.admin-nav__item:hover {
  color: #173a6a;
  background: rgba(222, 232, 243, 0.72);
}

.admin-nav__item.router-link-active,
.admin-nav__item.is-module-active {
  border-color: #bdd0e6;
  color: #0f438f;
  background: #e7eff9;
  box-shadow: inset 3px 0 0 #1463ff;
}

.admin-sidebar__status {
  display: grid;
  grid-template-columns: auto minmax(0, 1fr);
  gap: 9px;
  align-items: start;
  padding: 11px;
  border-top: 1px solid #d9e2eb;
  color: #4b6278;
}

.admin-sidebar__status svg {
  margin-top: 1px;
  color: #087b55;
}

.admin-sidebar__status span {
  display: grid;
  gap: 3px;
}

.admin-sidebar__status strong {
  font-size: 12px;
}

.admin-sidebar__status small {
  color: #7a8999;
  font-size: 12px;
  line-height: 1.45;
}

.admin-workspace {
  min-width: 0;
}

.admin-topbar {
  position: sticky;
  top: 0;
  z-index: 8;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 18px;
  min-height: 66px;
  padding: 0 clamp(20px, 3vw, 42px);
  border-bottom: 1px solid rgba(188, 201, 214, 0.74);
  background: rgba(238, 243, 247, 0.88);
  backdrop-filter: blur(16px);
}

.admin-topbar > div {
  display: grid;
  gap: 3px;
}

.admin-topbar strong {
  color: #182531;
  font-size: 14px;
}

.admin-topbar__service {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  color: #526478;
  font-size: 12px;
}

.admin-topbar__service i {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: #16a36f;
  box-shadow: 0 0 0 4px rgba(22, 163, 111, 0.12);
}

.admin-content {
  min-width: 0;
  padding: 28px clamp(20px, 3vw, 42px) 44px;
}

@media (max-width: 860px) {
  .admin-shell {
    grid-template-columns: 1fr;
  }

  .admin-sidebar {
    position: relative;
    grid-template-columns: auto minmax(0, 1fr);
    grid-template-rows: auto;
    height: auto;
    padding: 12px 14px;
    border-right: 0;
    border-bottom: 1px solid rgba(172, 188, 204, 0.72);
  }

  .admin-brand {
    padding: 0;
  }

  .admin-brand__copy,
  .admin-sidebar__status {
    display: none;
  }

  .admin-nav {
    display: flex;
    justify-content: flex-end;
    gap: 4px;
  }

  .admin-nav__item {
    grid-template-columns: auto;
    min-height: 38px;
    padding: 0 10px;
  }

  .admin-nav__item span {
    display: none;
  }

  .admin-nav__item.router-link-active,
  .admin-nav__item.is-module-active {
    box-shadow: inset 0 -3px 0 #1463ff;
  }

  .admin-topbar {
    min-height: 58px;
  }

  .admin-content {
    padding-top: 20px;
  }
}
</style>
