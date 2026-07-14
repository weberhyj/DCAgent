<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, RouterView, useRoute } from 'vue-router'

const route = useRoute()
const reportsActive = computed(() => route.path.startsWith('/quality/reports'))
</script>

<template>
  <section class="quality-module" data-testid="quality-module-layout">
    <nav class="quality-tabs" aria-label="质量评测模块导航">
      <RouterLink :to="{ name: 'quality-cases' }">评测集</RouterLink>
      <RouterLink
        :to="{ name: 'quality-reports' }"
        :class="{ 'is-active': reportsActive }"
        :aria-current="reportsActive ? 'page' : undefined"
      >评测报告</RouterLink>
    </nav>
    <RouterView />
  </section>
</template>

<style scoped>
.quality-module {
  min-width: 0;
  max-width: 1380px;
  margin: 0 auto;
}

.quality-tabs {
  display: flex;
  gap: 24px;
  min-width: 0;
  margin-bottom: 22px;
  border-bottom: 1px solid #ccd8e5;
}

.quality-tabs a {
  position: relative;
  min-height: 40px;
  padding: 9px 2px 10px;
  color: #667789;
  font-size: 13px;
  font-weight: 650;
  text-decoration: none;
}

.quality-tabs a:hover {
  color: #173a6a;
}

.quality-tabs a.router-link-active,
.quality-tabs a.is-active {
  color: #0f438f;
}

.quality-tabs a.router-link-active::after,
.quality-tabs a.is-active::after {
  position: absolute;
  right: 0;
  bottom: -1px;
  left: 0;
  height: 2px;
  background: #1463ff;
  content: '';
}
</style>
