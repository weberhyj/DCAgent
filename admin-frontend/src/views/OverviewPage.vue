<script setup lang="ts">
import { Activity, Bot, Database, FileCheck2, RefreshCcw, TriangleAlert } from 'lucide-vue-next'
import { computed, onMounted } from 'vue'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import { useChatKnowledgeManagement } from '@/composables/useChatKnowledgeManagement'

const {
  knowledgeSources,
  agentRuns,
  knowledgeSourcesLoading,
  agentRunsLoading,
  loadKnowledgeSources,
  loadAgentRuns,
} = useChatKnowledgeManagement()

const totalChunks = computed(() => knowledgeSources.value.reduce((total, source) => total + source.records, 0))
const indexedCount = computed(() => knowledgeSources.value.filter((source) => source.status === '已索引').length)
const failedCount = computed(() => knowledgeSources.value.filter((source) => source.status === '解析失败').length)
const recentSources = computed(() => knowledgeSources.value.slice(0, 5))
const recentRuns = computed(() => agentRuns.value.slice(0, 5))
const refreshing = computed(() => knowledgeSourcesLoading.value || agentRunsLoading.value)

onMounted(() => {
  void refreshOverview()
})

async function refreshOverview() {
  await Promise.all([loadKnowledgeSources(), loadAgentRuns()])
}
</script>

<template>
  <section class="module-page" data-testid="overview-page">
    <AdminPageHeader
      title="管理概览"
      description="集中查看知识库健康状态与 DCAgent 最近执行情况。"
    >
      <template #actions>
        <BaseButton
          type="button"
          variant="subtle"
          :disabled="refreshing"
          aria-label="刷新管理概览"
          @click="refreshOverview"
        >
          <RefreshCcw :size="16" />
          刷新
        </BaseButton>
      </template>
    </AdminPageHeader>

    <div class="metric-grid">
      <article class="metric-item">
        <span class="metric-item__icon"><Database :size="19" /></span>
        <div><small>资料总数</small><strong>{{ knowledgeSources.length }}</strong></div>
      </article>
      <article class="metric-item">
        <span class="metric-item__icon metric-item__icon--success"><FileCheck2 :size="19" /></span>
        <div><small>已完成索引</small><strong>{{ indexedCount }}</strong></div>
      </article>
      <article class="metric-item">
        <span class="metric-item__icon"><Activity :size="19" /></span>
        <div><small>知识片段</small><strong>{{ totalChunks }}</strong></div>
      </article>
      <article class="metric-item">
        <span class="metric-item__icon" :class="{ 'metric-item__icon--danger': failedCount }"><TriangleAlert :size="19" /></span>
        <div><small>解析失败</small><strong>{{ failedCount }}</strong></div>
      </article>
    </div>

    <div class="overview-grid">
      <section class="overview-panel">
        <header class="overview-panel__header">
          <div><Database :size="17" /><strong>最近资料</strong></div>
          <RouterLink :to="{ name: 'knowledge' }">进入知识库</RouterLink>
        </header>
        <div v-if="recentSources.length" class="overview-list">
          <RouterLink
            v-for="source in recentSources"
            :key="source.id"
            class="overview-row"
            :to="{ name: 'knowledge-source-detail', params: { sourceId: source.id } }"
          >
            <span class="overview-row__main">
              <strong>{{ source.name }}</strong>
              <small>{{ source.sourceType }} · {{ source.updatedAt }}</small>
            </span>
            <span class="overview-row__status" :class="`is-${source.status}`">{{ source.status }}</span>
          </RouterLink>
        </div>
        <p v-else class="overview-empty">暂无资料。</p>
      </section>

      <section class="overview-panel">
        <header class="overview-panel__header">
          <div><Bot :size="17" /><strong>最近 Agent 执行</strong></div>
          <RouterLink :to="{ name: 'agent-runs' }">查看审计</RouterLink>
        </header>
        <div v-if="recentRuns.length" class="overview-list">
          <RouterLink
            v-for="run in recentRuns"
            :key="run.id"
            class="overview-row"
            :to="{ name: 'agent-runs', query: { run: run.id } }"
          >
            <span class="overview-row__main">
              <strong>{{ run.query }}</strong>
              <small>{{ run.completedAt }} · {{ run.evidenceCount }} 个证据</small>
            </span>
            <span class="overview-row__status is-completed">已完成</span>
          </RouterLink>
        </div>
        <p v-else class="overview-empty">暂无 Agent 执行记录。</p>
      </section>
    </div>
  </section>
</template>

<style scoped>
.module-page {
  max-width: 1380px;
  margin: 0 auto;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  margin-bottom: 16px;
}

.metric-item {
  display: grid;
  grid-template-columns: 42px minmax(0, 1fr);
  gap: 12px;
  align-items: center;
  min-height: 92px;
  padding: 16px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.82);
  box-shadow: 0 10px 30px rgba(38, 57, 76, 0.05);
}

.metric-item__icon {
  display: grid;
  place-items: center;
  width: 42px;
  height: 42px;
  border-radius: 8px;
  color: #1454b8;
  background: #e7efff;
}

.metric-item__icon--success {
  color: #087b55;
  background: #e4f5ed;
}

.metric-item__icon--danger {
  color: #b42318;
  background: #fff0ed;
}

.metric-item div {
  display: grid;
  gap: 5px;
}

.metric-item small {
  color: #718193;
  font-size: 11px;
}

.metric-item strong {
  color: #14202b;
  font-family: var(--font-mono);
  font-size: 24px;
}

.overview-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}

.overview-panel {
  min-width: 0;
  overflow: hidden;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.84);
  box-shadow: 0 14px 36px rgba(38, 57, 76, 0.055);
}

.overview-panel__header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  min-height: 54px;
  padding: 0 16px;
  border-bottom: 1px solid #e0e7ef;
}

.overview-panel__header div {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: #1d2a37;
  font-size: 13px;
}

.overview-panel__header svg {
  color: #1463ff;
}

.overview-panel__header a {
  color: #1756a9;
  font-size: 11px;
  font-weight: 600;
  text-decoration: none;
}

.overview-list {
  display: grid;
}

.overview-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  min-height: 62px;
  padding: 10px 16px;
  border-bottom: 1px solid #e8edf2;
  color: inherit;
  text-decoration: none;
}

.overview-row:last-child {
  border-bottom: 0;
}

.overview-row:hover {
  background: #f6f9fc;
}

.overview-row__main {
  display: grid;
  gap: 5px;
  min-width: 0;
}

.overview-row__main strong,
.overview-row__main small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.overview-row__main strong {
  color: #1b2733;
  font-size: 13px;
}

.overview-row__main small {
  color: #758597;
  font-size: 10px;
}

.overview-row__status {
  padding: 4px 7px;
  border-radius: 999px;
  color: #087b55;
  background: #e5f5ee;
  font-size: 10px;
  white-space: nowrap;
}

.overview-row__status.is-解析失败 {
  color: #b42318;
  background: #fff0ed;
}

.overview-row__status.is-解析中 {
  color: #184985;
  background: #e8f0ff;
}

.overview-empty {
  margin: 0;
  padding: 34px 16px;
  color: #8291a1;
  font-size: 12px;
  text-align: center;
}

@media (max-width: 980px) {
  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .overview-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 520px) {
  .metric-grid {
    grid-template-columns: 1fr;
  }
}
</style>
