<script setup lang="ts">
import { Bot, RefreshCcw, Search } from 'lucide-vue-next'
import { computed, onMounted, shallowRef } from 'vue'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import { useChatKnowledgeManagement } from '@/composables/useChatKnowledgeManagement'

const { agentRuns, agentRunsLoading, loadAgentRuns } = useChatKnowledgeManagement()
const query = shallowRef('')

const toolLabels: Record<string, string> = {
  plan_retrieval: '规划检索',
  search_knowledge: '检索知识库',
  inspect_document: '深入检查资料',
  compare_evidence: '对比证据',
  compose_answer: '生成回答',
}

const modeLabels = {
  quick: '快速检索',
  deep: '深度分析',
  source: '全库检索',
} as const

const filteredRuns = computed(() => {
  const normalized = query.value.trim().toLocaleLowerCase()
  if (!normalized) return agentRuns.value
  return agentRuns.value.filter((run) => run.query.toLocaleLowerCase().includes(normalized))
})

onMounted(() => {
  void loadAgentRuns()
})

function toolLabel(toolName: string) {
  return toolLabels[toolName] ?? toolName
}
</script>

<template>
  <section class="module-page" data-testid="agent-audit-page">
    <AdminPageHeader
      title="Agent 执行审计"
      description="审查 DCAgent 的检索策略、资料检查、证据对比和回答生成过程。"
    >
      <template #actions>
        <BaseButton
          type="button"
          variant="subtle"
          :disabled="agentRunsLoading"
          data-testid="refresh-agent-runs"
          @click="loadAgentRuns"
        >
          <RefreshCcw :size="16" />
          刷新记录
        </BaseButton>
      </template>
    </AdminPageHeader>

    <div class="audit-toolbar">
      <Search :size="16" />
      <BaseInput v-model="query" placeholder="筛选用户问题" aria-label="筛选 Agent 执行记录" />
      <span>{{ filteredRuns.length }} 次执行</span>
    </div>

    <div v-if="agentRunsLoading" class="module-state">正在刷新 Agent 执行记录...</div>
    <div v-else-if="!filteredRuns.length" class="module-state">暂无匹配的 Agent 执行记录。</div>

    <div v-else class="audit-list">
      <details v-for="(run, runIndex) in filteredRuns" :key="run.id" class="audit-run" :open="runIndex === 0">
        <summary>
          <span class="audit-run__icon"><Bot :size="17" /></span>
          <span class="audit-run__main">
            <strong>{{ run.query }}</strong>
            <small>{{ run.completedAt }} · {{ modeLabels[run.mode] }}</small>
          </span>
          <span class="audit-run__metrics">
            <i>{{ run.evidenceCount }} 个证据</i>
            <i>{{ run.sourceCount }} 个来源</i>
            <i class="is-complete">已完成</i>
          </span>
        </summary>

        <ol class="audit-steps">
          <li v-for="step in run.steps" :key="step.id">
            <span class="audit-step__index">{{ step.stepIndex + 1 }}</span>
            <div>
              <header>
                <strong>{{ toolLabel(step.toolName) }}</strong>
                <span>只读</span>
                <time>{{ step.completedAt }}</time>
              </header>
              <p>{{ step.outputSummary }}</p>
            </div>
          </li>
        </ol>
      </details>
    </div>
  </section>
</template>

<style scoped>
.module-page {
  max-width: 1380px;
  margin: 0 auto;
}

.audit-toolbar {
  display: grid;
  grid-template-columns: auto minmax(0, 360px) 1fr;
  gap: 10px;
  align-items: center;
  min-height: 58px;
  margin-bottom: 14px;
  padding: 0 14px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.82);
}

.audit-toolbar svg {
  color: #718397;
}

.audit-toolbar > span {
  justify-self: end;
  color: #657688;
  font-size: 11px;
}

.audit-list {
  display: grid;
  gap: 10px;
}

.audit-run {
  overflow: hidden;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  background: rgba(255, 255, 255, 0.86);
  box-shadow: 0 12px 32px rgba(38, 57, 76, 0.05);
}

.audit-run summary {
  display: grid;
  grid-template-columns: 34px minmax(0, 1fr) auto;
  gap: 12px;
  align-items: center;
  min-height: 70px;
  padding: 12px 15px;
  cursor: pointer;
  list-style: none;
}

.audit-run summary::-webkit-details-marker {
  display: none;
}

.audit-run__icon {
  display: grid;
  place-items: center;
  width: 34px;
  height: 34px;
  border-radius: 7px;
  color: #1454b8;
  background: #e7efff;
}

.audit-run__main {
  display: grid;
  gap: 5px;
  min-width: 0;
}

.audit-run__main strong,
.audit-run__main small {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.audit-run__main strong {
  color: #17232e;
  font-size: 14px;
}

.audit-run__main small {
  color: #758597;
  font-size: 10px;
}

.audit-run__metrics {
  display: flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 6px;
}

.audit-run__metrics i {
  padding: 4px 7px;
  border-radius: 999px;
  color: #506477;
  background: #edf2f7;
  font-size: 10px;
  font-style: normal;
}

.audit-run__metrics .is-complete {
  color: #087b55;
  background: #e5f5ee;
}

.audit-steps {
  display: grid;
  margin: 0;
  padding: 0 15px 14px;
  list-style: none;
}

.audit-steps li {
  display: grid;
  grid-template-columns: 26px minmax(0, 1fr);
  gap: 10px;
  padding: 11px 0;
  border-top: 1px solid #e5ebf1;
}

.audit-step__index {
  display: grid;
  place-items: center;
  width: 24px;
  height: 24px;
  border-radius: 50%;
  color: #174e96;
  background: #e8f0ff;
  font-family: var(--font-mono);
  font-size: 10px;
}

.audit-steps header {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
}

.audit-steps header strong {
  color: #1d2a36;
  font-size: 12px;
}

.audit-steps header span {
  padding: 2px 6px;
  border-radius: 999px;
  color: #087b55;
  background: #e5f5ee;
  font-size: 9px;
}

.audit-steps time {
  margin-left: auto;
  color: #8090a0;
  font-family: var(--font-mono);
  font-size: 9px;
}

.audit-steps p {
  margin: 5px 0 0;
  color: #586a7b;
  font-size: 11px;
  line-height: 1.6;
}

.module-state {
  padding: 50px 20px;
  border: 1px dashed #b9c7d5;
  border-radius: 8px;
  color: #7b8b9b;
  background: rgba(255, 255, 255, 0.62);
  font-size: 12px;
  text-align: center;
}

@media (max-width: 720px) {
  .audit-toolbar {
    grid-template-columns: auto minmax(0, 1fr);
  }

  .audit-toolbar > span {
    display: none;
  }

  .audit-run summary {
    grid-template-columns: 34px minmax(0, 1fr);
  }

  .audit-run__metrics {
    grid-column: 2;
    justify-content: flex-start;
  }
}
</style>
