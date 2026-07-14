<script setup lang="ts">
import { CheckCircle2, Database, SearchCheck, XCircle } from 'lucide-vue-next'
import type { EvaluationRun } from '@/types/chat'

const props = defineProps<{
  run: EvaluationRun | null
}>()

function percentage(value: number) {
  return `${Math.round(value * 100)}%`
}
</script>

<template>
  <section class="diagnostic-panel" data-testid="evaluation-run-detail">
    <header class="panel-heading">
      <SearchCheck :size="18" />
      <span><h2>检索诊断</h2><p>查看最新一次 Top-K 命中、召回率和缺失条件。</p></span>
    </header>

    <template v-if="props.run">
      <div class="diagnostic-status" :class="`is-${props.run.status}`">
        <CheckCircle2 v-if="props.run.status === 'passed'" :size="18" />
        <XCircle v-else :size="18" />
        <span>
          <strong>
            {{ props.run.falsePositive ? '发生误召回' : props.run.status === 'passed' ? (props.run.expectAnswer ? '评测通过' : '无答案识别通过') : '评测未通过' }}
          </strong>
          <small>{{ props.run.completedAt }}</small>
        </span>
      </div>

      <div class="metric-strip">
        <span><strong>{{ percentage(props.run.sourceRecall) }}</strong>资料召回</span>
        <span><strong>{{ percentage(props.run.termRecall) }}</strong>关键词召回</span>
        <span><strong>{{ props.run.hitCount }}</strong>命中片段</span>
        <span><strong>{{ props.run.topScore.toFixed(2) }}</strong>最高分数</span>
      </div>

      <div v-if="props.run.missingSourceIds.length || props.run.missingTerms.length" class="missing-block">
        <strong>缺失条件</strong>
        <span v-for="sourceId in props.run.missingSourceIds" :key="sourceId">资料：{{ sourceId }}</span>
        <span v-for="term in props.run.missingTerms" :key="term">关键词：{{ term }}</span>
      </div>

      <div v-if="props.run.hits.length" class="hit-list">
        <article v-for="hit in props.run.hits" :key="hit.chunkId" class="hit-row">
          <div class="hit-rank">{{ hit.rank }}</div>
          <div class="hit-main">
            <div class="hit-title"><Database :size="15" /><strong>{{ hit.sourceName }}</strong><span>片段 {{ hit.chunkIndex + 1 }}</span></div>
            <p>{{ hit.excerpt }}</p>
            <div class="hit-terms">
              <span>关键词 {{ hit.keywordScore.toFixed(2) }}</span>
              <span>向量 {{ hit.vectorScore.toFixed(2) }}</span>
              <span>综合 {{ hit.score.toFixed(2) }}</span>
              <em v-for="term in hit.matchedTerms" :key="term">{{ term }}</em>
            </div>
          </div>
        </article>
      </div>
      <div v-else class="empty-state">本次检索没有返回可诊断的资料片段。</div>
    </template>
    <div v-else class="empty-state">选择并运行一个测试问题后，这里会显示检索诊断结果。</div>
  </section>
</template>

<style scoped>
.diagnostic-panel { min-width: 0; padding: 16px; border: 1px solid var(--color-border); border-radius: 8px; background: rgba(255,255,255,.88); box-shadow: 0 14px 36px rgba(38,57,76,.05); }
.panel-heading { display: flex; gap: 9px; align-items: start; margin-bottom: 14px; }
.panel-heading > svg { flex: 0 0 auto; color: #1463ff; }
.panel-heading > span { display: grid; gap: 4px; }
.panel-heading h2, .panel-heading p { margin: 0; }
.panel-heading h2 { color: #172431; font-size: 14px; }
.panel-heading p { color: #718294; font-size: 12px; line-height: 1.5; }
.diagnostic-status { display: flex; align-items: center; gap: 9px; padding: 11px 12px; border: 1px solid #cad6e2; border-radius: 7px; color: #526579; background: #f5f8fb; }
.diagnostic-status.is-passed { border-color: #b9ddcf; color: #087653; background: #ecf8f3; }
.diagnostic-status.is-failed { border-color: #efc7c1; color: #a92d22; background: #fff5f3; }
.diagnostic-status > span { display: grid; gap: 3px; }
.diagnostic-status strong { font-size: 13px; }
.diagnostic-status small { color: inherit; font-size: 12px; opacity: .78; }
.metric-strip { display: grid; grid-template-columns: repeat(4,minmax(0,1fr)); gap: 1px; margin-top: 12px; overflow: hidden; border: 1px solid #dce4ec; border-radius: 7px; background: #dce4ec; }
.metric-strip span { display: grid; gap: 3px; padding: 10px; color: #6a7b8d; background: #f8fafc; font-size: 12px; }
.metric-strip strong { color: #1d3f6c; font-family: var(--font-mono); font-size: 14px; font-variant-numeric: tabular-nums; }
.missing-block { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; padding: 10px; border: 1px solid #efd2c8; border-radius: 7px; color: #9e3528; background: #fff7f4; font-size: 12px; }
.missing-block strong { width: 100%; font-size: 12px; }
.missing-block span { padding: 4px 6px; border-radius: 5px; background: rgba(255,255,255,.72); }
.hit-list { display: grid; gap: 8px; margin-top: 12px; }
.hit-row { display: grid; grid-template-columns: 28px minmax(0,1fr); gap: 10px; padding: 11px; border: 1px solid #dfe7ef; border-radius: 7px; background: #fbfcfd; }
.hit-rank { display: grid; place-items: center; width: 28px; height: 28px; border-radius: 6px; color: #174a87; background: #e7efff; font-family: var(--font-mono); font-size: 12px; font-weight: 700; }
.hit-main { display: grid; gap: 7px; min-width: 0; }
.hit-main > * { min-width: 0; }
.hit-title { display: flex; align-items: center; gap: 6px; min-width: 0; color: #617387; font-size: 12px; }
.hit-title svg { flex: 0 0 auto; color: #1463ff; }
.hit-title strong { overflow: hidden; color: #1c2a38; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
.hit-title span { flex: 0 0 auto; }
.hit-main p { margin: 0; color: #465a6e; font-size: 12px; line-height: 1.6; overflow-wrap: anywhere; }
.hit-terms { display: flex; flex-wrap: wrap; gap: 6px; }
.hit-terms span, .hit-terms em { padding: 4px 6px; border-radius: 5px; font-size: 12px; font-style: normal; line-height: 1.2; }
.hit-terms span { color: #315c9d; background: #eaf1fb; }
.hit-terms em { color: #536579; background: #eef2f6; }
.empty-state { padding: 50px 18px; border: 1px dashed #bdcbd9; border-radius: 7px; color: #748596; background: #f8fafc; font-size: 12px; line-height: 1.55; text-align: center; }
@media (max-width: 620px) { .metric-strip { grid-template-columns: repeat(2,minmax(0,1fr)); } }
</style>
