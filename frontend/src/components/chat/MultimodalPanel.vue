<script setup lang="ts">
import { Image, Play } from 'lucide-vue-next'
import cityImage from '@/assets/city-intelligence.png'
import analysisImage from '@/assets/analysis-tunnel.png'
import type { Artifact } from '@/types/chat'

defineProps<{
  artifacts: readonly Artifact[]
}>()

const assetMap = {
  city: cityImage,
  analysis: analysisImage,
}
</script>

<template>
  <div class="artifact-grid">
    <section v-for="artifact in artifacts" :key="`${artifact.type}-${artifact.title}`" class="artifact-card">
      <template v-if="artifact.type === 'summary'">
        <h3>{{ artifact.title }}</h3>
        <ul class="summary-list">
          <li v-for="bullet in artifact.bullets" :key="bullet">{{ bullet }}</li>
        </ul>
      </template>

      <template v-else-if="artifact.type === 'image'">
        <div class="media-frame">
          <img :src="assetMap[artifact.assetKey]" :alt="artifact.title" />
          <span class="media-icon"><Image :size="15" /></span>
        </div>
        <h3>{{ artifact.title }}</h3>
      </template>

      <template v-else-if="artifact.type === 'video'">
        <div class="media-frame">
          <img :src="assetMap[artifact.assetKey]" :alt="artifact.title" />
          <button type="button" class="play-button" aria-label="播放视频"><Play :size="22" fill="currentColor" /></button>
          <span class="duration">{{ artifact.duration }}</span>
        </div>
        <h3>{{ artifact.title }}</h3>
      </template>

      <template v-else>
        <h3>{{ artifact.title }}</h3>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th v-for="column in artifact.columns" :key="column">{{ column }}</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="row in artifact.rows" :key="row.join('-')">
                <td v-for="cell in row" :key="cell">{{ cell }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </section>
  </div>
</template>

<style scoped>
.artifact-grid {
  display: grid;
  grid-template-columns: 1.15fr 0.7fr 0.95fr 1.05fr;
  gap: 10px;
  margin-top: 24px;
}

.artifact-card {
  min-width: 0;
  min-height: 210px;
  padding: 14px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 7px;
  background: rgba(255, 255, 255, 0.052);
  backdrop-filter: blur(10px);
}

.artifact-card h3 {
  margin: 0 0 10px;
  color: #e8f8fc;
  font-size: 13px;
  font-weight: 640;
}

.summary-list {
  display: grid;
  gap: 8px;
  margin: 0;
  padding-left: 18px;
  color: #aeb4bb;
  font-size: 13px;
  line-height: 1.58;
}

.media-frame {
  position: relative;
  overflow: hidden;
  aspect-ratio: 16 / 10;
  margin: -4px -4px 12px;
  border-radius: 5px;
  background: #050608;
}

.media-frame img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}

.media-icon {
  position: absolute;
  left: 10px;
  bottom: 10px;
  display: grid;
  place-items: center;
  width: 27px;
  height: 27px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  border-radius: 5px;
  color: #e8fbff;
  background: rgba(0, 0, 0, 0.55);
}

.play-button {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  width: 54px;
  height: 54px;
  margin: auto;
  border: 1px solid rgba(129, 229, 255, 0.72);
  border-radius: 50%;
  color: #b8f2ff;
  background: rgba(5, 6, 8, 0.66);
  cursor: pointer;
}

.duration {
  position: absolute;
  right: 10px;
  bottom: 9px;
  color: #e2e5e8;
  font-family: var(--font-mono);
  font-size: 11px;
}

.table-wrap {
  overflow: auto;
}

table {
  width: 100%;
  border-collapse: collapse;
  color: #b9c0c7;
  font-size: 11px;
}

th,
td {
  padding: 8px 7px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  text-align: left;
  white-space: nowrap;
}

th {
  color: #aeeeff;
  background: rgba(103, 216, 255, 0.08);
}

@media (max-width: 1180px) {
  .artifact-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 680px) {
  .artifact-grid {
    grid-template-columns: 1fr;
  }
}
</style>
