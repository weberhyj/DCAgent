<script setup lang="ts">
import GlitchText from '@/components/ui/GlitchText.vue'

const launchTitle = 'DC智识中...'
</script>

<template>
  <div
    class="knowledge-launch-loader"
    data-testid="knowledge-launch-loader"
    data-ignore-quantum-pulse
    role="status"
    aria-live="polite"
    aria-label="资料库检索中"
  >
    <GlitchText
      as="h1"
      class="knowledge-launch-loader__title"
      :text="launchTitle"
    />
  </div>
</template>

<style scoped>
.knowledge-launch-loader {
  position: relative;
  display: grid;
  place-items: center;
  width: min(760px, 100%);
  pointer-events: none;
}

.knowledge-launch-loader__title {
  font-size: 42px;
  font-weight: 800;
  line-height: 1.08;
}

@media (max-width: 920px) {
  .knowledge-launch-loader {
    width: 100%;
  }

  .knowledge-launch-loader__title {
    font-size: 30px;
  }
}
</style>
