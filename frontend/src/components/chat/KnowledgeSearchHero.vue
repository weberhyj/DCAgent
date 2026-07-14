<script setup lang="ts">
import ComposerBar from './ComposerBar.vue'
import KnowledgeLaunchLoader from './KnowledgeLaunchLoader.vue'
import type { ComposerMode } from '@/types/chat'

const heroTitle = '欢迎来到DC智识中枢'
const heroTitleCharacters = Array.from(heroTitle)

defineProps<{
  sending: boolean
  searching: boolean
  launching: boolean
  closing: boolean
  error: string | null
}>()

const emit = defineEmits<{
  send: [payload: { content: string; mode: ComposerMode }]
}>()
</script>

<template>
  <section
    class="knowledge-search-hero"
    :class="{
      'knowledge-search-hero--launching': launching,
      'knowledge-search-hero--closing': closing,
    }"
    :data-ignore-quantum-pulse="launching || closing ? '' : undefined"
  >
    <div v-if="!launching" class="knowledge-search-hero__idle" :class="{ 'knowledge-search-hero__idle--closing': closing }">
      <div class="knowledge-search-hero__copy" data-ignore-quantum-pulse>
        <h1 class="split-text-title" :aria-label="heroTitle">
          <span class="split-text-title__inner" aria-hidden="true">
            <span
              v-for="(character, index) in heroTitleCharacters"
              :key="`${character}-${index}`"
              class="split-text-title__char"
              :style="{ '--split-index': index }"
            >
              {{ character }}
            </span>
          </span>
        </h1>
      </div>

      <ComposerBar
        :closing="closing"
        :sending="sending"
        :searching="searching"
        variant="center"
        @send="emit('send', $event)"
      />

      <p v-if="error && !closing" class="knowledge-search-hero__error" data-testid="hero-search-error">
        {{ error }}
      </p>
    </div>

    <div v-else class="knowledge-search-hero__idle knowledge-search-hero__idle--launching">
      <div class="knowledge-search-hero__copy" data-ignore-quantum-pulse>
        <KnowledgeLaunchLoader class="knowledge-search-hero__loader" />
      </div>
      <div class="knowledge-search-hero__composer-spacer" aria-hidden="true" />
    </div>
  </section>
</template>

<style scoped>
.knowledge-search-hero {
  display: grid;
  place-items: center;
  align-content: center;
  min-height: 0;
  padding: 24px;
}

.knowledge-search-hero--launching {
  cursor: progress;
}

.knowledge-search-hero--closing {
  pointer-events: none;
  cursor: progress;
}

.knowledge-search-hero__idle {
  position: relative;
  display: grid;
  justify-items: center;
  gap: 24px;
  width: min(760px, 100%);
  transform-origin: 50% 52%;
}

.knowledge-search-hero__idle--closing {
  will-change: transform, opacity;
}

.knowledge-search-hero__copy {
  position: relative;
  display: grid;
  justify-items: center;
  max-width: 760px;
  padding: 4px 12px 8px;
  text-align: center;
}

.split-text-title {
  position: relative;
  margin: 0;
  color: #f7fcff;
  background: none;
  -webkit-text-fill-color: currentColor;
  -webkit-text-stroke: 0;
  font-family: var(--font-display);
  font-size: 42px;
  font-weight: 800;
  line-height: 1.08;
  letter-spacing: 0;
  text-wrap: balance;
  text-shadow:
    0 1px 0 rgba(4, 12, 18, 0.58),
    0 0 16px rgba(232, 252, 255, 0.44),
    0 0 34px rgba(103, 216, 255, 0.26),
    0 0 58px rgba(52, 160, 214, 0.16);
  filter:
    drop-shadow(0 10px 24px rgba(4, 16, 24, 0.42))
    drop-shadow(0 0 20px rgba(103, 216, 255, 0.18));
}

.split-text-title__inner {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  perspective: 780px;
}

.split-text-title__char {
  display: inline-block;
  min-width: 0.58em;
  padding-inline: 0.01em;
  transform-origin: 50% 100%;
  animation: split-text-reveal 760ms cubic-bezier(0.19, 1, 0.22, 1) both;
  animation-delay: calc(100ms + var(--split-index) * 44ms);
  backface-visibility: hidden;
}

.split-text-title__char:nth-child(2n) {
  animation-name: split-text-reveal-alt;
}

.knowledge-search-hero__loader {
  transform-origin: 50% 50%;
}

.knowledge-search-hero__error {
  max-width: min(640px, calc(100vw - 48px));
  margin: -8px 0 0;
  padding: 10px 14px;
  border: 1px solid rgba(245, 190, 104, 0.3);
  border-radius: 14px;
  color: #f3d9a4;
  background: rgba(34, 25, 17, 0.42);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.07),
    0 12px 34px rgba(0, 0, 0, 0.18);
  font-size: 13px;
  line-height: 1.6;
  text-align: center;
  backdrop-filter: blur(12px);
}

.knowledge-search-hero__composer-spacer {
  width: min(640px, 100%);
  height: 96px;
}

@keyframes split-text-reveal {
  0% {
    opacity: 0;
    filter: blur(9px) brightness(1.55);
    transform: translate3d(0, 24px, 0) rotateX(68deg) scale(0.88);
  }

  58% {
    opacity: 1;
    filter: blur(1px) brightness(1.22);
    transform: translate3d(0, -3px, 0) rotateX(-8deg) scale(1.02);
  }

  100% {
    opacity: 1;
    filter: none;
    transform: translate3d(0, 0, 0) rotateX(0deg) scale(1);
  }
}

@keyframes split-text-reveal-alt {
  0% {
    opacity: 0;
    filter: blur(9px) brightness(1.5);
    transform: translate3d(0, 20px, 0) rotateX(62deg) translateX(5px) scale(0.9);
  }

  58% {
    opacity: 1;
    filter: blur(1px) brightness(1.18);
    transform: translate3d(0, -2px, 0) rotateX(-6deg) translateX(-1px) scale(1.015);
  }

  100% {
    opacity: 1;
    filter: none;
    transform: translate3d(0, 0, 0) rotateX(0deg) translateX(0) scale(1);
  }
}

@media (prefers-reduced-motion: reduce) {
  .split-text-title__char {
    animation-duration: 1ms;
    animation-delay: 0ms;
  }
}

@media (max-width: 920px) {
  .knowledge-search-hero {
    padding: 24px 18px;
  }

  .split-text-title {
    font-size: 30px;
  }

  .knowledge-search-hero__composer-spacer {
    height: 86px;
  }
}
</style>
