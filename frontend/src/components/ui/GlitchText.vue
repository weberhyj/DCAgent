<script setup lang="ts">
defineOptions({ inheritAttrs: false })

withDefaults(defineProps<{
  text: string
  as?: 'span' | 'p' | 'h1' | 'h2' | 'h3'
}>(), {
  as: 'span',
})
</script>

<template>
  <component
    :is="as"
    v-bind="$attrs"
    class="glitch-text"
    :aria-label="text"
  >
    <span class="glitch-text__base">{{ text }}</span>
    <span class="glitch-text__layer glitch-text__layer--cyan" aria-hidden="true">{{ text }}</span>
    <span class="glitch-text__layer glitch-text__layer--violet" aria-hidden="true">{{ text }}</span>
  </component>
</template>

<style scoped>
.glitch-text {
  position: relative;
  display: inline-grid;
  place-items: center;
  margin: 0;
  color: #f7fcff;
  font-family: var(--font-display);
  font-size: clamp(36px, 5vw, 72px);
  font-weight: 850;
  line-height: 0.98;
  letter-spacing: 0;
  text-transform: uppercase;
  text-shadow:
    0 1px 0 rgba(1, 7, 11, 0.82),
    0 0 18px rgba(232, 252, 255, 0.46),
    0 0 42px rgba(103, 216, 255, 0.28);
  isolation: isolate;
}

.glitch-text__base,
.glitch-text__layer {
  grid-area: 1 / 1;
  min-width: 0;
  white-space: nowrap;
}

.glitch-text__base {
  position: relative;
  z-index: 2;
  animation: glitch-base 2.9s steps(1, end) infinite;
}

.glitch-text__layer {
  position: relative;
  z-index: 1;
  opacity: 0.82;
  mix-blend-mode: screen;
  pointer-events: none;
}

.glitch-text__layer--cyan {
  color: #55eaff;
  clip-path: inset(0 0 56% 0);
  transform: translate(-2px, -1px);
  animation: glitch-slice-cyan 1.8s steps(1, end) infinite;
}

.glitch-text__layer--violet {
  color: #b58cff;
  clip-path: inset(48% 0 0 0);
  transform: translate(2px, 1px);
  animation: glitch-slice-violet 2.14s steps(1, end) infinite;
}

@keyframes glitch-base {
  0%,
  87%,
  100% {
    filter: none;
    transform: translate(0);
  }

  88% {
    filter: brightness(1.35);
    transform: translate(1px, 0);
  }

  91% {
    transform: translate(-1px, 0);
  }

  94% {
    filter: brightness(1.12);
    transform: translate(0);
  }
}

@keyframes glitch-slice-cyan {
  0%,
  64%,
  100% {
    clip-path: inset(0 0 56% 0);
    transform: translate(-2px, -1px);
  }

  65% {
    clip-path: inset(12% 0 70% 0);
    transform: translate(-7px, 1px);
  }

  68% {
    clip-path: inset(2% 0 48% 0);
    transform: translate(4px, -1px);
  }

  71% {
    clip-path: inset(36% 0 42% 0);
    transform: translate(-3px, 0);
  }
}

@keyframes glitch-slice-violet {
  0%,
  58%,
  100% {
    clip-path: inset(48% 0 0 0);
    transform: translate(2px, 1px);
  }

  59% {
    clip-path: inset(62% 0 16% 0);
    transform: translate(6px, 0);
  }

  62% {
    clip-path: inset(42% 0 34% 0);
    transform: translate(-5px, 2px);
  }

  66% {
    clip-path: inset(74% 0 8% 0);
    transform: translate(3px, -1px);
  }
}

@media (prefers-reduced-motion: reduce) {
  .glitch-text__base,
  .glitch-text__layer {
    animation-duration: 1ms;
    animation-iteration-count: 1;
  }
}
</style>
