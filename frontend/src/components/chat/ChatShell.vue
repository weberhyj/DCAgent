<script setup lang="ts">
import { computed, defineAsyncComponent, nextTick, onBeforeUnmount, onMounted, shallowRef, useTemplateRef, watch } from 'vue'
import { gsap } from 'gsap'
import ChatTranscript from './ChatTranscript.vue'
import ComposerBar from './ComposerBar.vue'
import CompanyBrand from './CompanyBrand.vue'
import KnowledgeSearchHero from './KnowledgeSearchHero.vue'
import { useChat } from '@/composables/useChat'
import type { ComposerMode } from '@/types/chat'

const QuantumNetworkBackground = defineAsyncComponent(() => import('./QuantumNetworkBackground.vue'))
const HERO_CLOSING_DURATION_MS = 720
const HERO_LOADING_MIN_DURATION_MS = 3000
const shellRef = useTemplateRef<HTMLElement>('shell')

const {
  messages,
  loading,
  sending,
  error,
  loadFreshSession,
  sendMessage,
} = useChat()

const launchFromHero = shallowRef(false)
const heroComposerClosing = shallowRef(false)
const heroLoadingElapsed = shallowRef(true)
const hasMessages = computed(() => messages.value.length > 0)
const isEmptySession = computed(() => !hasMessages.value)
const isHeroLoadingHold = computed(() => launchFromHero.value && !heroLoadingElapsed.value)
const isHeroVisible = computed(() => isEmptySession.value || heroComposerClosing.value || isHeroLoadingHold.value)
const isHeroSearching = computed(() => launchFromHero.value && !heroComposerClosing.value && (sending.value || isHeroLoadingHold.value))
const isSearchTransitioning = computed(() => isHeroVisible.value && isHeroSearching.value)
const canShowAnswerPanel = computed(() => launchFromHero.value && hasMessages.value && heroLoadingElapsed.value)
const shouldCancelHeroLaunch = computed(() => {
  return launchFromHero.value && isEmptySession.value && !sending.value && heroLoadingElapsed.value
})
let answerPanelTimeline: gsap.core.Timeline | undefined
let heroClosingTimeline: gsap.core.Timeline | undefined
let heroLoadingTimer: number | undefined
let heroClosingTimer: number | undefined

onMounted(() => {
  void loadFreshSession()
})

onBeforeUnmount(() => {
  clearHeroClosingTimer()
  clearHeroLoadingTimer()
  heroClosingTimeline?.kill()
  answerPanelTimeline?.kill()
})

function prefersReducedMotion() {
  return window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
}

function clearHeroLoadingTimer() {
  if (heroLoadingTimer === undefined) return
  window.clearTimeout(heroLoadingTimer)
  heroLoadingTimer = undefined
}

function clearHeroClosingTimer() {
  if (heroClosingTimer === undefined) return
  window.clearTimeout(heroClosingTimer)
  heroClosingTimer = undefined
}

function startHeroLoadingTimer() {
  clearHeroLoadingTimer()
  heroLoadingElapsed.value = false
  heroLoadingTimer = window.setTimeout(() => {
    heroLoadingElapsed.value = true
    heroLoadingTimer = undefined
  }, HERO_LOADING_MIN_DURATION_MS)
}

function finishHeroClosingPhase() {
  clearHeroClosingTimer()
  heroClosingTimeline?.kill()
  heroClosingTimeline = undefined
  heroComposerClosing.value = false
  launchFromHero.value = true
  startHeroLoadingTimer()
}

async function startHeroClosingPhase() {
  clearHeroLoadingTimer()
  clearHeroClosingTimer()
  heroClosingTimeline?.kill()
  launchFromHero.value = false
  heroLoadingElapsed.value = true
  heroComposerClosing.value = true
  heroClosingTimer = window.setTimeout(finishHeroClosingPhase, HERO_CLOSING_DURATION_MS)
  await nextTick()
  playHeroComposerClose()
}

function playHeroComposerClose() {
  const hero = shellRef.value?.querySelector<HTMLElement>('.knowledge-search-hero')
  const composer = hero?.querySelector<HTMLElement>('.composer-wrap.center .composer')
  if (!hero || !composer) return

  const title = hero.querySelector<HTMLElement>('.split-text-title')
  const leftShutter = composer.querySelector<HTMLElement>('.composer-shutter--left')
  const rightShutter = composer.querySelector<HTMLElement>('.composer-shutter--right')
  const contentTargets = Array.from(composer.children).filter((child) => {
    return !child.classList.contains('composer-shutter')
  }) as HTMLElement[]
  const shutterTargets = [leftShutter, rightShutter].filter(Boolean) as HTMLElement[]

  heroClosingTimeline?.kill()
  gsap.set(composer, {
    transformOrigin: '50% 50%',
    overflow: 'hidden',
    willChange: 'transform,opacity,filter',
  })

  if (prefersReducedMotion()) {
    heroClosingTimeline = gsap.timeline()
      .to([title, composer].filter(Boolean) as HTMLElement[], {
        autoAlpha: 0,
        duration: 0.08,
        ease: 'power1.out',
      })
    return
  }

  if (shutterTargets.length) {
    gsap.set(shutterTargets, { autoAlpha: 1, scaleX: 0 })
  }

  heroClosingTimeline = gsap.timeline({ defaults: { ease: 'power3.inOut' } })
  if (title) {
    heroClosingTimeline.to(title, {
      autoAlpha: 0,
      y: -18,
      scale: 0.92,
      filter: 'blur(8px) brightness(1.45)',
      duration: 0.28,
    }, 0)
  }
  if (contentTargets.length) {
    heroClosingTimeline.to(contentTargets, {
      autoAlpha: 0,
      y: -2,
      duration: 0.22,
      stagger: { amount: 0.1, from: 'edges' },
    }, 0)
  }
  if (leftShutter) {
    heroClosingTimeline.to(leftShutter, { scaleX: 1, duration: 0.48 }, 0.08)
  }
  if (rightShutter) {
    heroClosingTimeline.to(rightShutter, { scaleX: 1, duration: 0.48 }, 0.08)
  }
  heroClosingTimeline.to(composer, {
    autoAlpha: 0,
    scaleX: 0.08,
    scaleY: 0.82,
    filter: 'brightness(1.4) saturate(1.2)',
    duration: 0.5,
  }, 0.2)
}

function playAnswerPanelIntro() {
  const workspace = shellRef.value?.querySelector<HTMLElement>('.knowledge-workspace')
  if (!workspace) return

  const header = workspace.querySelector<HTMLElement>('.query-header')
  const panel = workspace.querySelector<HTMLElement>('.answer-panel')
  const embeddedComposer = workspace.querySelector<HTMLElement>('.composer-wrap.embedded .composer')
  const messageItems = Array.from(workspace.querySelectorAll<HTMLElement>('.answer-panel .message')).slice(0, 8)
  const animatedTargets = [header, panel, embeddedComposer, ...messageItems].filter(Boolean) as HTMLElement[]

  if (!panel || !animatedTargets.length) return

  answerPanelTimeline?.kill()

  if (prefersReducedMotion()) {
    gsap.fromTo(animatedTargets, { autoAlpha: 0 }, {
      autoAlpha: 1,
      duration: 0.12,
      ease: 'power1.out',
      clearProps: 'opacity,visibility',
    })
    return
  }

  gsap.set(animatedTargets, { willChange: 'transform,opacity' })
  gsap.set(panel, { transformOrigin: '50% 100%' })

  answerPanelTimeline = gsap.timeline({
    defaults: { ease: 'power3.out' },
    onComplete: () => {
      gsap.set(animatedTargets, { clearProps: 'transform,transformOrigin,opacity,visibility,willChange' })
      answerPanelTimeline = undefined
    },
  })

  answerPanelTimeline
    .fromTo(panel, { autoAlpha: 0, y: 42, scale: 0.965 }, { autoAlpha: 1, y: 0, scale: 1, duration: 0.48 }, 0)

  if (header) {
    answerPanelTimeline.fromTo(header, { autoAlpha: 0, y: -10 }, { autoAlpha: 1, y: 0, duration: 0.3 }, 0.08)
  }

  if (embeddedComposer) {
    answerPanelTimeline.fromTo(
      embeddedComposer,
      { autoAlpha: 0, y: 22, scale: 0.98 },
      { autoAlpha: 1, y: 0, scale: 1, duration: 0.36 },
      0.16,
    )
  }

  if (messageItems.length) {
    answerPanelTimeline.fromTo(
      messageItems,
      { autoAlpha: 0, y: 14 },
      { autoAlpha: 1, y: 0, duration: 0.32, stagger: 0.045 },
      0.22,
    )
  }
}

watch(canShowAnswerPanel, async (canShow) => {
  if (!canShow) return

  launchFromHero.value = false
  await nextTick()
  playAnswerPanelIntro()
})

watch(shouldCancelHeroLaunch, (shouldCancel) => {
  if (shouldCancel) {
    launchFromHero.value = false
  }
})

function handleSend(payload: { content: string; mode: ComposerMode }) {
  if (isHeroVisible.value && !heroComposerClosing.value && !launchFromHero.value) {
    void startHeroClosingPhase()
  }
  void sendMessage(payload.content, payload.mode)
}
</script>

<template>
  <main ref="shell" class="app-shell" aria-label="DCAgent 公司资料库问答">
    <QuantumNetworkBackground class="quantum-bg" :auto-pulse="isSearchTransitioning" />

    <CompanyBrand />

    <section
      class="knowledge-workspace"
      :class="{ 'knowledge-workspace--empty': isHeroVisible }"
      aria-label="DCAgent资料库问答工作台"
    >
      <header v-if="!isHeroVisible" class="query-header">
        <h1>DC智识中枢</h1>
      </header>

      <KnowledgeSearchHero
        v-if="isHeroVisible"
        :closing="heroComposerClosing"
        :error="error"
        :launching="isHeroSearching"
        :sending="isHeroSearching || heroComposerClosing"
        :searching="isHeroSearching"
        @send="handleSend"
      />

      <div v-else class="answer-panel" data-ignore-quantum-pulse>
        <ChatTranscript
          :messages="messages"
          :loading="loading"
          :error="error"
        />
        <ComposerBar :sending="sending" variant="embedded" @send="handleSend" />
      </div>
    </section>
  </main>
</template>

<style scoped>
.app-shell {
  --answer-panel-width: min(1120px, calc(100% - 52px));
  --answer-panel-width-mobile: calc(100% - 24px);
  --answer-panel-border: rgba(103, 216, 255, 0.2);

  position: relative;
  isolation: isolate;
  display: block;
  height: 100vh;
  min-height: 100vh;
  overflow: hidden;
  background: #071017;
  color: var(--color-text);
}

.app-shell::before {
  position: absolute;
  inset: 0;
  z-index: 1;
  background:
    radial-gradient(circle at 70% 44%, rgba(103, 216, 255, 0.15), transparent 38%),
    radial-gradient(circle at 38% 18%, rgba(167, 139, 250, 0.08), transparent 30%),
    linear-gradient(90deg, rgba(7, 16, 23, 0.24) 0%, rgba(7, 16, 23, 0.09) 38%, rgba(7, 16, 23, 0.02) 100%),
    linear-gradient(180deg, rgba(255, 255, 255, 0.045), rgba(7, 16, 23, 0.05));
  pointer-events: none;
  content: "";
}

.quantum-bg {
  z-index: 0;
}

.knowledge-workspace {
  position: relative;
  z-index: 2;
  isolation: isolate;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  min-width: 0;
  height: 100vh;
  min-height: 100vh;
  overflow: hidden;
  background:
    radial-gradient(ellipse at 62% 48%, rgba(103, 216, 255, 0.07) 0%, rgba(13, 23, 31, 0.018) 54%, rgba(7, 16, 23, 0.12) 100%),
    linear-gradient(90deg, rgba(103, 216, 255, 0.035) 1px, transparent 1px),
    linear-gradient(180deg, rgba(255, 255, 255, 0.035), transparent 34%);
  background-size: 100% 100%, 96px 100%, 100% 100%;
}

.knowledge-workspace::before,
.knowledge-workspace::after {
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
  content: "";
}

.knowledge-workspace::before {
  background:
    radial-gradient(ellipse at 64% 48%, rgba(125, 211, 252, 0.055) 0%, rgba(13, 17, 23, 0.012) 42%, rgba(7, 16, 23, 0.08) 100%),
    linear-gradient(90deg, rgba(7, 16, 23, 0.12) 0%, rgba(7, 16, 23, 0.01) 52%, rgba(7, 16, 23, 0.09) 100%);
  opacity: 0.5;
}

.knowledge-workspace::after {
  background:
    radial-gradient(ellipse at center, transparent 0%, transparent 52%, rgba(7, 16, 23, 0.13) 100%),
    linear-gradient(90deg, rgba(103, 216, 255, 0.038) 1px, transparent 1px),
    linear-gradient(180deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px),
    repeating-linear-gradient(180deg, rgba(255, 255, 255, 0.012) 0 1px, transparent 1px 5px);
  background-size: 100% 100%, 96px 100%, 100% 72px, 100% 5px;
  opacity: 0.38;
}

.knowledge-workspace > * {
  position: relative;
  z-index: 1;
}

.knowledge-workspace--empty {
  grid-template-rows: minmax(0, 1fr);
}

.answer-panel {
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto;
  width: var(--answer-panel-width);
  min-height: 0;
  margin: 0 auto 50px;
  overflow: hidden;
  border: 1px solid var(--answer-panel-border);
  border-radius: 20px;
  background:
    linear-gradient(180deg, rgba(24, 38, 48, 0.56), rgba(10, 17, 24, 0.42)),
    rgba(12, 21, 29, 0.38);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.075),
    0 22px 62px rgba(0, 0, 0, 0.18),
    0 0 42px rgba(80, 198, 238, 0.06);
  backdrop-filter: blur(12px);
}

.query-header {
  width: var(--answer-panel-width);
  margin: 0 auto;
  padding: 28px 0 10px;
}

.query-header h1 {
  margin: 0;
  color: #edfaff;
  font-size: 18px;
  font-weight: 650;
}

@media (max-width: 920px) {
  .app-shell {
    height: auto;
    overflow: auto;
  }

  .app-shell::before {
    opacity: 0.56;
  }

  .knowledge-workspace {
    height: auto;
    min-height: auto;
  }

  .knowledge-workspace--empty {
    min-height: 100vh;
  }

  .query-header {
    width: var(--answer-panel-width-mobile);
    padding: 18px 0 12px;
  }

  .answer-panel {
    width: var(--answer-panel-width-mobile);
    margin: 0 12px 12px;
    border-radius: 18px;
  }
}
</style>
