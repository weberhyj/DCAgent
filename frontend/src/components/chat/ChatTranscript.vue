<script setup lang="ts">
import { Copy, ThumbsDown, ThumbsUp } from 'lucide-vue-next'
import { computed, nextTick, useTemplateRef, watch } from 'vue'
import MultimodalPanel from './MultimodalPanel.vue'
import type { ChatMessage } from '@/types/chat'

const props = defineProps<{
  messages: readonly ChatMessage[]
  loading: boolean
  error: string | null
}>()

const loadingPanelText = '正在调取资料库档案...'
const approveConclusionLabel = '赞同结论'
const rejectConclusionLabel = '反对结论'
const copyConclusionLabel = '复制结论'
const copyText = '复制'
const pendingFallbackText = 'DCAgent 正在思考'
const transcriptRef = useTemplateRef<HTMLElement>('transcript')
const transcriptScrollKey = computed(() => {
  return props.messages.map((message) => {
    const paragraphText = message.paragraphs.map((paragraph) => paragraph.text).join('\n')
    return [message.id, message.pending ? 'pending' : 'ready', message.streaming ? 'streaming' : 'settled', message.content ?? '', paragraphText].join('|')
  }).join('::')
})

watch(
  transcriptScrollKey,
  async () => {
    await nextTick()
    transcriptRef.value?.scrollTo({
      top: transcriptRef.value.scrollHeight,
      behavior: 'smooth',
    })
  },
)
</script>

<template>
  <div ref="transcript" class="transcript" aria-live="polite" data-ignore-quantum-pulse>
    <div v-if="props.error" class="error-banner">{{ props.error }}</div>
    <div v-if="props.loading && props.messages.length === 0" class="loading-panel">{{ loadingPanelText }}</div>

    <article v-for="message in props.messages" :key="message.id" class="message" :class="[message.role, { 'is-streaming': message.streaming }]">
      <div class="message-body">
        <p v-if="message.content" class="user-bubble">{{ message.content }}</p>

        <template v-if="message.role === 'assistant' && message.pending">
          <div class="assistant-pending" aria-live="polite">
            <span class="assistant-pending__scanner" aria-hidden="true" />
            <span class="assistant-pending__text">{{ message.paragraphs[0]?.text || pendingFallbackText }}</span>
            <span class="assistant-pending__dots" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
          </div>
        </template>

        <template v-else>
          <div
            v-for="(paragraph, paragraphIndex) in message.paragraphs"
            :key="`${message.id}-${paragraphIndex}-${paragraph.text}`"
            class="answer-block"
          >
            <p class="answer-paragraph" :class="{ 'answer-paragraph--streaming': message.streaming && paragraphIndex === message.paragraphs.length - 1 }">
              {{ paragraph.text }}<span v-if="message.streaming && paragraphIndex === message.paragraphs.length - 1" class="streaming-caret" aria-hidden="true" />
            </p>
          </div>

          <MultimodalPanel v-if="message.artifacts.length" :artifacts="message.artifacts" />
        </template>

        <div v-if="message.role === 'assistant' && !message.pending && !message.streaming" class="message-tools">
          <button type="button" :aria-label="approveConclusionLabel"><ThumbsUp :size="15" /></button>
          <button type="button" :aria-label="rejectConclusionLabel"><ThumbsDown :size="15" /></button>
          <button type="button" :aria-label="copyConclusionLabel"><Copy :size="15" />{{ copyText }}</button>
        </div>
      </div>
    </article>
  </div>
</template>

<style scoped>
.transcript {
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding: 18px 24px 24px 16px;
  scrollbar-color: rgba(103, 216, 255, 0.48) rgba(255, 255, 255, 0.04);
  scrollbar-gutter: stable;
  scrollbar-width: thin;
}

.transcript::-webkit-scrollbar {
  width: 9px;
}

.transcript::-webkit-scrollbar-track {
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.035);
}

.transcript::-webkit-scrollbar-thumb {
  border: 2px solid transparent;
  border-radius: 999px;
  background: rgba(103, 216, 255, 0.48);
  background-clip: padding-box;
}

.transcript::-webkit-scrollbar-thumb:hover {
  background: rgba(139, 232, 255, 0.68);
  background-clip: padding-box;
}

.error-banner,
.loading-panel {
  margin: 0 auto 18px;
  max-width: 820px;
  padding: 12px 14px;
  border: 1px solid rgba(216, 189, 122, 0.32);
  border-radius: 7px;
  color: #f2d99f;
  background: rgba(216, 189, 122, 0.09);
}

.message {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  max-width: 1090px;
  margin: 0 auto 22px;
}

.message.user {
  display: flex;
  justify-content: flex-end;
  box-sizing: border-box;
  max-width: 100%;
  padding-right: 30px;
}

.message.assistant {
  box-sizing: border-box;
  margin-left: 0;
  margin-right: auto;
  padding-left: 30px;
}

.message-body {
  min-width: 0;
}

.user-bubble {
  max-width: 920px;
  margin: 0 0 4px;
  padding: 7px 12px;
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 20px;
  color: #edf7fb;
  background: rgba(169, 225, 255, 0.1);
  line-height: 1.72;
}

.answer-block {
  margin: 0 0 14px;
}

.answer-paragraph {
  margin: 0 0 14px;
  color: #e4eef3;
  font-size: 15px;
  line-height: 1.86;
}

.answer-paragraph--streaming {
  min-height: 28px;
}

.streaming-caret {
  display: inline-block;
  width: 8px;
  height: 1.15em;
  margin-left: 3px;
  vertical-align: -0.18em;
  border-radius: 999px;
  background: #b8f2ff;
  box-shadow: 0 0 12px rgba(103, 216, 255, 0.52);
  animation: streaming-caret 780ms steps(2, end) infinite;
}

.assistant-pending {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 10px;
  max-width: min(620px, 100%);
  min-height: 42px;
  padding: 10px 14px;
  overflow: hidden;
  border: 1px solid rgba(103, 216, 255, 0.22);
  border-radius: 18px;
  color: #d7f7ff;
  background:
    linear-gradient(90deg, rgba(103, 216, 255, 0.13), rgba(255, 255, 255, 0.045)),
    rgba(10, 22, 30, 0.44);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 0 24px rgba(80, 198, 238, 0.08);
}

.assistant-pending__scanner {
  width: 16px;
  height: 16px;
  flex: 0 0 auto;
  border: 1px solid rgba(184, 242, 255, 0.5);
  border-radius: 50%;
  background:
    radial-gradient(circle, #b8f2ff 0 2px, transparent 3px),
    conic-gradient(from 45deg, rgba(184, 242, 255, 0), rgba(184, 242, 255, 0.85), rgba(184, 242, 255, 0));
  animation: pending-scan 1.1s linear infinite;
  box-shadow: 0 0 14px rgba(103, 216, 255, 0.32);
}

.assistant-pending__text {
  font-size: 14px;
  line-height: 1.55;
}

.assistant-pending__dots {
  display: inline-flex;
  gap: 4px;
  flex: 0 0 auto;
}

.assistant-pending__dots span {
  width: 4px;
  height: 4px;
  border-radius: 50%;
  background: #b8f2ff;
  animation: pending-dot 1s ease-in-out infinite;
}

.assistant-pending__dots span:nth-child(2) {
  animation-delay: 0.14s;
}

.assistant-pending__dots span:nth-child(3) {
  animation-delay: 0.28s;
}

.message-tools {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 16px;
}

.message-tools button {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  min-height: 30px;
  border: 0;
  color: #8b929b;
  background: transparent;
  cursor: pointer;
}

.message-tools button:hover {
  color: var(--color-accent-strong);
}

@media (max-width: 720px) {
  .transcript {
    overflow: visible;
    padding: 14px 14px 18px 10px;
    scrollbar-gutter: auto;
  }

  .message {
    grid-template-columns: 1fr;
  }

  .message.user {
    padding-right: 10px;
  }

  .message.assistant {
    padding-left: 10px;
  }
}

@keyframes pending-scan {
  to {
    transform: rotate(360deg);
  }
}

@keyframes pending-dot {
  0%,
  80%,
  100% {
    opacity: 0.28;
    transform: translateY(0);
  }

  40% {
    opacity: 1;
    transform: translateY(-3px);
  }
}

@keyframes streaming-caret {
  50% {
    opacity: 0;
  }
}
</style>
