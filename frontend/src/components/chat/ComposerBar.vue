<script setup lang="ts">
import { Paperclip, SendHorizontal } from 'lucide-vue-next'
import { computed, shallowRef } from 'vue'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import type { ComposerMode } from '@/types/chat'

const props = withDefaults(defineProps<{
  sending: boolean
  searching?: boolean
  closing?: boolean
  variant?: 'dock' | 'center' | 'embedded'
}>(), {
  closing: false,
  searching: false,
  variant: 'dock',
})

const emit = defineEmits<{
  send: [payload: { content: string; mode: ComposerMode }]
}>()

const content = shallowRef('')
const DEFAULT_COMPOSER_MODE: ComposerMode = 'deep'
const composerHintText = 'DCAgent 会基于资料库线索生成结论，请结合专业判断复核。'
const attachmentLabel = '添加附件'
const inputPlaceholder = '输入搜查问题，或追加检索条件'
const inputLabel = '输入搜查问题'
const searchingLabel = '资料库搜查中'
const sendLabel = '发起搜查'

const isInputDisabled = computed(() => props.sending || props.searching || props.closing)
const isSubmitDisabled = computed(() => isInputDisabled.value || !content.value.trim())

function submit() {
  const trimmed = content.value.trim()
  if (!trimmed) return
  emit('send', { content: trimmed, mode: DEFAULT_COMPOSER_MODE })
  content.value = ''
}
</script>

<template>
  <footer class="composer-wrap" :class="[props.variant, { searching: props.searching, closing: props.closing }]" data-ignore-quantum-pulse>
    <form class="composer" @submit.prevent="submit">
      <span v-if="props.variant === 'center'" class="composer-shutter composer-shutter--left" aria-hidden="true" />
      <span v-if="props.variant === 'center'" class="composer-shutter composer-shutter--right" aria-hidden="true" />
      <BaseButton type="button" class="tool-button" variant="ghost" size="icon" :disabled="isInputDisabled" :aria-label="attachmentLabel">
        <Paperclip :size="21" />
      </BaseButton>
      <BaseInput
        v-model="content"
        class="composer-input"
        type="text"
        :placeholder="inputPlaceholder"
        :aria-label="inputLabel"
        :disabled="isInputDisabled"
      />
      <div v-if="props.searching" class="composer-loading" data-testid="composer-loading" aria-live="polite">
        <span class="loading-ring" aria-hidden="true" />
        <span class="composer-loading-label">{{ searchingLabel }}</span>
      </div>
      <BaseButton class="send-button" variant="primary" size="icon" type="submit" :disabled="isSubmitDisabled" :aria-label="sendLabel">
        <SendHorizontal :size="20" />
      </BaseButton>
    </form>
    <div v-if="props.variant === 'dock'" class="composer-hint">{{ composerHintText }}</div>
  </footer>
</template>

<style scoped>
.composer-wrap {
  padding: 0 34px 24px;
}

.composer-wrap.center {
  width: min(640px, 100%);
  padding: 0 0 5%;
}

.composer-wrap.embedded {
  padding: 13px 18px 16px;
}

.composer {
  position: relative;
  display: grid;
  grid-template-columns: auto minmax(0, 1fr) 42px;
  align-items: center;
  gap: 10px;
  max-width: 900px;
  min-height: 60px;
  margin: 0 auto;
  padding: 0 12px;
  border: 1px solid rgba(103, 216, 255, 0.42);
  border-radius: 18px;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.095), rgba(255, 255, 255, 0.035)),
    rgba(17, 28, 37, 0.78);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.045),
    0 18px 44px rgba(0, 0, 0, 0.22),
    0 0 34px rgba(80, 198, 238, 0.08),
    inset 0 1px 0 rgba(255, 255, 255, 0.09);
  backdrop-filter: blur(18px);
}

.composer-shutter {
  position: absolute;
  top: 0;
  bottom: 0;
  z-index: 3;
  width: 50%;
  border-radius: inherit;
  opacity: 0;
  pointer-events: none;
  border: 1px solid rgba(166, 239, 255, 0.22);
  background:
    linear-gradient(90deg, rgba(213, 251, 255, 0.28), rgba(83, 215, 255, 0.13), rgba(10, 25, 35, 0.88)),
    rgba(11, 26, 35, 0.9);
  box-shadow:
    inset 0 0 24px rgba(196, 247, 255, 0.2),
    inset 0 1px 0 rgba(255, 255, 255, 0.12),
    0 0 30px rgba(80, 198, 238, 0.18);
  backdrop-filter: blur(16px);
}

.composer-shutter::after {
  position: absolute;
  top: 9px;
  bottom: 9px;
  width: 2px;
  content: '';
  background: linear-gradient(180deg, transparent, rgba(215, 252, 255, 0.92), transparent);
  box-shadow: 0 0 16px rgba(111, 225, 255, 0.82);
}

.composer-shutter--left {
  left: 0;
  border-top-right-radius: 10px;
  border-bottom-right-radius: 10px;
  transform-origin: 0 50%;
}

.composer-shutter--left::after {
  right: 0;
}

.composer-shutter--right {
  right: 0;
  border-top-left-radius: 10px;
  border-bottom-left-radius: 10px;
  transform-origin: 100% 50%;
}

.composer-shutter--right::after {
  left: 0;
}

.composer-wrap.closing .composer {
  pointer-events: none;
}

.composer-wrap.center .composer {
  width: 100%;
  max-width: 640px;
  min-height: 64px;
  border-radius: 30px;
  border-color: rgba(129, 229, 255, 0.52);
  background:
    linear-gradient(180deg, rgba(103, 216, 255, 0.12), rgba(255, 255, 255, 0.045)),
    rgba(17, 29, 38, 0.82);
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.055),
    0 24px 74px rgba(0, 0, 0, 0.24),
    0 0 52px rgba(80, 198, 238, 0.14),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
}

.composer-wrap.embedded .composer {
  max-width: 900px;
  min-height: 58px;
  border-radius: 22px;
  border-color: rgba(103, 216, 255, 0.28);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.075), rgba(255, 255, 255, 0.028)),
    rgba(13, 23, 31, 0.54);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.075),
    0 10px 28px rgba(0, 0, 0, 0.16);
  backdrop-filter: blur(12px);
}

.tool-button {
  grid-column: 1;
  grid-row: 1;
  width: 34px;
  height: 34px;
  border: 1px solid transparent;
  border-radius: 12px;
  color: #a9afb7;
  background: transparent;
}

.tool-button:hover {
  border-color: rgba(103, 216, 255, 0.24);
  background: rgba(103, 216, 255, 0.07);
}

.composer-input {
  grid-column: 2;
  grid-row: 1;
  height: 36px;
}

.composer-loading {
  grid-column: 2;
  grid-row: 1;
  justify-self: end;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 36px;
  min-width: 0;
  padding: 0 8px;
  color: #b8f2ff;
  font-size: 13px;
  white-space: nowrap;
  pointer-events: none;
}

.loading-ring {
  width: 14px;
  height: 14px;
  border: 2px solid rgba(184, 242, 255, 0.22);
  border-top-color: #b8f2ff;
  border-radius: 50%;
  animation: loading-spin 850ms linear infinite;
  box-shadow: 0 0 12px rgba(103, 216, 255, 0.28);
}

.composer-wrap.searching .composer {
  border-color: rgba(184, 242, 255, 0.62);
  box-shadow:
    0 0 0 1px rgba(184, 242, 255, 0.12),
    0 24px 74px rgba(0, 0, 0, 0.22),
    0 0 68px rgba(80, 198, 238, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.13);
}

.composer-wrap.searching .composer-input {
  padding-right: 150px;
}

.composer-input {
  border: 0;
  border-radius: 12px;
  outline: 0;
  color: var(--color-text);
  background: transparent;
  box-shadow: none;
}

.send-button {
  grid-column: 3;
  grid-row: 1;
  width: 38px;
  height: 38px;
  border-radius: 14px;
}

.send-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.composer-hint {
  margin: 11px auto 0;
  max-width: 900px;
  color: #6f767f;
  text-align: center;
  font-size: 12px;
}

@keyframes loading-spin {
  to {
    transform: rotate(360deg);
  }
}

@media (max-width: 720px) {
  .composer-wrap {
    padding: 0 18px 18px;
  }

  .composer-wrap.embedded {
    padding: 11px 12px 13px;
  }

  .composer {
    grid-template-columns: auto minmax(0, 1fr) 45px;
    min-height: 58px;
  }

  .composer-wrap.searching .composer-input {
    padding-right: 42px;
  }

  .composer-loading-label {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    clip-path: inset(50%);
    white-space: nowrap;
    border: 0;
  }
}
</style>
