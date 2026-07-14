<script setup lang="ts">
import { X } from 'lucide-vue-next'
import {
  DialogClose,
  DialogRoot,
  DialogTitle,
} from 'reka-ui'

const props = withDefaults(defineProps<{
  open: boolean
  title: string
  description?: string
  side?: 'center' | 'right'
}>(), {
  description: '',
  side: 'center',
})

const emit = defineEmits<{
  'update:open': [value: boolean]
}>()

function close() {
  emit('update:open', false)
}
</script>

<template>
  <DialogRoot
    :open="props.open"
    :modal="true"
    @update:open="emit('update:open', $event)"
  >
    <Teleport to="body">
      <div v-if="props.open" class="base-dialog-layer">
        <div class="base-dialog-overlay" @click.self="close" />
        <section
          class="base-dialog-content"
          :class="`base-dialog-content--${props.side}`"
          role="dialog"
          aria-modal="true"
          :aria-label="props.title"
        >
          <header class="base-dialog-header">
            <div>
              <DialogTitle class="base-dialog-title">{{ props.title }}</DialogTitle>
              <p v-if="props.description" class="base-dialog-description">
                {{ props.description }}
              </p>
            </div>
            <DialogClose as-child>
              <button
                type="button"
                class="base-dialog-close"
                data-testid="base-dialog-close"
                aria-label="关闭"
                @click="close"
              >
                <X :size="18" />
              </button>
            </DialogClose>
          </header>

          <div class="base-dialog-body">
            <slot />
          </div>
        </section>
      </div>
    </Teleport>
  </DialogRoot>
</template>

<style scoped>
.base-dialog-layer {
  position: fixed;
  inset: 0;
  z-index: 40;
}

.base-dialog-overlay {
  position: fixed;
  inset: 0;
  background: rgba(15, 23, 42, 0.28);
  backdrop-filter: blur(6px);
}

.base-dialog-content {
  position: fixed;
  z-index: 1;
  border: 1px solid var(--color-border);
  background: rgba(255, 255, 255, 0.96);
  color: var(--color-text);
  box-shadow:
    0 28px 76px rgba(27, 43, 60, 0.22),
    inset 0 1px 0 rgba(255, 255, 255, 0.88);
  outline: 0;
}

.base-dialog-content--center {
  top: 50%;
  left: 50%;
  width: min(520px, calc(100vw - 32px));
  max-height: calc(100vh - 48px);
  transform: translate(-50%, -50%);
  border-radius: 12px;
}

.base-dialog-content--right {
  top: 0;
  right: 0;
  width: min(460px, 100vw);
  height: 100vh;
  border-top: 0;
  border-right: 0;
  border-bottom: 0;
}

.base-dialog-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 22px 24px 0;
}

.base-dialog-title {
  margin: 0;
  color: #101820;
  font-size: 20px;
  font-weight: 720;
  line-height: 1.25;
}

.base-dialog-description {
  margin: 8px 0 0;
  color: var(--color-muted);
  font-size: 13px;
  line-height: 1.55;
}

.base-dialog-close {
  display: inline-grid;
  place-items: center;
  flex: 0 0 auto;
  width: 34px;
  height: 34px;
  border: 1px solid var(--color-border);
  border-radius: 8px;
  color: #536578;
  background: #ffffff;
  cursor: pointer;
}

.base-dialog-close:hover {
  border-color: #9fb2c6;
  background: #f4f7fb;
}

.base-dialog-close:focus-visible {
  outline: 2px solid rgba(20, 99, 255, 0.48);
  outline-offset: 2px;
}

.base-dialog-body {
  max-height: calc(100vh - 112px);
  overflow: auto;
  padding: 22px 24px 24px;
}
</style>
