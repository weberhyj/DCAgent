<script setup lang="ts">
import { computed } from 'vue'

defineOptions({ inheritAttrs: false })

const props = withDefaults(defineProps<{
  type?: 'button' | 'submit' | 'reset'
  variant?: 'primary' | 'subtle' | 'ghost'
  size?: 'sm' | 'md' | 'icon'
  disabled?: boolean
}>(), {
  type: 'button',
  variant: 'subtle',
  size: 'md',
  disabled: false,
})

const emit = defineEmits<{
  click: [event: MouseEvent]
}>()

const buttonClasses = computed(() => [
  'base-button',
  `base-button--${props.variant}`,
  `base-button--${props.size}`,
])
</script>

<template>
  <button
    v-bind="$attrs"
    :type="props.type"
    :class="buttonClasses"
    :disabled="props.disabled"
    @click="emit('click', $event)"
  >
    <slot />
  </button>
</template>

<style scoped>
.base-button {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  min-width: 0;
  border: 1px solid transparent;
  border-radius: 6px;
  color: var(--color-text);
  background: transparent;
  font: inherit;
  font-weight: 560;
  line-height: 1;
  white-space: nowrap;
  cursor: pointer;
  user-select: none;
}

.base-button--md {
  min-height: 40px;
  padding: 0 14px;
  font-size: 14px;
}

.base-button--sm {
  min-height: 34px;
  padding: 0 11px;
  font-size: 13px;
}

.base-button--icon {
  width: 38px;
  height: 38px;
  padding: 0;
}

.base-button--primary {
  color: #ffffff;
  border-color: #0d53dc;
  background: linear-gradient(135deg, #256eff, #0d53dc 58%, #0a3b9d);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.22),
    0 12px 26px rgba(20, 99, 255, 0.22);
}

.base-button--primary:hover:not(:disabled) {
  border-color: #0a3b9d;
  background: linear-gradient(135deg, #3a7cff, #1463ff 58%, #0a3b9d);
}

.base-button--subtle {
  color: #1f2c39;
  border-color: var(--color-border);
  background: #ffffff;
}

.base-button--subtle:hover:not(:disabled) {
  border-color: #9fb2c6;
  background: #f4f7fb;
}

.base-button--ghost {
  color: #536578;
  border-color: transparent;
  background: transparent;
}

.base-button--ghost:hover:not(:disabled) {
  color: var(--color-accent-strong);
  border-color: #cbd8e5;
  background: #eef4ff;
}

.base-button:focus-visible {
  outline: 2px solid rgba(20, 99, 255, 0.48);
  outline-offset: 2px;
}

.base-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}
</style>
