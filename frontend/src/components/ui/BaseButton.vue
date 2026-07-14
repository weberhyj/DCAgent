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
  color: #061118;
  border-color: rgba(129, 229, 255, 0.52);
  background: linear-gradient(135deg, #b8f2ff, #53c6ef 48%, #287c9e);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.42),
    0 12px 28px rgba(58, 196, 236, 0.18);
}

.base-button--primary:hover:not(:disabled) {
  border-color: rgba(184, 242, 255, 0.72);
  background: linear-gradient(135deg, #d5fbff, #66d7fa 48%, #2d91b7);
}

.base-button--subtle {
  color: #e8ecef;
  border-color: var(--color-border);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.032)),
    rgba(22, 29, 36, 0.78);
}

.base-button--subtle:hover:not(:disabled) {
  border-color: rgba(103, 216, 255, 0.34);
  background: rgba(187, 232, 255, 0.085);
}

.base-button--ghost {
  color: #a9afb7;
  border-color: transparent;
  background: transparent;
}

.base-button--ghost:hover:not(:disabled) {
  color: #eefaff;
  border-color: rgba(103, 216, 255, 0.24);
  background: rgba(160, 221, 255, 0.07);
}

.base-button:focus-visible {
  outline: 2px solid rgba(103, 216, 255, 0.78);
  outline-offset: 2px;
}

.base-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}
</style>
