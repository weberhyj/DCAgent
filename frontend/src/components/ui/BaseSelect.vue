<script setup lang="ts">
import { Check, ChevronDown } from 'lucide-vue-next'
import {
  SelectContent,
  SelectItem,
  SelectItemIndicator,
  SelectItemText,
  SelectPortal,
  SelectRoot,
  SelectTrigger,
  SelectValue,
  SelectViewport,
} from 'reka-ui'
import { computed, shallowRef } from 'vue'

defineOptions({ inheritAttrs: false })

export interface BaseSelectOption {
  label: string
  value: string
  disabled?: boolean
}

const props = withDefaults(defineProps<{
  modelValue: string
  options: readonly BaseSelectOption[]
  placeholder?: string
  disabled?: boolean
  ariaLabel?: string
}>(), {
  placeholder: '请选择',
  disabled: false,
  ariaLabel: '选择项',
})

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const isOpen = shallowRef(false)
const lastEmittedValue = shallowRef<string | null>(null)

const selectedLabel = computed(() => (
  props.options.find((option) => option.value === props.modelValue)?.label ?? props.placeholder
))

function emitValue(value: string) {
  if (lastEmittedValue.value === value) return
  lastEmittedValue.value = value
  emit('update:modelValue', value)
  queueMicrotask(() => {
    if (lastEmittedValue.value === value) {
      lastEmittedValue.value = null
    }
  })
}

function handleUpdate(value: unknown) {
  if (typeof value === 'string' && value !== props.modelValue) {
    emitValue(value)
  }
}

function handleOpenUpdate(value: boolean) {
  isOpen.value = value
}

function openSelect() {
  if (!props.disabled) {
    isOpen.value = true
  }
}

function handleOptionClick(option: BaseSelectOption) {
  if (!option.disabled && option.value !== props.modelValue) {
    emitValue(option.value)
  }
  isOpen.value = false
}
</script>

<template>
  <div v-bind="$attrs" class="base-select">
    <SelectRoot
      :model-value="props.modelValue"
      :open="isOpen"
      :disabled="props.disabled"
      @update:model-value="handleUpdate"
      @update:open="handleOpenUpdate"
    >
      <SelectTrigger
        class="base-select-trigger"
        data-testid="base-select-trigger"
        :aria-label="props.ariaLabel"
        @click="openSelect"
      >
        <SelectValue :placeholder="props.placeholder">
          {{ selectedLabel }}
        </SelectValue>
        <ChevronDown class="base-select-icon" :size="16" aria-hidden="true" />
      </SelectTrigger>

      <SelectPortal>
        <SelectContent class="base-select-content" position="popper" :side-offset="6" data-ignore-quantum-pulse>
          <SelectViewport class="base-select-viewport">
            <SelectItem
              v-for="option in props.options"
              :key="option.value"
              class="base-select-option"
              :value="option.value"
              :disabled="option.disabled"
              :data-testid="`base-select-option-${option.value}`"
              @click="handleOptionClick(option)"
            >
              <SelectItemIndicator class="base-select-indicator">
                <Check :size="14" />
              </SelectItemIndicator>
              <SelectItemText>{{ option.label }}</SelectItemText>
            </SelectItem>
          </SelectViewport>
        </SelectContent>
      </SelectPortal>
    </SelectRoot>
  </div>
</template>

<style scoped>
.base-select {
  width: 100%;
  min-width: 0;
}

.base-select-trigger {
  display: inline-flex;
  align-items: center;
  justify-content: space-between;
  gap: 9px;
  width: 100%;
  min-width: 0;
  height: 38px;
  padding: 0 10px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  color: #e3e7eb;
  background: rgba(23, 31, 39, 0.82);
  font: inherit;
  font-size: 13px;
  cursor: pointer;
}

.base-select-trigger:hover {
  border-color: rgba(103, 216, 255, 0.34);
  background: rgba(152, 220, 255, 0.075);
}

.base-select-trigger:focus-visible {
  outline: 0;
  box-shadow: none;
}

.base-select-trigger[data-disabled] {
  cursor: not-allowed;
  opacity: 0.5;
}

.base-select-icon {
  flex: 0 0 auto;
  color: #a9afb7;
}

:global(.base-select-content) {
  position: relative;
  z-index: 1000;
  min-width: var(--reka-select-trigger-width);
  overflow: hidden;
  border: 1px solid rgba(103, 216, 255, 0.28);
  border-radius: 7px;
  background:
    linear-gradient(180deg, rgba(103, 216, 255, 0.13), rgba(255, 255, 255, 0.055)),
    rgba(12, 19, 26, 0.98);
  box-shadow:
    0 18px 54px rgba(0, 0, 0, 0.5),
    inset 0 1px 0 rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(18px);
}

:global(.base-select-viewport) {
  padding: 5px;
}

:global(.base-select-option) {
  position: relative;
  display: flex;
  align-items: center;
  min-height: 32px;
  padding: 0 10px 0 30px;
  border-radius: 5px;
  color: #dfe3e7;
  font-size: 13px;
  outline: 0;
  cursor: pointer;
  user-select: none;
}

:global(.base-select-option[data-highlighted]) {
  color: #effbff;
  background: rgba(103, 216, 255, 0.17);
}

:global(.base-select-option[data-disabled]) {
  color: #656c75;
  cursor: not-allowed;
}

:global(.base-select-indicator) {
  position: absolute;
  left: 9px;
  display: inline-grid;
  place-items: center;
  color: var(--color-accent-strong);
}
</style>
