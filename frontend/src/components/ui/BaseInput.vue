<script setup lang="ts">
defineOptions({ inheritAttrs: false })

const model = defineModel<string>({ default: '' })

withDefaults(defineProps<{
  type?: string
  placeholder?: string
  disabled?: boolean
}>(), {
  type: 'text',
  placeholder: '',
  disabled: false,
})
</script>

<template>
  <input
    v-bind="$attrs"
    v-model="model"
    class="base-input"
    :type="type"
    :placeholder="placeholder"
    :disabled="disabled"
  />
</template>

<style scoped>
.base-input {
  width: 100%;
  min-width: 0;
  height: 38px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  color: var(--color-text);
  background: rgba(17, 28, 37, 0.72);
  padding: 0 11px;
  outline: 0;
  font: inherit;
}

.base-input::placeholder {
  color: #879ba8;
}

.base-input:focus {
  border-color: var(--color-border);
  box-shadow: none;
}

.base-input:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}
</style>
