<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  text: string
  terms?: readonly string[]
}>()

interface TextSegment {
  text: string
  highlighted: boolean
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

const segments = computed<TextSegment[]>(() => {
  const normalizedTerms = Array.from(new Set(
    (props.terms ?? [])
      .map((term) => term.trim())
      .filter(Boolean),
  )).sort((left, right) => right.length - left.length)

  if (!props.text || !normalizedTerms.length) {
    return [{ text: props.text, highlighted: false }]
  }

  const matcher = new RegExp(`(${normalizedTerms.map(escapeRegExp).join('|')})`, 'gi')
  const parts = props.text.split(matcher).filter((part) => part.length > 0)
  const termSet = new Set(normalizedTerms.map((term) => term.toLocaleLowerCase()))

  return parts.map((part) => ({
    text: part,
    highlighted: termSet.has(part.toLocaleLowerCase()),
  }))
})
</script>

<template>
  <template v-for="(segment, index) in segments" :key="`${index}-${segment.text}`">
    <mark v-if="segment.highlighted" class="match-highlight" data-testid="match-highlight">{{ segment.text }}</mark>
    <span v-else>{{ segment.text }}</span>
  </template>
</template>

<style scoped>
.match-highlight {
  padding: 0 2px;
  border-radius: 3px;
  color: #061118;
  background: rgba(174, 238, 255, 0.88);
  box-shadow: 0 0 14px rgba(103, 216, 255, 0.24);
}
</style>
