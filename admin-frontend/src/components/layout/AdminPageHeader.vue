<script setup lang="ts">
defineProps<{
  title: string
  description?: string
}>()
</script>

<template>
  <header class="page-header">
    <div class="page-header__copy">
      <h1>{{ title }}</h1>
      <p v-if="description">{{ description }}</p>
    </div>
    <div v-if="$slots.actions" class="page-header__actions">
      <slot name="actions" />
    </div>
  </header>
</template>

<style scoped>
.page-header {
  display: flex;
  align-items: end;
  justify-content: space-between;
  gap: 24px;
  max-width: 1380px;
  margin: 0 auto 22px;
}

.page-header__copy {
  min-width: 0;
}

.page-header h1 {
  margin: 0;
  color: #101b26;
  font-size: clamp(23px, 2.2vw, 31px);
  font-weight: 780;
  letter-spacing: 0;
}

.page-header p {
  max-width: 720px;
  margin: 8px 0 0;
  color: #667789;
  font-size: 13px;
  line-height: 1.65;
}

.page-header__actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 8px;
}

@media (max-width: 680px) {
  .page-header {
    align-items: start;
    flex-direction: column;
  }
}
</style>
