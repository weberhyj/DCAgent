<script setup lang="ts">
import { BookOpen, ChevronDown, MessageSquarePlus, Search, Settings, Shield, Trash2 } from 'lucide-vue-next'
import BaseButton from '@/components/ui/BaseButton.vue'
import BaseInput from '@/components/ui/BaseInput.vue'
import type { Conversation } from '@/types/chat'

const props = defineProps<{
  groups: Record<string, Conversation[]>
  activeId: string
  loading: boolean
}>()

const searchQuery = defineModel<string>('searchQuery', { required: true })

const emit = defineEmits<{
  select: [conversationId: string]
  newConversation: []
  delete: [conversationId: string]
  openKnowledge: []
}>()

function handleDelete(event: MouseEvent, conversationId: string) {
  event.stopPropagation()
  emit('delete', conversationId)
}
</script>

<template>
  <aside class="sidebar" aria-label="搜查档案列表">
    <div class="brand-row">
      <div>
        <div class="brand">DCAgent</div>
        <div class="brand-subtitle">资料库搜查员</div>
      </div>
      <BaseButton class="icon-button" type="button" variant="subtle" size="icon" aria-label="折叠菜单">
        <ChevronDown :size="16" />
      </BaseButton>
    </div>

    <BaseButton class="new-button" type="button" variant="primary" :disabled="props.loading" @click="emit('newConversation')">
      <MessageSquarePlus :size="18" />
      <span>新建搜查</span>
      <kbd>⌘ N</kbd>
    </BaseButton>

    <label class="search-box">
      <Search :size="16" />
      <BaseInput v-model="searchQuery" class="search-input" type="search" placeholder="检索搜查档案" aria-label="检索搜查档案" />
    </label>

    <nav class="conversation-nav">
      <section v-for="(items, group) in props.groups" :key="group" class="conversation-group">
        <h2>{{ group }}</h2>
        <button
          v-for="conversation in items"
          :key="conversation.id"
          type="button"
          class="conversation-item"
          :class="{ active: conversation.id === props.activeId }"
          @click="emit('select', conversation.id)"
        >
          <span class="conversation-title">{{ conversation.title }}</span>
          <span class="conversation-meta">{{ conversation.updatedAt }}</span>
          <Trash2
            class="delete-icon"
            :size="15"
            aria-label="删除搜查档案"
            @click="handleDelete($event, conversation.id)"
          />
        </button>
      </section>
    </nav>

    <div class="sidebar-actions">
      <BaseButton type="button" class="side-action" variant="ghost" @click="emit('openKnowledge')">
        <BookOpen :size="18" />
        <span>资料库管理</span>
        <strong>管理员</strong>
      </BaseButton>
      <BaseButton type="button" class="side-action" variant="ghost">
        <Settings :size="18" />
        <span>系统设置</span>
      </BaseButton>
    </div>

    <div class="account-row">
      <div class="avatar">W</div>
      <div>
        <div class="account-name">王总</div>
        <div class="account-role">管理员</div>
      </div>
      <Shield class="account-shield" :size="16" />
    </div>
  </aside>
</template>

<style scoped>
.sidebar {
  display: grid;
  grid-template-rows: auto auto auto minmax(0, 1fr) auto auto;
  height: 100vh;
  min-height: 100vh;
  overflow: hidden;
  padding: 28px 24px 22px;
  background:
    linear-gradient(180deg, rgba(103, 216, 255, 0.08), rgba(255, 255, 255, 0.026) 36%, transparent 74%),
    rgba(15, 24, 32, 0.86);
  box-shadow:
    inset -1px 0 0 rgba(161, 217, 238, 0.1),
    18px 0 60px rgba(0, 0, 0, 0.18);
  backdrop-filter: blur(14px);
}

.brand-row {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  margin-bottom: 22px;
}

.brand {
  color: var(--color-accent-strong);
  font-size: 28px;
  font-weight: 760;
}

.brand-subtitle {
  margin-top: 7px;
  color: var(--color-muted);
  font-size: 13px;
}

.icon-button,
.delete-icon {
  color: var(--color-muted);
}

.icon-button {
  display: inline-grid;
  place-items: center;
  width: 31px;
  height: 31px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.03);
}

.new-button {
  display: flex;
  align-items: center;
  gap: 10px;
  width: 100%;
  height: 45px;
  padding: 0 14px;
  border: 1px solid rgba(129, 229, 255, 0.52);
  border-radius: 7px;
  color: #061118;
  background: linear-gradient(135deg, #c7f7ff, #5ed2f4 48%, #287c9e);
  box-shadow:
    0 12px 28px rgba(0, 0, 0, 0.2),
    0 0 30px rgba(80, 198, 238, 0.12);
  font-size: 15px;
  cursor: pointer;
}

.new-button kbd {
  margin-left: auto;
  color: rgba(6, 17, 24, 0.72);
  font-family: var(--font-mono);
  font-size: 12px;
}

.search-box {
  display: flex;
  align-items: center;
  gap: 9px;
  height: 38px;
  margin: 16px 0 20px;
  padding: 0 12px;
  border: 1px solid var(--color-border);
  border-radius: 6px;
  color: var(--color-muted);
  background: rgba(255, 255, 255, 0.055);
}

.search-input {
  width: 100%;
  height: auto;
  padding: 0;
  border: 0;
  outline: 0;
  color: var(--color-text);
  background: transparent;
  box-shadow: none;
  font: inherit;
}

.conversation-nav {
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  overscroll-behavior: contain;
  padding-right: 8px;
  scrollbar-color: rgba(103, 216, 255, 0.48) rgba(255, 255, 255, 0.04);
  scrollbar-gutter: stable;
  scrollbar-width: thin;
}

.conversation-nav::-webkit-scrollbar {
  width: 8px;
}

.conversation-nav::-webkit-scrollbar-track {
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.035);
}

.conversation-nav::-webkit-scrollbar-thumb {
  border: 2px solid transparent;
  border-radius: 999px;
  background: rgba(103, 216, 255, 0.48);
  background-clip: padding-box;
}

.conversation-nav::-webkit-scrollbar-thumb:hover {
  background: rgba(139, 232, 255, 0.68);
  background-clip: padding-box;
}

.conversation-group {
  margin-bottom: 19px;
}

.conversation-group h2 {
  margin: 0 0 8px;
  color: #737982;
  font-size: 12px;
  font-weight: 520;
}

.conversation-item {
  position: relative;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: center;
  width: 100%;
  min-height: 45px;
  margin-bottom: 4px;
  padding: 0 32px 0 10px;
  border: 1px solid transparent;
  border-radius: 6px;
  color: var(--color-muted);
  background: rgba(255, 255, 255, 0.012);
  text-align: left;
  cursor: pointer;
}

.conversation-item:hover,
.conversation-item.active {
  color: var(--color-text);
  border-color: rgba(103, 216, 255, 0.3);
  background: rgba(139, 212, 255, 0.078);
}

.conversation-item.active::before {
  position: absolute;
  left: -7px;
  top: 9px;
  bottom: 9px;
  width: 2px;
  background: var(--color-accent);
  content: "";
}

.conversation-title {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
}

.conversation-meta {
  color: #777d85;
  font-size: 12px;
}

.delete-icon {
  position: absolute;
  right: 10px;
  opacity: 0;
  transition: opacity 160ms ease, color 160ms ease;
}

.conversation-item:hover .delete-icon,
.conversation-item.active .delete-icon {
  opacity: 1;
}

.delete-icon:hover {
  color: var(--color-accent-strong);
}

.sidebar-actions {
  display: grid;
  gap: 7px;
  padding: 18px 0 16px;
  border-top: 1px solid var(--color-border);
}

.side-action {
  display: flex;
  align-items: center;
  justify-content: flex-start;
  gap: 12px;
  width: 100%;
  height: 38px;
  border: 0;
  color: var(--color-text);
  background: transparent;
  font-size: 14px;
  cursor: pointer;
}

.side-action strong {
  margin-left: auto;
  padding: 3px 6px;
  border-radius: 4px;
  color: var(--color-accent-strong);
  background: rgba(103, 216, 255, 0.14);
  font-size: 11px;
  font-weight: 600;
}

.account-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding-top: 18px;
  border-top: 1px solid var(--color-border);
}

.avatar {
  display: grid;
  place-items: center;
  width: 46px;
  height: 46px;
  border: 1px solid rgba(103, 216, 255, 0.58);
  border-radius: 50%;
  color: var(--color-accent-strong);
  font-family: var(--font-mono);
  font-size: 19px;
}

.account-name {
  color: var(--color-text);
  font-size: 14px;
}

.account-role {
  color: var(--color-muted);
  font-size: 12px;
}

.account-shield {
  margin-left: auto;
  color: #747b84;
}

@media (max-width: 920px) {
  .sidebar {
    height: auto;
    min-height: auto;
    overflow: visible;
    padding: 18px;
  }

  .conversation-nav,
  .account-row {
    display: none;
  }
}
</style>
