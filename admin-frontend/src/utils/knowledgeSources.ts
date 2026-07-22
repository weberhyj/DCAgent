import type { KnowledgeSource } from '@/types/chat'

const STRUCTURED_STATUSES = new Set([
  '\u5f85\u786e\u8ba4\u8868\u7ed3\u6784',
  '\u7ed3\u6784\u5316\u5bfc\u5165\u4e2d',
])

export function isStructuredKnowledgeSource(
  source: Pick<KnowledgeSource, 'sourceType' | 'name' | 'status'> | undefined,
) {
  if (!source) return false
  const sourceType = source.sourceType.trim().toLowerCase()
  const sourceName = source.name.trim().toLowerCase()
  return sourceType === 'xlsx'
    || sourceType === 'csv'
    || sourceType === '\u8868\u683c'
    || sourceName.endsWith('.xlsx')
    || sourceName.endsWith('.csv')
    || STRUCTURED_STATUSES.has(source.status)
}
