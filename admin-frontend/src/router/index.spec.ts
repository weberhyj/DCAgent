import { describe, expect, it } from 'vitest'
import { createAdminRouter } from './index'

describe('administration router base', () => {
  it('resolves named routes below the configured public base', () => {
    const router = createAdminRouter('/operations/')

    expect(router.resolve({ name: 'overview' }).href).toBe('/operations/overview')
    expect(router.resolve({ name: 'knowledge' }).href).toBe('/operations/knowledge')
    expect(router.resolve({
      name: 'knowledge-source-detail',
      params: { sourceId: 'source-1' },
    }).href).toBe('/operations/knowledge/source-1')
  })
})
