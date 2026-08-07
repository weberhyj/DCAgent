import { effectScope } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useBackendVersion } from './useBackendVersion'

const fetchBackendVersionMock = vi.hoisted(() => vi.fn())

vi.mock('@/services/api', () => ({
  fetchBackendVersion: fetchBackendVersionMock,
}))

describe('useBackendVersion', () => {
  beforeEach(() => {
    fetchBackendVersionMock.mockReset()
  })

  it('formats a successfully loaded backend version', async () => {
    fetchBackendVersionMock.mockResolvedValue('0.1.9')
    const scope = effectScope()
    const state = scope.run(() => useBackendVersion())
    if (!state) throw new Error('failed to create backend version state')

    await state.load()

    expect(state.displayVersion.value).toBe('v0.1.9')
    scope.stop()
  })

  it('falls back without throwing when the request fails', async () => {
    fetchBackendVersionMock.mockRejectedValue(new Error('unavailable'))
    const scope = effectScope()
    const state = scope.run(() => useBackendVersion())
    if (!state) throw new Error('failed to create backend version state')

    await expect(state.load()).resolves.toBeUndefined()
    expect(state.displayVersion.value).toBe('版本未知')
    scope.stop()
  })
})
