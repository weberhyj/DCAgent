import { describe, expect, it } from 'vitest'
import { normalizeAdminBasePath, resolveAdminPublicAsset } from './adminBasePath'

describe('normalizeAdminBasePath', () => {
  it('defaults to the administration subpath', () => {
    expect(normalizeAdminBasePath()).toBe('/admin/')
    expect(normalizeAdminBasePath('')).toBe('/admin/')
    expect(normalizeAdminBasePath('   ')).toBe('/admin/')
  })

  it('normalizes custom and root deployments', () => {
    expect(normalizeAdminBasePath('operations')).toBe('/operations/')
    expect(normalizeAdminBasePath('/operations')).toBe('/operations/')
    expect(normalizeAdminBasePath('/operations/')).toBe('/operations/')
    expect(normalizeAdminBasePath('/')).toBe('/')
  })

  it('resolves public assets below the configured base', () => {
    expect(resolveAdminPublicAsset('favicon-logo.svg', '/admin/')).toBe('/admin/favicon-logo.svg')
    expect(resolveAdminPublicAsset('/favicon-logo.svg', '/operations/')).toBe(
      '/operations/favicon-logo.svg',
    )
    expect(resolveAdminPublicAsset('favicon-logo.svg', '/')).toBe('/favicon-logo.svg')
  })

  it.each([
    'https://intranet.example/admin/',
    '//intranet.example/admin/',
    '/admin/?debug=1',
    '/admin/#overview',
    '/admin/../root/',
    '/admin/./overview/',
    '\\admin\\',
  ])('rejects unsafe base path %s', (value) => {
    expect(() => normalizeAdminBasePath(value)).toThrow('Invalid VITE_ADMIN_BASE_PATH')
  })
})
