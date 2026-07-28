import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

const THREE_CDN_URL =
  'https://cdn.jsdelivr.net/npm/three@0.185.1/build/three.module.js'

describe('Three.js CDN loading', () => {
  it('externalizes the runtime to the pinned jsDelivr ESM build outside tests', () => {
    const config = readFileSync(join(process.cwd(), 'vite.config.ts'), 'utf8')

    expect(config).toContain(THREE_CDN_URL)
    expect(config).toContain("source === 'three'")
    expect(config).toContain('external: true')
    expect(config).toContain("mode !== 'test'")
    expect(config).not.toContain('unpkg.com')
  })
})
