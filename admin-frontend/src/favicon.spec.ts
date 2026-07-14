import { existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'

describe('administration application favicon', () => {
  it('declares the supplied public SVG favicon', () => {
    const html = readFileSync(join(process.cwd(), 'index.html'), 'utf8')
    const assetPath = join(process.cwd(), 'public', 'favicon-logo.svg')

    expect(html).toContain('<link rel="icon" type="image/svg+xml" href="/favicon-logo.svg" />')
    expect(existsSync(assetPath)).toBe(true)
    expect(readFileSync(assetPath, 'utf8')).toContain('viewBox="0 0 66.77 66.77"')
  })
})
