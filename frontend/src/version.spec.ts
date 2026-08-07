import manifest from '../package.json'
import { describe, expect, it } from 'vitest'
import { APP_VERSION, APP_VERSION_LABEL } from './version'

describe('user frontend version', () => {
  it('uses the independent frontend package version', () => {
    expect(APP_VERSION).toBe(manifest.version)
    expect(APP_VERSION_LABEL).toBe(`v${manifest.version}`)
  })
})
