import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import CompanyBrand from '../CompanyBrand.vue'

describe('CompanyBrand', () => {
  it('renders the supplied decorative logo and keeps the product name', () => {
    const wrapper = mount(CompanyBrand)
    const logo = wrapper.get('img.company-brand__mark')

    expect(logo.attributes('src')).toBe('/favicon-logo.svg')
    expect(logo.attributes('alt')).toBe('')
    expect(logo.attributes('aria-hidden')).toBe('true')
    expect(wrapper.get('.company-brand__name').text()).toBe('DC-Agent')
  })
})
