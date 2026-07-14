import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import AdminLayout from '../AdminLayout.vue'

describe('AdminLayout', () => {
  it('renders the supplied decorative logo and keeps the administration brand contract', () => {
    const wrapper = mount(AdminLayout, {
      global: {
        mocks: {
          $route: {
            path: '/',
            meta: { title: '管理概览' },
          },
        },
        stubs: {
          RouterLink: {
            template: '<a><slot /></a>',
          },
          RouterView: true,
        },
      },
    })
    const logo = wrapper.get('img.admin-brand__mark')

    expect(logo.attributes('src')).toBe('/favicon-logo.svg')
    expect(logo.attributes('alt')).toBe('')
    expect(logo.attributes('aria-hidden')).toBe('true')
    expect(wrapper.get('.admin-brand__copy strong').text()).toBe('DC-Agent')
    expect(wrapper.get('.admin-brand').attributes('aria-label')).toBe('返回管理概览')
  })
})
