import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import AdminLayout from '../AdminLayout.vue'

const fetchBackendVersionMock = vi.hoisted(() => vi.fn())

vi.mock('@/services/api', () => ({
  fetchBackendVersion: fetchBackendVersionMock,
}))

describe('AdminLayout', () => {
  beforeEach(() => {
    fetchBackendVersionMock.mockReset()
    fetchBackendVersionMock.mockResolvedValue('0.1.9')
  })

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

  it('shows the backend service version in the service status area', async () => {
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

    await flushPromises()

    expect(wrapper.get('[data-testid="backend-version"]').text()).toBe('v0.1.9')
    expect(fetchBackendVersionMock).toHaveBeenCalledTimes(1)
    expect(wrapper.text()).not.toContain('前端版本')
  })
})
