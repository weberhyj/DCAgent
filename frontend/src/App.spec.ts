import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import App from './App.vue'

describe('App', () => {
  it('mounts the user search shell directly without a shared router', () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          ChatShell: {
            template: '<main data-testid="user-search-app" />',
          },
          RouterView: {
            template: '<main data-testid="shared-router-view" />',
          },
        },
      },
    })

    expect(wrapper.find('[data-testid="user-search-app"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="shared-router-view"]').exists()).toBe(false)
  })
})
