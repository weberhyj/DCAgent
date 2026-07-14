import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import App from './App.vue'

describe('Admin App', () => {
  it('renders the router outlet for modular administrator pages', () => {
    const wrapper = mount(App, {
      global: {
        stubs: {
          RouterView: {
            template: '<main data-testid="shared-router-view" />',
          },
        },
      },
    })

    expect(wrapper.find('[data-testid="shared-router-view"]').exists()).toBe(true)
  })
})
