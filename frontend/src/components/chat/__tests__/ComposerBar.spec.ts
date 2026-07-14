import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import ComposerBar from '../ComposerBar.vue'

describe('ComposerBar', () => {
  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('uses user-safe search mode copy instead of source tracing copy', async () => {
    const wrapper = mount(ComposerBar, {
      props: {
        sending: false,
      },
    })

    await wrapper.get('[data-testid="base-select-trigger"]').trigger('click')

    expect(document.body.textContent).toContain('全库检索')
    expect(document.body.textContent).not.toContain('溯源')
  })

  it('shows a search loading state while the centered composer is submitting', () => {
    const wrapper = mount(ComposerBar, {
      props: {
        sending: true,
        searching: true,
        variant: 'center',
      },
    })

    expect(wrapper.text()).toContain('资料库搜查中')
    expect(wrapper.get('[data-testid="composer-loading"]').exists()).toBe(true)
    expect(wrapper.get('input').attributes('disabled')).toBeDefined()
    expect(wrapper.get('button[type="submit"]').attributes('disabled')).toBeDefined()
  })
})
