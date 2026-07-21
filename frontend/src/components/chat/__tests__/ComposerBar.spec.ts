import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import ComposerBar from '../ComposerBar.vue'

describe('ComposerBar', () => {
  afterEach(() => {
    document.body.innerHTML = ''
  })

  it('hides unfinished search mode choices', () => {
    const wrapper = mount(ComposerBar, {
      props: {
        sending: false,
      },
    })

    expect(wrapper.find('[data-testid="base-select-trigger"]').exists()).toBe(false)
    expect(wrapper.text()).not.toContain('快速检索')
    expect(wrapper.text()).not.toContain('深度分析')
    expect(wrapper.text()).not.toContain('全库检索')
  })

  it('submits trimmed content with the fixed deep mode and clears the input', async () => {
    const wrapper = mount(ComposerBar, {
      props: {
        sending: false,
      },
    })

    const input = wrapper.get('input')
    await input.setValue('  分析现金流风险  ')
    await wrapper.get('form').trigger('submit')

    expect(wrapper.emitted('send')).toEqual([[
      { content: '分析现金流风险', mode: 'deep' },
    ]])
    expect(input.element.value).toBe('')
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
