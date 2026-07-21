import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it } from 'vitest'
import ComposerBar from '../ComposerBar.vue'
import composerBarSource from '../ComposerBar.vue?raw'

function getStyleRule(source: string, selector: string) {
  const marker = `${selector} {`
  const ruleStart = source.indexOf(marker)
  if (ruleStart === -1) return ''
  const declarationsStart = ruleStart + marker.length
  return source.slice(declarationsStart, source.indexOf('}', declarationsStart))
}

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

  it('hides the attachment entry by default while preserving its implementation', () => {
    const wrapper = mount(ComposerBar, {
      props: {
        sending: false,
      },
    })

    expect(wrapper.find('.tool-button').exists()).toBe(false)
    expect(wrapper.find('button[aria-label="添加附件"]').exists()).toBe(false)
    expect(wrapper.find('input').exists()).toBe(true)
    expect(wrapper.find('button[type="submit"]').exists()).toBe(true)
    expect(composerBarSource).toContain('const isAttachmentEntryVisible = false')
    expect(composerBarSource).toContain('v-if="isAttachmentEntryVisible"')
    expect(composerBarSource).toContain('<Paperclip :size="21" />')
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

  it('keeps visible composer controls in a two-column layout while searching', () => {
    const composerRule = getStyleRule(composerBarSource, '.composer')
    const inputRule = getStyleRule(composerBarSource, '.composer-input')
    const loadingRule = getStyleRule(composerBarSource, '.composer-loading')
    const sendRule = getStyleRule(composerBarSource, '.send-button')

    expect(composerRule).toContain('grid-template-columns: minmax(0, 1fr) 42px')
    expect(inputRule).toContain('grid-column: 1')
    expect(inputRule).toContain('grid-row: 1')
    expect(loadingRule).toContain('grid-column: 1')
    expect(loadingRule).toContain('grid-row: 1')
    expect(sendRule).toContain('grid-column: 2')
    expect(sendRule).toContain('grid-row: 1')
  })

  it('visually hides the mobile loading label without removing accessible text', () => {
    const wrapper = mount(ComposerBar, {
      props: {
        sending: true,
        searching: true,
      },
    })
    const label = wrapper.get('.composer-loading-label')
    const mobileStyles = composerBarSource.split('@media (max-width: 720px)')[1] ?? ''
    const labelRule = getStyleRule(mobileStyles, '.composer-loading-label')

    expect(label.text()).toBe('资料库搜查中')
    expect(wrapper.get('[aria-live="polite"]').element.contains(label.element)).toBe(true)
    expect(labelRule).not.toMatch(/display\s*:\s*none/)
    expect(labelRule).toContain('position: absolute')
    expect(labelRule).toContain('width: 1px')
    expect(labelRule).toContain('height: 1px')
    expect(labelRule).toContain('overflow: hidden')
    expect(labelRule).toContain('clip: rect(0, 0, 0, 0)')
    expect(labelRule).toContain('clip-path: inset(50%)')
    expect(labelRule).toContain('white-space: nowrap')
  })
})
