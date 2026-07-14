import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import MultimodalPanel from '../MultimodalPanel.vue'
import type { Artifact } from '@/types/chat'

describe('MultimodalPanel', () => {
  it('never renders artifact source labels on the user-facing answer surface', () => {
    const artifacts: Artifact[] = [
      {
        type: 'summary',
        title: '经营摘要',
        source: 'finance-report.xlsx',
        bullets: ['回款周期需要关注'],
      },
    ]

    const wrapper = mount(MultimodalPanel, {
      props: { artifacts },
    })

    expect(wrapper.text()).toContain('经营摘要')
    expect(wrapper.text()).not.toContain('finance-report.xlsx')
    expect(wrapper.find('.artifact-source').exists()).toBe(false)
  })
})
