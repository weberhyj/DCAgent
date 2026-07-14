import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import BaseButton from '../BaseButton.vue'
import BaseInput from '../BaseInput.vue'
import BaseSelect from '../BaseSelect.vue'
import BaseDialog from '../BaseDialog.vue'
import AdminLayout from '@/components/layout/AdminLayout.vue'
import AdminPageHeader from '@/components/layout/AdminPageHeader.vue'

describe('base UI components', () => {
  it('renders page headers without eyebrow labels', () => {
    const wrapper = mount(AdminPageHeader, {
      props: {
        title: '知识库管理',
        description: '上传、筛选和维护公司内部资料。',
      },
    })

    expect(wrapper.find('.page-header__copy > span').exists()).toBe(false)
    expect(wrapper.text()).toContain('知识库管理')
  })

  it('keeps the admin shell free of decorative English subtitles', () => {
    const wrapper = mount(AdminLayout, {
      global: {
        mocks: {
          $route: { meta: { title: '知识库管理' } },
        },
        stubs: {
          RouterLink: { template: '<a><slot /></a>' },
          RouterView: { template: '<main />' },
        },
      },
    })

    expect(wrapper.text()).not.toContain('Knowledge Console')
    expect(wrapper.text()).not.toContain('DC-Agent Admin')
    expect(wrapper.find('.admin-topbar__eyebrow').exists()).toBe(false)
  })

  it('emits click from BaseButton while preserving slot content', async () => {
    const wrapper = mount(BaseButton, {
      slots: { default: '新建搜查' },
    })

    await wrapper.get('button').trigger('click')

    expect(wrapper.text()).toContain('新建搜查')
    expect(wrapper.emitted('click')).toHaveLength(1)
  })

  it('updates BaseInput through v-model contract', async () => {
    const wrapper = mount(BaseInput, {
      props: {
        modelValue: '旧问题',
        placeholder: '输入搜查问题',
        'onUpdate:modelValue': (value: string) => wrapper.setProps({ modelValue: value }),
      },
    })

    await wrapper.get('input').setValue('新的问题')

    expect(wrapper.get('input').element.value).toBe('新的问题')
    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['新的问题'])
  })

  it('updates BaseSelect when a visible option is chosen', async () => {
    const wrapper = mount(BaseSelect, {
      attachTo: document.body,
      props: {
        modelValue: 'deep',
        options: [
          { label: '快速检索', value: 'quick' },
          { label: '深度分析', value: 'deep' },
          { label: '溯源搜查', value: 'source' },
        ],
        'onUpdate:modelValue': (value: string) => wrapper.setProps({ modelValue: value }),
      },
    })

    expect(wrapper.text()).toContain('深度分析')

    await wrapper.get('[data-testid="base-select-trigger"]').trigger('click')
    await document.querySelector<HTMLElement>('[data-testid="base-select-option-source"]')?.click()
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('update:modelValue')?.at(-1)).toEqual(['source'])
  })

  it('renders BaseDialog content and emits close through v-model contract', async () => {
    const wrapper = mount(BaseDialog, {
      attachTo: document.body,
      props: {
        open: true,
        title: '资料库投喂',
        'onUpdate:open': (value: boolean) => wrapper.setProps({ open: value }),
      },
      slots: { default: '<p>资料名称</p>' },
    })

    expect(document.body.textContent).toContain('资料库投喂')
    expect(document.body.textContent).toContain('资料名称')

    document.querySelector<HTMLElement>('[data-testid="base-dialog-close"]')?.click()
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('update:open')?.at(-1)).toEqual([false])
  })
})
