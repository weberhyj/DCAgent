import { afterEach, describe, expect, it, vi } from 'vitest'

const axiosCreateMock = vi.hoisted(() => vi.fn())
const httpMock = vi.hoisted(() => ({
  delete: vi.fn(),
  get: vi.fn(),
  post: vi.fn(),
}))

vi.mock('axios', () => ({
  default: {
    create: axiosCreateMock,
  },
}))

async function loadApi() {
  axiosCreateMock.mockReturnValue(httpMock)
  return import('./api')
}

describe('user chat api service', () => {
  afterEach(() => {
    httpMock.delete.mockReset()
    httpMock.get.mockReset()
    httpMock.post.mockReset()
    axiosCreateMock.mockReset()
    vi.resetModules()
  })

  it('keeps the HTTP timeout long enough for real model answers', async () => {
    await loadApi()

    expect(axiosCreateMock).toHaveBeenCalledWith({
      baseURL: '/api',
      timeout: 70000,
    })
  })

  it('posts search messages to the active conversation endpoint', async () => {
    const bundle = {
      activeConversationId: 'conv-live',
      conversations: [],
      messages: [],
    }
    httpMock.post.mockResolvedValue({ data: bundle })
    const { sendConversationMessage } = await loadApi()

    const result = await sendConversationMessage('conv-live', '查询合同审批流程', 'deep')

    expect(result).toBe(bundle)
    expect(httpMock.post).toHaveBeenCalledWith('/conversations/conv-live/messages', {
      content: '查询合同审批流程',
      mode: 'deep',
    })
  })
})
