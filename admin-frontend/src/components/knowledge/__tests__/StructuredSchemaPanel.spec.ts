import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import StructuredSchemaPanel from '../StructuredSchemaPanel.vue'

type ColumnType = 'string' | 'integer' | 'decimal' | 'date' | 'datetime' | 'boolean'

interface TestColumn {
  physicalName: string
  originalName: string
  displayName: string
  dataType: ColumnType
  aliases: string[]
  examples: string[]
  sampledRows: number
  nullCount: number
}

interface TestDataset {
  datasetId: string
  sourceId: string
  worksheetName: string
  sampledRows: number
  schemaHash: string
  columns: TestColumn[]
}

interface TestPreview {
  sourceId: string
  datasets: TestDataset[]
  diagnostics: Array<{
    code: string
    message: string
    worksheetName?: string
    columnName?: string | null
    rowNumber?: number | null
  }>
}

const preview: TestPreview = {
  sourceId: 'source-1',
  datasets: [{
    datasetId: 'dataset-1',
    sourceId: 'source-1',
    worksheetName: 'Sheet1',
    sampledRows: 2,
    schemaHash: 'hash-1',
    columns: [
      {
        physicalName: 'amount',
        originalName: 'Amount',
        displayName: 'Amount',
        dataType: 'decimal',
        aliases: [],
        examples: ['1.2', '4.8'],
        sampledRows: 2,
        nullCount: 0,
      },
      {
        physicalName: 'region',
        originalName: 'Region',
        displayName: 'Region',
        dataType: 'string',
        aliases: ['area'],
        examples: ['East'],
        sampledRows: 2,
        nullCount: 1,
      },
    ],
  }],
  diagnostics: [],
}

function mountPanel(overrides: Partial<TestPreview> = {}) {
  return mount(StructuredSchemaPanel, {
    props: { preview: { ...preview, ...overrides } },
  })
}

describe('StructuredSchemaPanel', () => {
  it('renders worksheets, column metadata, examples, and diagnostics', () => {
    const wrapper = mountPanel({
      diagnostics: [{ code: 'duplicate_header', message: 'Duplicate header was normalized' }],
    })

    expect(wrapper.get('[data-testid="structured-schema-panel"]').exists()).toBe(true)
    expect(wrapper.text()).toContain('Sheet1')
    expect(wrapper.text()).toContain('amount')
    expect(wrapper.text()).toContain('Amount')
    expect(wrapper.text()).toContain('1.2')
    expect(wrapper.text()).toContain('2 sampled')
    expect(wrapper.text()).toContain('0 null')
    expect(wrapper.text()).toContain('Duplicate header was normalized')
    expect(wrapper.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeUndefined()
  })

  it('edits every column option and emits a complete camelCase submission', async () => {
    const wrapper = mountPanel()

    await wrapper.get('[data-testid="display-name-amount"]').setValue('Order amount')
    await wrapper.get('[data-testid="aliases-amount"]').setValue('amount, revenue')
    await wrapper.get('[data-testid="type-amount"]').setValue('integer')
    await wrapper.get('[data-testid="aggregate-amount"]').setValue(true)
    await wrapper.get('[data-testid="filter-amount"]').setValue(true)
    await wrapper.get('[data-testid="null-policy-amount"]').setValue('zero')
    await wrapper.get('[data-testid="structured-confirm-button"]').trigger('click')

    expect(wrapper.emitted('confirm')).toEqual([[
      {
        datasets: [{
          datasetId: 'dataset-1',
          columns: [
            {
              physicalName: 'amount',
              displayName: 'Order amount',
              dataType: 'integer',
              aliases: ['amount', 'revenue'],
              allowAggregate: true,
              allowFilter: true,
              nullPolicy: 'zero',
            },
            {
              physicalName: 'region',
              displayName: 'Region',
              dataType: 'string',
              aliases: ['area'],
              allowAggregate: false,
              allowFilter: false,
              nullPolicy: 'ignore',
            },
          ],
        }],
      },
    ]])
  })

  it('clears and disables aggregate when a column becomes non-numeric', async () => {
    const wrapper = mountPanel()
    const aggregate = wrapper.get<HTMLInputElement>('[data-testid="aggregate-amount"]')

    await aggregate.setValue(true)
    await wrapper.get('[data-testid="type-amount"]').setValue('string')

    expect(aggregate.attributes('disabled')).toBeDefined()
    expect(aggregate.element.checked).toBe(false)
    await wrapper.get('[data-testid="structured-confirm-button"]').trigger('click')

    const submission = wrapper.emitted('confirm')?.[0]?.[0] as {
      datasets: Array<{ columns: Array<{ allowAggregate: boolean }> }>
    }
    expect(submission.datasets[0].columns[0].allowAggregate).toBe(false)
  })

  it('blocks only known blocking diagnostics', () => {
    const warning = mountPanel({
      diagnostics: [{ code: 'duplicate_header', message: 'Duplicate header' }],
    })
    const blocking = mountPanel({
      diagnostics: [{ code: 'empty_sheet', message: 'Sheet is empty' }],
    })

    expect(warning.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeUndefined()
    expect(blocking.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()
  })

  it('rejects empty datasets, columns, and unreadable display names', async () => {
    const emptyDatasets = mountPanel({ datasets: [] })
    const emptyColumns = mountPanel({ datasets: [{ ...preview.datasets[0], columns: [] }] })
    expect(emptyDatasets.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()
    expect(emptyColumns.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()

    const blank = mountPanel()
    await blank.get('[data-testid="display-name-amount"]').setValue('  ')
    expect(blank.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()

    const generated = mountPanel({
      datasets: [{
        ...preview.datasets[0],
        columns: [{
          ...preview.datasets[0].columns[0],
          physicalName: 'column_1',
          originalName: '',
          displayName: 'column_1',
        }],
      }],
    })
    expect(generated.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()
    await generated.get('[data-testid="display-name-column_1"]').setValue('Unnamed amount')
    expect(generated.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeUndefined()
  })

  it('rejects blank, duplicate, and cross-column aliases', async () => {
    const blank = mountPanel()
    await blank.get('[data-testid="aliases-amount"]').setValue('revenue, ')
    expect(blank.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()

    const duplicate = mountPanel()
    await duplicate.get('[data-testid="aliases-amount"]').setValue('revenue, Revenue')
    expect(duplicate.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()

    const crossColumn = mountPanel()
    await crossColumn.get('[data-testid="aliases-amount"]').setValue('area')
    expect(crossColumn.get('[data-testid="structured-confirm-button"]').attributes('disabled')).toBeDefined()
  })
})
