import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { AssetPicker } from './AssetPicker'
import type { AssetOut } from '@/api/types'

const asset = (id: string, label: string, extra: Partial<AssetOut> = {}): AssetOut => ({
  id, kind: 'reference', name: `${id}.png`, label, label_source: 'human', note_caption: '',
  sha256: '', created_at: '', path: `/a/${id}.png`, url: `/api/projects/p1/assets/${id}/file`, ...extra,
})

describe('AssetPicker', () => {
  it('shows the caption under each thumbnail so a pick is informed', () => {
    render(<AssetPicker assets={[asset('a', 'headland', { note_caption: 'a white tower above grey sea' })]}
      selected={[]} onChange={() => {}} />)
    expect(screen.getByText('a white tower above grey sea')).toBeInTheDocument()
  })

  it('flags labels Gemma suggested but nobody has reviewed', () => {
    render(<AssetPicker assets={[asset('a', 'suggested', { label_source: 'auto' }), asset('b', 'confirmed')]}
      selected={[]} onChange={() => {}} />)
    // exactly one AI chip: the reviewed asset carries none
    expect(screen.getAllByText('AI')).toHaveLength(1)
  })

  it('still selects by id', async () => {
    const onChange = vi.fn()
    render(<AssetPicker assets={[asset('a', 'x')]} selected={[]} onChange={onChange} testId="pick" />)
    await userEvent.click(screen.getByTestId('pick-a'))
    expect(onChange).toHaveBeenCalledWith(['a'])
  })

  it('falls back to the file name when there is no label', () => {
    render(<AssetPicker assets={[asset('a', '')]} selected={[]} onChange={() => {}} />)
    expect(screen.getByText('a.png')).toBeInTheDocument()
  })
})
