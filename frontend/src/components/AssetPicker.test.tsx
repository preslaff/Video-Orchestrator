import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { AssetPicker } from './AssetPicker'
import type { AssetOut } from '@/api/types'

const asset = (id: string, label: string, extra: Partial<AssetOut> = {}): AssetOut => ({
  id, kind: 'reference', name: `${id}.png`, label, label_source: 'human', note_caption: '', note_tags: [], note_text: '', describe_state: '', describe_error: '', thumb_url: `/api/projects/p1/assets/${id}/thumb`,
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

  describe('large sets', () => {
    const many = Array.from({ length: 60 }, (_, i) => asset(`a${i}`, `photo ${i}`, {
      note_tags: i % 3 === 0 ? ['lighthouse'] : ['road'],
      note_caption: i === 42 ? 'A plaque on a grey wall' : '',
      note_text: i === 42 ? 'MIZEN HEAD VISITOR CENTRE' : '',
    }))

    it('keeps small sets as a plain grid without search', () => {
      render(<AssetPicker assets={many.slice(0, 5)} selected={[]} onChange={() => {}} testId="p" />)
      expect(screen.queryByTestId('p-search')).toBeNull()
      expect(screen.getAllByRole('img')).toHaveLength(5)
    })

    it('renders one page at a time and loads thumbnails lazily', async () => {
      render(<AssetPicker assets={many} selected={[]} onChange={() => {}} testId="p" />)
      expect(screen.getByTestId('p-count')).toHaveTextContent('60 of 60')
      const imgs = screen.getAllByRole('img')
      expect(imgs).toHaveLength(48)
      expect(imgs[0]).toHaveAttribute('src', '/api/projects/p1/assets/a0/thumb')
      expect(imgs[0]).toHaveAttribute('loading', 'lazy')
      await userEvent.click(screen.getByTestId('p-more'))
      expect(screen.getAllByRole('img')).toHaveLength(60)
    })

    it('finds an asset by its sign text and filters by a tag chip', async () => {
      const user = userEvent.setup()
      render(<AssetPicker assets={many} selected={[]} onChange={() => {}} testId="p" />)
      await user.type(screen.getByTestId('p-search'), 'mizen')
      expect(screen.getByTestId('p-count')).toHaveTextContent('1 of 60')
      expect(screen.getByTestId('p-a42')).toBeInTheDocument()
      await user.clear(screen.getByTestId('p-search'))
      await user.click(screen.getByTestId('p-tag-lighthouse'))
      expect(screen.getByTestId('p-count')).toHaveTextContent('20 of 60')
    })

    it('keeps the selection visible, in order, when a search hides it', async () => {
      const user = userEvent.setup()
      const onChange = vi.fn()
      render(<AssetPicker assets={many} selected={['a7', 'a3']} onChange={onChange} multi testId="p" />)
      await user.type(screen.getByTestId('p-search'), 'nothing matches this')
      expect(screen.getByText('No assets match.')).toBeInTheDocument()
      expect(screen.getByTestId('p-a7')).toHaveTextContent('1')   // order badge
      expect(screen.getByTestId('p-a3')).toHaveTextContent('2')
      await user.click(screen.getByTestId('p-a7'))
      expect(onChange).toHaveBeenCalledWith(['a3'])
    })
  })
})


describe('AssetPicker finding a newly added asset', () => {
  /** 20 assets (> SEARCH_THRESHOLD) so the search, chips and paging are on. */
  const many = (n: number) =>
    Array.from({ length: n }, (_, i) => asset(`a${i}`, `photo ${i}`, {
      created_at: `2026-09-0${(i % 9) + 1}T00:00:00+00:00`, note_tags: i % 2 ? ['coastal'] : ['tower'],
    }))

  it('lists the newest asset first, so a just-saved frame is the first tile', () => {
    const assets = [...many(20), asset('fresh', 'scene-8-keyframe.png', { created_at: '2026-09-30T00:00:00+00:00' })]
    render(<AssetPicker assets={assets} selected={[]} onChange={() => {}} testId="p" />)
    const tiles = screen.getByTestId('p').querySelectorAll('[data-testid^="p-a"], [data-testid="p-fresh"]')
    expect(tiles[0]).toHaveAttribute('data-testid', 'p-fresh')     // newest, though stored last
  })

  it('says how many are shown and clears an active filter', async () => {
    const user = userEvent.setup()
    render(<AssetPicker assets={many(20)} selected={[]} onChange={() => {}} testId="p" />)
    expect(screen.getByTestId('p-count')).toHaveTextContent('20 of 20 shown')
    expect(screen.queryByTestId('p-clear')).toBeNull()             // nothing to clear yet
    await user.type(screen.getByTestId('p-search'), 'photo 17')
    expect(screen.getByTestId('p-count')).toHaveTextContent('1 of 20 shown')
    await user.click(screen.getByTestId('p-clear'))
    expect(screen.getByTestId('p-count')).toHaveTextContent('20 of 20 shown')
    expect((screen.getByTestId('p-search') as HTMLInputElement).value).toBe('')
  })
})
