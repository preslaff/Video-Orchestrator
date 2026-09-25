import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'
import type { TextItem } from '@/api/types'
import { TextItems } from './TextItems'
import { blankText } from '@/lib/titles'

function Harness({ initial = [] as TextItem[], ...rest }: { initial?: TextItem[]; [k: string]: unknown }) {
  const [items, setItems] = useState<TextItem[]>(initial)
  return <TextItems items={items} onChange={setItems} {...rest} />
}

describe('TextItems', () => {
  it('adds, edits and removes a caption', async () => {
    const user = userEvent.setup()
    render(<Harness />)
    expect(screen.getByText('No captions yet.')).toBeInTheDocument()

    await user.click(screen.getByText('Add caption'))
    await user.type(screen.getByTestId('texts-text-0'), 'Открий своя аромат')
    expect(screen.getByTestId('texts-text-0')).toHaveValue('Открий своя аромат')

    await user.click(screen.getByTestId('texts-remove-0'))
    expect(screen.queryByTestId('texts-row-0')).toBeNull()
  })

  it('offers presets as a starting point, not a reference', async () => {
    const user = userEvent.setup()
    render(<Harness presets={{ kicker: { size: 0.03, colour: 'D8B26A', align: 8 } }} />)
    await user.click(screen.getByTestId('texts-preset-kicker'))
    // The preset's values are copied onto the item, so editing one afterwards moves nothing else.
    expect(screen.getByLabelText('caption 1 size')).toHaveValue(3)
    expect(screen.getByTestId('texts-colour-0')).toHaveValue('#d8b26a')
  })

  it('shows size as a percentage of the frame height', async () => {
    const user = userEvent.setup()
    render(<Harness initial={[blankText({ size: 0.048 })]} />)
    expect(screen.getByLabelText('caption 1 size')).toHaveValue(4.8)
    await user.clear(screen.getByLabelText('caption 1 size'))
    await user.type(screen.getByLabelText('caption 1 size'), '7.5')
    expect(screen.getByLabelText('caption 1 size')).toHaveValue(7.5)
  })

  it('warns when a caption runs past the end of what it is written over', () => {
    render(<Harness initial={[blankText({ start: 9, duration: 3 })]} limit={10} />)
    expect(screen.getByTestId('texts-overrun-0')).toHaveTextContent('past the end (10.0s)')
  })

  it('says nothing when the caption fits', () => {
    render(<Harness initial={[blankText({ start: 1, duration: 2 })]} limit={10} />)
    expect(screen.queryByTestId('texts-overrun-0')).toBeNull()
  })

  it('previews only the captions that are on screen', () => {
    render(<Harness videoUrl="/media/a.mp4" initial={[
      blankText({ text: 'сега', start: 0, duration: 5 }),
      blankText({ text: 'по-късно', start: 9, duration: 2 }),
    ]} />)
    // The video starts at 0, so the first is up and the second is not.
    expect(screen.getByTestId('texts-preview-0')).toHaveTextContent('сега')
    expect(screen.queryByTestId('texts-preview-1')).toBeNull()
  })

  it('is honest that only the timing is exact', () => {
    render(<Harness videoUrl="/media/a.mp4" />)
    expect(screen.getByText(/timing is exact/i)).toBeInTheDocument()
    expect(screen.getByText(/Placement is approximate/i)).toBeInTheDocument()
  })

  it('uses the effects the server offers', async () => {
    render(<Harness initial={[blankText()]} effects={['none', 'fade', 'typewriter']} />)
    const select = screen.getByLabelText('caption 1 effect')
    expect([...select.querySelectorAll('option')].map((o) => o.textContent))
      .toEqual(['none', 'fade', 'typewriter'])
  })
})
