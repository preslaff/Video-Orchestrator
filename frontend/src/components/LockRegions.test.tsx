import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import { LockRegions } from './LockRegions'

/** jsdom lays nothing out, so the frame has to be told how big it is. */
function sized(testId = 'locks') {
  const frame = screen.getByTestId(`${testId}-frame`)
  frame.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 200, height: 100, right: 200, bottom: 100, x: 0, y: 0, toJSON: () => ({}) })
  return frame
}

const drag = (frame: HTMLElement, from: [number, number], to: [number, number]) => {
  fireEvent.pointerDown(frame, { clientX: from[0], clientY: from[1], pointerId: 1 })
  fireEvent.pointerMove(frame, { clientX: to[0], clientY: to[1], pointerId: 1 })
  fireEvent.pointerUp(frame, { clientX: to[0], clientY: to[1], pointerId: 1 })
}

describe('LockRegions', () => {
  it('turns a drag into a rectangle in fractions of the frame', async () => {
    const onSave = vi.fn()
    render(<LockRegions url="/media/kf.png" regions={[]} onSave={onSave} onCancel={() => {}} />)
    drag(sized(), [40, 20], [120, 60])
    await userEvent.click(screen.getByTestId('locks-save'))
    expect(onSave).toHaveBeenCalledWith([{ x: 0.2, y: 0.2, w: 0.4, h: 0.4 }])
  })

  it('normalises a drag made in any direction and clamps it to the frame', async () => {
    const onSave = vi.fn()
    render(<LockRegions url="/media/kf.png" regions={[]} onSave={onSave} onCancel={() => {}} />)
    drag(sized(), [120, 60], [-40, 20])       // right-to-left, and off the left edge
    await userEvent.click(screen.getByTestId('locks-save'))
    expect(onSave).toHaveBeenCalledWith([{ x: 0, y: 0.2, w: 0.6, h: 0.4 }])
  })

  it('ignores a click that is not a rectangle', async () => {
    const onSave = vi.fn()
    render(<LockRegions url="/media/kf.png" regions={[]} onSave={onSave} onCancel={() => {}} />)
    drag(sized(), [40, 20], [40, 20])
    await userEvent.click(screen.getByTestId('locks-save'))
    expect(onSave).toHaveBeenCalledWith([])
    expect(screen.getByTestId('locks-save')).toHaveTextContent('Remove the marks')
  })

  it('shows the marks it was given and lets one be removed without touching the rest', async () => {
    const onSave = vi.fn()
    const marks = [{ x: 0.1, y: 0.1, w: 0.2, h: 0.2 }, { x: 0.5, y: 0.5, w: 0.2, h: 0.2 }]
    render(<LockRegions url="/media/kf.png" regions={marks} onSave={onSave} onCancel={() => {}} />)
    expect(screen.getByTestId('locks-save')).toHaveTextContent('Lock 2 regions')
    await userEvent.click(screen.getByTestId('locks-remove-0'))
    await userEvent.click(screen.getByTestId('locks-save'))
    expect(onSave).toHaveBeenCalledWith([marks[1]])
  })

  it('draws nothing while it is busy putting the pixels back', async () => {
    const onSave = vi.fn()
    render(<LockRegions url="/media/kf.png" regions={[]} busy onSave={onSave} onCancel={() => {}} />)
    drag(sized(), [40, 20], [120, 60])
    expect(screen.getByText('putting the pixels back…')).toBeInTheDocument()
    expect(screen.getByTestId('locks-save')).toBeDisabled()
  })
})
