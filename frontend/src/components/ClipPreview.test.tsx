import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { ClipPreview, playhead } from './ClipPreview'

/** jsdom implements neither playback nor currentTime side effects; drive them by hand. */
function stubMedia() {
  let t = 0
  Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
    configurable: true, get: () => t, set: (v: number) => { t = v },
  })
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockImplementation(function (this: HTMLMediaElement) {
    fireEvent.play(this); return Promise.resolve()
  })
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(function (this: HTMLMediaElement) {
    fireEvent.pause(this)
  })
  vi.stubGlobal('requestAnimationFrame', () => 1)
  vi.stubGlobal('cancelAnimationFrame', () => {})
  return { seek: (v: number) => { t = v } }
}

describe('playhead', () => {
  it('measures the elapsed time inside whichever span is playing', () => {
    // As cut: the counter starts at zero at the in-point, not at the clip's start.
    expect(playhead(1.5, 1.0, 2.0, false, 4)).toMatchObject({ elapsed: 0.5, total: 1 })
    expect(playhead(1.5, 1.0, 2.0, false, 4).at).toBeCloseTo(0.5)
    // Whole clip: the same instant is measured against the file.
    expect(playhead(1.5, 1.0, 2.0, true, 4)).toMatchObject({ elapsed: 1.5, total: 4 })
    expect(playhead(0.2, 1.0, 2.0, false, 4).at).toBe(0)        // before the in-point: clamped
    expect(playhead(9, 1.0, 2.0, false, 4).at).toBe(1)
  })
})

describe('ClipPreview', () => {
  beforeEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

  it('starts at the in-point and reports the time and frame as it plays', async () => {
    const m = stubMedia()
    render(<ClipPreview url="/c.mp4" duration={4} fps={16} start={1} end={2} />)
    fireEvent.loadedMetadata(document.querySelector('video')!)
    expect(screen.getByTestId('preview-time')).toHaveTextContent('0.00 / 1.00s · frame 16')

    await userEvent.click(screen.getByTestId('preview-play'))
    m.seek(1.5)
    fireEvent.timeUpdate(document.querySelector('video')!)       // paused-path sampling
    fireEvent.pause(document.querySelector('video')!)
    fireEvent.timeUpdate(document.querySelector('video')!)
    expect(screen.getByTestId('preview-time')).toHaveTextContent('0.50 / 1.00s · frame 24')
  })

  it('switches between what the timeline keeps and what the clip holds', async () => {
    stubMedia()
    render(<ClipPreview url="/c.mp4" duration={4} fps={16} start={1} end={2} />)
    const mode = screen.getByTestId('preview-mode')
    expect(mode).toHaveTextContent('as cut')
    expect(screen.getByTestId('preview-time')).toHaveTextContent('/ 1.00s')   // the kept second

    await userEvent.click(mode)
    expect(mode).toHaveTextContent('whole clip')
    expect(mode).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('preview-time')).toHaveTextContent('/ 4.00s')   // the whole file
  })

  it('marks the kept span on the bar, in the whole clip s own scale', () => {
    stubMedia()
    render(<ClipPreview url="/c.mp4" duration={4} fps={16} start={1} end={2} />)
    const span = screen.getByTestId('preview-bar').firstElementChild as HTMLElement
    expect(span.style.left).toBe('25%')                          // 1s of 4
    expect(span.style.width).toBe('25%')                         // 1s of 4
  })

  it('an untrimmed clip is its own span', () => {
    stubMedia()
    render(<ClipPreview url="/c.mp4" duration={4} fps={16} start={null} end={null} />)
    const span = screen.getByTestId('preview-bar').firstElementChild as HTMLElement
    expect(span.style.left).toBe('0%')
    expect(span.style.width).toBe('100%')
    expect(screen.getByTestId('preview-time')).toHaveTextContent('0.00 / 4.00s')
  })
})
