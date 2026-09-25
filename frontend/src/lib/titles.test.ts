import { describe, expect, it } from 'vitest'
import { anchorStyle, blankText, placeOnTimeline, textStyle, visibleAt } from './titles'

describe('titles placement', () => {
  it('agrees with the renderer on where a scene caption lands', () => {
    // The same numbers as tests/test_assembly.py: a caption 1.2s into a clip played at 2x, whose
    // scene starts at 5.0s on the finished film. If these two drift, the preview lies.
    const { start, duration } = placeOnTimeline(blankText({ start: 1.2, duration: 1.0 }), 5.0, 2)
    expect(start).toBeCloseTo(5.0 + 1.2 / 2)
    expect(duration).toBeCloseTo(0.5)
  })

  it('leaves a timeline caption alone', () => {
    const { start, duration } = placeOnTimeline(blankText({ start: 3, duration: 2 }), 0, 1)
    expect(start).toBe(3)
    expect(duration).toBe(2)
  })

  it('shows a caption for exactly its window', () => {
    const t = blankText({ start: 1, duration: 2 })
    expect(visibleAt(t, 0.99)).toBe(false)
    expect(visibleAt(t, 1)).toBe(true)
    expect(visibleAt(t, 2.99)).toBe(true)
    expect(visibleAt(t, 3)).toBe(false)
  })
})

describe('titles styling', () => {
  it('anchors from the alignment, the way the renderer does', () => {
    expect(anchorStyle(blankText({ align: 2, margin_y: 0.1 }))).toMatchObject({ bottom: '10%', textAlign: 'center' })
    expect(anchorStyle(blankText({ align: 7, margin_x: 0.05 }))).toMatchObject({ top: '8%', left: '5%', textAlign: 'left' })
    expect(anchorStyle(blankText({ align: 6 }))).toMatchObject({ top: '50%', textAlign: 'right' })
  })

  it('sizes type as a fraction of the box, so both orientations read the same', () => {
    // 0.048 of the height — 40px at 832, 23px at 480, exactly as titles.py computes it.
    expect(textStyle(blankText({ size: 0.048 }), 832).fontSize).toBe('39.936px')
    expect(textStyle(blankText({ size: 0.048 }), 480).fontSize).toBe('23.04px')
  })

  it('keeps a floor so a tiny preview is still legible', () => {
    expect(textStyle(blankText({ size: 0.001 }), 100).fontSize).toBe('8px')
  })

  it('writes the colour as the person picked it, not as ASS stores it', () => {
    expect(textStyle(blankText({ colour: 'D8B26A' }), 480).color).toBe('#D8B26A')
  })
})
