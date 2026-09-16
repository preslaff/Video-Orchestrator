import { describe, expect, it } from 'vitest'
import { estimateClip, fmtDuration, fmtRange, vramRatio, vramWarning } from './render'

describe('what a size costs', () => {
  it('measures a size against the tested 16 GB budget', () => {
    expect(vramRatio(832, 480, 49)).toBe(1)          // the baseline itself
    expect(vramWarning(832, 480, 49)).toBe('')
    expect(vramWarning(480, 832, 81)).toMatch(/1\.7× the tested budget \(portrait\)/)
    expect(vramWarning(832, 1248, 49)).toMatch(/2\.6× the tested 16 GB budget — expect RAM offload/)
    expect(vramWarning(1280, 720, 81)).toMatch(/3\.8×/)
  })

  it('estimates a clip from the measured local run', () => {
    // the calibration point: 832×480×81 took 13 min
    const base = estimateClip({ width: 832, height: 480, length: 81 })!
    expect(Math.round(base.seconds / 60)).toBe(13)
    expect(base.measured).toBe(true)

    // the size that prompted this: ~28 min measured by hand in ComfyUI at 81 frames
    const big = estimateClip({ width: 832, height: 1248, length: 81 })!
    expect(big.seconds / 60).toBeGreaterThan(25)
    expect(big.seconds).toBeCloseTo(base.seconds * (1248 / 480), -1)   // linear in pixels
  })

  it('uses each cloud workflow′s own measured rate', () => {
    const local = estimateClip({ width: 1280, height: 720, length: 81, workflow: 'animate' })!
    const h100 = estimateClip({ width: 1280, height: 720, length: 81, workflow: 'animate_wan22_fp16' })!
    const ltx = estimateClip({ width: 1280, height: 704, length: 97, workflow: 'animate_ltx25' })!
    expect(Math.round(h100.seconds / 60)).toBe(13)      // 12.5 min measured, rounded
    expect(h100.seconds).toBeLessThan(local.seconds)    // a rented H100 beats the 16 GB card
    expect(Math.round(ltx.seconds)).toBe(28)
    expect(h100.where).toBe('the H100')
  })

  it('falls back to the local rate for an unknown workflow, and says so', () => {
    const e = estimateClip({ width: 832, height: 480, length: 49, workflow: 'something_new' })!
    expect(e.measured).toBe(false)
    expect(e.where).toBe('this GPU')
    expect(estimateClip({ width: 0, height: 480, length: 49 })).toBeNull()
  })

  it('formats durations the way a person reads them', () => {
    expect(fmtDuration(28)).toBe('28 s')
    expect(fmtDuration(400)).toBe('7 min')
    expect(fmtDuration(5400)).toBe('1 h 30 min')
    expect(fmtDuration(0)).toBe('—')
    expect(fmtRange({ seconds: 780, low: 546, high: 1092, where: 'x', measured: true })).toBe('~9–18 min')
    expect(fmtRange({ seconds: 28, low: 20, high: 39, where: 'x', measured: true })).toBe('~20–39 s')
  })
})
