import { describe, expect, it } from 'vitest'
import { clampCut, clampTarget, defaultCuts, fmtSpeed, groupTransitions, maxTransition, retimed, speedFor, totalDuration } from './assembly'

describe('assembly helpers', () => {
  it('totalDuration ignores hard cuts', () => {
    expect(totalDuration([2, 2, 2, 2], [{ transition: 'fade', duration: 0.5 }, { transition: 'none', duration: 0.5 }, { transition: 'wipeleft', duration: 0.3 }])).toBe(7.2)
    expect(totalDuration([3.0625], [])).toBe(3.06)
  })
  it('maxTransition is shorter than both clips by a frame, floored and capped', () => {
    expect(maxTransition(3.0625, 3.0625, 16)).toBe(2)
    expect(maxTransition(1, 3, 8)).toBe(0.8)
    expect(maxTransition(0.1, 3, 8)).toBe(0.1)
  })
  it('clampCut keeps hard cuts and bounds durations', () => {
    expect(clampCut({ transition: 'none', duration: 9 }, 1, 1, 8)).toEqual({ transition: 'none', duration: 9 })
    expect(clampCut({ transition: 'fade', duration: 9 }, 1, 3, 8)).toEqual({ transition: 'fade', duration: 0.8 })
    expect(clampCut({ transition: 'fade', duration: 0.01 }, 3, 3, 8)).toEqual({ transition: 'fade', duration: 0.1 })
  })
  it('groupTransitions groups known names and keeps unknown ones in order', () => {
    const g = groupTransitions(['none', 'fade', 'wipeleft', 'zzz', 'aaa'])
    expect(g.map((x) => x.label)).toEqual(['cut', 'fades', 'wipes', 'other'])
    expect(g[3].names).toEqual(['zzz', 'aaa'])
  })
  it('speed helpers', () => {
    expect(speedFor(3.0625, 6.125)).toBe(0.5)
    expect(speedFor(3.0625, null)).toBe(1)
    expect(clampTarget(3.0625, 20)).toBe(12.25)
    expect(clampTarget(3.0625, 0.1)).toBe(0.77)
    expect(clampTarget(3.0625, 4.123456)).toBe(4.12)
    expect(retimed([2, 2], [4, null])).toEqual([4, 2])
    expect(fmtSpeed(1 / 3)).toBe('0.33×')
  })
  it('defaultCuts', () => {
    expect(defaultCuts(3)).toHaveLength(2)
    expect(defaultCuts(1)).toEqual([])
  })
})
