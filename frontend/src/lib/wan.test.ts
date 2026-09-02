import { describe, expect, it } from 'vitest'
import { clipSeconds, isWanLength, snapToWan } from './wan'
import { keyframeSize, orientation, parseAspect, wanSize } from './aspect'

describe('wan length helper', () => {
  it('validates 4n+1', () => {
    expect([49, 81, 121].every(isWanLength)).toBe(true)
    expect([48, 50, 80, 0].some(isWanLength)).toBe(false)
  })
  it('snaps to the nearest valid length', () => {
    expect(snapToWan(50)).toBe(49)
    expect(snapToWan(52)).toBe(53)
    expect(snapToWan(100)).toBe(101)
    expect(snapToWan(NaN)).toBe(49)
    expect(clipSeconds(49, 16)).toBe(3.06)
  })
})

describe('aspect helpers mirror formats.py', () => {
  it('presets and computed sizes', () => {
    expect(keyframeSize('9:16')).toEqual([576, 1024])
    expect(wanSize('1:1')).toEqual([640, 640])
    expect(wanSize('2:3')).toEqual([480, 720])
    const kf = keyframeSize('21:9')!
    expect(kf[0] % 16).toBe(0)
    expect(kf[1] % 16).toBe(0)
    expect(parseAspect('wide')).toBeNull()
    expect(orientation('9:16')).toBe('portrait')
    expect(orientation('1:1')).toBe('square')
  })
})
