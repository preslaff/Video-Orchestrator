import { describe, expect, it } from 'vitest'
import { costLabel } from './cost'

describe('costLabel', () => {
  it('shows one figure for a flat price and two for a range', () => {
    expect(costLabel({ min_usd: 0.03, max_usd: 0.03 })).toBe('$0.03')
    expect(costLabel({ min_usd: 0.04, max_usd: 0.12 })).toBe('$0.04–$0.12')
  })

  it('keeps enough precision for the cheap models', () => {
    // GPT-Image at quality=low is under a cent; rounding to 2dp would print "$0.01".
    expect(costLabel({ min_usd: 0.0058, max_usd: 0.0228 })).toBe('$0.0058–$0.0228')
    expect(costLabel({ min_usd: 0.143, max_usd: 0.143 })).toBe('$0.143')
    expect(costLabel({ min_usd: 1.5, max_usd: 1.5 })).toBe('$1.50')
    expect(costLabel({ min_usd: 0.1, max_usd: 0.1 })).toBe('$0.10')   // never fewer than 2dp
    expect(costLabel({ min_usd: 0.09, max_usd: 0.09 })).toBe('$0.09')
  })

  it('returns nothing rather than throwing on a missing or malformed price', () => {
    // It threw once, on the real API shape: a flat price arrived as {usd} with no min_usd, and
    // the exception took the whole gate panel down. A price is decoration; it must not do that.
    expect(costLabel(null)).toBe('')
    expect(costLabel(undefined)).toBe('')
    expect(costLabel({ usd: 0.03 } as unknown as { min_usd: number; max_usd: number })).toBe('')
    expect(costLabel({ min_usd: 0.03 } as unknown as { min_usd: number; max_usd: number })).toBe('')
  })
})
