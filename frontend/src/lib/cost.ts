/** Formatting for hosted-model prices, which the API declares as a USD range. */

export type Cost = { min_usd: number; max_usd: number }

/**
 * A price, to as many decimals as it actually needs.
 *
 * These span three orders of magnitude — $0.0058 for GPT-Image at quality=low, $0.143 for
 * Ideogram at quality speed — so a fixed precision either loses a digit or prints noise. Four
 * decimals with the trailing zeros dropped, never fewer than two.
 */
const money = (n: number): string => {
  if (n >= 1) return `$${n.toFixed(2)}`
  const trimmed = n.toFixed(4).replace(/0+$/, '')
  const [whole, frac = ''] = trimmed.split('.')
  return `$${whole}.${frac.padEnd(2, '0')}`
}

/**
 * "$0.03" for a flat price, "$0.04–$0.12" for a range, "" when there is no usable price.
 *
 * Deliberately tolerant of a malformed cost: a price is decoration on a select option, and it
 * must not be able to take the page down. It did once — the API served `{usd}` for a flat price
 * and `{min_usd, max_usd}` for a range, so reading `min_usd` threw on every flat-priced model.
 */
export function costLabel(cost: Cost | null | undefined): string {
  const lo = Number(cost?.min_usd)
  const hi = Number(cost?.max_usd)
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return ''
  return Math.abs(hi - lo) < 1e-9 ? money(lo) : `${money(lo)}–${money(hi)}`
}
