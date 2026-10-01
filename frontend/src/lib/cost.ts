/** Formatting for hosted-model prices, which vorch declares as a USD range. */

export type Cost = { min_usd: number; max_usd: number }

const money = (n: number): string =>
  n >= 1 ? `$${n.toFixed(2)}` : `$${n.toFixed(n < 0.01 ? 4 : 3).replace(/0+$/, '').replace(/\.$/, '')}`

/** "$0.03" for a flat price, "$0.04-$0.12" for a range. */
export function costLabel(cost: Cost | null | undefined): string {
  if (!cost) return ''
  const { min_usd: lo, max_usd: hi } = cost
  return Math.abs(hi - lo) < 1e-9 ? money(lo) : `${money(lo)}–${money(hi)}`
}
