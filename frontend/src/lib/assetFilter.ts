import type { AssetOut } from '@/api/types'

/** Everything a person might search an asset by: its label, file name, Gemma's caption,
 * the tags from its description and any legible sign or plaque text. */
function haystack(a: AssetOut): string {
  return [a.label, a.name, a.note_caption, (a.note_tags ?? []).join(' '), a.note_text].join(' ').toLowerCase()
}

/** True when every word of ``query`` occurs somewhere in the asset's searchable text. */
export function matchesQuery(a: AssetOut, query: string): boolean {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean)
  if (!words.length) return true
  const text = haystack(a)
  return words.every((w) => text.includes(w))
}

/** Assets matching the query and carrying every one of ``tags`` (order preserved). */
export function filterAssets(assets: AssetOut[], query: string, tags: string[] = []): AssetOut[] {
  return assets.filter((a) => matchesQuery(a, query) && tags.every((t) => (a.note_tags ?? []).includes(t)))
}

/** The ``n`` most frequent tags among ``assets`` (ties alphabetically), for quick filters. */
export function topTags(assets: AssetOut[], n = 8): string[] {
  const counts = new Map<string, number>()
  for (const a of assets) for (const t of new Set(a.note_tags ?? [])) counts.set(t, (counts.get(t) ?? 0) + 1)
  return [...counts.entries()]
    .filter(([, c]) => c > 1)                      // a tag on one asset filters nothing useful
    .sort((x, y) => y[1] - x[1] || x[0].localeCompare(y[0]))
    .slice(0, n)
    .map(([t]) => t)
}
