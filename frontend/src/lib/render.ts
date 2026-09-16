/** What a size costs: VRAM headroom and how long a clip takes.
 *
 * Mirrors `vorch/formats.py:vram_warnings` (the 832×480×49 baseline is the tested 16 GB budget),
 * and estimates render time from the runs measured in `docs/modal-backend.md`. Both are shown
 * while a profile's size is being chosen, because that is the moment the cost is decided —
 * 832×1248 is a perfectly legal size that happens to take ~30 minutes a clip on a 16 GB card.
 */

/** 832 × 480 × 49 frames — the local budget every other size is compared against. */
export const BASELINE_PIXEL_FRAMES = 832 * 480 * 49

export const vramRatio = (width: number, height: number, length: number): number =>
  BASELINE_PIXEL_FRAMES > 0 ? (width * height * length) / BASELINE_PIXEL_FRAMES : 0

/** The same soft warning the backend logs, or '' when the size is within budget. */
export function vramWarning(width: number, height: number, length: number): string {
  const r = vramRatio(width, height, length)
  if (r > 2) return `${r.toFixed(1)}× the tested 16 GB budget — expect RAM offload, long renders or OOM`
  if (r > 1.5 && height > width) return `${r.toFixed(1)}× the tested budget (portrait) — watch VRAM`
  return ''
}

/**
 * Seconds per pixel-frame, from measured end-to-end runs (docs/modal-backend.md).
 * Local is a Q5 GGUF cascade on an RTX 5070 Ti; the remote rates include upload and download
 * but not a cold container start.
 */
const RATES: Record<string, { rate: number; where: string }> = {
  // 81 frames at 832×480 in 13 min
  animate: { rate: 780 / (832 * 480 * 81), where: 'this GPU' },
  // 81 frames at 480×832 in 6 min 9 s on an L40S
  animate_wan22_fp8: { rate: 369 / (480 * 832 * 81), where: 'the L40S' },
  // 81 frames at 1280×720 in 12.5 min on an H100
  animate_wan22_fp16: { rate: 750 / (1280 * 720 * 81), where: 'the H100' },
  // distilled two-stage: ~28 s per clip once the container is warm
  animate_ltx25: { rate: 28 / (1280 * 704 * 97), where: 'the H100' },
}

export type RenderEstimate = {
  seconds: number          // the midpoint
  low: number
  high: number
  where: string
  measured: boolean        // false = extrapolated from a different workflow's rate
}

/** Roughly how long one clip takes. Linear in pixels × frames, which holds well enough
 *  until RAM offload starts; past ~2× the budget the true cost climbs faster than this. */
export function estimateClip(
  { width, height, length, workflow }: { width: number; height: number; length: number; workflow?: string },
): RenderEstimate | null {
  if (!(width > 0 && height > 0 && length > 0)) return null
  const known = RATES[workflow ?? 'animate']
  const { rate, where } = known ?? RATES.animate
  const seconds = width * height * length * rate
  return { seconds, low: seconds * 0.7, high: seconds * 1.4, where, measured: !!known }
}

/** "45 s" · "6 min" · "1 h 20 min" — a duration a person can act on. */
export function fmtDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return '—'
  if (seconds < 90) return `${Math.round(seconds)} s`
  const mins = Math.round(seconds / 60)
  if (mins < 90) return `${mins} min`
  const h = Math.floor(mins / 60)
  return `${h} h${mins % 60 ? ` ${mins % 60} min` : ''}`
}

/** "~20–35 min" for the estimate's range, collapsing when both ends read the same. */
export function fmtRange(e: RenderEstimate): string {
  const low = fmtDuration(e.low)
  const high = fmtDuration(e.high)
  if (low === high) return `~${low}`
  const unit = high.replace(/^[\d ]+/, '')            // "9 min" + "18 min" -> "~9–18 min"
  return low.endsWith(unit) ? `~${low.slice(0, -unit.length).trim()}–${high}` : `~${low}–${high}`
}
