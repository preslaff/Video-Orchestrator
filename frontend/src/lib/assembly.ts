/** Pure helpers for the Assemble panel (mirrors vorch/media.py rules). */

export type CutDraft = { transition: string; duration: number }

export const DEFAULT_TRANSITION = 'fade'
export const DEFAULT_CUT = 0.5
export const MIN_CUT = 0.1
export const MAX_CUT = 2

export const TRANSITION_GROUPS: Record<string, string[]> = {
  cut: ['none'],
  fades: ['fade', 'fadeblack', 'fadewhite', 'fadefast', 'fadeslow', 'fadegrays', 'dissolve', 'pixelize', 'hblur', 'distance'],
  wipes: ['wipeleft', 'wiperight', 'wipeup', 'wipedown', 'wipetl', 'wipetr', 'wipebl', 'wipebr', 'diagtl', 'diagtr', 'diagbl', 'diagbr', 'hlslice', 'hrslice', 'vuslice', 'vdslice'],
  slides: ['slideleft', 'slideright', 'slideup', 'slidedown', 'smoothleft', 'smoothright', 'smoothup', 'smoothdown', 'coverleft', 'coverright', 'coverup', 'coverdown', 'revealleft', 'revealright', 'revealup', 'revealdown', 'squeezeh', 'squeezev'],
  shapes: ['circlecrop', 'rectcrop', 'radial', 'circleopen', 'circleclose', 'vertopen', 'vertclose', 'horzopen', 'horzclose', 'zoomin'],
  wind: ['hlwind', 'hrwind', 'vuwind', 'vdwind'],
}

/** Group the API's transition list for <optgroup>s; unknown names land in "other", API order kept. */
export function groupTransitions(names: string[]): { label: string; names: string[] }[] {
  const out: { label: string; names: string[] }[] = []
  const placed = new Set<string>()
  for (const [label, members] of Object.entries(TRANSITION_GROUPS)) {
    const present = members.filter((m) => names.includes(m))
    if (present.length) {
      out.push({ label, names: present })
      present.forEach((m) => placed.add(m))
    }
  }
  const other = names.filter((n) => !placed.has(n))
  if (other.length) out.push({ label: 'other', names: other })
  return out
}

export const round1 = (x: number) => Math.round(x * 10) / 10

/** Total length = Σ clips − Σ transitions (hard cuts cost nothing). */
export function totalDuration(durations: number[], cuts: CutDraft[]): number {
  const clips = durations.reduce((a, b) => a + b, 0)
  const trans = cuts.reduce((a, c) => a + (c.transition === 'none' ? 0 : c.duration), 0)
  return Math.max(0, Math.round((clips - trans) * 100) / 100)
}

/** Longest allowed transition between two clips: shorter than both by one frame, capped. */
export function maxTransition(prev: number, next: number, fps: number): number {
  const frame = fps > 0 ? 1 / fps : 0
  const limit = Math.min(prev, next) - frame
  return Math.max(MIN_CUT, Math.min(MAX_CUT, Math.floor(limit * 10) / 10))
}

export function clampCut(cut: CutDraft, prev: number, next: number, fps: number): CutDraft {
  if (cut.transition === 'none') return cut
  const max = maxTransition(prev, next, fps)
  return { ...cut, duration: round1(Math.min(max, Math.max(MIN_CUT, cut.duration))) }
}

export function defaultCuts(n: number): CutDraft[] {
  return Array.from({ length: Math.max(0, n - 1) }, () => ({ transition: DEFAULT_TRANSITION, duration: DEFAULT_CUT }))
}

export function fmtSeconds(s: number | null | undefined): string {
  if (s == null) return '?'
  return `${Math.round(s * 100) / 100}s`
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}
