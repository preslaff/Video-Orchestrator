/** SSE contract with the backend (vorch/api/events.py). `id` is per-job and monotonic. */
export const EVENT_TYPES = [
  'status',
  'storyboard',
  'verdict',
  'keyframe',
  'variant',
  'video',
  'postprocess',
  'stills',
  'gate',
  'export',
  'error',
  'end',
] as const

export type EventType = (typeof EVENT_TYPES)[number]

export interface JobEvent {
  id?: number
  type: EventType | string
  job_id: string
  [key: string]: unknown
}

/** Events after which the jobs list (status column) should refresh. */
export const LIST_EVENTS = new Set<string>(['status', 'gate', 'end', 'error', 'export'])

export function describeEvent(e: JobEvent): string {
  switch (e.type) {
    case 'status':
      return `status → ${String(e.status)}`
    case 'storyboard':
      return `storyboard: ${String(e.scenes)} scene(s)`
    case 'verdict':
      return `scene ${String(e.index)} attempt ${String(e.attempt)}: ${String(e.decision)} (score ${String(e.score)})`
    case 'keyframe':
      return `scene ${String(e.index)}: keyframe ready`
    case 'variant':
      return `scene ${String(e.index)}: variant ${Number(e.variant) + 1} ready`
    case 'video':
      return `scene ${String(e.index)}: clip ready`
    case 'postprocess':
      return `scene ${String(e.index)}: post-step ${Number(e.step) + 1} done`
    case 'stills':
      return `scene ${String(e.index)}: ${String(e.variants)} still(s) finished`
    case 'gate':
      return `paused at gate "${String(e.gate)}" — approval needed`
    case 'export':
      return `exported ${Array.isArray(e.files) ? e.files.length : '?'} file(s) → ${String(e.path)}`
    case 'error':
      return `error: ${String(e.error)}`
    case 'end':
      return 'run segment finished'
    default:
      return e.type
  }
}
